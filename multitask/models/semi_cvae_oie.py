from typing import Dict, List, TextIO, Optional, Any, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F
import torch.distributions as distributions

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions


@Model.register('semi_cvae_oie')
class SemiConditionalVAEOIE(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 binary_feature_dim: int,
                 y_feature_dim: int,
                 discriminator: Seq2SeqEncoder, # p(y1|x)
                 encoder: Seq2SeqEncoder, # p(y1|x, y2)
                 decoder: Seq2SeqEncoder, # p(y2|x, y1) or p(y2|y1)
                 y1_ns: str = 'gt',
                 y2_ns: str = 'srl',
                 sample_num: int = 1, # number of samples generated from encoder
                 infer_algo: str = 'reinforce', # algorithm used in encoder optimization
                 # "all" means using both x and y1 to decode y2,
                 # "partial" means only using y1 to decode y2
                 decode_method: str = 'all',
                 beta: float = 1.0, # a coefficient that controls the strength of KL term (similar to beta-VAE)
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 use_crf: bool = False) -> None:
        super(SemiConditionalVAEOIE, self).__init__(vocab, regularizer)
        self._y1_ns = y1_ns
        self._y2_ns = y2_ns
        # label namespace for y1 and y2
        self._y1_label_ns = 'MT_{}_labels'.format(y1_ns)
        self._y2_label_ns = 'MT_{}_labels'.format(y2_ns)
        # task index for y1 and y2
        self._y1_ind = self.vocab.get_token_index(y1_ns, 'task_labels')
        self._y2_ind = self.vocab.get_token_index(y2_ns, 'task_labels')
        # num_class for y1 and y2
        self._y1_num_class = self.vocab.get_vocab_size(self._y1_label_ns)
        self._y2_num_class = self.vocab.get_vocab_size(self._y2_label_ns)
        assert sample_num > 0, 'sample_num should be a positive integer'
        self._sample_num = sample_num
        assert infer_algo in {'reinforce', 'gumbel_softmax'}, 'infer_algo not supported'
        self._infer_algo = infer_algo
        assert decode_method in {'all', 'partial'}, 'decode_method not supported'
        self._decode_method = decode_method
        assert beta >= 0, 'alpha should be non-negative'
        self._beta = beta
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        self.use_crf = use_crf  # TODO: add crf

        # dropout
        self.embedding_dropout = Dropout(p=embedding_dropout)

        # discriminator p(y1|x)
        self.text_field_embedder = text_field_embedder
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.discriminator = discriminator
        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               discriminator.get_input_dim(),
                               'text emb dim + verb indicator emb dim',
                               'discriminator input dim')
        self.disc_y1_proj = TimeDistributed(Linear(discriminator.get_output_dim(), self._y1_num_class))

        # encoder p(y1|x, y2)
        self.y2_embedding = Embedding(self._y2_num_class, y_feature_dim)
        self.encoder = encoder
        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim + y_feature_dim,
                               encoder.get_input_dim(),
                               'text emb dim + verb indicator emb dim + y2 emb dim',
                               'encoder input dim')
        self.enc_y1_proj = TimeDistributed(Linear(encoder.get_output_dim(), self._y1_num_class))

        # decoder p(y2|x, y1) or p(y2|y1)
        self.y1_embedding = Embedding(self._y1_num_class, y_feature_dim)
        self.decoder = decoder
        if self._decode_method == 'all':
            check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim + y_feature_dim,
                                   decoder.get_input_dim(),
                                   'text emb dim + verb indicator emb dim + y1 emb dim',
                                   'decoder input dim')
        elif self._decode_method == 'partial':
            check_dimensions_match(y_feature_dim, decoder.get_input_dim(),
                                   'y1 emb dim', 'decoder input dim')
        self.dec_y2_proj = TimeDistributed(Linear(decoder.get_output_dim(), self._y2_num_class))

        # metrics
        self.y1_span_metric = \
            SpanBasedF1Measure(vocab, tag_namespace=self._y1_label_ns, ignore_classes=['V'])
        self.y1_accuracy = CategoricalAccuracy(top_k=1, tie_break=False)

        initializer(self)


    def vae_encode(self, x_emb: torch.Tensor, y2: torch.LongTensor, mask: torch.LongTensor):
        ''' x,y2 -> y1 '''
        y2_emb = self.y2_embedding(y2)
        # concat_emb must be concatenated by the order of word, verb, y2
        concat_emb = torch.cat([x_emb, y2_emb], -1)
        enc = self.encoder(concat_emb, mask)
        logits = self.enc_y1_proj(enc)
        if self._sample_num == 1:
            # SHAPE: (batch_size, seq_len)
            y1 = distributions.Categorical(logits=logits).sample()
            # SHAPE: (batch_size,)
            # TODO: the sequence_cross_entropy_with_logits an average across tokens
            y1_nll = sequence_cross_entropy_with_logits(logits, y1, mask, average=None)
        else:
            # TODO: beam search
            raise NotImplementedError
        if self._infer_algo == 'reinforce':
            # in REINFORCE, no backpropagation will go through y1
            y1.requires_grad = False
            return x_emb, y2, y1, y1_nll
        elif self._infer_algo == 'gumbel_softmax':
            # TODO: gt softmax
            raise NotImplementedError


    def vae_decode(self, x_emb: torch.Tensor,
                   y1: torch.LongTensor,
                   y2: torch.LongTensor,
                   mask: torch.LongTensor):
        ''' y1 -> y2 or x, y1 -> y2 '''
        y1_emb = self.y1_embedding(y1)
        if self._decode_method == 'all':
            # concat_emb must be concatenated by the order of word, verb, y1
            concat_emb = torch.cat([x_emb, y1_emb], -1)
            enc = self.decoder(concat_emb, mask)
        elif self._decode_method == 'partial':
            enc = self.decoder(y1_emb, mask)
        logits = self.dec_y2_proj(enc)
        y2_nll = sequence_cross_entropy_with_logits(logits, y2, mask, average=None)
        return y2_nll


    def vae_discriminate(self, x_emb: torch.Tensor, mask: torch.LongTensor, y1: torch.LongTensor = None):
        ''' x -> y1 '''
        _, sl = mask.size()

        # discriminator
        enc = self.discriminator(x_emb, mask)
        logits = self.disc_y1_proj(enc)
        cp = F.softmax(logits.view(-1, self._y1_num_class), dim=-1).view([-1, sl, self._y1_num_class])

        if y1 is not None:
            # loss
            y1_nll = sequence_cross_entropy_with_logits(
                logits, y1, mask, average=None, label_smoothing=self._label_smoothing)
            return logits, cp, y1_nll

        return logits, cp, None


    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                task_labels: torch.LongTensor,
                weight: torch.FloatTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        output_dict = {}

        # emb text
        text_emb = self.embedding_dropout(self.text_field_embedder(tokens))
        verb_emb = self.binary_feature_embedding(verb_indicator.long())
        concat_emb = torch.cat([text_emb, verb_emb], -1)
        bs, sl, es = concat_emb.size()
        mask = get_text_field_mask(tokens)

        sup_unsup_loss = 0.0

        # supervised
        sup_tm = task_labels.eq(self._y1_ind)
        if sup_tm.sum().item() > 0:  # skip if no supervised data exists
            sup_x = concat_emb.masked_select(sup_tm.view(-1, 1, 1)).view(-1, sl, es)
            sup_mask = mask.masked_select(sup_tm.view(-1, 1)).view(-1, sl)
            sup_weight = weight.masked_select(sup_tm)
            sup_y1 = None
            if tags is not None:
                sup_y1 = tags.masked_select(sup_tm.view(-1, 1)).view(-1, sl)
            sup_y1_logits, sup_y1_cp, sup_y1_nll = self.vae_discriminate(sup_x, sup_mask, sup_y1)
            output_dict['logits'] = sup_y1_logits
            output_dict['class_probabilities'] = sup_y1_cp
            output_dict['mask'] = sup_mask
            if tags is not None:
                sup_unsup_loss += (sup_y1_nll * sup_weight).sum()
                # metrics
                if not self.ignore_span_metric:
                    self.y1_span_metric(sup_y1_cp, sup_y1, sup_mask)
                self.y1_accuracy(sup_y1_logits, sup_y1, sup_mask)

        # unsupervised
        unsup_tm = task_labels.eq(self._y2_ind)
        if unsup_tm.sum().item() > 0 and tags is not None: # skip if no unsupervised data exists
            upsup_x = concat_emb.masked_select(unsup_tm.view(-1, 1, 1)).view(-1, sl, es)
            unsup_y2 = tags.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_mask = mask.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_weight = weight.masked_select(unsup_tm)

            # sample
            # need to replace the old "upsup_x" and "unsup_y2" because there
            # could be multiple random samples for each data point
            unsup_x, unsup_y2, unsup_y1, enc_y1_nll = self.vae_encode(upsup_x, unsup_y2, unsup_mask)

            # decoder loss (reconstruction loss)
            y2_nll = self.vae_decode(unsup_x, unsup_y1, unsup_y2, unsup_mask)

            # discriminator loss
            _, _, dis_y1_nll = self.vae_discriminate(unsup_x, unsup_mask, unsup_y1)  # prior

            # kl divergence
            kl = self._beta * (dis_y1_nll - enc_y1_nll)  # beta * log(q(y1|x,y2) / p(y1|x))

            # encoder loss
            # only REINFORCE needs manually calculate encoder loss,
            # while gumbel_softmax could directly do bp
            if self._infer_algo == 'reinforce':
                # TODO: clip reward?
                encoder_reward = -y2_nll - kl  # log(p(y2|y1)) - log(q(y1|x,y2) / p(y1|x))
                y1_nll_with_reward = enc_y1_nll * (encoder_reward.detach() - 1) # be mindful of the 1

            # overall loss
            if self._infer_algo == 'reinforce':
                decoder_loss = (y2_nll * unsup_weight).sum()
                encoder_loss = (y1_nll_with_reward * unsup_weight).sum()
                discriminator_loss = (dis_y1_nll * unsup_weight).sum()
                sup_unsup_loss += decoder_loss + encoder_loss + discriminator_loss
            elif self._infer_algo == 'gumbel_softmax':
                vae_loss = y2_nll + kl
                sup_unsup_loss += vae_loss

        if tags is not None:
            #output_dict['loss'] = sup_unsup_loss / ((mask.sum(1) > 0).float().sum() + 1e-13)
            output_dict['loss'] = sup_unsup_loss / (weight.sum() + 1e-13) # TODO: use both weight and mask?

        return output_dict


    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        all_predictions = output_dict['class_probabilities']
        sequence_lengths = get_lengths_from_binary_sequence_mask(output_dict["mask"]).data.tolist()

        if all_predictions.dim() == 3:
            predictions_list = [all_predictions[i].detach().cpu() for i in range(all_predictions.size(0))]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        all_probs = []
        transition_matrix = self.get_viterbi_pairwise_potentials()
        for predictions, length in zip(predictions_list, sequence_lengths):
            max_likelihood_sequence, score = viterbi_decode(predictions[:length], transition_matrix)
            probs = [predictions[i, max_likelihood_sequence[i]].numpy().tolist()
                     for i in range(len(max_likelihood_sequence))]
            all_probs.append(probs)
            tags = [self.vocab.get_token_from_index(x, namespace='MT_gt_labels')
                    for x in max_likelihood_sequence] # TODO: add more task and avoid "gt"
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        output_dict['probs'] = all_probs
        return output_dict


    def get_metrics(self, reset: bool = False):
        metric_dict = {}
        # span metric
        sm = {'{}_{}'.format('y1', x): y for x, y in
              self.y1_span_metric.get_metric(reset=reset).items()
              if 'f1-measure-overall' in x}
        metric_dict.update(sm)
        # accuracy
        metric_dict['y1_accuracy'] = self.y1_accuracy.get_metric(reset=reset)
        return metric_dict


    def get_viterbi_pairwise_potentials(self):
        """
        Generate a matrix of pairwise transition potentials for the BIO labels.
        The only constraint implemented here is that I-XXX labels must be preceded
        by either an identical I-XXX tag or a B-XXX tag. In order to achieve this
        constraint, pairs of labels which do not satisfy this constraint have a
        pairwise potential of -inf.

        Returns
        -------
        transition_matrix : torch.Tensor
            A (num_labels, num_labels) matrix of pairwise potentials.
        """
        # TODO: add more task and avoid "gt"
        all_labels = self.vocab.get_index_to_token_vocabulary('MT_gt_labels')
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float('-inf')
        return transition_matrix