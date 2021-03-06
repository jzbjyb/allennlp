from typing import Dict, List, TextIO, Optional, Any, Tuple

from overrides import overrides
import torch
from torch.nn.modules import Linear, Dropout
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions

from allennlp.models.base import BaseModel


@Model.register("srl_mt")
class SemanticRoleLabelerMultiTask(BaseModel):
    """
    This model performs semantic role labeling using BIO tags using Propbank semantic roles.
    Specifically, it is an implementation of `Deep Semantic Role Labeling - What works
    and what's next <https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf>`_ .

    This implementation is effectively a series of stacked interleaved LSTMs with highway
    connections, applied to embedded sequences of words concatenated with a binary indicator
    containing whether or not a word is the verbal predicate to generate predictions for in
    the sentence. Additionally, during inference, Viterbi decoding is applied to constrain
    the predictions to contain valid BIO sequences.

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    encoder : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    binary_feature_dim : int, required.
        The dimensionality of the embedding of the binary verb predicate features.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    label_smoothing : ``float``, optional (default = 0.0)
        Whether or not to use label smoothing on the labels when computing cross entropy loss.
    ignore_span_metric: ``bool``, optional (default = False)
        Whether to calculate span loss, which is irrelevant when predicting BIO for Open Information Extraction.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,  # the base encoder shared across tasks
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 more_encoder: Seq2SeqEncoder = None,  # the additional encoder shared across tasks
                 task_encoder: Dict[str, Seq2SeqEncoder] = None,  # task-specific encoder
                 encoder_requires_grad: bool = True,
                 more_encoder_requires_grad: bool = True,
                 task_encoder_requires_grad: Dict[str, bool] = None,
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 # Whether to use the tag results of (viterbi) decoding to compute span metric,
                 # which is more consistent with test-time performance.
                 decode_span_metric: bool = True,
                 use_crf: bool = False) -> None:
        # TODO: avoid "gt"
        super(SemanticRoleLabelerMultiTask, self).__init__('MT_gt_labels', vocab, regularizer)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        self.decode_span_metric = decode_span_metric
        self.use_crf = use_crf  # whether to use CRF decoding

        # task-agnostic components
        self.text_field_embedder = text_field_embedder
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = encoder_requires_grad
        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               'text embedding dim + verb indicator embedding dim',
                               'encoder input dim')
        self.more_encoder = more_encoder
        if self.more_encoder is not None:
            for param in self.more_encoder.parameters():
                param.requires_grad = more_encoder_requires_grad

        # task-related components
        self.task_encoder = task_encoder
        self.num_tasks = self.vocab.get_vocab_size('task_labels')
        self.ind_task_map = self.vocab.get_index_to_token_vocabulary('task_labels').items()
        for task_ind, task_name in self.ind_task_map:
            # basic task properties
            label_ns = 'MT_{}_labels'.format(task_name)
            setattr(self, '{}_num_classes'.format(task_name),
                    self.vocab.get_vocab_size(label_ns))
            num_classes = getattr(self, '{}_num_classes'.format(task_name))
            # task encoder
            if self.task_encoder is not None and task_name in self.task_encoder:
                print('{} : {}'.format(task_name, task_encoder_requires_grad[task_name]))
                setattr(self, '{}_task_encoder'.format(task_name), self.task_encoder[task_name])
                for param in getattr(self, '{}_task_encoder'.format(task_name)).parameters():
                    param.requires_grad = task_encoder_requires_grad[task_name]
            else:
                setattr(self, '{}_task_encoder'.format(task_name), None)
            # task projection
            setattr(self, '{}_tag_projection_layer'.format(task_name),
                    TimeDistributed(Linear(self.encoder.get_output_dim(), num_classes)))
            if self.use_crf:
                labels = self.vocab.get_index_to_token_vocabulary(label_ns)
                constraints = allowed_transitions('BIO', labels)
                setattr(self, '{}_crf'.format(task_name), ConditionalRandomField(
                    num_classes, constraints=constraints, include_start_end_transitions=False))
            # metrics (track the performance of different tasks separately)
            setattr(self, '{}_span_metric'.format(task_name),
                    SpanBasedF1Measure(vocab, tag_namespace=label_ns, ignore_classes=['V']))
            setattr(self, '{}_accuracy'.format(task_name), CategoricalAccuracy(top_k=1, tie_break=False))

        initializer(self)


    def task_agnostic_comp(self, tokens: Dict[str, torch.LongTensor],
                           verb_indicator: torch.LongTensor,
                           task_labels: torch.LongTensor,
                           weight: torch.FloatTensor,
                           tags: torch.LongTensor = None,
                           metadata: List[Dict[str, Any]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        text_emb = self.embedding_dropout(self.text_field_embedder(tokens))
        verb_emb = self.binary_feature_embedding(verb_indicator.long())
        concat_emb = torch.cat([text_emb, verb_emb], -1)

        mask = get_text_field_mask(tokens)

        enc = self.encoder(concat_emb, mask)
        if self.more_encoder is not None:
            enc = self.more_encoder(enc, mask)

        return enc, mask


    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                task_labels: torch.LongTensor,
                weight: torch.FloatTensor,
                tags: torch.LongTensor = None,
                tag_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        # task-agnostic components
        enc, mask = self.task_agnostic_comp(
            tokens, verb_indicator, task_labels, weight, tags, metadata)

        # task-related components
        output_dict = {}
        all_loss = 0.0
        for task_ind, task_name in self.ind_task_map:
            # get samples of this task
            bs, sl, hs = enc.size()
            task_mask = task_labels.eq(task_ind)
            # SHAPE: (task_batch_size, seq_len, hidden_size)
            t_enc = enc.masked_select(task_mask.view(-1, 1, 1)).view(-1, sl, hs)
            if t_enc.size(0) == 0:
                continue # skip when the task is not found in the current batch
            # SHAPE: (task_batch_size, seq_len)
            t_mask = mask.masked_select(task_mask.view(-1, 1)).view(-1, sl)
            # SHAPE: (task_batch_size,)
            t_weight = weight.masked_select(task_mask)
            t_bs, t_sl, = t_mask.size()
            if t_weight.ne(0).sum().item() == 0:  # all weights are zero
                continue

            # get metadata of the current task
            num_classes = getattr(self, '{}_num_classes'.format(task_name))
            label_ns = 'MT_{}_labels'.format(task_name)

            if getattr(self, '{}_task_encoder'.format(task_name)) is not None: # task encoder
                t_enc = getattr(self, '{}_task_encoder'.format(task_name))(t_enc, t_mask)

            # prediction
            t_logits = getattr(self, '{}_tag_projection_layer'.format(task_name))(t_enc)

            output_dict['logits'] = t_logits # TODO
            output_dict['mask'] = t_mask # TODO
            # calculate prob
            if self.use_crf:
                t_pred_tags = [x for x, y in self.crf.viterbi_tags(t_logits, t_mask)]
                output_dict['tags'] = t_pred_tags # TODO
                # pseudo class prob
                t_pred_tags_tensor = t_mask * 0
                for i, inst_tags in enumerate(t_pred_tags_tensor):
                    for j, tag_id in enumerate(inst_tags):
                        t_pred_tags_tensor[i, j] = tag_id
                pll = self.crf(t_logits, t_pred_tags_tensor, t_mask, agg=None)
                pll /= t_mask.sum(1).float() + 1e-13  # average over words in each seq
                epll = torch.exp(pll)
                t_cp = t_logits * 0.
                for i, inst_tags in enumerate(t_pred_tags_tensor):
                    for j, tag_id in enumerate(inst_tags):
                        t_cp[i, j, tag_id] = epll[i]
                output_dict['class_probabilities'] = t_cp # TODO
            else:
                t_cp = F.softmax(t_logits.view(-1, num_classes), dim=-1).view([t_bs, t_sl, num_classes])
                output_dict['class_probabilities'] = t_cp # TODO

            if tags is not None:
                # SHAPE: (task_batch_size, seq_len)
                t_tags = tags.masked_select(task_mask.view(-1, 1)).view(-1, sl)
                # calculate loss
                if self.use_crf:
                    ll = self.crf(t_logits, t_tags, t_mask, agg=None)
                    ll /= t_mask.sum(1).float() + 1e-13 # average over words in each seq
                else:
                    ll = -sequence_cross_entropy_with_logits(
                        t_logits, t_tags, t_mask, average=None, label_smoothing=self._label_smoothing)
                t_loss = -(ll * t_weight).sum()
                #all_loss += t_loss
                all_loss += t_loss / ((t_mask.sum(1) > 0).float().sum() + 1e-13) # task average
                # calculate metrics
                if not self.ignore_span_metric:
                    if self.decode_span_metric and task_name == 'gt':  # TODO: avoid "gt"
                        od = {'class_probabilities': t_cp, 'mask': t_mask}
                        getattr(self, '{}_span_metric'.format(task_name))(
                            self.get_decode_pseudo_class_prob(od), t_tags, t_mask)
                    else:
                        getattr(self, '{}_span_metric'.format(task_name))(
                            t_cp, t_tags, t_mask)
                if self.use_crf:
                    getattr(self, '{}_accuracy'.format(task_name))(t_cp, t_tags, t_mask)
                else:
                    getattr(self, '{}_accuracy'.format(task_name))(t_logits, t_tags, t_mask)

        if tags is not None:
            # TODO: which version to use?
            # overall average by the number of samples
            #output_dict['loss'] = all_loss / ((mask.sum(1) > 0).float().sum() + 1e-13)
            # overall average by sample weights
            #output_dict['loss'] = all_loss / (weight.sum() + 1e-13)
            # use the loss as is, which is already average for each task
            output_dict['loss'] = all_loss
        if metadata is not None:
            words, verbs = zip(*[(x['words'], x['verb']) for x in metadata])
            output_dict['words'] = list(words)
            output_dict['verb'] = list(verbs)
        return output_dict


    def get_metrics(self, reset: bool = False):
        metric_dict = {}
        # span metric
        if not self.ignore_span_metric:
            for task_ind, task_name in self.ind_task_map:
                sm = getattr(self, '{}_span_metric'.format(task_name)).get_metric(reset=reset)
                sm = {'{}_{}'.format(task_name, x): y
                      for x, y in sm.items() if '-overall' in x}
                metric_dict.update(sm)
        # accuracy
        for task_ind, task_name in self.ind_task_map:
            metric_dict['{}_accuracy'.format(task_name)] = \
                getattr(self, '{}_accuracy'.format(task_name)).get_metric(reset=reset)
        return metric_dict
