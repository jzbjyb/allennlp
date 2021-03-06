import logging
from typing import Dict, List, TextIO, Optional, Any, Tuple

from overrides import overrides
import torch
import torch.nn as nn
from torch.nn.modules import Linear, Dropout, Dropout2d
import torch.nn.functional as F
import torch.distributions as distributions
from torch.autograd import Variable

import numpy as np

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode, n_best_viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.models.base import BaseModel

from multitask.metrics import MultipleLoss
from multitask.modules import PostElmo, CVAEEnDeCoder
from multitask.modules.util import share_weights, modify_req_grad, Rule

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def masked_select_dict_on_first_dim(dicts: Dict[str, torch.Tensor], mask: torch.Tensor):
    result = {}
    for k in dicts:
        os = dicts[k].size()[1:]  # original shape except for the first dim
        ns = [1] * len(dicts[k].size())
        ns[0] = -1
        kmask = mask.view(ns)  # reshape mask to make it have the same dimensions as the tensor
        result[k] = dicts[k].masked_select(kmask).view((-1,) + os)
    return result


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def gumbel_softmax_multiple(logits,  # SHAPE: (batch_size, seq_len, num_class)
                            temperature,
                            num_sample=1):
    # SHAPE: (num_sample, batch_size, seq_len, num_class)
    y = torch.stack([gumbel_softmax_sample(logits, temperature) for _ in range(num_sample)])
    # SHAPE: _, (num_sample, batch_size, seq_len)
    _, y_ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard.scatter_(-1, y_ind.unsqueeze(-1), 1)
    return (y_hard - y).detach() + y, y_ind


@Model.register('semi_cvae_oie')
class SemiConditionalVAEOIE(BaseModel):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 binary_feature_dim: int,
                 y_feature_dim: int,
                 discriminator: Seq2SeqEncoder, # p(y1|x)
                 encoder: Seq2SeqEncoder, # p(y1|x, y2)
                 decoder: Seq2SeqEncoder, # p(y2|x, y1) or p(y2|y1)
                 lang_model: Seq2SeqEncoder = None,  # oie lang model used as reward
                 share_param: bool = False,  # whether to share the params in three components
                 # when encoder is the same as prior, decoder is a late_add model
                 encoder_same_as_pior = False,
                 fix_prior = False,
                 enc_to_disc_share: Dict[str, str] = None,
                 dec_to_disc_share: Dict[str, str] = None,
                 dec_skip: List[str] = None,
                 enc_skip: List[str] = None,
                 y1_ns: str = 'gt',
                 y2_ns: str = 'srl',
                 kl_method: str = 'sample',  # whether to use sample to approximate kl or calculate exactly
                 sample_num: int = 1, # number of samples generated from encoder
                 sample_algo: str = 'beam',  # beam search or random
                 infer_algo: str = 'reinforce', # algorithm used in encoder optimization
                 use_rule: str = None,  # whether to use rule
                 baseline: str = 'wb',  # baseline used in reinforce
                 clip_reward: float = None,  # avoid very small learning signal
                 temperature: float = 1.0,  # temperature in gumbel softmax
                 # "all" means using both x and y1 to decode y2,
                 # "partial" means only using y1 to decode y2
                 decode_method: str = 'all',
                 beta: float = 1.0, # a coefficient that controls the strength of KL term (similar to beta-VAE)
                 embedding_dropout: float = 0.0,
                 word_dropout: float = 0.0,  # dropout a word
                 unsup_loss_type: str = 'all',  # how to compute unsupervised loss
                 unsup_loss_weight: float = 1.0,  # weight of the unsupervised loss
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 # Whether to use the tag results of (viterbi) decoding to compute span metric,
                 # which is more consistent with test-time performance.
                 decode_span_metric: bool = True,
                 use_crf: bool = False,
                 debug: bool = False) -> None:
        # TODO: avoid "gt"
        super(SemiConditionalVAEOIE, self).__init__('MT_gt_labels', vocab, regularizer)
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
        assert kl_method in {'sample', 'exact'}
        self._kl_method = kl_method
        assert sample_num > 0, 'sample_num should be a positive integer'
        self._sample_num = sample_num
        assert sample_algo in {'beam', 'random'}, 'sample_algo not supported'
        self._sample_algo = sample_algo
        assert infer_algo in {'reinforce', 'gumbel_softmax'}, 'infer_algo not supported'
        self._infer_algo = infer_algo
        if infer_algo == 'gumbel_softmax' and kl_method == 'sample':
            # gumbel softmax with sampled kl divergence is tricky.
            # the grad of cross entropy loss in q(y1|x, y2) and p(y1|x) need to be bp to both logits and targets.
            # TODO: gumbel softmax with sampled kl
            raise NotImplementedError
        self._use_rule = use_rule
        if use_rule:
            self.rule = Rule(vocab, y1_ns=self._y1_label_ns, y2_ns=self._y2_label_ns)
        assert baseline in {'wb', 'mean', 'non'}, 'baseline not supported'
        self._baseline = baseline
        if clip_reward is not None:
            assert clip_reward >= 0, 'clip_reward should be nonnegative'
        self._clip_reward = clip_reward
        self._temperature = temperature
        self._decode_method = decode_method
        assert beta >= 0, 'alpha should be non-negative'
        self._beta = beta
        assert unsup_loss_type in {'all', 'only_disc'}, 'unsup_loss_type not supported'
        self._unsup_loss_type = unsup_loss_type
        self._unsup_loss_weight = unsup_loss_weight
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        self.decode_span_metric = decode_span_metric
        self.use_crf = use_crf  # TODO: add crf
        self.debug = debug
        self._use_post_elmo = False
        if hasattr(text_field_embedder, 'token_embedder_elmo'):
            elmo_token_embedder = getattr(text_field_embedder, 'token_embedder_elmo')
            if elmo_token_embedder.get_use_post_elmo():
                self._use_post_elmo = True
        # use for sampling process at training time
        self.register_buffer('_pairwise_potential', self.get_viterbi_pairwise_potentials())

        # dropout
        self.embedding_dropout = Dropout(p=embedding_dropout)  # TODO: where to do emb dropout?
        self.word_dropout = Dropout2d(p=word_dropout)

        # text field embedder (handle elmo)
        self.text_field_embedder = text_field_embedder
        if self._use_post_elmo:
            # use separate elmo scalars
            self.disc_post_elmo_ = PostElmo(**elmo_token_embedder.get_post_elmo_params())
            self.enc_post_elmo_ = PostElmo(**elmo_token_embedder.get_post_elmo_params())
            self.dec_post_elmo_ = PostElmo(**elmo_token_embedder.get_post_elmo_params())
            self.disc_post_elmo = lambda x: self.disc_post_elmo_(**x)
            self.enc_post_elmo = lambda x: self.enc_post_elmo_(**x)
            self.dec_post_elmo = lambda x: self.dec_post_elmo_(**x)
            # use separate binary embeddings
            self.disc_bin_emb = Embedding(2, binary_feature_dim)
            self.enc_bin_emb = Embedding(2, binary_feature_dim)
            self.dec_bin_emb = Embedding(2, binary_feature_dim)
        else:
            # use the same elmo scalars
            self.disc_post_elmo = lambda x: x
            self.enc_post_elmo = lambda x: x
            self.dec_post_elmo = lambda x: x
            # use the same binary embedding
            self.binary_feature_embedding = Embedding(2, binary_feature_dim)
            self.disc_bin_emb = self.binary_feature_embedding
            self.enc_bin_emb = self.disc_bin_emb
            self.dec_bin_emb = self.disc_bin_emb

        # discriminator p(y1|x)
        self.discriminator = discriminator
        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               discriminator.get_input_dim(),
                               'text emb dim + verb indicator emb dim',
                               'discriminator input dim')
        self.disc_y1_proj = TimeDistributed(Linear(discriminator.get_output_dim(), self._y1_num_class))

        # encoder p(y1|x, y2)
        self.y2_embedding = Embedding(self._y2_num_class, y_feature_dim)
        self.encoder = encoder
        self.enc_y1_proj = TimeDistributed(Linear(encoder.get_output_dim(), self._y1_num_class))
        # encoder reward estimation
        if self._baseline == 'wb':
            self.w = nn.Parameter(torch.ones(1))
            self.b = nn.Parameter(torch.zeros(1))

        # decoder p(y2|x, y1) or p(y2|y1)
        self.y1_embedding = Embedding(self._y1_num_class, y_feature_dim)
        self.decoder = decoder
        # TODO: different binary emb size for decoder?
        self.dec_y2_proj = TimeDistributed(Linear(decoder.get_output_dim(), self._y2_num_class))

        # language model
        if lang_model is not None:
            # TODO: make sure that <s> and </s> are the last two tags
            self._lang_start = self._y1_num_class
            self._lang_end = self._y1_num_class + 1
            self.lang_embedding = Embedding(self._y1_num_class + 2, y_feature_dim)
            self.lang_model = lang_model
            self.lang_proj = TimeDistributed(Linear(lang_model.get_output_dim(), self._y1_num_class + 2))

        # share parameters
        self.encoder_same_as_pior = encoder_same_as_pior
        self.fix_prior = fix_prior
        if fix_prior and encoder_same_as_pior:
            raise ValueError('when prior is fixed, it can not be the same as the encoder')
        if self._use_post_elmo and share_param:
            raise ValueError('when share_param is true, use_post_elmo should be disabled')
        if share_param and not encoder_same_as_pior and not fix_prior:
            # share the base layers in three components
            if not isinstance(self.encoder, CVAEEnDeCoder) or not isinstance(self.decoder, CVAEEnDeCoder):
                raise NotImplementedError
            logging.info('sharing parameters in base layers')
            # share x_encoder in encoder, decoder, and discriminator
            # TODO: explicitly specify the name of the parameters?
            share_weights(self.encoder.x_encoder, self.discriminator)
            share_weights(self.decoder.x_encoder, self.discriminator)
        elif share_param and encoder_same_as_pior:
            if not isinstance(self.encoder, CVAEEnDeCoder) or not isinstance(self.decoder, CVAEEnDeCoder):
                raise NotImplementedError
            if self.encoder.combine_method != 'only_x' \
                    or self.decoder.combine_method != 'late_add':
                raise ValueError('the encoder should be only_x and the decoder should be late_add'
                                 ' when encoder_same_as_pior is set to true')
            logging.info('reuse discriminator as encoder')
            share_weights(self.encoder.x_encoder, self.discriminator)  # encoder and disc are the same
            share_weights(self.enc_y1_proj, self.disc_y1_proj)  # encoder and disc are the same
            share_weights(self.decoder.x_encoder, self.discriminator,
                          skip=dec_skip)  # decoder and disc share the base layers
            modify_req_grad(self.decoder.x_encoder, False, skip=dec_skip)  # fix base layer
            modify_req_grad(self.binary_feature_embedding, False, skip=dec_skip)  # fix binary embedding
        elif share_param and fix_prior:
            if not isinstance(self.encoder, CVAEEnDeCoder) or not isinstance(self.decoder, CVAEEnDeCoder):
                raise NotImplementedError
            if self.encoder.combine_method != 'only_x' \
                    or self.decoder.combine_method != 'late_add':
                raise ValueError('the encoder should be only_x and the decoder should be late_add'
                                 ' when fix_prior is set to true')
            logging.info('fix discriminator')
            share_weights(self.encoder.x_encoder, self.discriminator,
                          skip=enc_skip)  # encoder and disc share the base layers
            share_weights(self.decoder.x_encoder, self.discriminator,
                          skip=dec_skip)  # decoder and disc share the base layers
            modify_req_grad(self.discriminator, False)  # fix discriminator
            modify_req_grad(self.disc_y1_proj, False)  # fix discriminator
            modify_req_grad(self.binary_feature_embedding, False)  # fix binary embedding

        # metrics
        self.y1_span_metric = \
            SpanBasedF1Measure(vocab, tag_namespace=self._y1_label_ns, ignore_classes=['V'])
        self.y1_accuracy = CategoricalAccuracy(top_k=1, tie_break=False)
        extra_loss = []
        if fix_prior:
            extra_loss = ['enc_sup_l']
        if infer_algo == 'reinforce':
            self.y1_multi_loss = MultipleLoss(['sup_l', 'enc_l', 'dec_l', 'disc_l', 'base_l'] + extra_loss)
        elif infer_algo == 'gumbel_softmax':
            self.y1_multi_loss = MultipleLoss(['sup_l', 'elbo_l', 'recon_l', 'kl_l'] + extra_loss)

        if self.debug:
            self._sample_num = 1

        initializer(self)


    def language_model(self,
                       y1: torch.LongTensor,  # SHAPE: (beam_size, batch_size, seq_len)
                       mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                       ) -> torch.Tensor:
        # add start and end symbol
        inp_y1 = F.pad(y1, [1, 0], 'constant', self._lang_start)
        out_y1 = y1 * mask.unsqueeze(0) + (1 - mask.unsqueeze(0)) * self._lang_end
        out_y1 = F.pad(out_y1, [0, 1], 'constant', 0)
        mask = F.pad(mask, [1, 0], 'constant', 1)

        beam_size, batch_size, seq_len = inp_y1.size()
        inp_y1 = inp_y1.view(-1, seq_len)
        out_y1 = out_y1.view(-1, seq_len)
        mask = mask.repeat(beam_size, 1)

        inp_y1_emb = self.lang_embedding(inp_y1)
        enc = self.lang_model(None, None, inp_y1_emb, mask)
        logits = self.lang_proj(enc)

        lang_nll = sequence_cross_entropy_with_logits(
            logits, out_y1, mask, average='sum', label_smoothing=self._label_smoothing)
        lang_nll = lang_nll.view(beam_size, batch_size)
        return lang_nll


    def beam_search_sample(self,
                           logits: torch.Tensor,  # SHAPE: (batch_size, max_seq_len, num_class)
                           mask: torch.LongTensor,  # SHAPE: (batch_size, max_seq_len)
                           beam_size: int = 5) -> torch.Tensor:
        batch_size, max_seq_len, _ = logits.size()
        lp = F.log_softmax(logits, dim=-1)  # log prob
        samples = []
        for i in range(batch_size):
            seq_len = mask[i].sum().item() # length of this sample
            # SHAPE: (beam_size, seq_len)
            paths, _ = n_best_viterbi_decode(
                lp[i][:seq_len], self._pairwise_potential, n_best=beam_size)
            if paths.size(0) != beam_size:
                raise Exception('the number of samples is not equal to beam size')
            # SHPAE: (beam_size, max_seq_len)
            paths = F.pad(paths, [0, max_seq_len - seq_len], 'constant', 0)
            samples.append(paths)
        # SHAPE: (beam_size, batch_size, max_seq_len)
        return torch.stack(samples, 1)


    def vae_encode(self,
                   t_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, t_emb_size)
                   v_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, v_emb_size)
                   y2: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                   mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                   beam_size: int = 1,
                   y1_ground_truth: torch.LongTensor = None):  # SHAPE: (batch_size, seq_len)
        ''' x,y2 -> y1 '''
        if y2 is None:  # not use y2 in encoder
            y2_emb = None
        else:
            y2_emb = self.y2_embedding(y2)
        enc = self.encoder(t_emb, v_emb, y2_emb, mask)
        logits = self.enc_y1_proj(enc)

        if y1_ground_truth is not None:  # no sampling
            y1_nll = sequence_cross_entropy_with_logits(
                logits, y1_ground_truth, mask, average='sum', label_smoothing=self._label_smoothing)
            return logits, None, None, y1_nll

        if self._sample_algo == 'random':  # random sample from categorical distribution
            if self._infer_algo == 'reinforce':
                # SHAPE: (beam_size, batch_size, seq_len)
                y1 = distributions.Categorical(logits=logits).sample([beam_size])
            elif self._infer_algo == 'gumbel_softmax':
                # SHAPE: (beam_size, batch_size, seq_len, num_class), (beam_size, batch_size, seq_len)
                if self.debug:
                    y1_oh, y1 = gumbel_softmax_multiple(logits, self._temperature, 1)
                    _, y1 = logits.max(-1)
                    y1 = y1.unsqueeze(0)
                else:
                    y1_oh, y1 = gumbel_softmax_multiple(logits, self._temperature, beam_size)
        elif self._sample_algo == 'beam':  # beam search (deterministic)
            if self._infer_algo == 'reinforce':
                # SHAPE: (beam_size, batch_size, seq_len)
                y1 = self.beam_search_sample(logits, mask, beam_size=beam_size)
            elif self._infer_algo == 'gumbel_softmax':
                # TODO: gumbel softmax for beam search?
                raise NotImplementedError

        # SHAPE: (beam_size, batch_size)
        y1.requires_grad = False
        y1_nll = torch.stack([sequence_cross_entropy_with_logits(
            logits, y1[i], mask, average='sum') for i in range(beam_size)])

        if self._infer_algo == 'reinforce':
            return logits, y1, None, y1_nll
        elif self._infer_algo == 'gumbel_softmax':
            return logits, y1, y1_oh, y1_nll


    def rule_decode(self,
                    y1: torch.LongTensor,  # SHAPE: (beam_size, batch_size, seq_len)
                    y2: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                    y2_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                    mask: torch.LongTensor):
        ''' use rule to evaluate the goodness of y1 '''
        reward = getattr(self.rule, self._use_rule)(y1, y2, mask)
        return reward


    def vae_decode(self,
                   t_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, t_emb_size)
                   v_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, v_emb_size)
                   # SHAPE: (beam_size, batch_size, seq_len) or (beam_size, batch_size, seq_len, num_class)
                   y1: torch.LongTensor,
                   y2: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                   y2_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                   mask: torch.LongTensor):
        ''' y1 -> y2 or x, y1 -> y2 '''
        beam_size, batch_size, seq_len = y1.size()[:3]
        if len(y1.size()) == 3:  # no bp
            y1_emb = self.y1_embedding(y1)
        elif len(y1.size()) == 4:  # bp
            y1_emb = torch.matmul(y1, self.y1_embedding.weight)
        # SHAPE: (beam_size * batch_size, seq_len, y_emb_size)
        y1_emb = y1_emb.view(beam_size * batch_size, seq_len, -1)

        # SHAPE: (beam_size * batch_size, seq_len, t_emb_size)
        t_emb = t_emb.repeat(beam_size, 1, 1)
        # SHAPE: (beam_size * batch_size, seq_len, v_emb_size)
        v_emb = v_emb.repeat(beam_size, 1, 1)

        # SHAPE: (beam_size * batch_size, seq_len)
        y2 = y2.repeat(beam_size, 1)
        # SHAPE: (beam_size * batch_size, seq_len)
        y2_mask = y2_mask.repeat(beam_size, 1)
        # SHAPE: (beam_size * batch_size, seq_len)
        mask = mask.repeat(beam_size, 1)

        enc = self.decoder(t_emb, v_emb, y1_emb, mask)

        logits = self.dec_y2_proj(enc)
        loss_mask = ((mask * y2_mask) > 0).long()  # logical and on seq len mask and y2_mask
        y2_nll = sequence_cross_entropy_with_logits(logits, y2, loss_mask, average='sum')
        # SHAPE: (beam_size, batch_size)
        y2_nll = y2_nll.view(beam_size, batch_size)
        return y2_nll


    def vae_discriminate(self,
                         t_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, t_emb_size)
                         v_emb: torch.Tensor,  # SHAPE: (batch_size, seq_len, v_emb_size)
                         mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                         y1: torch.LongTensor = None):  # SHAPE: (beam_size, batch_size, seq_len)
        ''' x -> y1 '''
        enc = self.discriminator(torch.cat([t_emb, v_emb], -1), mask)
        logits = self.disc_y1_proj(enc)
        cp = F.softmax(logits, dim=-1)

        if y1 is not None: # loss
            if len(y1.size()) == 2:
                y1_nll = sequence_cross_entropy_with_logits(
                    logits, y1, mask, average='sum', label_smoothing=self._label_smoothing)
            elif len(y1.size()) == 3:
                beam_size, batch_size, seq_len = y1.size()
                # SHAPE: (beam_size, batch_size)
                y1_nll = torch.stack([sequence_cross_entropy_with_logits(
                    logits, y1[i], mask, average='sum', label_smoothing=self._label_smoothing)
                    for i in range(beam_size)])
            else:
                raise Exception('y1 dimension not correct')
            return logits, cp, y1_nll
        return logits, cp, None


    def exact_kl_div(self,
                     q_logits: torch.Tensor,  # logits of posterior, SHAPE: (batch_size, seq_len, num_class)
                     p_logits: torch.Tensor,  # logits of prior, SHAPE: (batch_size, seq_len, num_class)
                     mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                     ) -> torch.Tensor:  # SHAPE: (batch_size)
        log_p = F.log_softmax(p_logits, dim=-1)
        q = F.softmax(q_logits, dim=-1)
        if self._unsup_loss_type == 'only_disc':
            q = q.detach()
        kl = F.kl_div(log_p, q, reduction='none')
        return (kl * mask.unsqueeze(-1).float()).sum((1, 2))


    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                task_labels: torch.LongTensor,
                weight: torch.FloatTensor,
                tags: torch.LongTensor = None,
                tag_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        output_dict = {}
        bs, sl = verb_indicator.size()
        mask = get_text_field_mask(tokens)

        # tokens embedder (a wrapper)
        if self._use_post_elmo:
            # produce an intermediate result
            tokens_embber = lambda x: getattr(self.text_field_embedder, 'token_embedder_elmo')(x['elmo'])
        else:
            tokens_embber = lambda x: self.text_field_embedder(x)

        sup_unsup_loss = 0.0

        # supervised
        sup_tm = task_labels.eq(self._y1_ind)
        if sup_tm.sum().item() > 0:  # skip if no supervised data exists
            sup_tokens = masked_select_dict_on_first_dim(tokens, sup_tm)
            sup_t = tokens_embber(sup_tokens)
            sup_verb = verb_indicator.masked_select(sup_tm.view(-1, 1)).view(-1, sl)
            sup_mask = mask.masked_select(sup_tm.view(-1, 1)).view(-1, sl)
            sup_weight = weight.masked_select(sup_tm)
            sup_num = (sup_mask.sum(1) > 0).int().sum().item()
            sup_y1 = None
            if tags is not None:
                sup_y1 = tags.masked_select(sup_tm.view(-1, 1)).view(-1, sl)
            sup_y1_logits, sup_y1_cp, sup_y1_nll = self.vae_discriminate(
                self.disc_post_elmo(sup_t), self.disc_bin_emb(sup_verb), sup_mask, sup_y1)
            output_dict['logits'] = sup_y1_logits
            output_dict['class_probabilities'] = sup_y1_cp
            output_dict['mask'] = sup_mask
            if tags is not None:
                sup_loss = (sup_y1_nll * sup_weight).sum()
                sup_unsup_loss += sup_loss
                if self.fix_prior:  # test encoder on sup dataset
                    enc_sup_y1_logits, _, _, enc_sup_y1_nll = self.vae_encode(
                        self.disc_post_elmo(sup_t), self.disc_bin_emb(sup_verb), None, sup_mask,
                        beam_size=1, y1_ground_truth=sup_y1)
                    enc_sup_loss = (enc_sup_y1_nll * sup_weight).sum()
                    enc_sup_y1_cp = F.softmax(enc_sup_y1_logits, dim=-1)
                    od = {'logits': enc_sup_y1_logits, 'class_probabilities': enc_sup_y1_cp, 'mask': sup_mask}
                    # metrics
                    if not self.ignore_span_metric:
                        if self.decode_span_metric:
                            self.y1_span_metric(
                                self.get_decode_pseudo_class_prob(od), sup_y1, sup_mask)
                        else:
                            self.y1_span_metric(enc_sup_y1_cp, sup_y1, sup_mask)
                    self.y1_accuracy(enc_sup_y1_logits, sup_y1, sup_mask)
                    self.y1_multi_loss('enc_sup_l', enc_sup_loss.item(), count=sup_num)
                else:
                    # metrics
                    if not self.ignore_span_metric:
                        if self.decode_span_metric:
                            self.y1_span_metric(
                                self.get_decode_pseudo_class_prob(output_dict), sup_y1, sup_mask)
                        else:
                            self.y1_span_metric(sup_y1_cp, sup_y1, sup_mask)
                    self.y1_accuracy(sup_y1_logits, sup_y1, sup_mask)
                    self.y1_multi_loss('sup_l', sup_loss.item(), count=sup_num)


        # unsupervised
        unsup_tm = task_labels.eq(self._y2_ind)
        if unsup_tm.sum().item() > 0 and tags is not None: # skip if no unsupervised data exists
            unsup_tokens = masked_select_dict_on_first_dim(tokens, unsup_tm)
            unsup_t = tokens_embber(unsup_tokens)
            unsup_verb = verb_indicator.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_y2 = tags.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_y2_mask = tag_mask.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_mask = mask.masked_select(unsup_tm.view(-1, 1)).view(-1, sl)
            unsup_weight = weight.masked_select(unsup_tm)
            unsup_num = (unsup_mask.sum(1) > 0).int().sum().item()

            # sample
            # SHAPE: _, _, (beam_size, batch_size, seq_len),
            # (beam_size, batch_size, seq_len, num_class),
            # (beam_size, batch_size)
            enc_logits, unsup_y1, unsup_y1_oh, enc_y1_nll = \
                self.vae_encode(self.enc_post_elmo(unsup_t),
                                self.enc_bin_emb(unsup_verb),
                                unsup_y2, unsup_mask, beam_size=self._sample_num)

            # language model
            if self.lang_model and self._infer_algo == 'reinforce':
                lang_y1_nll = self.language_model(unsup_y1, unsup_mask)
            if self.lang_model and self._infer_algo == 'gumbel_softmax':
                # TODO add gumbel for language model
                raise NotImplementedError
                #lang_y1_nll = self.language_model(unsup_y1_oh, unsup_mask)

            # decoder loss (reconstruction loss)
            # SHAPE: (beam_size, batch_size)
            if self._infer_algo == 'reinforce':
                if self._use_rule:
                    y2_nll = self.rule_decode(unsup_y1, unsup_y2, unsup_y2_mask, unsup_mask)
                else:
                    y2_nll = self.vae_decode(self.dec_post_elmo(unsup_t),
                                             self.dec_bin_emb(unsup_verb),
                                             unsup_y1, unsup_y2,unsup_y2_mask, unsup_mask)
            elif self._infer_algo == 'gumbel_softmax':
                y2_nll = self.vae_decode(self.dec_post_elmo(unsup_t),
                                         self.dec_bin_emb(unsup_verb),
                                         unsup_y1_oh, unsup_y2, unsup_y2_mask, unsup_mask)

            if self.debug:
                unsup_metadata = np.array(metadata)[unsup_tm.cpu().numpy().astype(bool)]
                for i in range(unsup_mask.size(0)):
                    tl = unsup_mask[i].sum().item()
                    word_seq = unsup_metadata[i]['words']
                    verb_inds = unsup_verb[i].cpu().numpy()[:tl]
                    y2_seq = [self.vocab.get_token_from_index(t, namespace='MT_srl_labels') for t in
                              unsup_y2[i].cpu().numpy()[:tl]]
                    y2_mask_seq = [t for t in unsup_y2_mask[i].cpu().numpy()[:tl]]
                    assert len(word_seq) == len(y2_seq)
                    for j in range(1):
                        cur_score = y2_nll[j, i].item()
                        y1_seq = [self.vocab.get_token_from_index(t, namespace='MT_gt_labels') for t in
                                  unsup_y1[j, i].cpu().numpy()[:tl]]
                        assert len(y1_seq) == len(y2_seq)
                        comp  = [' '.join(map(str, t)) for t in zip(word_seq, verb_inds, y2_mask_seq, y2_seq, y1_seq)]
                        print('{}\t{}'.format(cur_score, '\t'.join(comp)))
                    c = input('next')
                    if c == 'c':
                        break

            # discriminator loss
            # SHAPE: (beam_size, batch_size)
            disc_logits, _, disc_y1_nll = self.vae_discriminate(
                self.disc_post_elmo(unsup_t), self.disc_bin_emb(unsup_verb), unsup_mask, unsup_y1)  # prior

            # kl divergence
            if self.encoder_same_as_pior:
                kl = torch.zeros_like(y2_nll)  # kl is zero when encoder is the same as pior (discriminator)
            else:
                if self._kl_method == 'sample':
                    # SHAPE: (beam_size, batch_size)
                    kl = self._beta * (disc_y1_nll - enc_y1_nll)  # beta * log(q(y1|x,y2) / p(y1|x))
                elif self._kl_method == 'exact':
                    # beta * \sum_{y1} {log(q(y1|x,y2) / p(y1|x))}
                    kl = self._beta * self.exact_kl_div(enc_logits, disc_logits, unsup_mask)
                    # SHAPE: (1, batch_size)
                    kl = kl.unsqueeze(0)

            # encoder loss
            # only REINFORCE needs manually calculate encoder loss,
            # while gumbel_softmax could directly do bp
            if self._infer_algo == 'reinforce':
                if self.lang_model:
                    encoder_reward = -lang_y1_nll
                else:
                    encoder_reward = 0.0
                if self._kl_method == 'sample':
                    # SHAPE: (beam_size, batch_size)
                    encoder_reward += -y2_nll - kl  # log(p(y2|y1)) - log(q(y1|x,y2) / p(y1|x))
                    encoder_reward = encoder_reward.detach() - self._beta  # be mindful of the beta
                elif self._kl_method == 'exact':
                    encoder_reward += -y2_nll  # log(p(y2|y1)
                    encoder_reward = encoder_reward.detach()
                # reduce variance
                if self._baseline == 'wb':
                    baseline = encoder_reward.mean(0, keepdim=True) * self.w + self.b
                elif self._baseline == 'mean':
                    baseline = encoder_reward.mean(0, keepdim=True)
                elif self._baseline == 'non':
                    baseline = 0.0
                encoder_reward = encoder_reward - baseline
                # clip reward
                if self._clip_reward is not None:
                    clipped_encoder_reward = torch.clamp(
                        encoder_reward, min=-self._clip_reward, max=self._clip_reward)
                else:
                    clipped_encoder_reward = encoder_reward
                y1_nll_with_reward = enc_y1_nll * clipped_encoder_reward.detach()
                baseline_loss = encoder_reward ** 2

            # overall loss
            unsup_loss = 0.0
            if self._infer_algo == 'reinforce':
                encoder_loss = (y1_nll_with_reward.mean(0) * unsup_weight).sum()
                decoder_loss = (y2_nll.mean(0) * unsup_weight).sum()
                baseline_loss = (baseline_loss.mean(0) * unsup_weight).sum()
                self.y1_multi_loss('enc_l', encoder_loss.item(), count=unsup_num)
                self.y1_multi_loss('dec_l', decoder_loss.item(), count=unsup_num)
                self.y1_multi_loss('base_l', baseline_loss.item(), count=unsup_num)
                if self._kl_method == 'sample':
                    # be mindful of the beta
                    discriminator_loss = (self._beta * disc_y1_nll.mean(0) * unsup_weight).sum()
                    self.y1_multi_loss('disc_l', discriminator_loss.item(), count=unsup_num)
                    if self._unsup_loss_type == 'all':
                        unsup_loss += decoder_loss + encoder_loss + discriminator_loss + baseline_loss
                    elif self._unsup_loss_type == 'only_disc':
                        unsup_loss += discriminator_loss
                elif self._kl_method == 'exact':
                    # part of the encoder loss and all of the discriminator loss
                    kl_loss = (kl.mean(0) * unsup_weight).sum()
                    self.y1_multi_loss('disc_l', kl_loss.item(), count=unsup_num)
                    if self._unsup_loss_type == 'all':
                        unsup_loss += decoder_loss + encoder_loss + kl_loss + baseline_loss
                    elif self._unsup_loss_type == 'only_disc':
                        unsup_loss += kl_loss
            elif self._infer_algo == 'gumbel_softmax':
                recon_loss = (y2_nll.mean(0) * unsup_weight).sum()
                kl_loss = (kl.mean(0) * unsup_weight).sum()
                elbo_loss = recon_loss + kl_loss
                unsup_loss += elbo_loss
                self.y1_multi_loss('elbo_l', elbo_loss.item(), count=unsup_num)
                self.y1_multi_loss('recon_l', recon_loss.item(), count=unsup_num)
                self.y1_multi_loss('kl_l', kl_loss.item(), count=unsup_num)
            sup_unsup_loss += self._unsup_loss_weight * unsup_loss

        if tags is not None:
            output_dict['loss'] = sup_unsup_loss / ((mask.sum(1) > 0).float().sum() + 1e-13)
            #output_dict['loss'] = sup_unsup_loss / (weight.sum() + 1e-13) # TODO: use both weight and mask?

        return output_dict


    def get_metrics(self, reset: bool = False):
        metric_dict = {}
        # span metric
        sm = {'{}_{}'.format('y1', x): y for x, y in
              self.y1_span_metric.get_metric(reset=reset).items()
              if '-overall' in x}
        metric_dict.update(sm)
        # accuracy
        metric_dict['y1_accuracy'] = self.y1_accuracy.get_metric(reset=reset)
        metric_dict.update(self.y1_multi_loss.get_metric(reset=reset))
        return metric_dict
