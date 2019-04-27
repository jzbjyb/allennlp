import logging
from typing import Dict, List, Optional, Any

import torch
from torch.nn.modules import Linear
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.models.base import BaseModel

from multitask.modules.util import modify_req_grad

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('srl_oie_retag')
class SrlOieRetag(BaseModel):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 tag_feature_dim: int,
                 # specify the input and output,
                 # including "xoie_srl", "xsrl_oie", "oie_srl", and "srl_oie"
                 mode: str,
                 binary_req_grad: bool = True,
                 tag_proj_req_grad: bool = True,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 use_crf: bool = False) -> None:
        # determine the input and output of the model
        assert mode in {'xoie_srl', 'xsrl_oie', 'oie_srl', 'srl_oie'}, 'mode not supported'
        self.mode = mode
        self.use_x = mode.split('_')[0][:1] == 'x'
        self.yin = mode.split('_')[0][-3:]
        self.yout = mode.split('_')[1]
        self.yin_ns = 'MT_{}_labels'.format('gt' if self.yin == 'oie' else self.yin)
        self.yout_ns = 'MT_{}_labels'.format('gt' if self.yout == 'oie' else self.yout)
        logger.info('The retag is {}{} -> {}'.format(
            'x,' if self.use_x else '', self.yin, self.yout))

        # init base model
        super(SrlOieRetag, self).__init__(self.yout_ns, vocab, regularizer)
        self._label_smoothing = label_smoothing
        self.use_crf = use_crf  # whether to use CRF decoding TODO: add crf

        # model
        self.text_field_embedder = text_field_embedder
        self.binary_feature_embedding = Embedding(2, binary_feature_dim, trainable=binary_req_grad)
        self.tag_feature_embedding = Embedding(
            self.vocab.get_vocab_size(self.yin_ns), tag_feature_dim)

        if self.yin == 'oie':
            self.encoder_name = 'decoder'
        elif self.yin == 'srl':
            self.encoder_name = 'encoder'
        setattr(self, self.encoder_name, encoder)

        self.tag_projection_layer = TimeDistributed(Linear(
            encoder.get_output_dim(), self.vocab.get_vocab_size(self.yout_ns)))
        modify_req_grad(self.tag_projection_layer, tag_proj_req_grad)

        # metrics
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=self.yout_ns, ignore_classes=['V'])
        self.accuracy = CategoricalAccuracy(top_k=1, tie_break=False)

        initializer(self)


    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                srl_tags: torch.LongTensor = None,
                oie_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        yin_tag = eval('{}_tags'.format(self.yin))
        yout_tag = eval('{}_tags'.format(self.yout))

        # embed
        t_emb = self.text_field_embedder(tokens)
        v_emb = self.binary_feature_embedding(verb_indicator.long())
        yin_emb = self.tag_feature_embedding(yin_tag)

        # encode
        enc = getattr(self, self.encoder_name)(t_emb, v_emb, yin_emb, mask)
        logits = self.tag_projection_layer(enc)
        cp = F.softmax(logits, dim=-1)
        output_dict = {'logits': logits, 'class_probabilities': cp, 'mask': mask}

        # loss
        if yout_tag is not None:
            loss = sequence_cross_entropy_with_logits(
                logits, yout_tag, mask, label_smoothing=self._label_smoothing)
            self.span_metric(cp, yout_tag, mask)
            self.accuracy(logits, yout_tag, mask)
            output_dict['loss'] = loss

        # metadata
        words, verbs = zip(*[(x['words'], x['verb']) for x in metadata])
        if metadata is not None:
            output_dict['words'] = list(words)
            output_dict['verb'] = list(verbs)
        return output_dict


    def get_metrics(self, reset: bool = False):
        metric_dict = {x: y for x, y in self.span_metric.get_metric(reset=reset).items()
                       if 'overall' in x}
        metric_dict['accuracy'] = self.accuracy.get_metric(reset=reset)
        return metric_dict
