import logging
from typing import Dict, List, Optional, Any

import torch
from torch.nn.modules import Linear, Dropout, Dropout2d
import torch.nn.functional as F

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy

from .base import BaseModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register('srl_oie_retag')
class SrlOieRetag(BaseModel):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,  # the base encoder shared across tasks
                 binary_feature_dim: int,
                 tag_feature_dim: int,
                 # specify the input and output,
                 # including "xoie_srl", "xsrl_oie", "oie_srl", and "srl_oie"
                 mode: str,
                 embedding_dropout: float = 0.0,
                 word_dropout: float = 0.0,  # dropout a word
                 word_proj_dim: int = None,  # the dim of projection of word embedding
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 use_crf: bool = False) -> None:
        # determine the input and output of the model
        assert mode in {'xoie_srl', 'xsrl_oie', 'oie_srl', 'srl_oie'}, 'mode not supported'
        self.mode = mode
        self.has_x = mode.split('_')[0][:1] == 'x'
        self.y1 = mode.split('_')[0][-3:]
        self.y2 = mode.split('_')[1]
        self.y1_ns = 'MT_{}_labels'.format('gt' if self.y1 == 'oie' else self.y1)
        self.y2_ns = 'MT_{}_labels'.format('gt' if self.y2 == 'oie' else self.y2)
        logger.info('The retag is {}{} -> {}'.format(
            'x,' if self.has_x else '', self.y1, self.y2))

        # init base model
        super(SrlOieRetag, self).__init__(self.y2_ns, vocab, regularizer)
        self._label_smoothing = label_smoothing
        self.use_crf = use_crf  # whether to use CRF decoding TODO: add crf

        # model
        self.text_field_embedder = text_field_embedder
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_feature_embedding = Embedding(
            self.vocab.get_vocab_size(self.y1_ns), tag_feature_dim)
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self.word_dropout = Dropout2d(p=word_dropout)
        self.word_proj_dim = word_proj_dim
        if word_proj_dim:
            self.word_projection_layer = TimeDistributed(Linear(
                text_field_embedder.get_output_dim(), word_proj_dim))
            x_dim = word_proj_dim + binary_feature_dim
        else:
            x_dim = text_field_embedder.get_output_dim() + binary_feature_dim

        self.encoder = encoder
        if mode in {'xoie_srl', 'xsrl_oie'}:
            check_dimensions_match(x_dim + tag_feature_dim, encoder.get_input_dim(),
                'text embedding dim + verb indicator embedding dim + tag embedding dim',
                'encoder input dim')
        elif mode in {'oie_srl', 'srl_oie'}:
            check_dimensions_match(tag_feature_dim, encoder.get_input_dim(),
                'tag embedding dim', 'encoder input dim')
        self.tag_projection_layer = TimeDistributed(Linear(
            self.encoder.get_output_dim(), self.vocab.get_vocab_size(self.y2_ns)))

        # metrics
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace=self.y2_ns, ignore_classes=['V'])
        self.accuracy = CategoricalAccuracy(top_k=1, tie_break=False)

        initializer(self)


    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                srl_tags: torch.LongTensor = None,
                oie_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        y1_tag = eval('{}_tags'.format(self.y1))
        y2_tag = eval('{}_tags'.format(self.y2))
        emb_inp = self.embedding_dropout(self.tag_feature_embedding(y1_tag))

        if self.has_x:
            emb_text = self.embedding_dropout(self.text_field_embedder(tokens))
            emb_text = self.word_dropout(emb_text)  # TODO: is Dropout2d problematic?
            if self.word_proj_dim:
                emb_text = self.word_projection_layer(emb_text)
            emb_verb = self.binary_feature_embedding(verb_indicator.long())
            # Concatenate the verb feature onto the embedded text. This now
            # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
            emb_inp = torch.cat([emb_text, emb_verb, emb_inp], -1)

        enc = self.encoder(emb_inp, mask)

        logits = self.tag_projection_layer(enc)
        cp = F.softmax(logits.view(-1, logits.size(-1)), dim=-1).view(list(logits.size()))
        output_dict = {'logits': logits, 'class_probabilities': cp}

        if y2_tag is not None:
            loss = sequence_cross_entropy_with_logits(
                logits, y2_tag, mask, label_smoothing=self._label_smoothing)
            self.span_metric(cp, y2_tag, mask)
            self.accuracy(logits, y2_tag, mask)
            output_dict['loss'] = loss

        output_dict['mask'] = mask

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
