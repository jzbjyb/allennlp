from typing import Dict, List, Optional, Any

from overrides import overrides
import torch
from torch.nn.modules import Linear

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import sequence_cross_entropy_with_logits
from torch.nn.functional import margin_ranking_loss
from multitask.models.semantic_role_labeler_mt import SemanticRoleLabelerMultiTask


@Model.register('reranker')
class Reranker(SemanticRoleLabelerMultiTask):
    '''
    This model tune the probability of a tagging sequence for better ranking performance.
    '''
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 task_encoder: Seq2SeqEncoder = None,
                 encoder_requires_grad: bool = True,
                 task_encoder_requires_grad: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False) -> None:
        super().__init__(vocab, text_field_embedder, encoder, binary_feature_dim, embedding_dropout,
                         initializer, task_encoder, encoder_requires_grad, task_encoder_requires_grad,
                         regularizer, label_smoothing, ignore_span_metric)
        self.score_layer = Linear(1, 1)
        self.alpha = 1.0 # weights for cross entropy loss

    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                task_labels: torch.LongTensor,
                tags: torch.LongTensor,
                labels: torch.LongTensor = None,
                weights: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(tokens, verb_indicator, task_labels, metadata=metadata) # don't need tags
        logits = output_dict['logits']
        probs = output_dict['class_probabilities']
        probs = torch.clamp(probs, 1e-5, 1 - 1e-5)
        log_probs = torch.log(probs)
        mask = output_dict['mask']
        # calculate confidence scores
        lpt = torch.gather(log_probs.view(-1, log_probs.size(-1)),
                           dim=1, index=tags.view(-1, 1)).view(*tags.size())
        lpt *= mask.float()
        alpt = lpt.sum(-1) / (mask.sum(-1).float() + 1e-13)
        scores = self.score_layer(alpt.unsqueeze(-1)).squeeze(-1)
        output_dict['scores'] = scores
        if labels is not None:
            labels = labels.float()
            # loss mask
            ce_mask = labels.eq(2).float() # cross entropy loss mask
            hi_mask = 1 - ce_mask # hinge loss mask
            # cross entropy loss
            ce_loss = sequence_cross_entropy_with_logits(logits, tags, mask, average=None,
                                                         label_smoothing=self._label_smoothing)
            ce_loss *= ce_mask
            # hinge loss
            # SHAPE: (batch_size, seq_len)
            hi_loss = margin_ranking_loss(scores / 2, -scores / 2, labels, margin=1.0, reduction='none')
            hi_loss *= hi_mask
            # combine two loss and weights
            loss = self.alpha * ce_loss + hi_loss
            if weights is not None:
                loss *= weights
            loss = loss.mean()
            output_dict['loss'] = loss
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # no decoding needed because we already have scores
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
