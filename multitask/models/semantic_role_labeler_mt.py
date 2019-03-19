from typing import Dict, List, TextIO, Optional, Any

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
from allennlp.nn.util import get_lengths_from_binary_sequence_mask, viterbi_decode
from allennlp.training.metrics import SpanBasedF1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions

@Model.register("srl_mt")
class SemanticRoleLabelerMultiTask(Model):
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
                 encoder: Seq2SeqEncoder,
                 binary_feature_dim: int,
                 embedding_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 task_encoder: Seq2SeqEncoder = None,
                 encoder_requires_grad: bool = True,
                 task_encoder_requires_grad: bool = True,
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 use_crf: bool = False) -> None:
        super(SemanticRoleLabelerMultiTask, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.num_tasks = self.vocab.get_vocab_size("task_labels")
        self.register_buffer('_task', torch.arange(self.num_tasks).view([1, -1]))

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=["V"])
        self.accuracy = CategoricalAccuracy(top_k=1, tie_break=False)

        self.encoder = encoder
        self.task_encoder = task_encoder
        for param in self.encoder.parameters():
            param.requires_grad = encoder_requires_grad
        if self.task_encoder is not None:
            for param in self.task_encoder.parameters():
                param.requires_grad = task_encoder_requires_grad
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer_mt = TimeDistributed(
            Linear(self.encoder.get_output_dim(), self.num_tasks * self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        # set up crf
        self.use_crf = use_crf  # whether to use CRF decoding
        if self.use_crf:
            labels = self.vocab.get_index_to_token_vocabulary('labels')
            constraints = allowed_transitions('BIO', labels)
            self.crf = ConditionalRandomField(
                self.num_classes, constraints=constraints, include_start_end_transitions=False)
        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "encoder input dim")
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
                task_labels: torch.LongTensor,
                tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        tokens : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``, which should typically be passed directly to a
            ``TextFieldEmbedder``. This output is a dictionary mapping keys to ``TokenIndexer``
            tensors.  At its most basic, using a ``SingleIdTokenIndexer`` this is: ``{"tokens":
            Tensor(batch_size, num_tokens)}``. This dictionary will have the same keys as were used
            for the ``TokenIndexers`` when you created the ``TextField`` representing your
            sequence.  The dictionary is designed to be passed directly to a ``TextFieldEmbedder``,
            which knows how to combine different word representations into a single vector per
            token in your input.
        verb_indicator: torch.LongTensor, required.
            An integer ``SequenceFeatureField`` representation of the position of the verb
            in the sentence. This should have shape (batch_size, num_tokens) and importantly, can be
            all zeros, in the case that the sentence has no verbal predicate.
        tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels
            of shape ``(batch_size, num_tokens)``
        metadata : ``List[Dict[str, Any]]``, optional, (default = None)
            metadata containg the original words in the sentence and the verb to compute the
            frame for, under 'words' and 'verb' keys, respectively.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            unnormalised log probabilities of the tag classes.
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_tokens, tag_vocab_size)`` representing
            a distribution of the tag classes per word.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.

        """
        embedded_text_input = self.embedding_dropout(self.text_field_embedder(tokens))
        mask = get_text_field_mask(tokens)
        embedded_verb_indicator = self.binary_feature_embedding(verb_indicator.long())
        # Concatenate the verb feature onto the embedded text. This now
        # has shape (batch_size, sequence_length, embedding_dim + binary_feature_dim).
        embedded_text_with_verb_indicator = torch.cat([embedded_text_input, embedded_verb_indicator], -1)
        batch_size, sequence_length, _ = embedded_text_with_verb_indicator.size()

        encoded_text = self.encoder(embedded_text_with_verb_indicator, mask)
        if self.task_encoder is not None:
            encoded_text = self.task_encoder(encoded_text, mask)

        logits = self.tag_projection_layer_mt(encoded_text)
        # get the logits of the corresponding task
        logits = logits.view([batch_size, sequence_length, self.num_tasks, self.num_classes])
        task_mask = task_labels.view([batch_size, 1]) == self._task
        logits = torch.masked_select(logits, task_mask.view([batch_size, 1, self.num_tasks, 1])).view(
            [batch_size, sequence_length, self.num_classes])
        output_dict = {'logits': logits}
        # calculate prob
        reshaped_log_probs = logits.view(-1, self.num_classes)
        if self.use_crf:
            predicted_tags = [x for x, y in self.crf.viterbi_tags(logits, mask)]
            output_dict['tags'] = predicted_tags
            # pseudo class prob
            vtags = verb_indicator * 0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    vtags[i, j] = tag_id
            vll = self.crf(logits, vtags, mask, agg=None)
            vll /= mask.sum(1).float() + 1e-13  # average over words in each seq
            evll = torch.exp(vll)
            class_probabilities = logits * 0.
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = evll[i]
            output_dict['class_probabilities'] = class_probabilities
        else:
            class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
                [batch_size, sequence_length, self.num_classes])
            output_dict['class_probabilities'] = class_probabilities
        # calculate loss
        if tags is not None:
            if self.use_crf:
                ll = self.crf(logits, tags, mask, agg=None)
                ll /= mask.sum(1).float() + 1e-13 # average over words in each seq
                loss = -ll.sum() / ((mask.sum(1) > 0).float().sum() + 1e-13) # average over seqs
            else:
                loss = sequence_cross_entropy_with_logits(
                    logits, tags, mask, label_smoothing=self._label_smoothing)
            output_dict["loss"] = loss
            if not self.ignore_span_metric:
                self.span_metric(output_dict['class_probabilities'], tags, mask)
            if self.use_crf:
                # TODO: we should use logits here, but class_probabilities seems to be a good replacement
                self.accuracy(output_dict['class_probabilities'], tags, mask)
            else:
                self.accuracy(logits, tags, mask)

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict["mask"] = mask

        if metadata is not None:
            words, verbs = zip(*[(x["words"], x["verb"]) for x in metadata])
            output_dict["words"] = list(words)
            output_dict["verb"] = list(verbs)
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does constrained viterbi decoding on class probabilities output in :func:`forward`.  The
        constraint simply specifies that the output tags must be a valid BIO sequence.  We add a
        ``"tags"`` key to the dictionary with the result.
        """
        if self.use_crf:
            # TODO: add prob
            output_dict['tags'] = [
                [self.vocab.get_token_from_index(tag, namespace='labels') for tag in instance_tags]
                for instance_tags in output_dict["tags"]]
            return output_dict
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
            tags = [self.vocab.get_token_from_index(x, namespace="labels")
                    for x in max_likelihood_sequence]
            all_tags.append(tags)
        output_dict['tags'] = all_tags
        output_dict['probs'] = all_probs
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            metric_dict = {}
        else:
            metric_dict = self.span_metric.get_metric(reset=reset)
            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            metric_dict = {x: y for x, y in metric_dict.items() if 'overall' in x}
        metric_dict['accuracy'] = self.accuracy.get_metric(reset=reset)
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
        all_labels = self.vocab.get_index_to_token_vocabulary("labels")
        num_labels = len(all_labels)
        transition_matrix = torch.zeros([num_labels, num_labels])

        for i, previous_label in all_labels.items():
            for j, label in all_labels.items():
                # I labels can only be preceded by themselves or
                # their corresponding B tag.
                if i != j and label[0] == 'I' and not previous_label == 'B' + label[1:]:
                    transition_matrix[i, j] = float("-inf")
        return transition_matrix
