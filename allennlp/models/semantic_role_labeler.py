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
from allennlp.training.metrics import SpanBasedF1Measure
from allennlp.models.base import BaseModel


@Model.register("srl")
class SemanticRoleLabeler(BaseModel):
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
                 only_train_proj: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 label_smoothing: float = None,
                 ignore_span_metric: bool = False,
                 decode_span_metric: bool = True) -> None:
        super(SemanticRoleLabeler, self).__init__('labels', vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")

        # For the span based evaluation, we don't want to consider labels
        # for verb, because the verb index is provided to the model.
        self.span_metric = SpanBasedF1Measure(vocab, tag_namespace="labels", ignore_classes=["V"])

        self.encoder = encoder
        # There are exactly 2 binary features for the verb predicate embedding.
        self.binary_feature_embedding = Embedding(2, binary_feature_dim)
        self.tag_projection_layer = TimeDistributed(Linear(self.encoder.get_output_dim(),
                                                           self.num_classes))
        self.embedding_dropout = Dropout(p=embedding_dropout)
        self._only_train_proj = only_train_proj
        self._label_smoothing = label_smoothing
        self.ignore_span_metric = ignore_span_metric
        self.decode_span_metric = decode_span_metric

        check_dimensions_match(text_field_embedder.get_output_dim() + binary_feature_dim,
                               encoder.get_input_dim(),
                               "text embedding dim + verb indicator embedding dim",
                               "encoder input dim")
        initializer(self)

    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                verb_indicator: torch.LongTensor,
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

        if self._only_train_proj: # only train the projection layer
            encoded_text = encoded_text.detach()

        logits = self.tag_projection_layer(encoded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view([batch_size,
                                                                          sequence_length,
                                                                          self.num_classes])

        # We need to retain the mask in the output dictionary
        # so that we can crop the sequences to remove padding
        # when we do viterbi inference in self.decode.
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities,
                       "mask": mask}

        if tags is not None:
            loss = sequence_cross_entropy_with_logits(logits,
                                                      tags,
                                                      mask,
                                                      label_smoothing=self._label_smoothing)
            if not self.ignore_span_metric:
                if self.decode_span_metric:
                    self.span_metric(self.get_decode_pseudo_class_prob(output_dict), tags, mask)
                else:
                    self.span_metric(class_probabilities, tags, mask)
            output_dict["loss"] = loss

        words, verbs, verb_inds = zip(*[(x['words'], x['verb'], x['verb_inds']) for x in metadata])
        if metadata is not None:
            output_dict['words'] = list(words)
            output_dict['verb'] = list(verbs)
            output_dict['verb_inds'] = list(verb_inds)
        return output_dict

    def get_metrics(self, reset: bool = False):
        if self.ignore_span_metric:
            # Return an empty dictionary if ignoring the
            # span metric
            return {}

        else:
            metric_dict = self.span_metric.get_metric(reset=reset)

            # This can be a lot of metrics, as there are 3 per class.
            # we only really care about the overall metrics, so we filter for them here.
            return {x: y for x, y in metric_dict.items() if "overall" in x}


def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str]):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    verb_only_sentence = ["-"] * len(sentence)
    if verb_index:
        verb_only_sentence[verb_index] = sentence[verb_index]

    conll_format_predictions = convert_bio_tags_to_conll_format(prediction)
    conll_format_gold_labels = convert_bio_tags_to_conll_format(gold_labels)

    for word, predicted, gold in zip(verb_only_sentence,
                                     conll_format_predictions,
                                     conll_format_gold_labels):
        prediction_file.write(word.ljust(15))
        prediction_file.write(predicted.rjust(15) + "\n")
        gold_file.write(word.ljust(15))
        gold_file.write(gold.rjust(15) + "\n")
    prediction_file.write("\n")
    gold_file.write("\n")


def convert_bio_tags_to_conll_format(labels: List[str]):
    """
    Converts BIO formatted SRL tags to the format required for evaluation with the
    official CONLL 2005 perl script. Spans are represented by bracketed labels,
    with the labels of words inside spans being the same as those outside spans.
    Beginning spans always have a opening bracket and a closing asterisk (e.g. "(ARG-1*" )
    and closing spans always have a closing bracket (e.g. "*)" ). This applies even for
    length 1 spans, (e.g "(ARG-0*)").

    A full example of the conversion performed:

    [B-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, I-ARG-1, O]
    [ "(ARG-1*", "*", "*", "*", "*)", "*"]

    Parameters
    ----------
    labels : List[str], required.
        A list of BIO tags to convert to the CONLL span based format.

    Returns
    -------
    A list of labels in the CONLL span based format.
    """
    sentence_length = len(labels)
    conll_labels = []
    for i, label in enumerate(labels):
        if label == "O":
            conll_labels.append("*")
            continue
        new_label = "*"
        # Are we at the beginning of a new span, at the first word in the sentence,
        # or is the label different from the previous one? If so, we are seeing a new label.
        if label[0] == "B" or i == 0 or label[1:] != labels[i - 1][1:]:
            new_label = "(" + label[2:] + new_label
        # Are we at the end of the sentence, is the next word a new span, or is the next
        # word not in a span? If so, we need to close the label span.
        if i == sentence_length - 1 or labels[i + 1][0] == "B" or label[1:] != labels[i + 1][1:]:
            new_label = new_label + ")"
        conll_labels.append(new_label)
    return conll_labels
