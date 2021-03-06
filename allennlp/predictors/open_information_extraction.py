from typing import List, Dict

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import WordTokenizer
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from allennlp.nn.util import n_best_viterbi_decode

import torch
import numpy as np
import spacy

def join_mwp(tags: List[str]) -> List[str]:
    """
    Join multi-word predicates to a single
    predicate ('V') token.
    """
    ret = []
    verb_flag = False
    for tag in tags:
        if "V" in tag:
            # Create a continuous 'V' BIO span
            prefix, _ = tag.split("-")
            if verb_flag:
                # Continue a verb label across the different predicate parts
                prefix = 'I'
            ret.append(f"{prefix}-V")
            verb_flag = True
        else:
            ret.append(tag)
            verb_flag = False

    return ret

def make_oie_string(tokens: List[Token], tags: List[str]) -> str:
    """
    Converts a list of model outputs (i.e., a list of lists of bio tags, each
    pertaining to a single word), returns an inline bracket representation of
    the prediction.
    """
    frame = []
    chunk = []
    words = [token.text for token in tokens]

    for (token, tag) in zip(words, tags):
        if tag.startswith("I-"):
            chunk.append(token)
        else:
            if chunk:
                frame.append("[" + " ".join(chunk) + "]")
                chunk = []

            if tag.startswith("B-"):
                chunk.append(tag[2:] + ": " + token)
            elif tag == "O":
                frame.append(token)

    if chunk:
        frame.append("[" + " ".join(chunk) + "]")

    return " ".join(frame)

def get_predicate_indices(tags: List[str]) -> List[int]:
    """
    Return the word indices of a predicate in BIO tags.
    """
    return [ind for ind, tag in enumerate(tags) if 'V' in tag]

def get_predicate_text(sent_tokens: List[Token], tags: List[str]) -> str:
    """
    Get the predicate in this prediction.
    """
    return " ".join([sent_tokens[pred_id].text
                     for pred_id in get_predicate_indices(tags)])

def predicates_overlap(tags1: List[str], tags2: List[str]) -> bool:
    """
    Tests whether the predicate in BIO tags1 overlap
    with those of tags2.
    """
    # Get predicate word indices from both predictions
    pred_ind1 = get_predicate_indices(tags1)
    pred_ind2 = get_predicate_indices(tags2)

    # Return if pred_ind1 pred_ind2 overlap
    return any(set.intersection(set(pred_ind1), set(pred_ind2)))

def get_coherent_next_tag(prev_label: str, cur_label: str) -> str:
    """
    Generate a coherent tag, given previous tag and current label.
    """
    if cur_label == "O":
        # Don't need to add prefix to an "O" label
        return "O"

    if prev_label == cur_label:
        return f"I-{cur_label}"
    else:
        return f"B-{cur_label}"

def merge_overlapping_predictions(tags1: List[str], tags2: List[str]) -> List[str]:
    """
    Merge two predictions into one. Assumes the predicate in tags1 overlap with
    the predicate of tags2.
    """
    ret_sequence = []
    prev_label = "O"

    # Build a coherent sequence out of two
    # spans which predicates' overlap

    for tag1, tag2 in zip(tags1, tags2):
        label1 = tag1.split("-")[-1]
        label2 = tag2.split("-")[-1]
        if (label1 == "V") or (label2 == "V"):
            # Construct maximal predicate length -
            # add predicate tag if any of the sequence predict it
            cur_label = "V"

        # Else - prefer an argument over 'O' label
        elif label1 != "O":
            cur_label = label1
        else:
            cur_label = label2

        # Append cur tag to the returned sequence
        cur_tag = get_coherent_next_tag(prev_label, cur_label)
        prev_label = cur_label
        ret_sequence.append(cur_tag)
    return ret_sequence

def consolidate_predictions(outputs: List[List[str]], sent_tokens: List[Token],
                            probs: List[List[float]] = None) -> Dict[str, List[str]]:
    """
    Identify that certain predicates are part of a multiword predicate
    (e.g., "decided to run") in which case, we don't need to return
    the embedded predicate ("run").
    """
    pred_dict: Dict[str, List[str]] = {}
    pred_dict_wp: Dict[str, (List[str], List[float])] = {}
    merged_outputs = [join_mwp(output) for output in outputs]
    predicate_texts = [get_predicate_text(sent_tokens, tags)
                       for tags in merged_outputs]
    if probs is None:
        probs = [None] * len(predicate_texts)

    assert len(predicate_texts) == len(merged_outputs) and \
           len(predicate_texts) == len(probs), 'length not equal'

    for pred1_text, tags1, prob in zip(predicate_texts, merged_outputs, probs):
        # A flag indicating whether to add tags1 to predictions
        add_to_prediction = True

        #  Check if this predicate overlaps another predicate
        for pred2_text, tags2 in pred_dict.items():
            if predicates_overlap(tags1, tags2):
                # tags1 overlaps tags2
                pred_dict[pred2_text] = merge_overlapping_predictions(tags1, tags2)
                add_to_prediction = False

        # This predicate doesn't overlap - add as a new predicate
        if add_to_prediction:
            pred_dict[pred1_text] = tags1
            if prob is not None:
                pred_dict_wp[pred1_text] = (tags1, prob)
            else:
                pred_dict_wp[pred1_text] = tags1

    return pred_dict_wp


def sanitize_label(label: str) -> str:
    """
    Sanitize a BIO label - this deals with OIE
    labels sometimes having some noise, as parentheses.
    """
    if "-" in label:
        prefix, suffix = label.split("-")
        suffix = suffix.split("(")[-1]
        return f"{prefix}-{suffix}"
    else:
        return label

@Predictor.register('open-information-extraction')
class OpenIePredictor(Predictor):
    """
    Predictor for the :class: `models.SemanticRolelabeler` model (in its Open Information variant).
    Used by online demo and for prediction on an input file using command line.
    """
    def __init__(self,
                 model: Model,
                 dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = WordTokenizer(word_splitter=SpacyWordSplitter(pos_tags=True))
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "...", "predicate_index": "..."}``.
        Assumes sentence is tokenized, and that predicate_index points to a specific
        predicate (word index) within the sentence, for which to produce Open IE extractions.
        """
        tokens = json_dict["sentence"]
        predicate_index = int(json_dict["predicate_index"])
        verb_labels = [0 for _ in tokens]
        verb_labels[predicate_index] = 1
        return self._dataset_reader.text_to_instance(tokens, verb_labels)


    def _tag_tokens(self, tokens):
        tag = self.nlp.tagger(self.nlp.tokenizer.tokens_from_list(tokens))
        return tag


    def _beam_search(self, prob: np.ndarray, mask: np.ndarray, n_best: int = 1):
        log_prob = np.log(prob)
        seq_lens = mask.sum(-1).tolist()
        one_sam = False
        if log_prob.ndim == 2:
            one_sam = True
            p_li, lp_li, seq_lens = [prob], [log_prob], [seq_lens]
        else:
            p_li, lp_li, seq_lens = prob, log_prob, seq_lens
        all_tags, all_probs = [], []
        trans_mat = self._model.get_viterbi_pairwise_potentials() # to eliminate invalid tag sequence
        for p, lp, slen in zip(p_li, lp_li, seq_lens):
            # viterbi decoding (based on torch tensor)
            vpaths, vscores = n_best_viterbi_decode(
                torch.from_numpy(lp[:slen]), trans_mat, n_best=n_best)
            vpaths = vpaths.numpy()
            # collect tags and corresponding probs
            cur_tags, cur_probs = [], []
            for vpath in vpaths:
                probs = [p[i, vpath[i]] for i in range(len(vpath))]
                # TODO: any better way to handle the mismatch between normal model and multitask model?
                if 'labels' in self._model.vocab._index_to_token:
                    tags = [self._model.vocab.get_token_from_index(x, namespace='labels') for x in vpath]
                elif 'MT_gt_labels' in self._model.vocab._index_to_token:
                    tags = [self._model.vocab.get_token_from_index(x, namespace='MT_gt_labels') for x in vpath]
                else:
                    raise KeyError('what is the namespace of the tags?')
                cur_probs.append(probs)
                cur_tags.append(tags)
            all_probs.append(cur_probs)
            all_tags.append(cur_tags)
        if one_sam:
            return all_tags[0], all_probs[0]
        return all_tags, all_probs


    def predict_conll_file(self, filepath: str, batch_size: int = 256, is_dir: bool = True):
        ontonotes_reader = Ontonotes()
        batch_inst, tokens_li, verb_ind_li, srl_tags_li = [], [], [], []
        def dir_iter(filepath):
            if is_dir:
                for conll_file in ontonotes_reader.dataset_path_iterator(filepath):
                    for sentence in ontonotes_reader.sentence_iterator(conll_file):
                        yield sentence
            else:
                for sentence in ontonotes_reader.sentence_iterator_direct(filepath):
                    yield sentence
        for sentence in dir_iter(filepath):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                continue
            for (_, srl_tags) in sentence.srl_frames:
                verb_ind = [1 if label[-2:] == '-V' else 0 for label in srl_tags]
                inst = self._dataset_reader.text_to_instance(tokens, verb_ind)
                batch_inst.append(inst)
                tokens_li.append(sentence.words)
                verb_ind_li.append(verb_ind)
                srl_tags_li.append(srl_tags)
                if len(batch_inst) >= batch_size:
                    for i, pred in enumerate(self._model.forward_on_instances(batch_inst)):
                        oie_tags, _ = self._beam_search(
                            pred['class_probabilities'], pred['mask'], n_best=1)
                        yield {
                            'words': tokens_li[i],
                            'verb': verb_ind_li[i],
                            'description': None,
                            'srl_tags': srl_tags_li[i],
                            'oie_tags': oie_tags[0]
                        }
                    batch_inst, tokens_li, verb_ind_li, srl_tags_li = [], [], [], []


    def predict_batch(self, sents: List[List[str]], batch_size: int = 256, warm_up: int = 0,
                      beam_search: int = 1) -> JsonDict:
        sents_token = [self._tag_tokens(sent) for sent in sents]

        instances, insts_st, insts_ed = [], [], []
        # Find all verbs in the input sentence
        for sent_token in sents_token:
            pred_ids = [i for (i, t) in enumerate(sent_token) if t.pos_ == 'VERB']
            insts_st.append(len(instances))
            instances.extend([self._json_to_instance(
                {'sentence': sent_token, 'predicate_index': pid}) for pid in pred_ids])
            insts_ed.append(len(instances))

        # Warm up the model using warm_up batch (mainly because of non-determinism of ELMO).
        if warm_up:
            for b in range(0, min(warm_up * batch_size, len(instances)), batch_size):
                batch_inst = instances[b:b + batch_size]
                self._model.forward_on_instances(batch_inst)

        # Run model
        outputs, probs = [], []
        for b in range(0, len(instances), batch_size):
            batch_inst = instances[b:b+batch_size]
            for prediction in self._model.forward_on_instances(batch_inst):
                all_tags, all_probs = self._beam_search(
                    prediction['class_probabilities'], prediction['mask'], n_best=beam_search)
                outputs.append(all_tags)
                probs.append(all_probs)

        results_li = []
        for sent_token, st, ed in zip(sents_token, insts_st, insts_ed):
            # Consolidate predictions
            cur_o = [e for o in outputs[st:ed] for e in o]
            cur_p = [e for o in probs[st:ed] for e in o]
            # Build and return output dictionary
            results = {'verbs': [], 'words': [token.text for token in sent_token]}

            for tags, prob in zip(cur_o, cur_p):
                # Create description text
                description = make_oie_string(sent_token, tags)
                # Add a predicate prediction to the return dictionary.
                results['verbs'].append({
                    'verb': get_predicate_text(sent_token, tags),
                    'description': description,
                    'tags': tags,
                    'probs': prob,
                })
            results_li.append(results)

        return sanitize(results_li)


    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Create instance(s) after predicting the format. One sentence containing multiple verbs
        will lead to multiple instances.

        Expects JSON that looks like ``{"sentence": "..."}``

        Returns a JSON that looks like

        .. code-block:: js

            {"tokens": [...],
             "tag_spans": [{"ARG0": "...",
                            "V": "...",
                            "ARG1": "...",
                             ...}]}
        """
        sent_tokens = self._tokenizer.tokenize(inputs["sentence"])

        # Find all verbs in the input sentence
        pred_ids = [i for (i, t)
                    in enumerate(sent_tokens)
                    if t.pos_ == "VERB"]

        # Create instances
        instances = [self._json_to_instance({"sentence": sent_tokens,
                                             "predicate_index": pred_id})
                     for pred_id in pred_ids]

        # Run model
        outputs = [[sanitize_label(label) for label in self._model.forward_on_instance(instance)["tags"]]
                   for instance in instances]

        # Consolidate predictions
        pred_dict = consolidate_predictions(outputs, sent_tokens)

        # Build and return output dictionary
        results = {"verbs": [], "words": sent_tokens}

        for tags in pred_dict.values():
            # Join multi-word predicates
            tags = join_mwp(tags)

            # Create description text
            description = make_oie_string(sent_tokens, tags)

            # Add a predicate prediction to the return dictionary.
            results["verbs"].append({
                    "verb": get_predicate_text(sent_tokens, tags),
                    "description": description,
                    "tags": tags,
            })

        return sanitize(results)
