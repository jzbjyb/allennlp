from typing import List

from overrides import overrides
from spacy.tokens import Doc

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.data.tokenizers import Token


@Predictor.register("semantic-role-labeling")
class SemanticRoleLabelerPredictor(Predictor):
    """
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)

    def predict(self, sentence: str) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence and returns a dictionary
        with the results.

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}

        Parameters
        ----------
        sentence, ``str``
            The sentence to parse via semantic role labeling.

        Returns
        -------
        A dictionary representation of the semantic roles in the sentence.
        """
        return self.predict_json({"sentence": sentence})

    def predict_tokenized(self, tokenized_sentence: List[str]) -> JsonDict:
        """
        Predicts the semantic roles of the supplied sentence tokens and returns a dictionary
        with the results.

        Parameters
        ----------
        tokenized_sentence, ``List[str]``
            The sentence tokens to parse via semantic role labeling.

        Returns
        -------
        A dictionary representation of the semantic roles in the sentence.
        """
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokenized_sentence)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)

        tokens = [token for token in spacy_doc]
        instances = self.tokens_to_instances(tokens)

        if not instances:
            return sanitize({"verbs": [], "words": tokens})

        return self.predict_instances(instances)

    @staticmethod
    def make_srl_string(words: List[str], tags: List[str]) -> str:
        frame = []
        chunk = []

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

    @overrides
    def _json_to_instance(self, json_dict: JsonDict):
        raise NotImplementedError("The SRL model uses a different API for creating instances.")

    def tokens_to_instances(self, tokens):
        words = [token.text for token in tokens]
        instances: List[Instance] = []
        for i, word in enumerate(tokens):
            if word.pos_ == "VERB":
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    def _str_tokens_to_spcay_tokens(self, tokens: List[str]) -> List[Token]:
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokens)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)
        return [token for token in spacy_doc]

    def _tokens_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        tokens = self._str_tokens_to_spcay_tokens(json_dict['sentence'])
        return self.tokens_to_instances(tokens)

    def _sentence_to_srl_instances(self, json_dict: JsonDict) -> List[Instance]:
        """
        The SRL model has a slightly different API from other models, as the model is run
        forward for every verb in the sentence. This means that for a single sentence, we need
        to generate a ``List[Instance]``, where the length of this list corresponds to the number
        of verbs in the sentence. Additionally, all of these verbs share the same return dictionary
        after being passed through the model (as really we care about all the frames of the sentence
        together, rather than separately).

        Parameters
        ----------
        json_dict : ``JsonDict``, required.
            JSON that looks like ``{"sentence": "..."}``.

        Returns
        -------
        instances : ``List[Instance]``
            One instance per verb.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self.tokens_to_instances(tokens)

    def predict_conll_file(self, conll_filepath: str, batch_size: int = 256):
        # get instances and raw tags
        instances = []
        raw_tags_li = []
        verb_indicator_li = []
        ontonotes_reader = Ontonotes()
        emp_ct = 0
        for sentence in ontonotes_reader.sentence_iterator(conll_filepath):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                emp_ct += 1
                continue # skip sentence without predicate
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == '-V' else 0 for label in tags]
                    if sum(verb_indicator) != 1:
                        emp_ct += 1
                        continue
                    instances.append(self._dataset_reader.text_to_instance(tokens, verb_indicator))
                    raw_tags_li.append(tags)
                    verb_indicator_li.append(verb_indicator)
        print('{} empty sentences'.format(emp_ct))
        print('{} instances'.format(len(instances)))
        # run the model
        outputs = []
        for batch in range(0, len(instances), batch_size):
            batch = instances[batch:batch + batch_size]
            outputs.extend(self._model.forward_on_instances(batch))
        result_li = []
        for output, raw_tags, verb_indicator in zip(outputs, raw_tags_li, verb_indicator_li):
            words = output['words']
            tags = output['tags']
            description = self.make_srl_string(words, tags)
            result = {
                'words': words,
                'verb': verb_indicator,
                'description': description,
                'tags': tags,
                'raw_tags': raw_tags
            }
            result_li.append(result)
        return result_li

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict],
                           batch_size: int = None,
                           tokenized: bool = False) -> List[JsonDict]:
        """
        Expects JSON that looks like ``[{"sentence": "..."}, {"sentence": "..."}, ...]``
        and returns JSON that looks like

        .. code-block:: js

            [
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]},
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]}
            ]
        """
        # For SRL, we have more instances than sentences, but the user specified
        # a batch size with respect to the number of sentences passed, so we respect
        # that here by taking the batch size which we use to be the number of sentences
        # we are given.
        if batch_size is None:
            batch_size = len(inputs)
        if tokenized:
            instances_per_sentence = [self._tokens_to_srl_instances(json) for json in inputs]
            tokenize_func = self._str_tokens_to_spcay_tokens
        else:
            instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]
            tokenize_func = self._tokenizer.split_words

        flattened_instances = [instance for sentence_instances in instances_per_sentence
                               for instance in sentence_instances]

        if not flattened_instances:
            return sanitize([{"verbs": [], "words": tokenize_func(x["sentence"])}
                             for x in inputs])

        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [instance for instance in batched_instances[-1]
                                 if instance is not None]
        # Run the model on the batches.
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))

        verbs_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts: List[JsonDict] = [{"verbs": []} for x in inputs]

        output_index = 0
        for sentence_index, verb_count in enumerate(verbs_per_sentence):
            if verb_count == 0:
                # We didn't run any predictions for sentences with no verbs,
                # so we don't have a way to extract the original sentence.
                # Here we just tokenize the input again.
                original_text = tokenize_func(inputs[sentence_index]["sentence"])
                return_dicts[sentence_index]["words"] = original_text
                continue

            for _ in range(verb_count):
                output = outputs[output_index]
                words = output["words"]
                tags = output['tags']
                description = self.make_srl_string(words, tags)
                return_dicts[sentence_index]["words"] = words
                return_dicts[sentence_index]["verbs"].append({
                        "verb": output["verb"],
                        "description": description,
                        "tags": tags,
                })
                output_index += 1

        return sanitize(return_dicts)

    def predict_instances(self, instances: List[Instance]) -> JsonDict:
        outputs = self._model.forward_on_instances(instances)

        results = {"verbs": [], "words": outputs[0]["words"]}
        for output in outputs:
            tags = output['tags']
            description = self.make_srl_string(output["words"], tags)
            results["verbs"].append({
                    "verb": output["verb"],
                    "description": description,
                    "tags": tags,
            })

        return sanitize(results)

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}
        """
        instances = self._sentence_to_srl_instances(inputs)

        if not instances:
            return sanitize({"verbs": [], "words": self._tokenizer.split_words(inputs["sentence"])})

        return self.predict_instances(instances)
