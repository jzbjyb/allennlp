from typing import List

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token
from spacy.tokens import Doc


@Predictor.register('sentence-tagger')
class SentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader, language: str = 'en_core_web_sm') -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=language, pos_tags=True)

    def _str_tokens_to_spcay_tokens(self, tokens: List[str]) -> List[Token]:
        spacy_doc = Doc(self._tokenizer.spacy.vocab, words=tokens)
        for pipe in filter(None, self._tokenizer.spacy.pipeline):
            pipe[1](spacy_doc)
        return [token for token in spacy_doc]

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})

    def predict_batch_tokenized(self, sents: List[List[str]], batch_size=256) -> List[JsonDict]:
        insts = []
        for sent in sents:
            tokens = self._str_tokens_to_spcay_tokens(sent)
            inst = self._dataset_reader.text_to_instance(tokens)
            insts.append(inst)
        result = []
        for b in range(0, len(insts), batch_size):
            result.extend(self.predict_batch_instance(insts[b:b + batch_size]))
        return result

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer.split_words(sentence)
        return self._dataset_reader.text_to_instance(tokens)
