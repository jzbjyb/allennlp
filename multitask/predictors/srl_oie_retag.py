from typing import List, Union

from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token
from allennlp.common.util import JsonDict
from multitask.dataset_readers.rerank_reader import SupOieConll


@Predictor.register('srl-oie-retag')
class SrlOieRetagPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)


    def predict_batch_tokenized(self,
                                inputs: List[JsonDict],
                                batch_size: int = 256) -> List[JsonDict]:
        # collect instance
        inp_tags_name = '{}_tags'.format(self._model.yin)
        insts = []
        for inp in inputs:
            tokens: List[str] = inp['tokens']
            tokens = [Token(t) for t in tokens]
            verb_inds: List[int] = inp['verb_inds']
            inp_tags: List[str] = inp[inp_tags_name]
            if inp_tags == None:  # check the input tags
                raise Exception('{} is required as input'.format(inp_tags_name))
            assert len(tokens) == len(verb_inds) and len(tokens) == len(inp_tags), \
                'length of the inputs is not consistent'
            insts.append(self._dataset_reader.text_to_instance_parallel(
                tokens, verb_inds, **{inp_tags_name: inp_tags}))
        # predict
        outputs = []
        for batch in range(0, len(insts), batch_size):
            batch = insts[batch:batch + batch_size]
            outputs.extend(self._model.forward_on_instances(batch))
        return outputs
