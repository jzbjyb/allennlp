import logging
from typing import Dict, List, Iterable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class IntField(Field[int]):
    def __init__(self, value: int) -> None:
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.long)

    def __str__(self) -> str:
        return 'IntFeild'


class FloadField(Field[float]):
    def __init__(self, value: float) -> None:
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        return torch.tensor(self.value, dtype=torch.float)

    def __str__(self) -> str:
        return 'FloatFeild'


class SupOieConllExtraction:
    def __init__(self,
                 words: List[str],
                 tags: List[str],
                 pred_ind: int,
                 label: int = None,
                 weight: float = None) -> None:
        self.words = words
        self.tags = tags
        self.pred_ind = pred_ind
        self.label = label
        self.weight = weight


class SupOieConll:
    def _conll_rows_to_extraction(self, rows):
        words = []
        tags = []
        pred_ind = None
        label = None
        weight = None
        for row in rows:
            row = row.split('\t')
            words.append(row[1])
            tags.append(row[7])
            pred_ind = int(row[4])
            if len(row) > 8:
                label = int(row[8])
            if len(row) > 9:
                weight = float(row[9])
        return SupOieConllExtraction(words, tags, pred_ind, label, weight)


    def sentence_iterator(self, filepath: str) -> Iterable[SupOieConllExtraction]:
        with open(filepath, 'r') as fin:
            rows = []
            _ = fin.readline() # skip csv head
            for l in fin:
                l = l.strip()
                if l == '':
                    if rows:
                        yield self._conll_rows_to_extraction(rows)
                        rows = []
                else:
                    rows.append(l)
            if rows:
                yield self._conll_rows_to_extraction(rows)


    def map_tags_reverse(self, tags):
        ''' Map sup oie tags to conll (used in SRL) tags '''
        def mapper(tag):
            if tag == 'O':
                return tag
            else:
                bio, pa = tag.split('-')
                if bio not in {'B', 'I'}:
                    raise ValueError('tag error')
                if pa.startswith('ARG'):
                    pos = pa[3:]
                    return 'A{}-{}'.format(int(pos), bio)
                elif pa =='V':
                    return 'P-B'
                else:
                    return 'O'
        return [mapper(tag) for tag in tags]


    def map_tags(self, tags, one_verb=True):
        ''' Map sup oie tags to conll (used in SRL) tags '''
        def mapper(tag):
            nv = 0
            if tag == 'O':
                return tag
            else:
                pa, bio = tag.split('-')
                if bio not in {'B', 'I'}:
                    raise ValueError('tag error')
                if pa.startswith('A'):
                    pos = pa[1:]
                    return '{}-ARG{}'.format(bio, int(pos))
                elif pa == 'P':
                    if one_verb and (bio != 'B' or nv > 0):
                        return 'O'
                    nv += 1
                    return '{}-{}'.format(bio, 'V')
                else:
                    raise ValueError('tag error')
        return [mapper(tag) for tag in tags]


@DatasetReader.register('rerank')
class RerankReader(DatasetReader):
    '''
    Designed to read conll file used in supervised-oie project.
    '''
    def __init__(self,
                 default_task = 'gt', # "gt" is the default task (ground truth)
                 one_verb = True, # each instance only have one "V" tag which is "B-V"
                 token_indexers: Dict[str, TokenIndexer] = None,
                 skip_neg: bool = False, # if True, negative samples are dropped
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._default_task = default_task
        self._one_verb = one_verb
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._skip_neg = skip_neg

    @overrides
    def _read(self, filepath: str):
        filepath = cached_path(filepath)
        soc = SupOieConll()
        for ext in soc.sentence_iterator(filepath):
            verb_ind = [int(i == ext.pred_ind) for i in range(len(ext.words))]
            tokens = [Token(t) for t in ext.words]
            if not any(verb_ind):
                continue # skip extractions without predicate
            if self._skip_neg and ext.label == 0:
                continue # skip negative examples
            yield self.text_to_instance(tokens, verb_ind,
                                        tags=soc.map_tags(ext.tags, one_verb=self._one_verb),
                                        label=ext.label, weight=ext.weight)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         tags: List[str],
                         task: str = None,
                         label: int = None,
                         weight: float = None) -> Instance:
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        task = task or self._default_task
        fields['task_labels'] = LabelField(task, label_namespace='task_labels')
        fields['tags'] = SequenceLabelField(tags, text_field)
        if label is not None:
            fields['labels'] = IntField(label if label != 0 else -1)
        if weight is not None:
            fields['weights'] = FloadField(weight)
        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields['metadata'] = MetadataField({'words': [x.text for x in tokens],
                                            'verb': verb})
        return Instance(fields)
