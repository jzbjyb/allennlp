import logging
from typing import Dict, List, Iterable, Union

from overrides import overrides
from operator import itemgetter
import numpy as np

from allennlp.common.file_utils import cached_path
from allennlp.common.params import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
from multitask.dataset_readers.util import FloadField

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("srl_mt")
class SrlReaderMultiTask(DatasetReader):
    """
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 # "gt" is the default task (ground truth)
                 default_task: str = 'gt',
                 # yield up samples from multiple files uniformly
                 multiple_files: bool = False,
                 # whether to restart iterating a file when it is exhausted.
                 # Only effective when `multiple_files` is True
                 restart_file: bool = True,
                 # The ratio of samples drawn from each file.
                 # Only effective when `multiple_files` is True
                 multiple_files_sample_rate: List[int] = None,
                 # weights of each task with the number of samples considered
                 task_weight: Union[Params, Dict[str, float]] = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 domain_identifier: str = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._default_task = default_task
        self._multiple_files = multiple_files
        self._restart_file = restart_file
        self._multiple_files_sample_rate = multiple_files_sample_rate
        if type(task_weight) is Params: # directly configured by json file
            task_weight = task_weight.as_dict()
        self._task_weight = task_weight
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    @overrides
    def _read(self, file_path: str):
        logger.info('Reading multitask instances from dataset files at: {}'.format(file_path))
        if self._domain_identifier is not None:
            logger.info('Filtering to only include file paths containing the {} domain'.format(
                self._domain_identifier))
        if self._multiple_files:
            file_path_li = file_path.split(':')
            # iterate through multiple files accroding to the ratio of `multiple_files_sample_rate`
            if self._multiple_files_sample_rate is None:
                # use uniform sampling as default
                self._multiple_files_sample_rate = [1] * len(file_path_li)
            assert len(self._multiple_files_sample_rate) == len(file_path_li), \
                'number of items in multiple_files_sample_rate should be the same as the number of files'
            for sr in self._multiple_files_sample_rate:
                assert type(sr) is int, 'multiple_files_sample_rate must be a list of int'
            readers = [self._read_one_file(file_path) for file_path in file_path_li]
            stop_set = set()
            restart = 0
            while True:
                buf: List[Instance] = []
                for i, (reader, sr) in enumerate(zip(readers, self._multiple_files_sample_rate)):
                    for j in range(sr): # yield up `sr` samples from the current file
                        try:
                            buf.append(reader.__next__())
                        except StopIteration:
                            stop_set.add(i)
                            if self._restart_file:
                                restart += 1
                                # restart the current file
                                readers[i] = self._read_one_file(file_path_li[i])
                                buf.append(readers[i].__next__())
                if len(stop_set) >= len(file_path_li): # exit when all the files are exhausted
                    break
                yield from buf
        else:
            # only one file
            yield from self._read_one_file(file_path)

    def _read_one_file(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        #file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        for sentence in ontonotes_reader.sentence_iterator(file_path):
            tokens = [Token(t) for t in sentence.words]
            if sentence.srl_frames:
                # skip sentence with no predicates because we don't know which task it should be
                for (_, tts) in sentence.srl_frames:
                    # "#" is used to separate tag and task (for example, "ARG0#openie4")
                    tags, tasks = [], []
                    for tt in tts:
                        tt_li = tt.split('#')
                        if len(tt_li) > 2:
                            print(tt_li)
                            raise ValueError('tag is not in valid multi-task format')
                        tags.append(tt_li[0]) # tag should not include task
                        if len(tt_li) == 2:
                            tasks.append(tt_li[1])
                    verb_indicator = [1 if label[-2:] == '-V' else 0 for label in tags]
                    task = tasks[0] if len(tasks) > 0 else None # if None, use default task
                    if task is not None and len(np.unique(tasks)) != 1:
                        raise ValueError('inconsistent task')
                    yield self.text_to_instance(tokens, verb_indicator, task=task, tags=tags)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader: Ontonotes,
                          file_path: str,
                          domain_identifier: str) -> Iterable[OntonotesSentence]:
        """
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or f"/{domain_identifier}/" in conll_file:
                yield from ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         task: str = None,
                         tags: List[str] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields['tokens'] = text_field
        fields['verb_indicator'] = SequenceLabelField(verb_label, text_field)
        task = task or self._default_task
        fields['task_labels'] = LabelField(task, label_namespace='task_labels')
        weight = self._task_weight[task] or 1.0
        fields['weight'] = FloadField(weight)
        if tags:
            # use different namespaces for different task
            fields['tags'] = SequenceLabelField(tags, text_field, 'MT_{}_labels'.format(task))
        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields['metadata'] = MetadataField({'words': [x.text for x in tokens], 'verb': verb})
        return Instance(fields)
