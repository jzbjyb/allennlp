import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.iterators.bucket_iterator import sort_by_padding

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def split_by_task(instance_li: List[Instance], task_namespace: str) -> Dict[str, List[Instance]]:
    result = {}
    for inst in instance_li:
        task = inst[task_namespace].label
        if task not in result:
            result[task] = []
        result[task].append(inst)
    return result


def interleave_by_task(inst_li_by_task: Dict[str, List[Instance]]):
    task_len_li = [(k, len(inst_li_by_task[k])) for k in inst_li_by_task]
    num_inst = np.sum([l[1] for l in task_len_li])
    ideal_dist = np.array([l[1] / num_inst for l in task_len_li])
    cur_dist = np.zeros_like(ideal_dist)
    task_ind = np.zeros_like(ideal_dist, dtype=int)
    result = []
    for i in range(num_inst):
        task = np.argmax(ideal_dist - cur_dist / (np.sum(cur_dist) + 1e-5))
        result.append(inst_li_by_task[task_len_li[task][0]][task_ind[task]])
        task_ind[task] += 1
        cur_dist[task] += 1
    for i, tl in enumerate(task_ind):
        assert len(inst_li_by_task[task_len_li[i][0]]) == tl, 'task interleave failure'
    return result


@DataIterator.register('task_bucket')
class TaskBucketIterator(DataIterator):
    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 task_namespace: str = 'task_labels',
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        if not sorting_keys:
            raise ConfigurationError('TaskBucketIterator requires sorting_keys to be specified')

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._sorting_keys = sorting_keys
        self._task_namespace = task_namespace
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        for instance_list in self._memory_sized_lists(instances):

            inst_li_by_task = split_by_task(instance_list, task_namespace=self._task_namespace)
            inst_li_by_task = dict((k, sort_by_padding(
                inst_li_by_task[k], self._sorting_keys, self.vocab, self._padding_noise)) for k in inst_li_by_task)
            instance_list = interleave_by_task(inst_li_by_task)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))

            # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
            # num_gpu batches together, shuffle and then expand the groups.
            # This guards against imbalanced batches across GPUs.
            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches
