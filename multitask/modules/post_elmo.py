import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn.util import remove_sentence_boundaries

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PostElmo(torch.nn.Module):
    def __init__(self,
                 pre_elmo_output_dim: int,
                 pre_elmo_num_layers: int,
                 num_output_representations: int,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 keep_sentence_boundaries: bool = False,
                 scalar_mix_parameters: List[float] = None,
                 projection_dim: int = None) -> None:
        super(PostElmo, self).__init__()
        '''
        components in Elmo
        '''
        self._keep_sentence_boundaries = keep_sentence_boundaries
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(
                pre_elmo_num_layers,
                do_layer_norm=do_layer_norm,
                initial_scalar_parameters=scalar_mix_parameters,
                trainable=scalar_mix_parameters is None)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)
        '''
        components in ElmoTokenEmbedder
        '''
        if projection_dim:
            self._projection = torch.nn.Linear(pre_elmo_output_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = pre_elmo_output_dim


    def get_output_dim(self):
        return self.output_dim


    def forward(self,    # pylint: disable=arguments-differ
                layer_activations: List[torch.Tensor],
                mask_with_bos_eos: torch.LongTensor,
                original_shape: torch.Size,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        '''
        steps in Elmo
        '''
        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                        representation_with_bos_eos, mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if len(original_word_size) > 2:
                mask = processed_mask.view(original_word_size)
                elmo_representations = [representation.view(original_word_size + (-1, ))
                                        for representation in representations]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1, ))
                                    for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations

        '''
        steps in ElmoTokenEmbedder
        '''
        elmo_representations = elmo_representations[0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations
