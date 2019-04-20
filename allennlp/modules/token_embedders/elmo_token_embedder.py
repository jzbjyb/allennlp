from typing import List, Any, Dict
import torch

from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.data import Vocabulary


@TokenEmbedder.register("elmo_token_embedder")
class ElmoTokenEmbedder(TokenEmbedder):
    """
    Compute a single layer of ELMo representations.

    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...)

    Parameters
    ----------
    options_file : ``str``, required.
        An ELMo JSON options file.
    weight_file : ``str``, required.
        An ELMo hdf5 weight file.
    do_layer_norm : ``bool``, optional.
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional.
        The dropout value to be applied to the ELMo representations.
    requires_grad : ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    projection_dim : ``int``, optional
        If given, we will project the ELMo embedding down to this dimension.  We recommend that you
        try using ELMo with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particularly where there is very limited training data).
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, the ElmoTokenEmbedder expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    scalar_mix_parameters : ``List[int]``, optional, (default=None)
        If not ``None``, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training.
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 requires_grad: bool = False,
                 projection_dim: int = None,
                 vocab_to_cache: List[str] = None,
                 scalar_mix_parameters: List[float] = None,
                 stateful: bool = True,
                 use_post_elmo: bool = False) -> None:
        super(ElmoTokenEmbedder, self).__init__()

        self._use_post_elmo = use_post_elmo
        self._elmo = Elmo(options_file,
                          weight_file,
                          1,
                          do_layer_norm=do_layer_norm,
                          dropout=dropout,
                          requires_grad=requires_grad,
                          vocab_to_cache=vocab_to_cache,
                          scalar_mix_parameters=scalar_mix_parameters,
                          stateful=stateful,
                          use_post_elmo=use_post_elmo)
        if use_post_elmo:
            self.output_dim = self._elmo.get_output_dim()
            self._post_elmo_params = {
                "pre_elmo_output_dim": self.output_dim,
                "pre_elmo_num_layers": self._elmo._elmo_lstm.num_layers,
                "num_output_representations": 1,
                "do_layer_norm": do_layer_norm,
                "dropout": dropout,
                "keep_sentence_boundaries": False,
                "scalar_mix_parameters": scalar_mix_parameters,
                "projection_dim": projection_dim,
            }
        else:
            if projection_dim:
                self._projection = torch.nn.Linear(self._elmo.get_output_dim(), projection_dim)
                self.output_dim = projection_dim
            else:
                self._projection = None
                self.output_dim = self._elmo.get_output_dim()

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_use_post_elmo(self) -> bool:
        return self._use_post_elmo

    def get_post_elmo_params(self) -> Dict[str, Any]:
        return self._post_elmo_params

    def forward(self, # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                word_inputs: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs, word_inputs)
        if self._use_post_elmo:  # return intermediate results
            return elmo_output
        elmo_representations = elmo_output['elmo_representations'][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ElmoTokenEmbedder':  # type: ignore
        # pylint: disable=arguments-differ
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('weight_file')
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        requires_grad = params.pop('requires_grad', False)
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        dropout = params.pop_float("dropout", 0.5)
        namespace_to_cache = params.pop("namespace_to_cache", None)
        if namespace_to_cache is not None:
            vocab_to_cache = list(vocab.get_token_to_index_vocabulary(namespace_to_cache).keys())
        else:
            vocab_to_cache = None
        projection_dim = params.pop_int("projection_dim", None)
        scalar_mix_parameters = params.pop('scalar_mix_parameters', None)
        stateful = params.pop('stateful', True)
        use_post_elmo = params.pop("use_post_elmo", False)
        params.assert_empty(cls.__name__)
        return cls(options_file=options_file,
                   weight_file=weight_file,
                   do_layer_norm=do_layer_norm,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim,
                   vocab_to_cache=vocab_to_cache,
                   scalar_mix_parameters=scalar_mix_parameters,
                   stateful=stateful,
                   use_post_elmo=use_post_elmo)
