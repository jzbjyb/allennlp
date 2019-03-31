from typing import Dict
from overrides import overrides
from allennlp.data.fields import Field
import torch


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