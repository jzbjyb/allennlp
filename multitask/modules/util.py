from typing import Dict, List

import torch


class SeqWordDropout(torch.nn.Module):
    def __init__(self, p: float = 0.0):
        super(SeqWordDropout, self).__init__()
        self.p = p


    def forward(self, seq: torch.Tensor):
        mask = torch.rand_like(seq[:, :, 0])
        mask = (mask >= self.p).type(seq.dtype)
        return seq * mask.unsqueeze(-1)


def modify_req_grad(module: torch.nn.Module,
                    req_grad: bool,
                    skip: List[str] = None):
    for name, param in module.named_parameters():
        if skip is not None and name in skip:
            continue
        param.requires_grad = req_grad


def hasattr_recurse(inst, name):
    names = name.split('.')
    for name in names:
        if not hasattr(inst, name):
            return False
        inst = getattr(inst, name)
    return True


def getattr_recurse(inst, name):
    names = name.split('.')
    for name in names:
        inst = getattr(inst, name)
    return inst


def setattr_recurse(inst, name, newattr):
    names = name.split('.')
    for name in names[:-1]:
        inst = getattr(inst, name)
    setattr(inst, names[-1], newattr)


def share_weights(module1: torch.nn.Module,
                  module2: torch.nn.Module,
                  name_mapping: Dict[str, str] = None,
                  skip: List[str] = None):
    '''
    share/copy all the parameters from module2 to module1
    '''
    for name, param in module1.named_parameters():
        if skip is not None and name in skip:
            continue
        name2 = name
        if name_mapping is not None and name in name_mapping:
            name2 = name_mapping[name]
        if not hasattr_recurse(module2, name2):
            raise Exception('{} is not contained in {}'.format(name2, module2))
        setattr_recurse(module1, name, getattr_recurse(module2, name2))
