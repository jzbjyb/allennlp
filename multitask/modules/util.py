from typing import Dict

import torch


def modify_req_grad(module: torch.nn.Module, req_grad: bool):
    for param in module.parameters():
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


def share_weights(module1: torch.nn.Module, module2: torch.nn.Module, name_mapping: Dict = {}):
    '''
    share/copy all the parameters from module2 to module1
    '''
    for name, param in module1.named_parameters():
        print(name)
    for name, param in module2.named_parameters():
        print(name)
    for name, param in module1.named_parameters():
        if name in name_mapping:
            name = name_mapping[name]
        if not hasattr_recurse(module2, name):
            raise Exception('{} is not contained in {}'.format(name, module2))
        setattr_recurse(module1, name, getattr_recurse(module2, name))
