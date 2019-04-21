import torch


def modify_req_grad(module: torch.nn.Module, req_grad: bool):
    for param in module.parameters():
        param.requires_grad = req_grad
