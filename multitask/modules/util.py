from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary


class SeqWordDropout(torch.nn.Module):
    def __init__(self, p: float = 0.0):
        super(SeqWordDropout, self).__init__()
        self.p = p


    def forward(self, seq: torch.Tensor):
        # TODO: remove dropout at test time
        #if not self.training:
        #    return seq
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


def read_oie_srl_ana_file(filename, n=2000):
    results: List[List[Tuple]] = []
    with open(filename, 'r') as fin:
        for i, l in enumerate(fin):
            if i >= n:
                break
            r = [w.split(' ') for w in l.strip().split('\t')[1:]]
            results.append(r)
    return results


def oie_srl_ana(tags_li: List[List[Tuple]], mask_filter=None):
    tsd = defaultdict(lambda: 0)
    tod = defaultdict(lambda: 0)
    all = 0
    exact = 0
    srlbnum = 0
    bnum, inum, onum = 0, 0, 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for tags in tags_li:
        for w, vi, tsm, ts, to in tags:
            if mask_filter is not None and tsm != mask_filter:
                continue
            all += 1
            # count tags
            tsd[ts] += 1
            tod[to] += 1
            # collect O
            if ts == 'O' and to == 'O':
                tn += 1
            elif ts == 'O' and to != 'O':
                fp += 1
            elif ts != 'O' and to != 'O':
                tp += 1
            else:
                fn += 1
            # bio of srl
            if ts.startswith('B'):
                srlbnum += 1
            # bio
            if to == 'O':
                onum += 1
            elif to.startswith('B'):
                bnum += 1
            elif to.startswith('I'):
                inum += 1
            # exact
            if ts == to:
                exact += 1

    print('#sent {}, #toknes {}'.format(len(tags_li), all))
    print('exact {}'.format(exact / all))
    print('tp, tn, fp, fn: {}, {}, {}, {}'.format(tp / all, tn / all, fp / all, fn / all))
    print('bio: {} {} {}'.format(bnum / all, inum / all, onum / all))
    print('#b per sent (oie): {}'.format(bnum / len(tags_li)))
    print('#b per sent (srl): {}'.format(srlbnum / len(tags_li)))
    print(sorted(tod.items(), key=lambda x: -x[1]))


class Rule():
    def __init__(self, vocab: Vocabulary, y1_ns: str, y2_ns: str):
        self.vocab = vocab
        self.y1_o = self.vocab.get_token_index('O', namespace=y1_ns)
        self.y2_o = self.vocab.get_token_index('O', namespace=y2_ns)


    def diff_boundary_rule(self,
                           y1: torch.LongTensor,  # SHAPE: (beam_size, batch_size, seq_len)
                           y2: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                           mask: torch.LongTensor  # SHAPE: (batch_size, seq_len)
                           ) -> torch.LongTensor:
        y1_b = y1[:, :, :-1].ne(y1[:, :, 1:]).long() * mask[:, :-1].unsqueeze(0)
        y2_b = y2[:, :-1].ne(y2[:, 1:]).long() * mask[:, :-1]
        ratio = (y1_b & y2_b.unsqueeze(0)).sum(-1).float() / (y1_b.sum(-1).float() + 1e-10)
        return 1.0 - ratio


    def o_boundary_rule(self,
                        y1: torch.LongTensor,  # SHAPE: (beam_size, batch_size, seq_len)
                        y2: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                        mask: torch.LongTensor  # SHAPE: (batch_size, seq_len)
                        ) -> torch.LongTensor:
        y1 = y1.ne(self.y1_o).long() * mask.unsqueeze(0)
        y2 = y2.ne(self.y2_o).long() * mask
        y1 = F.pad(y1, [1, 1], 'constant', 0)
        y2 = F.pad(y2, [1, 1], 'constant', 0)
        y1_b = y1[:, :, 1:-1] & (1 - (y1[:, :, :-2] & y1[:, :, 2:]))
        y2_b = y2[:, 1:-1] & (1 - (y2[:, :-2] & y2[:, 2:]))
        ratio = (y1_b & y2_b.unsqueeze(0)).sum(-1).float() / (y1_b.sum(-1).float() + 1e-10)
        return 1.0 - ratio
