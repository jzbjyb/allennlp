import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, root_dir)
import argparse
from functools import reduce
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules
from typing import List, Dict
import numpy as np
import itertools


def read_raw_sents(filepath: str) -> List[List[str]]:
    sents_token = []
    with open(filepath, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l == '':
                continue
            sents_token.append(l.split(' '))
    return sents_token


class Extraction:
    def __init__(self, sent, pred, args, probs,
                 calc_prob=lambda probs: 1.0 / (reduce(lambda x, y: x * y, probs) + 0.001)):
        self.sent = sent
        self.calc_prob = calc_prob
        self.probs = probs
        self.prob = self.calc_prob(self.probs)
        self.pred = pred
        self.args = args


    def _format_aop(self, aop):
        return ' '.join(map(lambda x: x[0], aop)) + '##' + str(aop[0][1])


    def __str__(self):
        return '\t'.join(map(str, [' '.join(self.sent), self.prob, self._format_aop(self.pred),
            '\t'.join([self._format_aop(arg) for arg in self.args])]))


def allennlp_prediction_to_extraction(preds: List[Dict],
                                      max_n_arg: int = 10, merge=True) -> List[Extraction]:
    '''
    Assume the tag sequence is reasonable (no validation check)
    When unmerge=True, spans with the same argument index will be separated into different extractions.
    '''
    exts = []
    for pred in preds:
        tokens = pred['words']
        for ext in pred['verbs']:
            probs = []
            pred = []
            args = [[] for _ in range(max_n_arg)]
            last_ai = -1
            for i, w, t, p in zip(range(len(tokens)), tokens, ext['tags'], ext['probs']):
                probs.append(p)
                if t.find('V') >= 0:
                    pred.append((w, i))
                    last_ai = -1
                elif t.find('ARG') >= 0:
                    ai = int(t[t.find('ARG')+3:])
                    if ai >= len(args):
                        raise ValueError('too many args')
                    if last_ai != ai:
                        args[ai].append([]) # create new coordination argument placeholder
                    args[ai][-1].append((w, i))
                    last_ai = ai
                else:
                    last_ai = -1
            # Remove empty argument position (for example, arg2 exists without arg1).
            args = [arg for arg in args if len(arg) > 0]
            if len(pred) <= 0 or len(args) <= 0:
                continue
            # merge all the arguments at the same position
            if merge:
                args = [[[w for a in arg for w in a]] for arg in args]
            # iterate through all the combinations
            for arg in itertools.product(*args):
                exts.append(Extraction(sent=tokens, pred=pred, args=arg, probs=probs, calc_prob=np.mean))
    return exts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract openie extractions')
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input file of raw sentences.', required=True)
    parser.add_argument('--out', type=str, help='output file, where extractions should be written.', required=True)
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    parser.add_argument('--unmerge', help='whether to generate multiple extraction for one predicate',
                        action='store_true')
    args = parser.parse_args()

    import_submodules('multitask')

    arc = load_archive(args.model, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(arc, predictor_name='open-information-extraction')
    sents_tokens = read_raw_sents(args.inp)
    preds = predictor.predict_batch(sents_tokens, batch_size=256, warm_up=3)
    exts = allennlp_prediction_to_extraction(preds, max_n_arg=10, merge=not args.unmerge)
    with open(args.out, 'w') as fout:
        for ext in exts:
            fout.write('{}\n'.format(ext))
