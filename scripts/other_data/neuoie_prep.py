from typing import List, Tuple
from operator import itemgetter
import argparse, re
from tqdm import tqdm


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


class CannotIdentifySpan(Exception):
    pass


def len_in_char(text):
    return len(text)


def arg_to_openie4(sub_tokens: List[Tuple[str, int]], tokens: List[str]):
    st = len_in_char(' '.join(tokens[:sub_tokens[0][1]])) + (sub_tokens[0][1] > 0)
    sub = ' '.join(map(itemgetter(0), sub_tokens))
    ed = st + len_in_char(sub)
    return '{}({},List([{}, {})))'.format('SimpleArgument', sub, st, ed)


def pred_to_openie4(sub_tokens: List[Tuple[str, int]], tokens: List[str]):
    sts, eds = [], []
    for w in sub_tokens:
        st = len_in_char(' '.join(tokens[:w[1]])) + (w[1] > 0)
        ed = st + len_in_char(w[0])
        sts.append(st)
        eds.append(ed)
    return '{}({},List({}))'.format(
        'Relation', ' '.join(map(itemgetter(0), sub_tokens)),
        ', '.join('[{}, {})'.format(st, ed) for st, ed in zip(sts, eds)))


def span_to_sub_tokens(span: str, sentence: str):
    #ind = [m.start() for m in re.finditer(' ' + span + ' ', ' ' + sentence + ' ')]
    ind = list(find_all(' ' + sentence + ' ', ' ' + span + ' '))
    if len(ind) > 1:
        raise CannotIdentifySpan
    if len(ind) == 0:
        raise CannotIdentifySpan
    ind = ind[0]
    st = len(list(re.finditer(' ', sentence[:ind])))  # span start token index
    return [(w, st + i) for i, w in enumerate(span.split(' '))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess NeuralOpenIE dataset released by Cui, et al.')
    parser.add_argument('--sent_file', type=str, required=True, help='input sentence file')
    parser.add_argument('--triple_file', type=str, required=True, help='input triple file')
    parser.add_argument('--out', type=str, required=True, help='path to the output file')
    parser.add_argument('--format', choices=['openie'], default='openie', help='output format')
    parser.add_argument('--first', type=int, default=0, help='only use first n lines')
    args = parser.parse_args()

    num, skip_num = 0, 0
    with open(args.sent_file, 'r') as sent_fin, \
            open(args.triple_file, 'r') as tri_fin, \
            open(args.out, 'w') as fout:
        for i, sent in enumerate(sent_fin):
            sent = sent.strip()
            tokens = sent.split(' ')
            tri = tri_fin.readline().strip()[7:-8]  # skip '<arg1> ' and ' </arg2>'
            arg1 = tri.split('</arg1> <rel>')
            rel, arg2 = arg1[1].split('</rel> <arg2>')
            arg1 = arg1[0].strip()
            rel = rel.strip()
            arg2 = arg2.strip()
            try:
                arg1 = span_to_sub_tokens(arg1, sent)
                arg2 = span_to_sub_tokens(arg2, sent)
                rel = span_to_sub_tokens(rel, sent)
            except CannotIdentifySpan:
                skip_num += 1
                continue
            arg1 = arg_to_openie4(arg1, tokens)
            arg2 = arg_to_openie4(arg2, tokens)
            rel = pred_to_openie4(rel, tokens)

            num += 1
            if args.first and num >= args.first:
                break
            fout.write('{}\t\t{}\t{}\t{}\t{}\n'.format(
                0, arg1, rel, arg2, sent))
    print('skip {} to get {}'.format(skip_num, num))
