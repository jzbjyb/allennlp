import argparse
from tqdm import tqdm
from allennlp.data.dataset_readers.dataset_utils import Ontonotes


def convert_conll_tag_to_tag_with_task(tag, task):
    ind = tag.find('*')
    if ind < 0:
        raise ValueError('malformed tag in conll file')
    if ind > 0: # add task if some tag exists before "*"
        return '{}#{}{}'.format(tag[:ind], task, tag[ind:])
    else:
        return tag


def comb_conll_with_task(args):
    tasks = args.task.split(':')
    inps = args.inp.split(':')
    assert len(inps) == len(tasks), 'the number of inps should be equal to the number of tasks'

    with open(args.out, 'w') as fout:
        nl = True  # used to add newline between different files
        for i, (inp, task) in enumerate(zip(inps, tasks)):
            if not nl:
                print('add newline between {} and {}'.format(inp, inps[i - 1]))
                fout.write('\n')
            with open(inp, 'r') as fin:
                for l in fin:
                    if l.startswith('#'): # comment line
                        nl = False
                    elif len(l.strip()) > 0:
                        # add task to non-empty and non-comment lines
                        nl = False
                        conll_components = l.strip().split()  # use any whitespace to split
                        conll_components[11:-1] = [convert_conll_tag_to_tag_with_task(tag, task)
                                                   for tag in conll_components[11:-1]]
                        l = '\t'.join(conll_components) + '\n'  # use tab to concat
                    else:  # newline
                        nl = True
                    fout.write(l)


def comb_conll_in_ontonotes(args):
    domain_identifier = None
    ontonotes_reader = Ontonotes()
    num_files = len([1 for _ in ontonotes_reader.dataset_path_iterator(args.inp)])
    with open(args.out, 'w') as fout:
        for conll_file in tqdm(ontonotes_reader.dataset_path_iterator(args.inp), total=num_files):
            if domain_identifier is not None and f'/{domain_identifier}/' not in conll_file:
                continue
            with open(conll_file) as fin:
                fout.write(fin.read()) # don't have to insert newline between files in ontonotes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine multiple conll files into one file')
    parser.add_argument('--op', choices=['ontonotes', 'merge'], required=True,
                        help='''"ontonotes" combine all the gold_conll files in a directory,
                        "merge" combine several conll files along with their tasks''')
    parser.add_argument('--inp', type=str, required=True,
                        help='input conll files separated by ":", or a dir')
    parser.add_argument('--out', type=str, required=True, help='path to the output file')
    parser.add_argument('--task', type=str, help='task of each file separated by :')
    args = parser.parse_args()

    if args.op == 'ontonotes':
        comb_conll_in_ontonotes(args)
    elif args.op == 'merge':
        comb_conll_with_task(args)

