import argparse


def convert_conll_tag_to_tag_with_task(tag, task):
    ind = tag.find('*')
    if ind < 0:
        raise ValueError('malformed tag in conll file')
    if ind > 0: # add task if some tag exists before "*"
        return '{}#{}{}'.format(tag[:ind], task, tag[ind:])
    else:
        return tag


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combine multiple conll files into one file by adding task indicators')
    parser.add_argument('--inp', type=str, help='input conll files separated by ":"', required=True)
    parser.add_argument('--out', type=str, help='path to the output file', required=True)
    parser.add_argument('--task', type=str, help='task of each file separated by :', required=True)
    args = parser.parse_args()

    tasks = args.task.split(':')
    inps = args.inp.split(':')
    assert len(inps) == len(tasks), 'the number of inps should be equal to the number of tasks'

    with open(args.out, 'w') as fout:
        nl = True # used to add newline between different files
        for i, (inp, task) in enumerate(zip(inps, tasks)):
            if not nl:
                print('add newline between {} and {}'.format(inp, inps[i-1]))
                fout.write('\n')
            with open(inp, 'r') as fin:
                for l in fin:
                    if len(l.strip()) > 0 and not l.startswith('#'):
                        # add task to non-empty and non-comment lines
                        nl = False
                        conll_components = l.strip().split() # use any whitespace to split
                        conll_components[11:-1] = [convert_conll_tag_to_tag_with_task(tag, task)
                                                   for tag in conll_components[11:-1]]
                        l = '\t'.join(conll_components) + '\n' # use tab to concat
                    else: # newline
                        nl = True
                    fout.write(l)
