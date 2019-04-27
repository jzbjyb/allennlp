import argparse
import random
from tqdm import tqdm
random.seed(2019)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shuffle one or several conll files randomly')
    parser.add_argument('--inp', type=str, required=True,
                        help='input conll files separated by ":"')
    parser.add_argument('--out', type=str, required=True, help='path to the output file')
    args = parser.parse_args()

    filenames = args.inp.split(':')
    samples, sample = [], []
    for filename in filenames:
        print('read from "{}"'.format(filename))
        with open(filename, 'r') as fin:
            for l in tqdm(fin):
                if l.startswith('#'):  # comment line: remove
                    continue
                elif len(l.strip()) == 0: # newline: split
                    if len(sample) > 0:
                        samples.append(''.join(sample))
                    sample = []
                else:  # content line: keep
                    sample.append(l)
    if len(sample) > 0: # the last sample
        samples.append(''.join(sample))
        sample = []

    print('{} samples'.format(len(samples)))
    random.shuffle(samples)
    print('write')
    with open(args.out, 'w') as fout:
        for sample in tqdm(samples):
            fout.write('{}\n'.format(sample))
