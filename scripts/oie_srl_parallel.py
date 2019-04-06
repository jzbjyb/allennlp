import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, root_dir)
import argparse
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.common.util import import_submodules
import numpy as np
from tqdm import tqdm

def get_coordination_ext(conll_filepath, debug=False):
    '''
    Analyse coordination gold standard extractions
    '''
    ontonotes_reader = Ontonotes()
    coordination_count = 0
    for sentence in ontonotes_reader.sentence_iterator(conll_filepath):
        if sentence.srl_frames:
            tokens = sentence.words
            for (_, tags) in sentence.srl_frames:
                arg_tags = [tag for tag in tags if 'B-ARG' in tag]
                #verb_indicator = [1 if tag[-2:] == "-V" else 0 for tag in tags]
                if len(np.unique(arg_tags)) != len(arg_tags):
                    coordination_count += 1
                    if debug:
                        arg_tags, counts = np.unique(arg_tags, return_counts=True)
                        args_tags = sorted(zip(arg_tags, counts), key=lambda x: x[0])
                        print(args_tags)
                        print(' '.join(tokens))
                        print(' '.join(map(lambda x: '{}/{}'.format(*x), zip(tokens, tags))))
                        input()
    print('{} coordination extractions'.format(coordination_count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert oie conll file to srl predictions')
    parser.add_argument('--direction', type=str, help='direction of the convertion',
                        choices=['oie2srl', 'srl2oie'], required=True)
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input conll file or directory.', required=True)
    parser.add_argument('--out', type=str, help='output file.', required=True)
    parser.add_argument('--format', type=str, help='format of the output file',
                        default='tagging', choices=['tagging', 'conll'])
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()

    import_submodules('multitask')

    arc = load_archive(args.model, cuda_device=args.cuda_device)

    # generate srl-oie parallel data
    if args.direction == 'oie2srl':
        predictor = Predictor.from_archive(arc, predictor_name='semantic-role-labeling')
    elif args.direction == 'srl2oie':
        predictor = Predictor.from_archive(arc, predictor_name='open-information-extraction')
    results = predictor.predict_conll_file(args.inp, batch_size=128) # result generator

    # save to file
    if type(results) is list:
        print('{} srl-oie parallel samples'.format(len(results)))
    with open(args.out, 'w') as fout:
        for result in tqdm(results):
            if args.format == 'tagging':
                words = result['words']  # word sequence
                verb = result['verb']  # verb indicators (0/1)
                srl_tags = result['srl_tags']
                oie_tags = result['oie_tags']
                assert len(words) == len(verb) and len(words) == len(srl_tags) and len(words) == len(oie_tags), \
                    'seq len inconsistent'
                # word verb srl oie
                sent = '\t'.join(map(lambda x: '{} {} {} {}'.format(*x),
                                     zip(words, verb, srl_tags, oie_tags)))
                fout.write('{}\n'.format(sent))
            else:
                raise NotImplementedError
