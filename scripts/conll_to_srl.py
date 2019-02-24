import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, root_dir)
import argparse
from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence
import numpy as np

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
    parser = argparse.ArgumentParser(description='convert conll file to srl predictions')
    parser.add_argument('--model', type=str, help='model file', required=True)
    parser.add_argument('--inp', type=str, help='input conll file.', required=True)
    parser.add_argument('--out', type=str, help='output file.', required=True)
    parser.add_argument('--format', type=str, help='format of the output file',
                        default='tagging', choices=['tagging', 'conll'])
    parser.add_argument('--cuda_device', type=int, default=0, help='id of GPU to use (if any)')
    args = parser.parse_args()


    arc = load_archive(args.model, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(arc, predictor_name='semantic-role-labeling')
    results = predictor.predict_conll_file(args.inp, batch_size=256)
    with open(args.out, 'w') as fout:
        for result in results:
            if args.format == 'tagging':
                raw_tags = result['raw_tags'] # supposed to be openie tags
                tags = result['tags'] # supposed to be srl tags
                assert len(tags) == len(raw_tags), 'tag seq len inconsistent'
                sent = ' '.join(map(lambda x: '{}/{}'.format(*x), zip(tags, raw_tags)))
                fout.write('{}\n'.format(sent))

