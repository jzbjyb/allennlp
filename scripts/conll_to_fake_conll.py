import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../..'))
sys.path.insert(0, root_dir)
import argparse
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from multitask.dataset_readers.rerank_reader import SupOieConll


if __name__ == '__main__':
    # only conll file with one extraction per fragment can be converted
    parser = argparse.ArgumentParser(description='convert conll file to fake conll file used in sup oie')
    parser.add_argument('--inp', type=str, help='input conll file.', required=True)
    parser.add_argument('--out', type=str, help='output fake conll file.', required=True)
    args = parser.parse_args()

    ontonotes_reader = Ontonotes()
    coordination_count = 0
    soc = SupOieConll()
    ind = 0
    with open(args.out, 'w') as fout:
        fout.write('word_id\tword\tpred\tpred_id\thead_pred_id\tsent_id\trun_id\tlabel\n')
        for sentence in ontonotes_reader.sentence_iterator(args.inp):
            if not sentence.srl_frames:
                continue
            _, labels = sentence.srl_frames[0]
            tokens = sentence.words
            labels = soc.map_tags_reverse(labels)
            head_pred = [int(l == 'P-B') for l in labels]
            head_pred = head_pred.index(1)
            for i, t, l in zip(range(len(tokens)), tokens, labels):
                fout.write('{}\t{}\tpred\tpred_id\t{}\t{}\t{}\t{}\n'.format(i, t, head_pred, ind, ind, l))
            fout.write('\n')
            ind += 1
