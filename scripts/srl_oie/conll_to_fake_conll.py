import sys, os
from tqdm import tqdm
root_dir = os.path.abspath(os.path.join(__file__, '../../..'))
sys.path.insert(0, root_dir)
import argparse
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from multitask.dataset_readers.rerank_reader import SupOieConll


if __name__ == '__main__':
    # only conll file with one extraction per fragment can be converted
    parser = argparse.ArgumentParser(description='convert conll file to fake conll file used in sup oie')
    parser.add_argument('--inp', type=str, help='input conll dir.', required=True)
    parser.add_argument('--out', type=str, help='output fake conll file.', required=True)
    parser.add_argument('--domain', type=str, help='domain of conll', default=None)
    args = parser.parse_args()

    ontonotes_reader = Ontonotes()
    coordination_count = 0
    soc = SupOieConll()
    ind = 0
    with open(args.out, 'w') as fout:
        fout.write('word_id\tword\tpred\tpred_id\thead_pred_id\tsent_id\trun_id\tlabel\n')
        for conll_file in tqdm(ontonotes_reader.dataset_path_iterator(args.inp)):
            if args.domain is None or f'/{args.domain}/' in conll_file:
                for sentence in ontonotes_reader.sentence_iterator(conll_file):
                    if not sentence.srl_frames:
                        continue
                    for _, labels in sentence.srl_frames:
                        tokens = sentence.words
                        head_pred = [int(l[-2:] == '-V') for l in labels]
                        head_pred_id = head_pred.index(1)
                        head_pred_word = tokens[head_pred_id]
                        for i, t, l in zip(range(len(tokens)), tokens, labels):
                            fout.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                                i, t, head_pred_word, [head_pred_id], head_pred_id, ind, ind, l))
                        fout.write('\n')
                        ind += 1
