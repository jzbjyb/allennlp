from typing import Tuple, List
import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../../..'))
sys.path.insert(0, root_dir)
import argparse
from collections import defaultdict
from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from tqdm import tqdm


def get_samples(conll_filepath: str,
                is_dir=False,
                domain_identifier=None) -> Tuple[List[str], int, List[str]]:
    '''
    get tokens, tags, and verb indicators from conll file
    '''
    def dir_iter(filepath):
        if is_dir:
            for conll_file in ontonotes_reader.dataset_path_iterator(filepath):
                if domain_identifier is None or f'/{domain_identifier}/' in conll_file:
                    for sentence in ontonotes_reader.sentence_iterator(conll_file):
                        yield sentence
        else:
            for sentence in ontonotes_reader.sentence_iterator_direct(filepath):
                yield sentence
    ontonotes_reader = Ontonotes()
    for sentence in dir_iter(conll_filepath):
        if sentence.srl_frames:
            tokens = sentence.words
            for (_, tags) in sentence.srl_frames:
                verb_ind = [i for i, label in enumerate(tags) if label[-2:] == '-V']
                if len(verb_ind) != 1:
                    raise Exception('verb not unique')
                yield tokens, verb_ind[0], tags, sentence.document_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='find srl samples corresponding to oie samples')
    parser.add_argument('--oie', type=str, help='oie conll file', required=True)
    parser.add_argument('--srl', type=str, help='srl conll file', required=True)
    parser.add_argument('--out', type=str, help='output file', required=True)
    args = parser.parse_args()

    # get oie samples
    oie_samples = defaultdict(lambda: defaultdict(lambda: {}))
    num_oie_sam = 0
    for sam in get_samples(args.oie, is_dir=False):
        sent = ' '.join(sam[0])
        verb_ind = sam[1]
        oie_tags = sam[2]
        num_oie_sam += 1
        oie_samples[sent][verb_ind]['oie'] = oie_tags

    # get corresponding srl samples
    num_srl_sam = 0
    num_overlap = 0
    docids = defaultdict(lambda: 0)
    for sam in tqdm(get_samples(args.srl, is_dir=True, domain_identifier='nw')):
        sent = ' '.join(sam[0])
        verb_ind = sam[1]
        srl_tags = sam[2]
        docid = '/'.join(sam[3].split('/')[:2])
        num_srl_sam += 1
        if sent in oie_samples and verb_ind in oie_samples[sent]:
            num_overlap += 1
            oie_samples[sent][verb_ind]['srl'] = srl_tags
            oie_samples[sent][verb_ind]['srl_from'] = docid
            docids[docid] += 1
        if num_overlap >= num_oie_sam:
            print('find all oie samples')
            break

    print('totally {} oie samples, {} srl samples, {} overlap'.format(
        num_oie_sam, num_srl_sam, num_overlap))
    docids = sorted(docids.items(), key=lambda x: -x[1])
    print('overlap srl mainly come from {} out of {} docs'.format(docids[:10], len(docids)))

    # save samples
    with open(args.out, 'w') as fout:
        for sent in oie_samples:
            for verb_ind in oie_samples[sent]:
                tokens = sent.split(' ')
                oie_tags = oie_samples[sent][verb_ind]['oie']
                if 'srl' in oie_samples[sent][verb_ind]:
                    skip = False
                    srl_tags = oie_samples[sent][verb_ind]['srl']
                else:
                    skip = True
                    srl_tags = ['-'] * len(tokens)
                verb_inds = [str(int(i == verb_ind)) for i in range(len(tokens))]
                assert len(tokens) == len(oie_tags) and len(tokens) == len(srl_tags), 'length not consistent'
                line = ('# ' if skip else '') + '\t'.join(
                    map(lambda x: ' '.join(x), zip(tokens, verb_inds, srl_tags, oie_tags)))
                fout.write(line + '\n')
