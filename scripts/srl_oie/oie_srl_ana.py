from typing import Tuple, List
import sys, os
root_dir = os.path.abspath(os.path.join(__file__, '../../..'))
sys.path.insert(0, root_dir)
import argparse
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from tqdm import tqdm
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def pmi_matrix(tag1, tag2, thres=0):
    label1, count1 = np.unique(tag1, return_counts=True)
    l1c = len(label1)  # number of unique labels of tag1
    label2, count2 = np.unique(tag2, return_counts=True)
    l2c = len(label2)  # count of unique labels of tag2
    ac = len(tag1)  # number of pairs
    label12, count12 = np.unique(list(map(lambda x: '{}+{}'.format(*x), zip(tag1, tag2))), return_counts=True)
    lc12 = defaultdict(lambda: 0)
    lc12.update(dict(zip(label12, count12)))
    pmi = np.zeros((len(label1), len(label2)))
    coocc = np.zeros((len(label1), len(label2)))
    for i in range(l1c):
        for j in range(l2c):
            c = lc12[label1[i] + '+' + label2[j]]
            if c <= thres:  # replace small value with zero
                c = 0
            coocc[i, j] = np.log(c + 1e-10)
            pmi[i, j] = np.log((ac *  lc12[label1[i] + '+' + label2[j]] / (count1[i] * count2[j])) + 1e-10)
    return pmi, coocc, label1, label2


def plot_correlation(matrix, yclasses, xclasses, title, saveto):
    fig, ax = plt.subplots(figsize=(0.5 * len(xclasses), 0.5 * len(yclasses)))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.bwr)

    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           xticklabels=xclasses, yticklabels=yclasses,
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=90, ha='right',
             rotation_mode='anchor')
    plt.savefig(saveto)


def read_oie_srl_parallel(filepath: str, method='') -> Tuple[List[str], List[str], List[str]]:
    assert method in {None, 'pos_oie', 'pos_srl', 'pos_all'}
    srls, oies = [], []
    with open(filepath, 'r') as fin:
        for i, l in tqdm(enumerate(fin)):
            l = l.strip()
            if len(l) == 0:
                continue
            if l.startswith('#'):  # comment line
                continue
            tokens = l.split('\t')
            tokens = [t.split(' ') for t in tokens]
            tokens, verb_inds, srl_tags, oie_tags = zip(*tokens)
            verb_inds = list(map(int, verb_inds))
            if method.startswith('pos_'):
                # get pos
                tokens_tag = nlp.tagger(nlp.tokenizer.tokens_from_list(list(tokens)))
                pos = [t.tag_ for t in tokens_tag]
                if method == 'pos_all' or method == 'pos_oie':
                    oie_tags = list(map(lambda x: '|'.join(x), zip(oie_tags, pos)))
                if method == 'pos_all' or method == 'pos_srl':
                    srl_tags = list(map(lambda x: '|'.join(x), zip(srl_tags, pos)))
            srls.extend(srl_tags)
            oies.extend(oie_tags)
    return srls, oies


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyse the correlation between srl tags and oie tags')
    parser.add_argument('--inp', type=str, help='parallel data (separated by :)', required=True)
    args = parser.parse_args()

    srls, oies = [], []
    for inp in args.inp.split(':'):
        s, o = read_oie_srl_parallel(inp, method='')
        srls.extend(s)
        oies.extend(o)

    labels = np.unique(srls + oies)
    cm = confusion_matrix(srls, oies, labels=labels)

    pmi, coocc, oie_labels, srl_labels = pmi_matrix(oies, srls, thres=100)

    with np.printoptions(precision=3, suppress=True):
        print('totally {} pairs of labels'.format(len(srls)))
        print('{} SRL tags: {}'.format(len(srl_labels), srl_labels))
        print('{} OIE tags: {}'.format(len(oie_labels), oie_labels))
        print('{} confusion matrix:'.format(cm.shape))
        print(cm)
        #print('{} pmi matrix'.format(pmi.shape))
        #print(pmi)
        #print('{} co-occur matrix'.format(coocc.shape))
        #print(coocc)
        plot_correlation(pmi, oie_labels, srl_labels, 'pmi', 'srl_oie_ana/pmi.png')
        plot_correlation(coocc, oie_labels, srl_labels, 'co-occure', 'srl_oie_ana/coocc.png')
