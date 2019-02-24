import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from typing import List, Union
import logging
from pprint import pprint
from pprint import pformat
from collections import defaultdict
from operator import itemgetter
from collections import namedtuple
import regex
from tqdm import tqdm
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.tokenizers import WordTokenizer
import argparse
import spacy
import numpy as np

class OverlapError(Exception):
    pass

Extraction = namedtuple("Extraction",  # Open IE extraction
                        ["sent",       # Sentence in which this extraction appears
                         "toks",       # spaCy toks
                         "arg1",       # Subject
                         "rel",        # Relation
                         "args2",      # A list of arguments after the predicate
                         "task",       # for multi task learning
                         "confidence"] # Confidence in this extraction
)

MergedExtraction = namedtuple("MergedExtraction",  # Open IE extraction
                        ["sent",       # Sentence in which this extraction appears
                         "toks",       # spaCy toks
                         "arg1",       # Subject (list)
                         "rel",        # Relation
                         "args2",      # A list of arguments after the predicate (list of list)
                         "task",       # for multi task learning
                         "confidence"] # Confidence in this extraction
)

Element = namedtuple("Element",    # An element (predicate or argument) in an Open IE extraction
                     ["elem_type", # Predicate or argument ID
                      "span",      # The element's character span in the sentence
                      "text"]      # The textual representation of this element
)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def main(inp_fn: str,
         domain: str,
         out_fn: str,
         task: str,
         split: bool = False,
         dedup: bool = False,
         rm_coor: bool = False) -> None:
    """
    inp_fn: str, required.
       Path to file from which to read Open IE extractions in Open IE4's format.
    domain: str, required.
       Domain to be used when writing CoNLL format.
    out_fn: str, required.
       Path to file to which to write the CoNLL format Open IE extractions.
    task: str, required.
        Used for multi-task learning
    split: bool.
        When equals to True, each extraction for the same sentences
        will be separated into different data fragments.
    dedup: bool.
        When dedup equals to True, only keep one extraction per predicate.
    rm_coor: bool.
        When rm_coor equals to True, predicate with multiple extractions will be removed.
    """
    print('rm_coor: {}, dedup: {}, split: {}'.format(rm_coor, dedup, split))
    inp_fn_li = inp_fn.split(':')
    if task is None:
        task_li = [None] * len(inp_fn_li)
    else:
        task_li = task.split(':')
    assert len(inp_fn_li) == len(task_li), 'should have the same length'
    n_sent, n_sam = 0, 0
    with open(out_fn, 'w') as fout:
        for inp_fn, task in zip(inp_fn_li, task_li):
            print('file {} with task {}'.format(inp_fn, task))
            for exts_per_sen in read(inp_fn, task):
                n_sent += 1
                if rm_coor:
                    pred_set = {}
                    filter_exts = []
                    for ext in exts_per_sen:
                        sp = tuple(ext.rel.span)
                        if sp not in pred_set:
                            pred_set[sp] = 0
                        pred_set[sp] += 1
                    for ext in exts_per_sen:
                        sp = tuple(ext.rel.span)
                        if pred_set[sp] == 1:
                            filter_exts.append(ext)
                    exts_per_sen = filter_exts
                if dedup:
                    pred_set = set()
                    filter_exts = []
                    for ext in exts_per_sen:
                        sp = tuple(ext.rel.span)
                        if sp not in pred_set:
                            pred_set.add(sp)
                            filter_exts.append(ext)
                    exts_per_sen = filter_exts
                if split:
                    exts_per_sen = [[ext] for ext in exts_per_sen]
                else:
                    exts_per_sen = [exts_per_sen] if len(exts_per_sen) > 0 else []
                for exts in exts_per_sen:
                    ls = ['\t'.join(map(str, pad_line_to_ontonotes(line, domain)))
                          for line in convert_sent_to_conll(exts, merge=True)]
                    if len(ls) > 0:
                        n_sam += len(ls[0].split('\t')) - 12
                        fout.write("{}\n\n".format('\n'.join(ls)))
    print('{} sentences and {} samples'.format(n_sent, n_sam))

def safe_zip(*args):
    """
    Zip which ensures all lists are of same size.
    """
    assert (len(set(map(len, args))) == 1)
    return zip(*args)

def char_to_word_index(char_ind: int,
                       sent: str) -> int:
    """
    Convert a character index to
    word index in the given sentence.
    """
    return sent[: char_ind].count(' ')

def element_from_span(span: List[int],
                      span_type: str) -> Element:
    """
    Return an Element from span (list of spacy toks)
    """
    return Element(span_type,
                   [span[0].idx,
                    span[-1].idx + len(span[-1])],
                   ' '.join(map(str, span)))

def split_predicate(ex: Union[Extraction, MergedExtraction]) -> Union[Extraction, MergedExtraction]:
    """
    Ensure single word predicate
    by adding "before-predicate" and "after-predicate"
    arguments.
    """
    rel_toks = ex.toks[char_to_word_index(ex.rel.span[0], ex.sent) \
                       : char_to_word_index(ex.rel.span[1], ex.sent) + 1]
    if not rel_toks:
        return ex

    verb_inds = [tok_ind for (tok_ind, tok)
                 in enumerate(rel_toks)
                 if tok.tag_.startswith('VB')]

    last_verb_ind = verb_inds[-1] if verb_inds \
                    else (len(rel_toks) - 1)

    rel_parts = [element_from_span([rel_toks[last_verb_ind]],
                                   'V')]

    before_verb = rel_toks[ : last_verb_ind]
    after_verb = rel_toks[last_verb_ind + 1 : ]

    if before_verb:
        rel_parts.append(element_from_span(before_verb, "BV"))

    if after_verb:
        rel_parts.append(element_from_span(after_verb, "AV"))

    if type(ex) is Extraction:
        return Extraction(ex.sent, ex.toks, ex.arg1, rel_parts, ex.args2, ex.task, ex.confidence)
    elif type(ex) is MergedExtraction:
        return MergedExtraction(ex.sent, ex.toks, ex.arg1, rel_parts, ex.args2, ex.task, ex.confidence)

def extraction_to_conll(ex: Union[Extraction, MergedExtraction]) -> List[str]:
    """
    Return a conll representation of a given input Extraction.
    """
    ex = split_predicate(ex)
    toks = ex.sent.split(' ')
    ret = ['*'] * len(toks)
    ret_touched = [False] * len(toks)
    args = [ex.arg1] + ex.args2
    rels_and_args = [(rel_part.elem_type, rel_part) for rel_part in ex.rel]
    if type(ex) is Extraction:
        rels_and_args += [("ARG{}".format(arg_ind), arg)
                          for arg_ind, arg in enumerate(args)]
    elif type(ex) is MergedExtraction:
        rels_and_args += [("ARG{}".format(arg_ind), arg)
                          for arg_ind, argu in enumerate(args) for arg in argu]
    else:
        raise ValueError

    for rel, arg in rels_and_args:
        # Add brackets
        cur_start_ind = char_to_word_index(arg.span[0],
                                           ex.sent)
        cur_end_ind = char_to_word_index(arg.span[1],
                                         ex.sent)
        for ci in range(cur_start_ind, cur_end_ind + 1):
            if ret_touched[ci]:
                # no overlap allowed
                raise OverlapError
        if ex.task is None:
            ret[cur_start_ind] = "({}{}".format(rel, ret[cur_start_ind])
        else:
            ret[cur_start_ind] = "({}#{}{}".format(rel, ex.task, ret[cur_start_ind])
        ret[cur_end_ind] += ')'
        for ci in range(cur_start_ind, cur_end_ind + 1):
            ret_touched[ci] = True
    return ret

def interpret_span(text_spans: str) -> List[int]:
    """
    Return an integer tuple from
    textual representation of closed / open spans.
    """
    m = regex.match("^(?:(?:([\(\[]\d+, \d+[\)\]])|({\d+}))[,]?\s*)+$",
                    text_spans)

    spans = m.captures(1) + m.captures(2)

    int_spans = []
    for span in spans:
        ints = list(map(int,
                        span[1: -1].split(',')))
        if span[0] == '(':
            ints[0] += 1
        if span[-1] == ']':
            ints[1] += 1
        if span.startswith('{'):
            assert(len(ints) == 1)
            ints.append(ints[0] + 1)

        assert(len(ints) == 2)

        int_spans.append(ints)

    # Merge consecutive spans
    ret = []
    cur_span = int_spans[0]
    for (start, end) in int_spans[1:]:
        if start - 1 == cur_span[-1]:
            cur_span = (cur_span[0],
                        end)
        else:
            ret.append(cur_span)
            cur_span = (start, end)

    if (not ret) or (cur_span != ret[-1]):
        ret.append(cur_span)

    return ret[0]

def interpret_element(element_type: str, text: str, span: str) -> Element:
    """
    Construct an Element instance from regexp
    groups.
    """
    return Element(element_type,
                   interpret_span(span),
                   text)

def parse_element(raw_element: str) -> List[Element]:
    """
    Parse a raw element into text and indices (integers).
    """
    elements = [regex.match("^(([a-zA-Z]+)\(([^\t]+),List\(([^\t]*)\)\))$",
                            elem.lstrip().rstrip())
                for elem
                in raw_element.split('|;|;|')]
    return [interpret_element(*elem.groups()[1:])
            for elem in elements
            if elem]


def read(fn: str, task=None) -> List[Extraction]:
    prev_sent = []

    with open(fn) as fin:
        for line in tqdm(fin):
            line = line.replace('\xa0', ' ') # a bug found in stanford openie
            data = line.strip().split('\t')
            confidence = data[0]
            if not all(data[2:5]):
                # Make sure that all required elements are present
                continue
            arg1, rel, args2 = map(parse_element,
                                   data[2:5])

            # Exactly one subject and one relation
            # and at least one object
            if ((len(rel) == 1) and \
                (len(arg1) == 1) and \
                (len(args2) >= 1)):
                sent = data[5]
                tokens = nlp.tokenizer.tokens_from_list(sent.split(' '))
                tokens = nlp.tagger(tokens)
                cur_ex = Extraction(sent = sent,
                                    #toks = tokenizer.tokenize(sent),
                                    toks = tokens,
                                    arg1 = arg1[0],
                                    rel = rel[0],
                                    args2 = args2,
                                    task=task,
                                    confidence = confidence)


                # Decide whether to append or yield
                if (not prev_sent) or (prev_sent[0].sent == sent):
                    prev_sent.append(cur_ex)
                else:
                    yield prev_sent
                    prev_sent = [cur_ex]
            else:
                raise ValueError('parsing error')
    if prev_sent:
        # Yield last element
        yield prev_sent

def merge_extraction(ext_li: List[Extraction]) -> List[MergedExtraction]:
    def arg_uniq_add(args, arg):
        for a in args:
            if a.span == arg.span:
                return
        args.append(arg)
    ext_d = {}
    # group extractions with the same predicate
    for ext in ext_li:
        sp = ext.rel.span
        if type(sp) is list:
            sp = tuple(sp)
        if len(sp) != 2:
            raise ValueError('span format error')
        if sp not in ext_d:
            ext_d[sp] = []
        ext_d[sp].append(ext)
    mext_li = []
    for sp, exts in ext_d.items():
        arg1 = []
        args2 = [[] for i in range(np.max([len(ext.args2) for ext in exts]))]
        for ext in exts:
            arg_uniq_add(arg1, ext.arg1)
            for i in range(len(ext.args2)):
                arg_uniq_add(args2[i], ext.args2[i])
        mext = MergedExtraction(exts[0].sent, exts[0].toks, arg1, exts[0].rel,
                                args2, exts[0].task, exts[0].confidence)
        mext_li.append(mext)
    return mext_li

def convert_sent_to_conll(sent_ls: List[Extraction], merge=False):
    """
    Given a list of extractions for a single sentence -
    convert it to conll representation.
    When merge=True, multiple extractions with the same predicate will be merged
    and invalid extractions will be skipped.
    """
    # Sanity check - make sure all extractions are on the same sentence
    assert(len(set([ex.sent for ex in sent_ls])) == 1)
    toks = sent_ls[0].sent.split(' ')
    if merge:
        # merge extractions with the same predicate
        sent_ls = merge_extraction(sent_ls)
    fields = [range(len(toks)), toks]
    for ex in sent_ls:
        try:
            fields.append(extraction_to_conll(ex))
        except OverlapError:
            # skip problematic extractions
            continue
    return safe_zip(*fields)

def pad_line_to_ontonotes(line, domain) -> List[str]:
    """
    Pad line to conform to ontonotes representation.
    """
    word_ind, word = line[ : 2]
    pos = 'XX'
    oie_tags = line[2 : ]
    line_num = 0
    parse = "-"
    lemma = "-"
    return [domain, line_num, word_ind, word, pos, parse, lemma, '-',\
            '-', '-', '*'] + list(oie_tags) + ['-', ]

def convert_sent_dict_to_conll(sent_dic, domain) -> str:
    """
    Given a dictionary from sentence -> extractions,
    return a corresponding CoNLL representation.
    """
    return '\n\n'.join(['\n'.join(['\t'.join(map(str, pad_line_to_ontonotes(line, domain)))
                                   for line in convert_sent_to_conll(sent_ls)])
                        for sent_ls
                        in sent_dic.iteritems()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Open IE4 extractions to CoNLL (ontonotes) format.")
    parser.add_argument("--inp", type=str, help="input file from which to read Open IE extractions (separated by :)", required = True)
    parser.add_argument("--domain", type=str, help="domain to use when writing the ontonotes file.", required = True)
    parser.add_argument("--out", type=str, help="path to the output file, where CoNLL format should be written.", required = True)
    parser.add_argument("--task", type=str, help="task of each input file (separated by :).", default=None)
    parser.add_argument('--split', help='whether to separate extractions for the same sentence', action='store_true')
    parser.add_argument('--dedup', help='whether to keep only one extraction per predicate', action='store_true')
    parser.add_argument('--rm_coor', help='whether to remove predicates with multiple extractions', action='store_true')
    args = parser.parse_args()
    main(args.inp, args.domain, args.out, args.task, split=args.split, dedup=args.dedup, rm_coor=args.rm_coor)

