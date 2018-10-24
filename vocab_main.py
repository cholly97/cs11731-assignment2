from typing import List
import argparse
import os

import pickle

from util import *

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-src', dest='train_src',
                        type=str, default="",
                        help="train source text ")
    # parser.add_argument('--train-tar', dest='train_tar',
    #                     type=str, default="",
    #                     help="train target text ")
    parser.add_argument('--size', dest='size', type=int,
                        default=50000, help="max size of vocab")
    parser.add_argument('--freq-cutoff', dest='freq_cutoff', type=int,
                        default=2, help="The frequency of the word being cutoff")
    parser.add_argument('--vocab-src', dest='vocab_src', type=str,
                        default="", help="The path of vocab file output")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print( 'read in source sentences: %s' % args.train_src )
    # print('read in target sentences: %s' % args.train_tar)

    src_sents = read_corpus(args.train_src, source='src')
    # tgt_sents = read_corpus(args.train_tar, source='tgt')

    # vocab = Vocab(src_sents, tgt_sents, int(args.size), int(args.freq_cutoff))
    vocab = VocabEntry.from_corpus( src_sents, args.size, args.freq_cutoff )
    # print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    pickle.dump(vocab, open(args.vocab_src, 'wb'))
    # print('vocabulary saved to %s' % args.vocab_file)