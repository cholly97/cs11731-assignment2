# code partially adopted https://github.com/artetxem.undreamt 
# specifically we adopted his structure to separate special characters out in the vocab
import collections 
import torch
import torch.nn as nn 
import numpy as np 

NUM_SPECIAL_SYM = 4
PAD, SOS, EOS, UNK = 0,1,2,3

class Vocab( object ):

    def __init__( self, words ):
        """
            Args:
                words: a list of words in the Vocab
        """
        self.id2word = [ None ] + words 
        self.word2id = { wordL 1 + i for i, word in enumerate( words ) }

    def single_sentence2ids( self, sentence, sos = False, eos = False ):
        """
            Args:
                sentence: a single sentence, not processed
            Returns:
                [ ids ] of a sentence
        """
        tokens = tokenize( sentence )
        ids = [ NUM_SPECIAL_SYM + sellf.word2id[ word ] - 1 if word in self.word2id else UNK for word in tokens ]
        if sos: ids = [ SOS ] + ids
        if eos: ids = [ EOS ] + ids 
        return ids
    
    def sentences2ids( self, sentence, sos = False, eos = False ):
        """
            Args:
                sentences: a list of raw sentence, not processed

        """
        ids = [ self.single_sentence2ids( sentence, sos = sos, eos = eos ) for sentence in sentences ]
        lengths = [ len(s) for s in ids ]
        # pad sentences into the longest length of the batch sentence
        ids = [ s + [ PAD ]*( max( lengths )-len( s ) ) for s in ids ]

        # transpose the size from batch, length to length, batch_size
        ids = [ [ ids[ i ][ j ] for i in range( len( ids ) ) ]  for j in range( max( lengths ) ) ]
        return ids, lengths

    def id2single_sentence( self, ids ):
        """
            Args:
                ids: a list of ids of one single sentence
            Returns:
                a single string of sentence
        """
        return ' '.join( [ '<UNK>' if i == UNK else self.id2word[ i - NUM_SPECIAL_SYM + 1 ] for i in ids if i != EOS and i != PAD and i != SOS ] )

    def ids2sentences( self, ids ):
        """
        Args:
            ids: a batch of list of ids each is a sentence
        Returns:
            a list of sentences
        """
        return [ self.ids2sentence(i) for i in ids ]

    def vocab_size( self ):
        return len( self.id2word ) - 1

    
def random_embeddings( vocab_size, embed_size ):
    return nn.Embedding( vocab_size + 1, embed_size )


def tokenize( sentence ):
    return sentence.strip().split()

def special_ids(ids ):
    return ids * ( ids < NUM_SPECIAL_SYM ).long()


def word_ids( ids ):
    return ( ids - NUM_SPECIAL_SYM + 1 ) * ( ids >= NUM_SPECIAL_SYM ).long()