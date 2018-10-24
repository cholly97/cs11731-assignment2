import vocab 
from attention import GlobalAttention

import torch 
import torch.nn as nn 
from torch.autograd import Variable

# input feed adopted https://github.com/artetxem.undreamt 

# Based on OpenNMT-py
class StackedGRU( nn.Module ):

    def __init__( self, input_size, hidden_size, layers, dropout ):
        super( StackedGRU, self ).__init__()
        self.num_layers = layers 
        self.dropout = nn.Dropout( dropout )
        self.layers = nn.ModuleList()
        for i in range( layers ):
            self.layers.append( nn.GRUCell( input_size, hidden_size ) )
            input_size = hidden_size
    
    def forward( self, inputs, hidden ):
        h_1 = [] 
        for i, layer in enumerate( self.layers ):
            # print( "inputs before", inputs.size() )
            h_1_i = layer( inputs, hidden[ i ] )
            inputs = h_1_i
            # print( "inputs after", inputs.size() )
            if i+1 != self.num_layers:    
                inputs = self.dropout( inputs )
            h_1 += [ h_1_i ]
        h_1 = torch.stack( h_1 )
        return inputs, h_1 

class AttentionDecoder( nn.Module ):

    def __init__( self, embed_size, hidden_size, num_layer, dropout, input_feed = True, encoder_bidir = True ):
        super( AttentionDecoder, self ).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size # if not encoder_bidir else hidden_size * 2
        self.special_embedding = nn.Embedding( vocab.NUM_SPECIAL_SYM + 1, embed_size, padding_idx = 0 )
        self.attention = GlobalAttention( hidden_size, attention_type = "general" )
        self.input_feed = input_feed
        self.input_size = embed_size + hidden_size if input_feed else embed_size
        self.stacked_rnn = StackedGRU( self.input_size, hidden_size, num_layer, dropout )
        self.dropout = nn.Dropout( dropout )

    def forward( self, ids, lengths, word_embedder, hidden, context, context_mask, prev_output, generator ):
        # print( len( ids ), len( ids[ 0 ] ) )
        # print( "vocab", vocab.word_ids(ids).size() )
        # print( "special", vocab.special_ids( ids ).size() )
        embeddings = word_embedder( vocab.word_ids(ids).cuda() ) + self.special_embedding( vocab.special_ids( ids ).cuda() )
        output = prev_output
        # print( "output", output.size() )
        # print( "embeddings", embeddings.size() )
        # if self.input_feed:
        embeddings = embeddings.permute( 1,0,2 )
        scores = []
        for emb in embeddings.split(1):
            if self.input_feed:
                # print( "emb", emb.size() )
                inputs = torch.cat([emb.squeeze(0), output], 1)
            else:
                inputs = emb.squeeze(0)
            output, hidden = self.stacked_rnn(inputs, hidden)
            output = self.attention(output, context, context_mask)
            output = self.dropout(output)
            scores.append(generator(output))
        return torch.stack(scores), hidden, output

    def initial_output( self, batch_size ):
        return Variable( torch.zeros( batch_size, self.hidden_size ), requires_grad=False )
    
    def save_weight( self, path ):
        cpt  = dict()
        cpt[ "out" ] = self.state_dict()
        torch.save( cpt, path )
        print( "Successfully saved embedding" )

    def load_weight( self, path ):
        cpt = torch.load( path )
        self.load_state_dict( cpt[ "out" ] )
        print( "Successfully loaded embedding" )