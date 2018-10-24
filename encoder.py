
import torch 
import torch.nn as nn 
from torch.autograd import Variable 
import vocab 

class GRUEncoder( nn.Module ):

    def __init__( self, embed_size, hidden_size, bidirectional = True, layers = 1, dropout = 0 ):
        super( GRUEncoder, self ).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directional = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size  # // self.directional
        self.special_embeddings = nn.Embedding( vocab.NUM_SPECIAL_SYM + 1, embed_size, padding_idx = 0 )
        self.rnn = nn.GRU( embed_size, hidden_size, bidirectional = bidirectional, num_layers = layers, dropout = dropout )

        if self.bidirectional:
            self.combi_0 = nn.Linear( self.hidden_size, self.hidden_size, bias = False )
            self.combi_1 = nn.Linear( self.hidden_size, self.hidden_size, bias = False )
    
    def forward( self, ids, lengths, word_embedder, hidden ):
        sorted_lengths = sorted( lengths, reverse = True )
        is_sorted = sorted_lengths == lengths
        is_varlen = sorted_lengths[0] != sorted_lengths[-1]
        if not is_sorted:
            true2sorted = sorted(range(len(lengths)), key=lambda x: -lengths[x])
            sorted2true = sorted(range(len(lengths)), key=lambda x: true2sorted[x])
            ids = torch.stack([ids[:, i] for i in true2sorted], dim=1)
            lengths = [lengths[i] for i in true2sorted]
        # print( "lengths", lengths )
        embeddings = word_embedder(vocab.word_ids(ids)) + self.special_embeddings(vocab.special_ids(ids))
        # print( "encoder embeddings", embeddings.size() )
        if is_varlen:
            embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths)
        # print( "encoder hidden", hidden.size() )
        
        output, hidden = self.rnn(embeddings, hidden)
        if is_varlen:
            output = nn.utils.rnn.pad_packed_sequence(output)[0]
        if self.bidirectional:
            # hidden = torch.stack([torch.cat((hidden[2*i], hidden[2*i+1]), dim=1) for i in range(self.layers)])
            
            hidden = torch.stack( [ self.combi_0( hidden[2*i] ) + self.combi_1( hidden[2*i+1 ] ) for i in range(self.layers) ] )
            # sum up 
            var_len = list( output.size() )
            # print( "varlen", var_len )
            var_len[-1] = var_len[ -1 ] // 2
            split = output.split( var_len[ -1 ], dim = 2 )
            output = self.combi_0( split[0] ) + self.combi_1( split[1] )
        # print( "hidden_size after", hidden.size() )
        
        # print( "output before", output.size() )
        if not is_sorted:
            hidden = torch.stack([hidden[:, i, :] for i in sorted2true], dim=1)
            output = torch.stack([output[:, i, :] for i in sorted2true], dim=1)
        # print( "output after", output.size() )
        return hidden, output

    def initial_hidden(self, batch_size):
        return Variable(torch.zeros(self.layers*self.directional, batch_size, self.hidden_size), requires_grad=False)

    def save_weight( self, path ):
        cpt  = dict()
        cpt[ "out" ] = self.state_dict()
        torch.save( cpt, path )
        print( "Successfully saved embedding" )

    def load_weight( self, path ):
        cpt = torch.load( path )
        self.load_state_dict( cpt[ "out" ] )
        print( "Successfully loaded embedding" )

