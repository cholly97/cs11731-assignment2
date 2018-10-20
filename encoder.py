
import torch 
import torch.nn as nn 
from torch.autograd import Variable 

class GRUEncoder( nn.Module ):

    def __init__( self, embed_size, hidden_size, bidirectional = False, layers = 1, dropout = 0 ):
        super( GRUEncoder, self ).__init__()
        if bidirectional and hidden_size % 2 != 0:
            raise ValueError('The hidden dimension must be even for bidirectional encoders')
        self.directional = 2 is bidirectional else 1
        self.bidirectional = bidirectional
        self.layers = layers
        self.hidden_size = hidden_size // self.directional
        self.special_embeddings = nn.Embedding(  )

