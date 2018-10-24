import torch 
import torch.nn as nn 

class Embedder( nn.Module ):

    def __init__( self, dictionary_size, embed_size):
        super( Embedder, self ).__init__()
        self.dict_size = dictionary_size
        self.out = nn.Embedding( dictionary_size, embed_size )
    
    def forward( self, ids ):
        return self.out( ids )

    def save_weight( self, path ):
        cpt  = dict()
        cpt[ "out" ] = self.state_dict()
        torch.save( cpt, path )
        print( "Successfully saved embedding" )

    def load_weight( self, path ):
        cpt = torch.load( path )
        self.load_state_dict( cpt[ "out" ] )
        print( "Successfully loaded embedding" )
