# global attention adopted from homwwork 1, only implemented general attention

import torch.nn  as nn 

class GlobalAttention( nn.Module ):

    def __init__( self, hidden_size, attention_type = "general" ):
        if attention_type != "general":
            raise ValueError( "Methods other than general attention has not been implemented" )
        
        self.attention_type = attention_type
        