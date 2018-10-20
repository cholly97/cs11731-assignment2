# global attention adopted from homwwork 1, only implemented general attention

import torch.nn  as nn 

class GlobalAttention( nn.Module ):

    def __init__( self, hidden_size, attention_type = "general" ):
        super( GlobalAttention, self ).__init__()
        if attention_type != "general":
            raise ValueError( "Methods other than general attention has not been implemented" )
        
        self.attention_type = attention_type
        if self.attention_type == "general":
            self.linear_align = nn.Linear( hidden_size, hidden_size, bias = False )
        self.linear_context = nn.Linear( hidden_size, hidden_size, bias = False )
        self.linear_query = nn.Linear( hidden_size, hidden_size, bias = False )
        self.softmax = nn.Softmax( dim = 1 )

    def forward( self, query, context, mask ):
        """
        Args:
            query: batch_size, hidden_size
            context: len, batch_size, hidden_size
            

        Returns:
            batch, hidden_size
        """
        context_transposed = context.transpose( 0, 1 )
        # batch_size, len, hidden_size 

        if self.attention_type == "general":
            q = self.linear_align( query )
        else:
            q = query 
        
        if mask is not None:
            align.data.masked_fill_( mask, -float( "inf" ) )
        
        attention = self.softmax( align )
        # batch_size, len 
        weighted_context = attention.unsqueeze( 1 )
        weighted_context = weighted_context.bmm( context_transposed ).squeeze( 1 )
        # batch_size, hidden_size 
        weighted_context = self.linear_context( weighted_context )
        weighted_context  = weighted_context + self.linear_query( query )

        return self.tanh( weighted_context )
