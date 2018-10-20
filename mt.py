
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Semisupervised mt from one language to another
"""

class MT( object ):

    def __init__( self, vocab, src_embedder, tar_embedder,  )