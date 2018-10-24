
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import vocab 
from util import *
"""
Semisupervised mt from one language to another
"""
# this part, especially with the masking is heavily adopted from https://github.com/artetxem.undreamt 
# as we have no idea our homework 1 implementation went wrong so we started from somw other's work.
# we started looking from their code, and wrote every piece ourselves despite the fact that most of the code are adopted from theirs

class MT( object ):

    def __init__( self, src_vocab, tar_vocab, encoder_embedder, decoder_embedder, generator, encoder, decoder, denoising = True, multi_gpu = False ):
        self.encoder_embedder =encoder_embedder
        self.decoder_embedder = decoder_embedder
        self.generator = generator
        self.src_dict = src_vocab
        self.tar_dict = tar_vocab
        self.encoder = encoder
        self.decoder = decoder
        self.denoising = denoising 
        self.multi_gpu = multi_gpu
        if multi_gpu:
            weight = torch.ones( self.generator.module.num_output_class() ).cuda()
        else:
            weight = torch.ones( self.generator.num_output_class() ).cuda()
        weight[ vocab.PAD ] = 0
        self.loss = nn.NLLLoss( weight, size_average = False )

    def _train( self, mode ):
        self.encoder_embedder.train( mode )
        self.decoder_embedder.train( mode )
        self.generator.train( mode )
        self.encoder.train( mode )
        self.decoder.train( mode )
        self.loss.train( mode )

    def encode( self, sentences, mode = False ):
        self._train( mode )
        ids, lengths = self.src_dict.sentences2ids( sentences, sos = False, eos = True, transpose = True )
        # add noise as indicated in the paper
        if mode and self.denoising:
            for i, length in enumerate( lengths ):
                if length > 2:
                    for it in range( length // 2 ):
                        j = random.randint( 0, length - 2 )
                        ids[ j ][ i ], ids[ j + 1 ][ i ] = ids[ j + 1 ][ i ], ids[ j ][ i ]
        
        var_ids = Variable( torch.LongTensor( ids ), requires_grad = False, volatile = not mode ).cuda()
        hidden = self.encoder.initial_hidden( len( sentences ) ).cuda()
        hidden, context = self.encoder( var_ids, lengths, self.encoder_embedder, hidden )
        return hidden, context, lengths

    def mask( self, lengths ):
        batch_size = len( lengths )
        max_length = max( lengths )
        if max_length == min( lengths ):
            return None 
        mask = torch.ByteTensor( batch_size, max_length ).fill_( 0 )
        for i in range( batch_size ):
            for j in range( lengths[ i ], max_length ):
                mask[ i, j ] = 1
        return mask.cuda()

    def get_loss( self, src, trg, train = False ):
        self._train( train )
        # Check batch sizes
        if len(src) != len(trg):
            raise Exception('Sentence and hypothesis lengths do not match')

        # Encode
        hidden, context, context_lengths = self.encode(src, mode = train)
        context_mask = self.mask(context_lengths)

        # Decode
        initial_output = self.decoder.initial_output(len(src)).cuda()
        input_ids, lengths = self.tar_dict.sentences2ids(trg, eos=False, sos=True)
        input_ids_var = Variable(torch.LongTensor(input_ids), requires_grad=False).cuda()
        # print( "input ids-------------", input_ids_var.size() )
        logprobs, hidden, _ = self.decoder(input_ids_var, lengths, self.decoder_embedder, hidden, context, context_mask, initial_output, self.generator)

        # Compute loss
        output_ids, lengths = self.tar_dict.sentences2ids(trg, eos=True, sos=False)
        output_ids_var = Variable(torch.LongTensor(output_ids), requires_grad=False).cuda()
        loss = self.loss(logprobs.view(-1, logprobs.size()[-1]), output_ids_var.view(-1))

        return loss

    def greedy( self, sentences, max_ratio, mode = False ):
        self._train( mode )
        with torch.no_grad():
            input_lengths = [ len(  sentence  ) for sentence in sentences ]
            hidden, context, context_lengths = self.encode( sentences, mode )
            context_mask = self.mask( context_lengths )
            translations = [ [] for sentence in sentences ]
            prev_words = len( sentences ) * [ vocab.SOS ]
            pending = set( range( len( sentences ) ) )
            output = self.decoder.initial_output( len( sentences ) ).cuda()
            while len( pending ) > 0:
                decoder_in = Variable( torch.LongTensor( [ prev_words ] ), requires_grad = False ).view( len( sentences ), 1 )
                log_prob, hidden, output = self.decoder( decoder_in, [ 1 ] * len( sentences ), self.decoder_embedder,
                                                        hidden, context, context_mask, output, self.generator )
                prev_words = log_prob.max( dim=2 )[ 1 ].squeeze().data.cpu().numpy().tolist()
                for i in pending.copy():
                    if prev_words[i] == vocab.EOS:
                        pending.discard( i )
                    else:
                        translations[ i ].append( prev_words[ i ] )
                        if len( translations[ i ] ) >= max_ratio*input_lengths[ i ]:
                            pending.discard( i )
            return self.tar_dict.ids2sentences( translations )

    def beam_search( self, sentences, beam_size = 10, max_ratio = 2, mode = False ):
        batch_size = len(sentences)
        input_lengths = [len(sentence) for sentence in sentences]
        hidden, context, context_lengths = self.encode(sentences, train)
        translations = [[] for sentence in sentences]
        pending = set(range(batch_size))

        hidden = hidden.repeat(1, beam_size, 1)
        context = context.repeat(1, beam_size, 1)
        context_lengths *= beam_size
        context_mask = self.mask(context_lengths)
        ones = beam_size*batch_size*[1]
        prev_words = beam_size*batch_size*[vocab.SOS]
        output = self.decoder.initial_output(beam_size*batch_size).cuda()

        translation_scores = batch_size*[-float('inf')]
        hypotheses = batch_size*[(0.0, [])] + (beam_size-1)*batch_size*[(-float('inf'), [])]  # (score, translation)

        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            var = Variable(torch.LongTensor([prev_words]), requires_grad=False).cuda()
            logprobs, hidden, output = self.decoder(var, ones, self.decoder_embedder, hidden, context, context_mask, output, self.generator)
            prev_words = logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()

            word_scores, words = logprobs.topk(k=beam_size+1, dim=2, sorted=False)
            word_scores = word_scores.squeeze(0).data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            words = words.squeeze(0).data.cpu().numpy().tolist()

            for sentence_index in pending.copy():
                candidates = []  # (score, index, word)
                for rank in range(beam_size):
                    index = sentence_index + rank*batch_size
                    for i in range(beam_size + 1):
                        word = words[index][i]
                        score = hypotheses[index][0] + word_scores[index][i]
                        if word != vocab.EOS:
                            candidates.append((score, index, word))
                        elif score > translation_scores[sentence_index]:
                            translations[sentence_index] = hypotheses[index][1] + [word]
                            translation_scores[sentence_index] = score
                best = []  # score, word, translation, hidden, output
                for score, current_index, word in sorted(candidates, reverse=True)[:beam_size]:
                    translation = hypotheses[current_index][1] + [word]
                    best.append((score, word, translation, hidden[:, current_index, :].data, output[current_index].data))
                for rank, (score, word, translation, h, o) in enumerate(best):
                    next_index = sentence_index + rank*batch_size
                    hypotheses[next_index] = (score, translation)
                    prev_words[next_index] = word
                    hidden[:, next_index, :] = h
                    output[next_index, :] = o
                if len(hypotheses[sentence_index][1]) >= max_ratio*input_lengths[sentence_index] or translation_scores[sentence_index] > hypotheses[sentence_index][0]:
                    pending.discard(sentence_index)
                    if len(translations[sentence_index]) == 0:
                        translations[sentence_index] = hypotheses[sentence_index][1]
                        translation_scores[sentence_index] = hypotheses[sentence_index][0]
        return self.tar_dict.ids2sentences(translations)
    
    def evaluate_ppl(self, dev_data, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = self.get_loss(src_sents, tgt_sents).cpu().detach().numpy().sum()

            cum_loss += loss
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

        return ppl

    

