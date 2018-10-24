from util import *
from mt import *
from embedder import *
from generator import *
from encoder import *
from decoder import *
from vocab import *
import argparse
import os
import torch.nn as nn

def add_to_optimizer( module, param_list ):
    for para in module.parameters():
        for l in param_list:
            l.append( para )
    return param_list

def train(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tar, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tar, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args.batch_size)
    valid_niter = int(args.valid_iter)
    log_every = int(args.log_every)
    model_save_path = args.save_path

    "Vocab dict"
    vocab_src = pickle.load(open(args.vocab_src, 'rb'))
    vocab_tar = pickle.load(open(args.vocab_tar, 'rb'))

    "Optimizer params"
    s2s_param = []
    t2t_param = []
    s2t_param = []
    t2s_param = []

    "Embed"
    if args.encoder_bidir:
        embedder_src = Embedder( vocab_src.dict_size(), args.embed_size ).cuda()
        embedder_tar = Embedder( vocab_tar.dict_size(), args.embed_size ).cuda()
    else:
        embedder_src = Embedder( vocab_src.dict_size(), args.embed_size ).cuda()
        embedder_tar = Embedder( vocab_tar.dict_size(), args.embed_size ).cuda()

    if args.embed_src != "":
        embedder_src.load_weight( args.embed_src )
    else:
        [ s2s_param, s2t_param, t2s_param ] = add_to_optimizer( embedder_src, [ s2s_param, s2t_param, t2s_param ] )
    if args.embed_tar != "":
        embedder_tar.load_weight( args.embed_tar )
    else:
        [ s2t_param, t2s_param, t2t_param ] = add_to_optimizer( embedder_tar, [ s2t_param, t2s_param, t2t_param ] )
    
    if args.multi_gpu:
        embedder_src = nn.DataParallel( embedder_src,device_ids=[0,1] ) 
        embedder_tar = nn.DataParallel( embedder_tar,device_ids=[0,1] ) 

    "Generator"
    gen_src = EmbeddingGenerator( args.hidden_size, args.embed_size ).cuda()
    gen_src_wrapper = WrapperEmbeddingGenerator( gen_src, embedder_src ).cuda()
    gen_tar = EmbeddingGenerator( args.hidden_size, args.embed_size ).cuda()
    gen_tar_wrapper = WrapperEmbeddingGenerator( gen_tar, embedder_tar ).cuda()

    if args.gen_src != "":
        gen_src_wrapper.load_weight( args.gen_src )
    else:
        [ s2s_param, s2t_param, t2s_param ] = add_to_optimizer( gen_src_wrapper, [ s2s_param, s2t_param, t2s_param ] )
    if args.gen_tar != "":
        gen_tar_wrapper.load_weight( args.gen_tar ) 
    else:
        [ s2t_param, t2s_param, t2t_param ] = add_to_optimizer( gen_tar_wrapper, [ s2t_param, t2s_param, t2t_param ] )
    
    if args.multi_gpu:
        gen_src_wrapper = nn.DataParallel( gen_src_wrapper,device_ids=[0,1] ) 
        gen_tar_wrapper = nn.DataParallel( gen_tar_wrapper,device_ids=[0,1] ) 

    "encoder"
    encoder_src = GRUEncoder( args.embed_size, args.hidden_size, 
                              bidirectional = args.encoder_bidir, layers = args.encoder_layer, dropout = args.dropout ).cuda()
    if args.multi_gpu:
        encoder_src = nn.DataParallel( encoder_src,device_ids=[0,1] ) 
    encoder_tar = GRUEncoder( args.embed_size, args.hidden_size, 
                              bidirectional = args.encoder_bidir, layers = args.encoder_layer, dropout = args.dropout ).cuda()
    if args.multi_gpu:
        encoder_src = nn.DataParallel( encoder_tar,device_ids=[0,1] ) 

    [ s2s_param, s2t_param, t2s_param ] = add_to_optimizer( encoder_src, [ s2s_param, s2t_param, t2s_param ] )
    [ s2t_param, t2s_param, t2t_param ] = add_to_optimizer( encoder_tar, [ s2t_param, t2s_param, t2t_param ] )

    "Decoder"
    decoder_src = AttentionDecoder( args.embed_size, args.hidden_size, 1, args.dropout, input_feed = True ).cuda()
    decoder_tar = AttentionDecoder( args.embed_size, args.hidden_size, 1, args.dropout, input_feed = True ).cuda()
    if args.multi_gpu:
        decoder_src = nn.DataParallel( decoder_src,device_ids=[0,1] ) 
        decoder_tar = nn.DataParallel( decoder_tar,device_ids=[0,1] ) 

    [ s2s_param, s2t_param, t2s_param ] = add_to_optimizer( decoder_src, [ s2s_param, s2t_param, t2s_param ] )
    [ s2t_param, t2s_param, t2t_param ] = add_to_optimizer( decoder_tar, [ s2t_param, t2s_param, t2t_param ] )

    "Translators"
    s2s_model = MT( vocab_src, vocab_src, embedder_src, embedder_src, gen_src_wrapper, encoder_src, decoder_src, denoising=True, multi_gpu = args.multi_gpu )
    t2t_model = MT( vocab_tar, vocab_tar, embedder_tar, embedder_tar, gen_tar_wrapper, encoder_tar, decoder_tar, denoising=True, multi_gpu = args.multi_gpu )
    s2t_model = MT( vocab_src, vocab_tar, embedder_src, embedder_tar, gen_tar_wrapper, encoder_src, decoder_tar, denoising=False, multi_gpu = args.multi_gpu )
    t2s_model = MT( vocab_tar, vocab_src, embedder_tar, embedder_src, gen_src_wrapper, encoder_tar, decoder_src, denoising=False, multi_gpu = args.multi_gpu )

    "optimizers"
    s2s_optimizer = torch.optim.Adam( s2s_param, lr=args.lr )
    t2t_optimizer = torch.optim.Adam( t2t_param, lr=args.lr )
    s2t_optimizer = torch.optim.Adam( s2t_param, lr=args.lr )
    t2s_optimizer = torch.optim.Adam( t2s_param, lr=args.lr )

    def save_model():
        # save embedder
        if args.embed_src == "":
             embedder_src.save_weight( args.save_path + "/embed_src.bin" )
        if args.embed_tar == "":
             embedder_tar.save_weight( args.save_path + "/embed_tar.bin" )

        # save generator
        if args.gen_src == "":
            gen_src_wrapper.save_weight( args.save_path + "/gen_src.bin" )
        if args.gen_tar == "":
            gen_tar_wrapper.save_weight( args.save_path + "/gen_tar.bin" )

        # save encoder
        encoder_src.save_weight( args.save_path + "/encoder_src.bin" )
        encoder_tar.save_weight( args.save_path + "/encoder_tar.bin" )

        # save decoder
        decoder_src.save_weight( args.save_path + "/decoder_src.bin" )
        decoder_tar.save_weight( args.save_path + "/decoder_tar.bin" )

        # save optimizer

        print( "all models saved" )
    
    def train_step( mt, optimizer, src_sents, tar_sents ):
        optimizer.zero_grad()
        loss = mt.get_loss( src_sents, tar_sents, train = True )
        loss += loss.data[0]
        res = loss.cpu().detach().item()
        loss.div( args.batch_size ).backward()
        optimizer.step()
        return res
    
    def train_step_backtranslate( mt, optimizer, src_sents, max_ratio ):
        tar_sents = mt.greedy( src_sents, max_ratio, mode = False )
        res = train_step( mt, optimizer, src_sents, tar_sents )
        return res

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            batch_size = len(src_sents)

            srclen = max( map( len, src_sents ) )
            tar_len = max( map( len, tgt_sents ) )
            print( "SRCLEN {} TARLEN {}".format( srclen, tar_len ) )


            model = s2t_model
            # (batch_size)
            train_step( s2s_model, s2s_optimizer, src_sents, src_sents )
            print( "finish s2s" )
            train_step( t2t_model, t2t_optimizer, tgt_sents, tgt_sents )  
            print( "finish t2t" )
            
            train_step( t2s_model, t2s_optimizer, tgt_sents, src_sents )
            print( "finish t2s" )
            loss = -train_step( model, s2t_optimizer, src_sents, tgt_sents )

            train_step_backtranslate( s2t_model, s2t_optimizer, src_sents, 2 )
            print( "finish s2t back" )
            train_step_backtranslate( t2s_model, t2s_optimizer, tgt_sents, 0.8 )
            print( "finish t2s back" )
            os.system( "nvidia-smi" )
            report_loss += loss
            cum_loss += loss

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         np.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                save_model()

                # if is_better:
                #     patience = 0
                #     print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                #     model.save(model_save_path)

                #     # You may also save the optimizer's state
                # elif patience < int(args.patience):
                #     patience += 1
                #     print('hit patience %d' % patience, file=sys.stderr)

                #     if patience == int(args.patience):
                #         num_trial += 1
                #         print('hit #%d trial' % num_trial, file=sys.stderr)
                #         if num_trial == int(args.max_num_trail):
                #             print('early stop!', file=sys.stderr)
                #             exit(0)

                #         # decay learning rate, and restore from previously best checkpoint
                #         lr = lr * float(args.lr_decay)
                #         print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                #         # load model
                #         model_save_path

                #         print('restore parameters of the optimizers', file=sys.stderr)
                #         # You may also need to load the state of the optimizer saved before

                #         # reset patience
                #         patience = 0

                if epoch == int(args.max_epoch):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model-config-path', dest='model_config_path',
    #                     type=str, default='LunarLander-v2-config.json',
    #                     help="Path to the model config file.")
    parser.add_argument('--train-src', dest='train_src',
                        type=str, default="",
                        help="train source text ")
    parser.add_argument('--train-tar', dest='train_tar',
                        type=str, default="",
                        help="train target text ")
    parser.add_argument('--dev-src', dest='dev_src',
                        type=str, default="",
                        help="dev source text ")
    parser.add_argument('--dev-tar', dest='dev_tar',
                        type=str, default="",
                        help="dev target text ")
    parser.add_argument('--embed-src', dest='embed_src',
                        type=str, default="",
                        help="embed source text ")
    parser.add_argument('--embed-tar', dest='embed_tar',
                        type=str, default="",
                        help="embed target text ")

    parser.add_argument('--gen-src', dest='gen_src',
                        type=str, default="",
                        help="gen source text ")
    parser.add_argument('--gen-tar', dest='gen_tar',
                        type=str, default="",
                        help="gen target text ")

    parser.add_argument('--batch-size', dest='batch_size',
                        type=int, default=64,
                        help="dev target text ")
    parser.add_argument('--valid-iter', dest='valid_iter', type=int,
                        default=1000, help="validate every * iter")
    parser.add_argument('--log-every', dest='log_every', type=int,
                        default=5000, help="The frequency to save model")
    parser.add_argument('--save-path', dest='save_path', type=str,
                        default="", help="Model save path")
    parser.add_argument('--vocab-src', dest='vocab_src', type=str,
                        default="", help="model source vocab path")
    parser.add_argument('--vocab-tar', dest='vocab_tar', type=str,
                        default="", help="model target vocab path")
    parser.add_argument('--embed-size', dest='embed_size', type=int,
                        default=256, help="Model embed_size")
    parser.add_argument('--hidden-size', dest='hidden_size', type=int,
                        default=256, help="Model hidden_size")
    parser.add_argument('--dropout', dest='dropout', type=float,
                        default=0.8, help="Model dropout size")
    parser.add_argument('--patience', dest='patience', type=int,
                        default=8, help="patience")
    parser.add_argument('--max-num-trial', dest='max_num_trial', type=int,
                        default=3, help="Max number of trial before being early stopped")
    parser.add_argument('--max-epoch', dest='max_epoch', type=int,
                        default=50, help="Model max epoch to train")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=0.0001, help="Model lr")
    parser.add_argument('--lr-decay', dest='lr_decay', type=float,
                        default=0.02, help="Model lr decay")
    parser.add_argument('--encoder-bidir', dest='encoder_bidir', type=int,
                        default=1, help="is encoder bidirectional") 
    parser.add_argument('--encoder-layer', dest='encoder_layer', type=int,
                        default=1, help="is encoder layers") 
    parser.add_argument('--multi-gpu', dest='multi_gpu', type=int,
                        default=0, help="is encoder layers")  
    return parser.parse_args()         

if __name__ == '__main__':
    args = parse_arguments()
    train( args )