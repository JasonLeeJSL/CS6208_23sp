from utils import set_seed
from model import DeepWalk
import os
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--source_file', type=str, default='data/sample_small.tsv', help='data source')
parser.add_argument('--device', type=str, default='cpu', help='device of torch')
parser.add_argument('--embedding_size', type=int, default=256, help='the size of embedding')
parser.add_argument('--window_size', type=int, default=5, help='the size of window')
parser.add_argument('--walk_length', type=int, default=20, help='the length of each walk')
parser.add_argument('--walk_num', type=int, default=5, help='the number of walks of each node')
parser.add_argument('--train_epochs', type=int, default=10000, help='optimization steps of skipgram')
parser.add_argument('--lr', type=float, default=1, help='learning rate of skipgram')
parser.add_argument('--lr_decay', type=float, default=0.99, help='lr decay rate')
parser.add_argument('--eval_step', type=int, default=100, help='eval per # steps')
parser.add_argument('--decay_step', type=int, default=100, help='decay per # steps')
parser.add_argument('--savepath', type=str, default="results", help='folder path to save the model')
parser.add_argument('--seed', type=int, default=304, help='the random seed')
parser.add_argument('--mode', type=int, default=0, help='skipgram mode, mode 0 is hand written skipgram, mode 1 is Word2Vec from gensim')
args = parser.parse_args()



if __name__ == "__main__":
    # set the random seed
    set_seed(args.seed)

    # constract the saving path
    figsavepath = None
    if args.savepath is not None:
        if not os.path.exists(args.savepath):
            os.mkdir(args.savepath)
        figsavepath = os.path.join(args.savepath,"pca.png")
        modelsavepath = os.path.join(args.savepath,"model.pkl")
    
    # build the deepwork model
    dw_model = DeepWalk(args.source_file, args.embedding_size, args.window_size, args.walk_length, 
                        args.walk_num, args.device, args.train_epochs, args.lr, args.lr_decay, args.eval_step, args.decay_step, figsavepath, args.mode)
    
    # run skipgram procedure
    dw_model.do_skip_gram()

    # result visualization
    dw_model.viz()

    # save the model
    pickle.dump(dw_model,open(modelsavepath,'wb'))
