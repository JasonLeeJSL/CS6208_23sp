from utils import read_graph_from_csv
from utils import get_random_walks
from utils import SkipGram
from gensim.models import Word2Vec
from utils import viz_2d_1

# the deepwalk model
# for skipgram part, we use both self-written model and word2vec model
# can be switch by setting mode to 0 or 1

class DeepWalk:
    def __init__(self, source_file, embedding_size = 256, window_size = 5, walk_length = 10, walk_num = 5, 
                 device = 'cpu', train_epochs = 2000, lr = 5e-1, lr_decay = 0.99, eval_step = 100, decay_step = 10, savepath = None, mode = 0):
        # initialize the DeepWalk class with necessary hyper-parameters
        self.source_file = source_file
        self.G = read_graph_from_csv(source_file)
        self.nodelist = list(self.G.nodes())
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.walk_length = walk_length
        self.walk_num = walk_num
        self.device = device
        self.train_epochs = train_epochs
        self.lr = lr
        self.lr_decay =lr_decay
        self.eval_step = eval_step
        self.decay_step = decay_step
        self.savepath = savepath
        self.mode = mode

        self.get_random_walks()
    
    def get_embedding_size(self):
        # return the current embedding size
        return self.embedding_size
    def get_window_size(self):
        # return the current window size
        return self.window_size
    def get_walk_length(self):
        # return the current walk length
        return self.walk_length
    def get_walk_num(self):
        # return the current walk per vertex
        return self.walk_num
    def set_embedding_size(self, newval):
        # return the current embedding size
        self.embedding_size = newval
    def set_window_size(self, newval):
        # return the current embedding size
        self.window_size = newval
    def set_walk_length(self, newval):
        # return the current embedding size
        self.walk_length = newval
    def set_walk_num(self, newval):
        # return the current embedding size
        self.walk_num = newval
    
    # different from the paper, we first get all walks, then use skip gram to optimize the vectors
    def get_random_walks(self):
        # get random walks from the graph based on the hyper-parameters
        self.walks = get_random_walks(self.G, self.walk_num, self.walk_num) 
        return self.walks
    
    def do_skip_gram(self):
        # mode 0: use written skipgram
        if self.mode == 0:
            self.model = SkipGram(self.walks, self.window_size, self.device, self.embedding_size, self.train_epochs, self.lr, self.lr_decay, self.eval_step, self.decay_step)
            self.model.model_train()
            self.embedding_matrix = self.model.M_node_to_embedding
            self.inverse_matrix = self.model.M_embedding_to_node
        # mode 1: use gensim word2vec skipgram
        elif self.mode == 1:
            self.model = Word2Vec(window=self.window_size, sg=1)
            self.model.build_vocab(self.walks, progress_per=10)
            self.model.train(self.walks, total_examples=self.model.corpus_count, epochs=self.train_epochs)
            self.embedding_matrix = []
            for node in self.nodelist:
                self.embedding_matrix.append(self.model.wv[node])
            
    
    def viz(self):
        # visualization of results
        if self.mode == 0:
            self.model.viz(self.embedding_matrix, self.savepath)
        elif self.mode == 1:
            viz_2d_1(self.embedding_matrix, self.nodelist, self.savepath)


    
    

