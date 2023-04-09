import torch
from torch.autograd import Variable
import warnings
from tqdm import tqdm
from .viz import viz_2d_0

# skipgram codes

class SkipGram:
    def __init__(self, walks, window_size, device, embedding_size, train_epochs, lr, lr_decay, eval_step, decay_step):
        warnings.filterwarnings("ignore")
        self.walks = walks
        self.window_size = window_size
        self.device = device
        self.embedding_size = embedding_size
        self.train_epochs = train_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.eval_step = eval_step
        self.decay_step = decay_step
        
        self.initpara = 0.5 / self.embedding_size
        print("Creating node map...")
        self._create_nodemap()
        self._create_train_data()
        print("Model initializing...")
        self._model_init()
            
    def _create_nodemap(self):
        # create a nodemap, mapping node to index
        # similar with a vocabulary
        self.nodemap = {}
        idx = 0
        for rw in self.walks:
            for node in rw:
                if node not in self.nodemap:
                    self.nodemap[node] = idx
                    idx += 1
        self.node_num = len(self.nodemap)
        
    def _create_train_data(self):
        # create training data for optimizing
        # similar with word2vec method
        self.traindata = []
        for rw in self.walks:
            for i, node in enumerate(rw):
                center = node
                for n in range(1, self.window_size + 1):
                    if (i-n)>=0:
                        pred = rw[i-n]
                        self.traindata.append([self.nodemap[center], self.nodemap[pred]])
                    if (i+n)<len(rw):
                        pred = rw[i+n]
                        self.traindata.append([self.nodemap[center], self.nodemap[pred]])
        self.traindata = torch.tensor(self.traindata, device=self.device)

    
    def _model_init(self):
        # initialize the model and parameters
        self.M_node_to_embedding = Variable(torch.randn(self.node_num, self.embedding_size, device = self.device).uniform_(-self.initpara, self.initpara).float(), requires_grad=True)
        self.M_embedding_to_node = Variable(torch.randn(self.embedding_size, self.node_num, device = self.device).uniform_(-self.initpara, self.initpara).float(), requires_grad=True)
    
    def model_train(self):
        # the training process of the model
        print("Optimizing...")
        def get_input_tensor(x):
            # a helper function
            # transfer a node index into one-hot code
            ret = []
            for idx in x:
                node = []
                for e in range(self.node_num):
                    node.append(0)
                node[idx] = 1
                ret.append(node)
            return Variable(torch.tensor(ret)).float().to(self.device)
        
        learning_rate = self.lr
        x, y = self.traindata[:,0], self.traindata[:,1]
        input_tensor = get_input_tensor(x)
        y = y.reshape((input_tensor.shape[0],))

        for epoch in tqdm(range(self.train_epochs)):
            embedding = input_tensor.mm(self.M_node_to_embedding)
            y_pred = embedding.mm(self.M_embedding_to_node)
            loss_f = torch.nn.CrossEntropyLoss() # note: softmax included
            loss = loss_f(y_pred, y)
            loss.backward()
            # GD based update
            with torch.no_grad():
                self.M_node_to_embedding -= learning_rate * self.M_node_to_embedding.grad.data
                self.M_embedding_to_node -= learning_rate * self.M_embedding_to_node.grad.data
                self.M_node_to_embedding.grad.data.zero_()
                self.M_embedding_to_node.grad.data.zero_()
            if epoch % self.decay_step == 0:
                learning_rate *= self.lr_decay
            if epoch % self.eval_step == 0:
                print(f'Epoch {epoch}, loss = {loss}')
        self.M_node_to_embedding = self.M_node_to_embedding.detach().cpu().numpy()
        self.M_embedding_to_node = self.M_embedding_to_node.detach().cpu().numpy()
    
    def viz(self, matrix, savepath = None):
        self.inverse_nodemap = dict()
        for key in self.nodemap:
            self.inverse_nodemap[self.nodemap[key]] = key
        viz_2d_0(matrix, self.inverse_nodemap, savepath)            


