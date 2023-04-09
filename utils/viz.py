from sklearn import decomposition
import warnings
from matplotlib import pyplot as plt

# helper functions of visualization

def viz_2d_0(matrix, inverse_nodemap, savepath):
    # visualization of written skipgram

    plt.rcParams['figure.figsize'] = (15,12)
    warnings.filterwarnings("ignore")
    pca = decomposition.PCA(2)
    dec_mat = pca.fit_transform(matrix)
    x, y = dec_mat[:,0], dec_mat[:,1]
    plt.scatter(x, y)
    for i in inverse_nodemap:
        plt.text(x[i],y[i],inverse_nodemap[i])
    #plt.show()
    if savepath is not None:
        plt.savefig(savepath)

def viz_2d_1(matrix, nodelist, savepath):
    # visualization of gensim Word2Vec skipgram

    plt.rcParams['figure.figsize'] = (15,12)
    warnings.filterwarnings("ignore")
    pca = decomposition.PCA(n_components=2)
    dec_mat = pca.fit_transform(matrix)
    x, y = dec_mat[:,0], dec_mat[:,1]
    plt.scatter(x, y)
    for i, node in enumerate(nodelist):
        plt.text(x[i],y[i],node)
    #plt.show()
    if savepath is not None:
        plt.savefig(savepath)


