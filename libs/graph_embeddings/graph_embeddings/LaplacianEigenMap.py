import numpy as np
from scipy import sparse
from graph_embeddings import utils

class LaplacianEigenMap():
    def __init__(self):
        self.in_vec = None
        self.L = None
        self.deg = None

    def fit(self, G):
        A = utils.to_adjacency_matrix(G)

        # Compute the (inverse) normalized laplacian matrix
        deg = np.array( A.sum(axis=1) ).reshape(-1)
        N = A.shape[0]
        Dsqrt =sparse.diags(1/np.maximum(np.sqrt(deg), 1e-12), format = "csr")
        L = Dsqrt @ A @ Dsqrt

        self.L = L
        self.deg = deg
        return self

    def transform(self, dim):
        if self.in_vec is None:
            self.update_embedding(dim)
        elif self.in_vec.shape[1] != dim:
            self.update_embedding(dim)
        return self.in_vec

    def update_embedding(self, dim):
        u, s, _ = utils.rSVD(self.L, dim + 1) # add one for the trivial solution
        order = np.argsort(s)[::-1][1:]
        u = u[:, order]

        Dsqrt =sparse.diags(1/np.maximum(np.sqrt(self.deg), 1e-12), format = "csr")
        self.in_vec = Dsqrt @ u
        self.in_vec = u
