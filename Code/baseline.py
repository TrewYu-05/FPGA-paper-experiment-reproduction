import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def DIS(X, k=5):
    """
    Distance-based Outlier Detection.
    """
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    return distances[:, -1]

def ODIN(X, k=5):
    """
    Outlier Detection using In-degree Number.
    """
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    graph = nn.kneighbors_graph(X, mode='connectivity')
    graph.setdiag(0)
    in_degree = np.array(graph.sum(axis=0)).flatten()
    anomaly_score = 1.0 / (in_degree + 1e-5)
    return anomaly_score

def LDOF(X, k=5):
    """
    Local Distance-Based Outlier Factor.
    """
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    n_samples = X.shape[0]
    ldof_scores = np.zeros(n_samples)

    for i in range(n_samples):
        knn_indices = indices[i, 1:]
        if len(knn_indices) == 0:
            ldof_scores[i] = 0
            continue

        d_p = np.mean(distances[i, 1:])

        knn_points = X[knn_indices]
        if len(knn_points) > 1:
            D_in = pdist(knn_points)
            d_in = np.mean(D_in)
        else:
            d_in = 0

        if d_in > 0:
            ldof_scores[i] = d_p / d_in
        else:
            ldof_scores[i] = 0

    return ldof_scores

def OutRanka(X, k=5):
    """
    OutRanka using Markov random walk.
    """
    n = X.shape[0]
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, 0)

    sim_k = np.zeros_like(sim)
    for i in range(n):
        idx = np.argsort(sim[i])[-k:]
        sim_k[i, idx] = sim[i, idx]

    totSim = sim_k.sum(axis=1, keepdims=True)
    totSim[totSim == 0] = 1e-5
    S = sim_k / totSim

    d = 0.1
    c = np.ones(n) / n
    c_prev = np.zeros(n)

    iters = 0
    while np.linalg.norm(c - c_prev, 1) > 1e-4 and iters < 1000:
        c_prev = c.copy()
        c = d / n + (1 - d) * np.dot(S.T, c_prev)
        iters += 1

    outlier_scores = -c
    return (outlier_scores - outlier_scores.min()) / (outlier_scores.max() - outlier_scores.min() + 1e-10)

def WNINOD(X, h=2):
    """
    Weighted Neighbourhood Information Network Outlier Detection
    """
    n, m = X.shape
    X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)

    std_dev = X_std.std(axis=0)
    mean_val = X_std.mean(axis=0)

    r_h = np.zeros(m)
    for i in range(m):
        if mean_val[i] != 0:
            r_h[i] = std_dev[i] / (h * mean_val[i])
        else:
            r_h[i] = 0

    E_h = np.zeros(m)
    e_ij = np.zeros((n, n, m))
    s_ij = np.zeros((n, n, m))

    for attr in range(m):
        col = X_std[:, attr].reshape(-1, 1)
        dist_mat = squareform(pdist(col, 'euclidean'))

        neighbors = dist_mat <= r_h[attr]
        np.fill_diagonal(neighbors, False)

        e_ij[:, :, attr] = neighbors.astype(int)
        s_ij[:, :, attr] = 1 - dist_mat

        sum_e_ij = e_ij[:, :, attr].sum()
        if sum_e_ij > 0:
            p_e = e_ij[:, :, attr] / sum_e_ij
            p_e = p_e[p_e > 0]
            denom = n**2 - n
            if denom > 1:
                E_h[attr] = - (1.0 / np.log(denom)) * np.sum(p_e * np.log(p_e))
            else:
                E_h[attr] = 0
        else:
            E_h[attr] = 0

    sum_E_h = E_h.sum()
    if sum_E_h != m:
        w_h = (1 - E_h) / (m - sum_E_h)
    else:
        w_h = np.ones(m) / m

    S = np.zeros((n, n))
    for attr in range(m):
        S += w_h[attr] * s_ij[:, :, attr]
    np.fill_diagonal(S, 0)

    U = np.zeros((n, n))
    for attr in range(m):
        U = np.logical_or(U, e_ij[:, :, attr]).astype(int)

    A = S * U

    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0] = 1e-5
    P = A / row_sums[:, np.newaxis]

    p = np.ones(n) / n
    p_prev = np.zeros(n)
    iters = 0
    while np.linalg.norm(p - p_prev, 1) > 1e-4 and iters < 1000:
        p_prev = p.copy()
        p = np.dot(p_prev, P)
        iters += 1

    a, b = 1, 100
    p_min, p_max = p.min(), p.max()
    if p_max - p_min > 0:
        u = (b - a) * (p - p_min) / (p_max - p_min) + a
    else:
        u = np.ones(n) * a

    sum_U_ij = U.sum(axis=1)
    sum_U_ij[sum_U_ij == 0] = 1
    inlier_scores = u / sum_U_ij

    outlier_scores = 1.0 / (inlier_scores + 1e-10)
    return outlier_scores

class INFLO:
    """
    INFLO implemented using scikit-learn's NearestNeighbors.
    Influenced Outlierness (INFLO) considers both k-NN and reverse k-NN.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        nn = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        n_samples = X.shape[0]
        # Calculate local reachability density (1 / average distance to k-NN)
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            # Reachability distance in INFLO paper max(k-dist(j), dist(i, j))
            # However, PyOD uses exact distances in some cases or max.
            # Using max(k-dist(j), dist(i,j)) for reachability distance:
            reach_dist = np.maximum(distances[indices[i, 1:], -1], distances[i, 1:])
            sum_reach_dist = np.sum(reach_dist)
            if sum_reach_dist > 0:
                lrd[i] = self.n_neighbors / sum_reach_dist
            else:
                lrd[i] = 1e10 # Very high density if duplicate points

        # INFLO computes reverse nearest neighbors
        from collections import defaultdict
        rnn = defaultdict(list)
        for i in range(n_samples):
            for j in indices[i, 1:]:
                rnn[j].append(i)

        self.decision_scores_ = np.zeros(n_samples)
        for i in range(n_samples):
            # Influence space: kNN union RNN
            is_space = list(set(list(indices[i, 1:]) + rnn[i]))
            if len(is_space) > 0:
                self.decision_scores_[i] = np.mean(lrd[is_space]) / lrd[i]
            else:
                self.decision_scores_[i] = 1.0
        return self
