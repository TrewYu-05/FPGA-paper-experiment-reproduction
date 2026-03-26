import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics.pairwise import cosine_similarity
import warnings

def DIS(X, k=5):
    """
    Distance-based Outlier Detection.
    Computes the average distance to the k nearest neighbors.
    Often standard k-NN outlier is distance to k-th neighbor, but DIS can be average distance.
    """
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    # Average distance to the k neighbors (excluding self which is at index 0)
    return np.mean(distances[:, 1:], axis=1)

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
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        nn = NearestNeighbors(n_neighbors=self.n_neighbors+1)
        nn.fit(X)
        distances, indices = nn.kneighbors(X)

        n_samples = X.shape[0]
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            reach_dist = np.maximum(distances[indices[i, 1:], -1], distances[i, 1:])
            sum_reach_dist = np.sum(reach_dist)
            if sum_reach_dist > 0:
                lrd[i] = self.n_neighbors / sum_reach_dist
            else:
                lrd[i] = 1e10

        from collections import defaultdict
        rnn = defaultdict(list)
        for i in range(n_samples):
            for j in indices[i, 1:]:
                rnn[j].append(i)

        self.decision_scores_ = np.zeros(n_samples)
        for i in range(n_samples):
            is_space = list(set(list(indices[i, 1:]) + rnn[i]))
            if len(is_space) > 0:
                self.decision_scores_[i] = np.mean(lrd[is_space]) / lrd[i]
            else:
                self.decision_scores_[i] = 1.0
        return self

class FastABOD:
    """
    Fast Angle-Based Outlier Detection.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        n_samples = X.shape[0]
        self.decision_scores_ = np.zeros(n_samples)

        k = min(self.n_neighbors, n_samples - 1)
        if k < 2:
            return self

        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        for i in range(n_samples):
            knn_idx = indices[i, 1:]
            angles = []

            # Extract points
            A = X[knn_idx] - X[i]

            # Compute pairwise dot products and norms
            # We want angle variance: dot(a, b) / (norm(a)*norm(b))^2 or similar.
            # Classic ABOD score for a pair (a,b) is <a,b> / (||a||^2 ||b||^2)
            # Or <a,b> / (||a||^2 ||b||^2) or variance of angles?
            # In ABOD paper, ABOF is the variance of the angle weights.
            # Angle weight = <AB, AC> / (||AB||^2 * ||AC||^2)

            for u in range(k):
                for v in range(u+1, k):
                    AB = A[u]
                    AC = A[v]

                    norm_AB_sq = np.dot(AB, AB)
                    norm_AC_sq = np.dot(AC, AC)

                    if norm_AB_sq > 0 and norm_AC_sq > 0:
                        weight = np.dot(AB, AC) / (norm_AB_sq * norm_AC_sq)
                        angles.append(weight)

            if len(angles) > 1:
                # ABOF is the variance of these weights
                abof = np.var(angles)
            else:
                abof = 0.0

            # Inverse of ABOF is the outlier score (smaller variance -> more anomalous)
            if abof > 0:
                self.decision_scores_[i] = 1.0 / (abof + 1e-10)
            else:
                self.decision_scores_[i] = 1e10

        return self


class kNN:
    """
    Standard k-Nearest Neighbors Outlier Detection.
    Returns distance to the k-th nearest neighbor.
    """
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        k = min(self.n_neighbors, X.shape[0] - 1)
        if k < 1:
            self.decision_scores_ = np.zeros(X.shape[0])
            return self

        nn = NearestNeighbors(n_neighbors=k+1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        self.decision_scores_ = distances[:, -1]
        return self

from sklearn.ensemble import IsolationForest

class IForest:
    """
    Isolation Forest wrapper.
    """
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X):
        clf = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state)
        clf.fit(X)
        # decision_function returns positive for inliers and negative for outliers.
        # we negate it so higher score = more anomalous
        self.decision_scores_ = -clf.decision_function(X)
        return self
