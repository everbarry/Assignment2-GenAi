from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv


def load_dataset(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file, delimiter=',')
        dataset = np.array(list(reader)).astype(np.float32)
    return dataset


def print_tree(root, parents, children, node_names=None, indent="", is_last=True):
    if node_names is None:
        node_names = [str(i) for i in range(len(parents))]
    prefix = "└── " if is_last else "├── "
    print(f"{indent}{prefix}Node {node_names[root]}")
    child_indent = indent + ("    " if is_last else "│   ")
    for i, child in enumerate(children[root]):
        print_tree(child, parents, children, node_names, child_indent, i == len(children[root]) - 1)


class BinaryCLT:
    def __init__(self, data, root=None, alpha: float = 0.01):
        self.data = data
        self.n_samples, self.n_vars = data.shape
        self.alpha = alpha
        if root is None:
            self.root = np.random.randint(0, self.n_vars)
        else:
            assert 0 <= root < self.n_vars
            self.root = root
        self._learn_structure()
        self._compute_parameters()

    def _learn_structure(self):
        mi = np.zeros((self.n_vars, self.n_vars))
        counts = lambda i, j, vi, vj: np.sum((self.data[:, i] == vi) & (self.data[:, j] == vj)) + self.alpha
        for i in range(self.n_vars):
            for j in range(i + 1, self.n_vars):
                c00 = counts(i, j, 0, 0)
                c01 = counts(i, j, 0, 1)
                c10 = counts(i, j, 1, 0)
                c11 = counts(i, j, 1, 1)
                p00 = c00 / (self.n_samples + 4 * self.alpha)
                p01 = c01 / (self.n_samples + 4 * self.alpha)
                p10 = c10 / (self.n_samples + 4 * self.alpha)
                p11 = c11 / (self.n_samples + 4 * self.alpha)
                pi0 = (np.sum(self.data[:, i] == 0) + self.alpha) / (self.n_samples + 2 * self.alpha)
                pi1 = (np.sum(self.data[:, i] == 1) + self.alpha) / (self.n_samples + 2 * self.alpha)
                pj0 = (np.sum(self.data[:, j] == 0) + self.alpha) / (self.n_samples + 2 * self.alpha)
                pj1 = (np.sum(self.data[:, j] == 1) + self.alpha) / (self.n_samples + 2 * self.alpha)
                m = 0
                for pxy, px, py in [(p00, pi0, pj0), (p01, pi0, pj1), (p10, pi1, pj0), (p11, pi1, pj1)]:
                    if pxy > 0:
                        m += pxy * np.log(pxy / (px * py))
                mi[i, j] = mi[j, i] = m
        mst = minimum_spanning_tree(-mi).toarray()
        self.parents = np.full(self.n_vars, -1, int)
        self.children = [[] for _ in range(self.n_vars)]
        _, pred = breadth_first_order(mst, self.root, directed=False, return_predecessors=True)
        for i in range(self.n_vars):
            if i != self.root:
                p = int(pred[i])
                self.parents[i] = p
                self.children[p].append(i)

    def _compute_parameters(self):
        self.log_params = np.zeros((self.n_vars, 2, 2))
        # root
        rc = np.zeros(2)
        rc[0] = np.sum(self.data[:, self.root] == 0) + self.alpha
        rc[1] = np.sum(self.data[:, self.root] == 1) + self.alpha
        rp = rc / (self.n_samples + 2 * self.alpha)
        self.log_params[self.root, :, :] = np.log(rp)
        # others
        for i in range(self.n_vars):
            if i == self.root: continue
            p = self.parents[i]
            joint = np.zeros((2, 2))
            for vi in (0, 1):
                for vp in (0, 1):
                    joint[vp, vi] = np.sum((self.data[:, p] == vp) & (self.data[:, i] == vi)) + self.alpha
            joint /= (self.n_samples + 4 * self.alpha)
            pc = np.zeros(2)
            pc[0] = np.sum(self.data[:, p] == 0) + 2 * self.alpha
            pc[1] = np.sum(self.data[:, p] == 1) + 2 * self.alpha
            pc /= (self.n_samples + 4 * self.alpha)
            for vp in (0, 1):
                for vi in (0, 1):
                    self.log_params[i, vp, vi] = np.log(joint[vp, vi] / pc[vp])

    def get_tree(self):
        return self.parents.copy()

    def get_log_params(self):
        return self.log_params.copy()

    def _log_prob_exhaustive(self, row):
        obs = ~np.isnan(row)
        miss = np.where(~obs)[0]
        if miss.size == 0:
            return self._compute_tree_log_probs(row.reshape(1, -1))[0]
        acc = []
        for comb in itertools.product((0, 1), repeat=miss.size):
            r = row.copy()
            r[miss] = comb
            acc.append(self._compute_tree_log_probs(r.reshape(1, -1))[0])
        return logsumexp(acc)

    def _log_prob_sumprod(self, row):  
        obs = {i: int(row[i]) for i in range(self.n_vars) if not np.isnan(row[i])} 
        def pass_up(j):  
            msgs = [pass_up(c) for c in self.children[j]]  
            out = np.full(2, -np.inf)  
            for pv in (0,1):  
                tmp = []  
                for xj in (0,1):  
                    val = self.log_params[j, pv, xj]  
                    for m in msgs: val += m[xj]  
                    if j in obs and obs[j] != xj:  
                        val = -np.inf  
                    tmp.append(val)  
                out[pv] = logsumexp(tmp)  
            return out  
        children_msgs = [pass_up(c) for c in self.children[self.root]]  
        if self.root in obs:  
            xr = obs[self.root]  
            res = self.log_params[self.root, 0, xr] + sum(m[xr] for m in children_msgs)  
        else:  
            tmp = []  
            for xr in (0,1):  
                val = self.log_params[self.root,0,xr]  
                for m in children_msgs: val += m[xr]  
                tmp.append(val)  
            res = logsumexp(tmp)  
        return res  

    def log_prob(self, X, exhaustive: bool = False):
        if X.ndim == 1: X = X.reshape(1, -1)
        out = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            if exhaustive:
                out[i] = self._log_prob_exhaustive(row)
            else:
                out[i] = self._log_prob_sumprod(row) 
        return out

    def _compute_tree_log_probs(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        lp = np.zeros(X.shape[0])
        for i, row in enumerate(X):
            val = self.log_params[self.root,0,int(row[self.root])]
            for j in range(self.n_vars):
                if j == self.root: continue
                p = self.parents[j]
                val += self.log_params[j, int(row[p]), int(row[j])]
            lp[i] = val
        return lp

    def sample(self, n):
        order = [self.root] + list(itertools.chain(*[self.children[i] for i in order]))
        S = np.zeros((n, self.n_vars))
        for t in range(n):
            for j in order:
                if j == self.root:
                    p = np.exp(self.log_params[j,0])
                    S[t,j] = np.random.choice((0,1), p=p)
                else:
                    pv = int(S[t,self.parents[j]])
                    p = np.exp(self.log_params[j,pv])
                    S[t,j] = np.random.choice((0,1), p=p)
        return S

    def visualize_tree(self):
        print(f"Tree rooted at {self.root}")
        print_tree(self.root, self.parents, self.children)

if __name__ == "__main__":
    data = load_dataset("datasets/nltcs/nltcs.train.data")

    model = BinaryCLT(data, root=0, alpha=0.01)
    print("Tree:", model.get_tree())
    print("LL:", model.log_prob(data[0:3], exhaustive=False))
