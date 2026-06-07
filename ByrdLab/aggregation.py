import copy
import itertools
import math
from collections import deque
import sklearn.metrics.pairwise as smp
from sklearn.cluster import KMeans
import numpy as np
import torch
from ByrdLab import FEATURE_TYPE, DEVICE
from scipy import stats

from ByrdLab.library.tool import MH_rule, flatten_list, unflatten_vector

class CentraliedAggregation():
    def __init__(self, name, honest_nodes, byzantine_nodes):
        self.name = name
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
    def run(self, messages):
        raise NotImplementedError


def mean(wList):
    return torch.mean(wList, dim=0)


def geometric_median(wList, max_iter=80, err=1e-5):
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack(
            [w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= err:
            break
    return guess


def medoid_index(wList):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()

            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data
    dist_sum = dist.sum(dim=1)
    return dist_sum.argmax()


def medoid(wList):
    return wList[medoid_index(wList)]


def Krum_index(wList, byzantine_size):
    node_size = wList.size(0)
    dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
    for i in range(node_size):
        for j in range(i):
            distance = (wList[i].data - wList[j].data).norm()**2

            distance = -distance
            dist[i][j] = distance.data
            dist[j][i] = distance.data

    k = node_size - byzantine_size - 2 + 1
    topv, _ = dist.topk(k=k, dim=1)
    scores = topv.sum(dim=1)
    return scores.argmax()


def Krum(wList, byzantine_size):
    index = Krum_index(wList, byzantine_size)
    return wList[index]


def mKrum(wList, byzantine_size, m=1):
    remain = wList
    result = torch.zeros_like(wList[0], dtype=FEATURE_TYPE)
    for _ in range(m):
        res_index = Krum_index(remain, byzantine_size)
        result += remain[res_index]
        remain = remain[torch.arange(remain.size(0)) != res_index]
    return result / m


def median(wList):
    return wList.median(dim=0)[0]


def pairwise(data):

    n = len(data)
    for i in range(n - 1):
        for j in range(i + 1, n):
            yield (data[i], data[j])


def brute_selection(gradients, f, **kwargs):

    n = len(gradients)

    distances = [0] * (n * (n - 1) // 2)
    for i, (x, y) in enumerate(pairwise(tuple(range(n)))):
        distances[i] = gradients[x].sub(gradients[y]).norm().item()

    sel_iset = None
    sel_diam = None
    for cur_iset in itertools.combinations(range(n), n - f):

        cur_diam = 0.
        for x, y in pairwise(cur_iset):

            cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]

            if not math.isfinite(cur_dist):
                break

            if cur_dist > cur_diam:
                cur_diam = cur_dist
    else:

            if sel_iset is None or cur_diam < sel_diam:
                sel_iset = cur_iset
                sel_diam = cur_diam

    assert sel_iset is not None, "Too many non-finite gradients: a non-Byzantine gradient must only contain finite coordinates"
    return sel_iset


def brute(gradients, byzantine_size, **kwargs):

    sel_iset = brute_selection(gradients, byzantine_size, **kwargs)
    return sum(gradients[i] for i in sel_iset).div_(len(gradients) - byzantine_size)









def trimmed_mean(wList, byzantine_size):

    sorted_wList, _ = torch.sort(wList, dim=0)


    if byzantine_size == 0:
        trimmed_data = sorted_wList
    elif byzantine_size > 0:
        trimmed_data = sorted_wList[byzantine_size:-byzantine_size, :]
    else:
        assert False, 'Byzantine size should be equal or larger than 0!'


    if trimmed_data.nelement() > 0:
        tm = torch.mean(trimmed_data, dim=0)
    else:
        tm = 0

    return tm


def remove_outliers(wList, byzantine_size):
    mean = torch.mean(wList, dim=0)

    distances = torch.tensor([
        -torch.norm(model - mean) for model in wList
    ])
    node_size = wList.size(0)
    remain_cnt = node_size - byzantine_size
    (_, remove_index) = torch.topk(distances, k=remain_cnt)
    return wList[remove_index].mean(dim=0)


def faba(wList, byzantine_size):
    remain = wList
    for _ in range(byzantine_size):
        mean = remain.mean(dim=0)

        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        remove_index = distances.argmax()
        remain = remain[torch.arange(remain.size(0)) != remove_index]
    return remain.mean(dim=0)


def bulyan(wList, byzantine_size):
    remain = wList
    selected_ls = []
    node_size = wList.size(0)
    selection_size = node_size-2*byzantine_size
    for _ in range(selection_size):
        res_index = Krum_index(remain, byzantine_size)
        selected_ls.append(remain[res_index])
        remain = remain[torch.arange(remain.size(0)) != res_index]
    selection = torch.stack(selected_ls)
    m = median(selection)
    dist = -(selection - m).abs()
    indices = dist.topk(k=selection_size-2*byzantine_size, dim=0)[1]
    if len(wList.size()) == 1:
        result = selection[indices].mean()
    else:
        result = torch.stack([
            selection[indices[:, d], d].mean() for d in range(wList.size(1))])
    return result


class C_mean(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='mean', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):
        return torch.mean(messages, dim=0)


class C_trimmed_mean(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='trimmed_mean', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):

        sorted_wList, _ = torch.sort(messages, dim=0)
        byzantine_size = len(self.byzantine_nodes)


        if byzantine_size == 0:
            trimmed_data = sorted_wList
        elif byzantine_size > 0:
            trimmed_data = sorted_wList[byzantine_size:-byzantine_size, :]
        else:
            assert False, 'Byzantine size should be equal or larger than 0!'


        if trimmed_data.nelement() > 0:
            tm = torch.mean(trimmed_data, dim=0)
        else:
            tm = 0

        return tm


class C_faba(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='faba', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)

    def run(self, messages):
        remain = messages
        byzantine_size = len(self.byzantine_nodes)

        for _ in range(byzantine_size):
            mean = torch.mean(remain, dim=0)

            distances = torch.tensor([
                torch.norm(model - mean) for model in remain
            ])
            remove_index = distances.argmax()
            remain = remain[torch.arange(remain.size(0)) != remove_index]
        return remain.mean(dim=0)


class C_centered_clipping(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        self.memory = None
        self.threshold = threshold

    def run(self, messages):
        if self.memory == None:
            self.memory = torch.zeros_like(messages[0])

        diff = torch.zeros_like(self.memory)
        for n in self.honest_nodes + self.byzantine_nodes:
            grad = messages[n]
            norm = (grad - self.memory).norm()
            if norm > self.threshold:
                diff += self.threshold * (grad - self.memory) / norm
            else:
                diff += grad - self.memory
        diff /= (len(self.honest_nodes) + len(self.byzantine_nodes))
        self.memory = self.memory + diff
        return self.memory


class C_LFighter(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__('LFighter', honest_nodes, byzantine_nodes)

    def clusters_dissimilarity(self, clusters):
        n0 = len(clusters[0])
        n1 = len(clusters[1])
        m = n0 + n1
        cs0 = smp.cosine_similarity(clusters[0]) - np.eye(n0)
        cs1 = smp.cosine_similarity(clusters[1]) - np.eye(n1)
        mincs0 = np.min(cs0, axis=1)
        mincs1 = np.min(cs1, axis=1)
        ds0 = n0/m * (1 - np.mean(mincs0))
        ds1 = n1/m * (1 - np.mean(mincs1))
        return ds0, ds1

    def run(self, messages):
        m = len(messages)
        dw = [[] for _ in range(m)]
        for i in range(m):
            dw[i].append(messages[i][-2].cpu().data.numpy())
        dw = np.asarray(dw)
        dw = np.squeeze(dw)
        norms = np.linalg.norm(dw, axis=-1)
        memory = np.sum(norms, axis=0)
        max_two_freq_classes = memory.argsort()[-2:]

        data = []
        for i in range(m):
            data.append(dw[i][max_two_freq_classes].reshape(-1))

        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
        labels = kmeans.labels_

        clusters = {0:[], 1: []}
        for i, l in enumerate(labels):
            clusters[l].append(data[i])

        good_cl = 0
        cs0, cs1 = self.clusters_dissimilarity(clusters)
        if cs0 < cs1:
            good_cl = 1

        remain_worker_grad = []
        for i, l in enumerate(labels):
            if l == good_cl:
                remain_worker_grad.append(messages[i])

        mean = [torch.zeros_like(para, requires_grad=False) for para in messages[0]]
        for grad in remain_worker_grad:
            for i, g in enumerate(grad):
                mean[i].add_(g, alpha=1 / len(remain_worker_grad))

        return mean


class DecentralizedAggregation():
    def __init__(self, name, graph, superparameter={}):
        self.name = name
        self.graph = graph


        self.global_state = {}
        self.required_info = set()
        self.superparam = superparameter

    def run(self, local_models, node):
        raise NotImplementedError

    def all_neighbor_models(self, local_models, node):
        return local_models[self.graph.neighbors[node]]

    def neighbor_models_and_itself(self, local_model, node):
        li = list(self.graph.neighbors[node]) + [node]
        return local_model[li]


class D_mean(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='mean', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return neighbor_models.mean(axis=0)



class D_no_communication(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='no_communication',
                                                 graph=graph)

    def run(self, local_models, node):
        return local_models[node]


class D_meanW(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='meanW', graph=graph)
        self.W = MH_rule(graph)
        self.W = self.W.to(DEVICE)

    def run(self, local_models, node):
        return torch.tensordot(self.W[node], local_models, dims=1)


class D_median(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='median', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return median(neighbor_models)


class D_geometric_median(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='geometric_median',
                                                 graph=graph)

    def run(self,  local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return geometric_median(neighbor_models)


class D_Krum(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='Krum', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        return Krum(neighbor_models, byzantine_size=self.graph.byzantine_sizes[node])


class D_mKrum(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='mKrum', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.neighbor_models_and_itself(local_models, node)
        m = self.graph.neighbor_sizes[node] - \
            2*self.graph.byzantine_sizes[node]-3
        return mKrum(neighbor_models,
                     byzantine_size=self.graph.byzantine_sizes[node],
                     m=m)


class D_trimmed_mean(DecentralizedAggregation):
    def __init__(self, graph, exact_byz_cnt=True, byz_cnt=-1):
        if exact_byz_cnt:
            name = 'trimmed_mean'
        else:
            if byz_cnt < 0:
                name = 'trimmed_mean_max'
            else:
                name = f'trimmed_mean_{byz_cnt}'
        super().__init__(name=name, graph=graph)

        self.exact_byz_cnt = exact_byz_cnt
        self.Byz_cnt = byz_cnt

    def run(self, local_models, node):
        if self.exact_byz_cnt:
            estimate_byz_cnt = self.graph.byzantine_sizes[node]
        else:
            if self.Byz_cnt < 0:
                estimate_byz_cnt = max(self.graph.byzantine_sizes)
            else:
                estimate_byz_cnt = self.Byz_cnt
        neighbor_models = self.all_neighbor_models(local_models, node)
        tm = trimmed_mean(neighbor_models, byzantine_size=estimate_byz_cnt)
        trimmed_neighbor_size = len(neighbor_models) - 2 * estimate_byz_cnt
        local_model = local_models[node]
        return (tm * trimmed_neighbor_size + local_model) / (trimmed_neighbor_size + 1)


class D_remove_outliers(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='remove_outliers',
                                                graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        rm = remove_outliers(neighbor_models,
                             byzantine_size=self.graph.byzantine_sizes[node])
        neighbor_size = len(neighbor_models)
        res = (rm * neighbor_size + local_model) / (neighbor_size + 1)
        return res


class D_faba(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='FABA', graph=graph)

    def run(self, local_models, node):
        neighbor_models = self.all_neighbor_models(local_models, node)
        local_model = local_models[node]
        agg = faba(neighbor_models,
                   byzantine_size=self.graph.byzantine_sizes[node])
        hoenst_neighbor_size = self.graph.honest_sizes[node]
        res = (agg * hoenst_neighbor_size + local_model) / \
            (hoenst_neighbor_size + 1)
        return res


class D_ios(DecentralizedAggregation):
    def __init__(self, graph, exact_byz_cnt=True, byz_cnt=-1):
        if exact_byz_cnt:
            name = 'IOS'
        else:
            if byz_cnt < 0:
                name = 'IOS_max'
            else:
                name = f'IOS_{byz_cnt}'
        super().__init__(name=name, graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE).to(DEVICE)
        self.exact_byz_cnt = exact_byz_cnt
        self.Byz_cnt = byz_cnt
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                i_n = self.graph.neighbor_sizes[i] + 1
                j_n = self.graph.neighbor_sizes[j] + 1
                self.W[i][j] = 1 / max(i_n, j_n)

                self.W[i][i] -= self.W[i][j]

    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        if self.exact_byz_cnt:
            estimate_byz_cnt = self.graph.byzantine_sizes[node]
        else:
            if self.Byz_cnt < 0:
                estimate_byz_cnt = max(self.graph.byzantine_sizes)
            else:
                estimate_byz_cnt = self.Byz_cnt
        for _ in range(estimate_byz_cnt):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]

            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]
        return res


class D_ios_equal_neigbor_weight(DecentralizedAggregation):
    def __init__(self, graph):
        super().__init__(name='IOS_equal_neigbor_weight', graph=graph)
        node_size = graph.number_of_nodes()
        self.W = torch.eye(node_size, dtype=FEATURE_TYPE)
        max_degree = -1
        for i in range(node_size):
            if self.graph.neighbor_sizes[i] > max_degree:
                max_degree = self.graph.neighbor_sizes[i] + 1
        for i in range(node_size):
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i):
                    continue
                self.W[i][j] = 1 / max_degree

                self.W[i][i] -= self.W[i][j]

    def run(self, local_models, node):
        remain_models = local_models[self.graph.neighbors[node]]
        remain_weight = self.W[node][self.graph.neighbors[node]]
        for _ in range(self.graph.byzantine_sizes[node]):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean += self.W[node][node]*local_models[node]
            mean /= remain_weight.sum() + self.W[node][node]

            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res += self.W[node][node]*local_models[node]
        res /= remain_weight.sum() + self.W[node][node]
















        return res


class D_brute(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super().__init__(name='Brute', graph=graph)
    def run(self, local_models, node):
        local_model = local_models[node]
        agg = brute(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg


class D_bulyan(DecentralizedAggregation):
    def __init__(self, graph):
        self.byzantine_sizes = graph.byzantine_sizes
        super().__init__(name='Bulyan', graph=graph)

    def run(self, local_models, node):
        local_model = local_models[node]
        agg = bulyan(local_model, byzantine_size=self.byzantine_sizes[node])
        return agg


class D_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold=10):
        super().__init__(name=f'CC_tau={threshold}', graph=graph)
        self.memory = None
        self.threshold = threshold

    def run(self, local_models, node):
        if self.memory == None:
            self.memory = torch.zeros_like(local_models)

        diff = torch.zeros_like(self.memory[node])
        for n in self.graph.neighbors[node] + [node]:
            model = local_models[n]
            norm = (model - self.memory[node]).norm()
            if norm > self.threshold:
                diff += self.threshold * (model - self.memory[node]) / norm
            else:
                diff += model - self.memory[node]
        diff /= (self.graph.neighbor_sizes[node] + 1)
        self.memory[node] = self.memory[node] + diff
        return self.memory[node]


class D_self_centered_clipping(DecentralizedAggregation):
    def __init__(self, graph, threshold_selection='estimation', threshold=10):
        if threshold_selection == 'estimation':
            name = 'SCClip'
        elif threshold_selection == 'true':
            name = 'SCClip_T'
        elif threshold_selection == 'parameter':
            name = f'SCClip_tau={threshold}'
        else:
            raise ValueError('invalid threshold setting')
        super().__init__(name=name, graph=graph)
        self.W = MH_rule(graph)
        self.threshold = threshold
        self.threshold_selection = threshold_selection

    def get_threshold_estimate(self, local_models, node):

        local_model = local_models[node]
        node_size = local_models.size(0)
        norm_list = torch.tensor([
            -(local_models[n]-local_model).norm()
            if n in self.graph.neighbors[node] and n != node else 1
            for n in range(node_size)
        ])

        honest_size = self.graph.honest_sizes[node]
        _, bottom_index = norm_list.topk(k=honest_size)
        top_index = [
            n for n in self.graph.neighbors[node]
            if n not in bottom_index and n != node
        ]
        weighted_avg_norm = sum([
            self.W[node][n]*norm_list[n] for n in bottom_index
        ])
        cum_weight = sum([
            self.W[node][n] for n in top_index
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)

    def get_true_threshold(self, local_models, node):

        local_model = local_models[node]

        weighted_avg_norm = sum([
            self.W[node][n]*(local_models[n]-local_model).norm()**2
            for n in self.graph.honest_neighbors[node]
        ])
        cum_weight = sum([
            self.W[node][n] for n in self.graph.byzantine_neighbors[node]
        ])
        return torch.sqrt(weighted_avg_norm/cum_weight)

    def run(self, local_models, node):
        if self.threshold_selection == 'estimation':
            threshold = self.get_threshold(local_models, node)
        elif self.threshold_selection == 'true':
            threshold = self.get_true_threshold(local_models, node)
        elif self.threshold_selection == 'parameter':
            threshold = self.threshold
        else:
            raise ValueError('invalid threshold setting')
        local_model = local_models[node]
        cum_diff = torch.zeros_like(local_model)
        for n in self.graph.neighbors[node]:
            model = local_models[n]
            diff = model - local_model
            norm = diff.norm()
            weight = self.W[node][n]
            if norm > threshold:
                cum_diff += weight * threshold * diff / norm
            else:
                cum_diff += weight * diff
        return local_model + cum_diff


class C_HSM_FedAvg(CentraliedAggregation):
    def __init__(self, honest_nodes, byzantine_nodes,
                 rho=0.95, beta=0.9, k=2.5,
                 use_residual_consistency=True,
                 use_cluster_ref=True,
                 enable_logging=True, log_interval=100, eps=1e-12, log_file=None):
        super().__init__(name='HSM-Lite', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        self.rho = rho
        self.beta = beta
        self.k = k
        self.use_residual_consistency = use_residual_consistency
        self.use_cluster_ref = use_cluster_ref
        self.use_robust_normalization = True  # always use robust z; legacy branch unused
        self.rho_s = 0.9
        self.lam_sus = 0.5
        self.enable_logging = enable_logging
        self.log_interval = log_interval
        self.eps = eps
        self.log_file = log_file
        self.server_momentum = None
        self.client_delta_ema = {}
        self.client_raw_norm_ema = {}
        self.client_score_ema = {}
        self.client_suspicion_ema = {}
        self.g_ref_prev = None
        self.iteration_count = 0
        self.round_count = 0
        self.current_accuracy = None
        self.last_agg = None
        self.detection_recall_history = []
        self.detection_precision_history = []
    def set_accuracy(self, accuracy):
        self.current_accuracy = accuracy
    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        if norm1 < self.eps or norm2 < self.eps:
            return torch.tensor(0.0, dtype=vec1.dtype, device=vec1.device)
        return torch.dot(vec1, vec2) / (norm1 * norm2 + self.eps)
    def run(self, messages, client_ids=None):
        """
        HSM-Lite v2: Fixed version with:
        1. Median+MAD robust clipping
        2. Coordinate-wise median as robust reference
        3. Robust normalization (removes need for gamma/lam/T)
        4. Proper order: compute scores BEFORE updating EMA (fixes info leakage)
        """
        N, D = messages.shape
        device, dtype = messages.device, messages.dtype
        if client_ids is None:
            client_ids = list(range(N))
        else:
            if len(client_ids) != N:
                raise ValueError(f"client_ids length ({len(client_ids)}) must match messages size ({N})")
            client_ids = [int(x) for x in client_ids]
        if self.server_momentum is None:
            self.server_momentum = torch.zeros(D, device=device, dtype=dtype)
        # ---- (1) Robust clipping by median+MAD over norms ----
        # Record raw norms BEFORE clipping (for s calculation)
        raw_norms = torch.norm(messages, dim=1) + self.eps
        norms = raw_norms.clone()
        med = torch.median(norms)
        mad = torch.median(torch.abs(norms - med)) + self.eps
        clip = med + self.k * mad
        scales = torch.clamp(clip / norms, max=1.0)
        deltas = messages * scales[:, None]
        # Initialize new clients (AFTER clipping, use clipped deltas)
        # Compute g_ref first (needed for residual initialization)
        g_ref = torch.median(deltas, dim=0).values
        new_clients = []
        init_raw_norm = torch.median(raw_norms).clone().detach()
        for i, cid in enumerate(client_ids):
            if cid not in self.client_delta_ema:
                self.client_delta_ema[cid] = deltas[i].detach().clone()
                self.client_raw_norm_ema[cid] = raw_norms[i].detach().clone()
                new_clients.append(cid)
            if cid not in self.client_raw_norm_ema:
                self.client_raw_norm_ema[cid] = init_raw_norm.clone().detach()
        # ---- (2) Robust reference: coordinate-wise median (already computed above) ----
        g_ref_norm = torch.norm(g_ref) + self.eps
        g_ref_dir = g_ref / g_ref_norm
        # ---- (3) Compute a_i, h_i, s_i, c_i using OLD history (before EMA update) ----
        a_vec = torch.empty(N, device=device, dtype=dtype)
        h_vec = torch.empty(N, device=device, dtype=dtype)
        s_vec = torch.empty(N, device=device, dtype=dtype)
        c_vec = torch.empty(N, device=device, dtype=dtype) # Contribution consistency
        for i, cid in enumerate(client_ids):
            di = deltas[i]
            di_norm = torch.norm(di) + self.eps
            # a_i: align to robust reference
            a_vec[i] = torch.dot(di / di_norm, g_ref_dir)
            # h_i: history consistency (residual-based or direct delta-based)
            if self.use_residual_consistency:
                # Residual-based: more robust to MSA
                # Fixed: Use consistent reference frame to avoid drift
                g_ref_prev = self.g_ref_prev if self.g_ref_prev is not None else g_ref
                g_ref_prev = g_ref_prev.detach() # Ensure detached
                # Current residual (relative to current g_ref)
                resid_i = di - g_ref
                resid_i_norm = torch.norm(resid_i)
                # Historical residual (relative to previous g_ref)
                # Always store delta_ema, residualize on-the-fly
                delta_ema_old = self.client_delta_ema[cid]
                resid_ema_old = delta_ema_old - g_ref_prev
                resid_ema_old_norm = torch.norm(resid_ema_old)
                # Fixed: Adaptive threshold (relative to update scale, not absolute)
                # Use relative scale: residual < 1e-3 * max(di_norm, g_ref_norm)
                threshold = 1e-3 * torch.maximum(di_norm, g_ref_norm)
                if resid_i_norm < threshold or resid_ema_old_norm < threshold:
                    # Residual too small relative to update scale: treat as consistent
                    h_vec[i] = 1.0
                else:
                    # Normal case: compute cosine similarity of residuals
                    h_vec[i] = torch.dot(resid_i / (resid_i_norm + self.eps), resid_ema_old / (resid_ema_old_norm + self.eps))
            else:
                # Direct delta-based (original method)
                hi_old = self.client_delta_ema[cid]
                hi_old_norm = torch.norm(hi_old) + self.eps
                h_vec[i] = torch.dot(di / di_norm, hi_old / hi_old_norm)
            # s_i: scale deviation vs own raw norm EMA (use raw norm for more sensitive detection)
            # Use raw norm instead of clipped norm to detect scale changes more accurately
            raw_norm_i = raw_norms[i]
            r_i_old_raw = self.client_raw_norm_ema[cid]
            s_vec[i] = torch.abs(torch.log((raw_norm_i + self.eps) / (r_i_old_raw + self.eps)))
            # c_i: contribution consistency with server optimization trajectory
            # Use server_momentum as reference (or last_agg if momentum is too small)
            if self.server_momentum is not None:
                momentum_norm = torch.norm(self.server_momentum)
                if momentum_norm > 1e-6:
                    # Use server momentum as reference
                    c_vec[i] = torch.dot(di / di_norm, self.server_momentum / (momentum_norm + self.eps))
                elif self.last_agg is not None:
                    # Fallback to last aggregation if momentum is too small
                    last_agg_norm = torch.norm(self.last_agg) + self.eps
                    c_vec[i] = torch.dot(di / di_norm, self.last_agg / last_agg_norm)
                else:
                    # First round: use g_ref as fallback
                    c_vec[i] = a_vec[i] # Same as alignment to g_ref
            else:
                # No momentum yet: use g_ref
                c_vec[i] = a_vec[i] # Same as alignment to g_ref
        # ---- (4) Compute scores: robust normalization with Non-IID improvements ----
        # Initialize penalty vectors (for scaling attack defense)
        clip_penalty_vec = torch.zeros(N, device=device, dtype=dtype) # Default: no penalty
        z_ln_penalty = torch.zeros(N, device=device, dtype=dtype) # Default: no penalty (symmetric norm outlier)
        if self.use_robust_normalization:
            # Robust normalization: z(x) = (x - median(x)) / (MAD(x) + eps)
            # Fixed: Adaptive MAD clamp to avoid numerical explosion (scale-relative)
            def robust_z(x):
                med = torch.median(x)
                mad = torch.median(torch.abs(x - med))
                # Adaptive clamp: min = 5% of median absolute value (or 1e-3 if median is very small)
                median_abs = torch.median(torch.abs(x)) + self.eps
                mad_min = torch.maximum(0.05 * median_abs, torch.tensor(1e-3, device=x.device, dtype=x.dtype))
                mad = torch.maximum(mad, mad_min) + self.eps # Use torch.maximum for compatibility
                return (x - med) / mad
            # Improvement 2: Relative alignment (Non-IID friendly)
            # Use relative alignment: a_rel = a - median(a), h_rel = h - median(h), c_rel = c - median(c)
            a_med = torch.median(a_vec)
            h_med = torch.median(h_vec)
            c_med = torch.median(c_vec)
            a_rel = a_vec - a_med
            h_rel = h_vec - h_med
            c_rel = c_vec - c_med
            z_a = robust_z(a_rel)
            z_h = robust_z(h_rel)
            z_c = robust_z(c_rel) # Contribution consistency (relative)
            # Optional: clip z_h since it's cosine similarity (concentrated range)
            z_h = torch.clamp(z_h, -3.0, 3.0)
            # Critical fix: Only penalize negative z_c (clients that go against server momentum)
            # Don't reward positive z_c (clients that align with server momentum)
            # This prevents misclassifying honest Non-IID clients whose gradients naturally diverge
            z_c_penalty = torch.clamp(-z_c, min=0.0) # Only negative z_c (against momentum) contributes to penalty
            # Fixed: log1p transform for s_vec to reduce right-tail skew
            s_t = torch.log1p(s_vec) # log1p(s) = log(1+s), more stable than log
            z_s = robust_z(s_t)
            # Critical fix: Only penalize large s, don't reward small s
            # This prevents attackers from gaining high scores by maintaining stability
            z_s_penalty = torch.clamp(z_s, min=0.0) # Only positive z_s (large s) contributes to penalty
            # Scaling attack defense: Penalize clients that are frequently clipped (large norm outliers)
            # clip_penalty = 1 - scale (scale越小，penalty越大，表示被clip得越多)
            # Note: This only catches "amplification" attacks, not "shrinkage" attacks
            clip_penalty_vec = 1.0 - scales # scale越小，penalty越大
            # Relative centering for Non-IID friendliness
            clip_penalty_med = torch.median(clip_penalty_vec)
            clip_penalty_rel = clip_penalty_vec - clip_penalty_med
            z_clip_penalty = robust_z(clip_penalty_rel)
            # Only penalize positive z (clients that are clipped more than median)
            z_clip_penalty = torch.clamp(z_clip_penalty, min=0.0) # Only positive contributes to penalty
            # Symmetric scaling defense: Penalize raw norm outliers (both amplification and shrinkage)
            # Use log1p for better numerical stability and symmetry
            ln = torch.log1p(raw_norms) # ln = log(1 + raw_norm)
            ln_med = torch.median(ln)
            ln_rel = ln - ln_med # Relative to median
            z_ln = robust_z(ln_rel)
            # Symmetric penalty: penalize both too large and too small (|z| > 2)
            # Only penalize outliers beyond threshold (constant 2.0, not tunable)
            z_ln_penalty = torch.clamp(torch.abs(z_ln) - 2.0, min=0.0)
            # Improvement 4: Anti-mimic penalty for "too perfect" alignment
            # Use z-space (already normalized) instead of raw values + quantiles
            # Penalize clients with extremely high z_a, high z_h, and extremely low z_s simultaneously
            penalty = torch.zeros(N, device=device, dtype=dtype)
            for i in range(N):
                # If client has "too perfect" profile in z-space (z_a > 2, z_h > 2, z_s < -2)
                # This is more stable than quantile-based and consistent with robust normalization
                if (z_a[i] > 2.0 and z_h[i] > 2.0 and z_s[i] < -2.0):
                    penalty[i] = 0.5 # Reduce score by 0.5 (moderate penalty)
            # Compute raw scores (before clipping) for MAD calculation
            # Use z_s_penalty (only penalize large s, don't reward small s)
            # Add contribution consistency penalty (0.5 weight): only penalize clients going against momentum
            # Add scaling attack penalties:
            #   - z_clip_penalty (0.5 weight): penalize amplification (frequently clipped clients)
            #   - z_ln_penalty (0.5 weight): symmetric penalty for both amplification and shrinkage outliers
            scores_raw = z_a + z_h - 0.5 * z_c_penalty - z_s_penalty - 0.5 * z_clip_penalty - 0.5 * z_ln_penalty - penalty
            # Improvement 1: Use raw_score for MAD/threshold calculation (before clipping)
            # This avoids distortion from clipped values
            # Improvement 3: Score EMA smoothing (Non-IID stability)
            scores_ema = torch.empty(N, device=device, dtype=dtype)
            for i, cid in enumerate(client_ids):
                if cid not in self.client_score_ema:
                    self.client_score_ema[cid] = scores_raw[i].detach().clone()
                else:
                    # EMA with rho (same as other EMA updates)
                    self.client_score_ema[cid] = (self.rho * self.client_score_ema[cid] + (1.0 - self.rho) * scores_raw[i]).detach()
                scores_ema[i] = self.client_score_ema[cid]
            # Use EMA scores for final weighting (more stable in Non-IID)
            scores_for_softmax = scores_ema
            # Fixed: clip scores before softmax to avoid one-hot distribution
            scores_for_softmax = torch.clamp(scores_for_softmax, -5.0, 5.0)
            # Use fixed temperature 2.0 for smoother distribution
            w = torch.softmax(scores_for_softmax / 2.0, dim=0)
            # Keep raw scores for logging and detection (use raw, not EMA, for detection)
            scores = scores_raw
            scores_ema_for_log = scores_ema # EMA scores for logging
        else:
            # Legacy method (for backward compatibility)
            scores = a_vec + self.gamma * h_vec - self.lam * s_vec
            w = torch.softmax(scores / self.T, dim=0)
            scores_ema_for_log = scores # No EMA in legacy mode
        # ---- (5) Aggregate and update server momentum ----
        agg = (w[:, None] * deltas).sum(dim=0).detach()
        self.server_momentum = (self.beta * self.server_momentum + (1.0 - self.beta) * agg).detach()
        self.last_agg = agg
        self.round_count += 1
        # ---- (6) Update client histories (AFTER computing scores) ----
        for i, cid in enumerate(client_ids):
            di = deltas[i]
            raw_norm_i = raw_norms[i]
            hi_old = self.client_delta_ema[cid]
            r_i_old_raw = self.client_raw_norm_ema[cid]
            self.client_delta_ema[cid] = (self.rho * hi_old + (1.0 - self.rho) * di).detach()
            self.client_raw_norm_ema[cid] = (self.rho * r_i_old_raw + (1.0 - self.rho) * raw_norm_i).detach()
        # Update g_ref_prev for next round's residual consistency calculation
        self.g_ref_prev = g_ref.detach().clone()
        # ---- (7) Logging (with correct a/h/s values) ----
        if self.enable_logging and self.iteration_count % self.log_interval == 0:
            norms_clipped = torch.norm(deltas, dim=1)
            self._log_statistics(
                a_vec,
                h_vec,
                s_vec,
                c_vec,
                scores,
                scores_ema_for_log,
                w,
                norms_clipped,
                agg,
                client_ids,
                new_clients,
                self_alignments=None, # Not needed, we have h_vec
                accuracy=self.current_accuracy,
                scales=scales,
                clip_penalty_vec=clip_penalty_vec,
                z_ln_penalty=z_ln_penalty,
            )
        self.iteration_count += 1
        return agg
    def _log_statistics(self, a_vec, h_vec, s_vec, c_vec, scores, scores_ema, weights_tensor, norm_deltas, aggregated_update, client_ids, new_clients, self_alignments=None, accuracy=None, scales=None, clip_penalty_vec=None, z_ln_penalty=None):
        """Simplified logging for HSM-Lite v2 with correct a/h/s/c values and scaling defense info."""
        from ByrdLab.library.tool import log
        import os
        num_nodes = a_vec.size(0)
        # Detection metrics using adaptive threshold (MAD-based, no hard quantile)
        recall_adaptive = 0.0
        precision_adaptive = 0.0
        recall_bottomk = 0.0
        precision_bottomk = 0.0
        quantile_results = []
        num_attackers = 0
        current_byzantine_indices = []
        for i, cid in enumerate(client_ids):
            if cid in self.byzantine_nodes:
                current_byzantine_indices.append(i)
        num_attackers = len(current_byzantine_indices)
        if num_attackers > 0:
            weights_cpu = weights_tensor.detach().cpu()
            scores_cpu = scores.detach().cpu()
            ground_truth_set = set(current_byzantine_indices)
            # Method 1: Adaptive threshold based on MAD (no hard quantile)
            # Predict: score < median(score) - 2 * MAD(score) (constant 2, not a tunable hyperparameter)
            score_med = torch.median(scores_cpu)
            score_mad = torch.median(torch.abs(scores_cpu - score_med))
            # Adaptive MAD clamp (relative to score scale, consistent with robust_z)
            score_med_abs = torch.median(torch.abs(scores_cpu)) + 1e-12
            score_mad_min = torch.maximum(0.05 * score_med_abs, torch.tensor(1e-3, device=scores_cpu.device, dtype=scores_cpu.dtype))
            score_mad = torch.maximum(score_mad, score_mad_min) # Use relative scale
            adaptive_threshold = score_med - 2.0 * score_mad # Constant 2.0, not tunable
            predicted_adaptive = set(torch.where(scores_cpu <= adaptive_threshold)[0].tolist())
            correctly_detected_adaptive = len(predicted_adaptive.intersection(ground_truth_set))
            precision_adaptive = correctly_detected_adaptive / len(predicted_adaptive) if len(predicted_adaptive) > 0 else 0.0
            recall_adaptive = correctly_detected_adaptive / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
            # Method 2: Traditional bottom-k for comparison
            _, lowest_weight_indices = torch.topk(weights_cpu, k=num_attackers, largest=False)
            predicted_bottomk = set(lowest_weight_indices.tolist())
            correctly_detected_bottomk = len(predicted_bottomk.intersection(ground_truth_set))
            precision_bottomk = correctly_detected_bottomk / len(predicted_bottomk) if len(predicted_bottomk) > 0 else 0.0
            recall_bottomk = correctly_detected_bottomk / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
            # Method 3: Report multiple quantiles for analysis (not used for main metric)
            quantiles_to_report = [0.1, 0.2, 0.3]
            quantile_results = []
            for q in quantiles_to_report:
                q_threshold = torch.quantile(scores_cpu, q)
                predicted_q = set(torch.where(scores_cpu <= q_threshold)[0].tolist())
                correct_q = len(predicted_q.intersection(ground_truth_set))
                recall_q = correct_q / len(ground_truth_set) if len(ground_truth_set) > 0 else 0.0
                precision_q = correct_q / len(predicted_q) if len(predicted_q) > 0 else 0.0
                quantile_results.append((q, recall_q, precision_q))
            if self.enable_logging and self.iteration_count % self.log_interval == 0:
                self.detection_recall_history.append((self.iteration_count, recall_adaptive))
                self.detection_precision_history.append((self.iteration_count, precision_adaptive))
            log(f" [Defense Metric] Adaptive (MAD-based): Recall={recall_adaptive:.2%}, Precision={precision_adaptive:.2%}")
            log(f" [Defense Metric] Bottom-k (k={num_attackers}): Recall={recall_bottomk:.2%}, Precision={precision_bottomk:.2%}")
            log(f" [Defense Metric] Quantiles (for analysis): " + ", ".join([f"q={q:.1f}: R={r:.2%}/P={p:.2%}" for q, r, p in quantile_results]))
        else:
            log(f" [Defense Metric] No attackers in this round.")
        # Statistics summary
        def stats_summary(tensor, name):
            if tensor.numel() == 0:
                return f"{name}: N/A (empty)"
            mean_val = tensor.mean().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            std_val = tensor.std().item()
            median_val = torch.median(tensor).item()
            return f"{name}: mean={mean_val:.4f}, min={min_val:.4f}, max={max_val:.4f}, std={std_val:.4f}, median={median_val:.4f}"
        momentum_norm = torch.norm(self.server_momentum).item() if self.server_momentum is not None else 0.0
        aggregated_norm = torch.norm(aggregated_update).item()
        log(f"[HSM-Lite v2] Iteration {self.iteration_count} Statistics:")
        log(f" Round: {self.round_count}")
        log(f" Active clients: {len(client_ids)}, New clients: {len(new_clients)}")
        log(f" Server Momentum Norm: {momentum_norm:.6e}, Aggregated Update Norm: {aggregated_norm:.6e}")
        if accuracy is not None:
            log(f" Accuracy: {accuracy:.4f}")
        if self.use_robust_normalization:
            log(f" Method: Robust Normalization (z(a_rel) + z(h_rel) - 0.5*z_c_penalty - z_s_penalty - 0.5*z_clip_penalty - 0.5*z_ln_penalty - penalty) + EMA smoothing")
        else:
            log(" Method: Weighted Sum (a + gamma*h - lambda*s)")
        log(f" Raw Scores (for detection): {stats_summary(scores, 'raw_score')}")
        if self.use_robust_normalization and not torch.allclose(scores, scores_ema, atol=1e-6):
            log(f" EMA Scores (for weighting): {stats_summary(scores_ema, 'ema_score')}")
        log(f" Alignment to Robust Ref (a, relative): {stats_summary(a_vec, 'a')}")
        log(f" Self-Consistency (h, relative): {stats_summary(h_vec, 'h')}")
        log(f" Contribution Consistency (c, relative): {stats_summary(c_vec, 'c')}")
        log(f" Scale Deviations (s): {stats_summary(s_vec, 's')}")
        if clip_penalty_vec is not None:
            log(f" Clip Penalty (1-scale, relative): {stats_summary(clip_penalty_vec, 'clip_penalty')}")
        if z_ln_penalty is not None:
            log(f" Norm Outlier Penalty (symmetric, |z|>2): {stats_summary(z_ln_penalty, 'z_ln_penalty')}")
        if scales is not None:
            log(f" Clip Scales: {stats_summary(scales, 'scale')}")
        log(f" Soft Weights: {stats_summary(weights_tensor, 'weight')}, Sum: {weights_tensor.sum().item():.4f}")
        log(f" Update Norms: {stats_summary(norm_deltas, 'norm')}")
        log(" [Per-Client Values]:")
        if scales is not None:
            log(" Client_ID | Score | Align (a) | Self (h) | Contrib (c) | Scale (s) | Clip | Weight | Norm | Attacker")
            log(" " + "-" * 120)
        else:
            log(" Client_ID | Score | Align (a) | Self (h) | Contrib (c) | Scale (s) | Weight | Norm | Attacker")
            log(" " + "-" * 110)
        a_cpu = a_vec.cpu() if isinstance(a_vec, torch.Tensor) else a_vec
        h_cpu = h_vec.cpu() if isinstance(h_vec, torch.Tensor) else h_vec
        c_cpu = c_vec.cpu() if isinstance(c_vec, torch.Tensor) else c_vec
        s_cpu = s_vec.cpu() if isinstance(s_vec, torch.Tensor) else s_vec
        scores_cpu = scores.cpu() if isinstance(scores, torch.Tensor) else scores
        weights_cpu = weights_tensor.cpu() if isinstance(weights_tensor, torch.Tensor) else weights_tensor
        norms_cpu = norm_deltas.cpu() if isinstance(norm_deltas, torch.Tensor) else norm_deltas
        scales_cpu = scales.cpu() if scales is not None and isinstance(scales, torch.Tensor) else None
        for i, cid in enumerate(client_ids):
            score_val = scores_cpu[i].item()
            a_val = a_cpu[i].item()
            h_val = h_cpu[i].item()
            c_val = c_cpu[i].item()
            s_val = s_cpu[i].item()
            weight_val = weights_cpu[i].item()
            norm_val = norms_cpu[i].item()
            is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
            if scales_cpu is not None:
                scale_val = scales_cpu[i].item()
                log(f" {cid:9d} | {score_val:5.3f} | {a_val:9.4f} | {h_val:8.4f} | {c_val:10.4f} | {s_val:9.4f} | {scale_val:4.3f} | {weight_val:6.4f} | {norm_val:6.4f} | {is_attacker}")
            else:
                log(f" {cid:9d} | {score_val:5.3f} | {a_val:9.4f} | {h_val:8.4f} | {c_val:10.4f} | {s_val:9.4f} | {weight_val:6.4f} | {norm_val:6.4f} | {is_attacker}")
        log("")
        # File logging
        if hasattr(self, 'log_file') and self.log_file is not None:
            try:
                log_dir = os.path.dirname(self.log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
                log_content = f"[HSM-Lite v2] Iteration {self.iteration_count} Statistics:\n"
                log_content += f" Round: {self.round_count}\n"
                # Always log accuracy (write N/A if not available)
                if accuracy is not None:
                    log_content += f" Accuracy: {accuracy:.4f}\n"
                else:
                    log_content += f" Accuracy: N/A\n"
                if self.use_robust_normalization:
                    log_content += f" Method: Robust Normalization (z(a_rel) + z(h_rel) - 0.5*z_c_penalty - z_s_penalty - 0.5*z_clip_penalty - 0.5*z_ln_penalty - penalty) + EMA smoothing\n"
                else:
                    log_content += f" Method: Weighted Sum\n"
                log_content += f" Raw Scores mean: {scores.mean().item():.4f}, std: {scores.std().item():.4f}\n"
                if self.use_robust_normalization and not torch.allclose(scores, scores_ema, atol=1e-6):
                    log_content += f" EMA Scores mean: {scores_ema.mean().item():.4f}, std: {scores_ema.std().item():.4f}\n"
                log_content += f" Alignment (a, relative) mean: {a_vec.mean().item():.4f}, std: {a_vec.std().item():.4f}\n"
                log_content += f" Self-Consistency (h, relative) mean: {h_vec.mean().item():.4f}, std: {h_vec.std().item():.4f}\n"
                log_content += f" Contribution Consistency (c, relative) mean: {c_vec.mean().item():.4f}, std: {c_vec.std().item():.4f}\n"
                log_content += f" Scale Deviations (s) mean: {s_vec.mean().item():.4f}, std: {s_vec.std().item():.4f}\n"
                if clip_penalty_vec is not None:
                    log_content += f" Clip Penalty (1-scale, relative) mean: {clip_penalty_vec.mean().item():.4f}, std: {clip_penalty_vec.std().item():.4f}\n"
                if z_ln_penalty is not None:
                    log_content += f" Norm Outlier Penalty (symmetric, |z|>2) mean: {z_ln_penalty.mean().item():.4f}, std: {z_ln_penalty.std().item():.4f}\n"
                if scales is not None:
                    log_content += f" Clip Scales mean: {scales.mean().item():.4f}, std: {scales.std().item():.4f}\n"
                log_content += f" Weights mean: {weights_tensor.mean().item():.4f}, std: {weights_tensor.std().item():.4f}\n"
                log_content += f" Update Norms mean: {norm_deltas.mean().item():.4f}, std: {norm_deltas.std().item():.4f}\n"
                log_content += f"\n [Per-Client Values]:\n"
                if scales is not None:
                    log_content += f" Client_ID | Score | Align (a) | Self (h) | Contrib (c) | Scale (s) | Clip | Weight | Norm | Attacker\n"
                    log_content += f" " + "-" * 120 + "\n"
                else:
                    log_content += f" Client_ID | Score | Align (a) | Self (h) | Contrib (c) | Scale (s) | Weight | Norm | Attacker\n"
                    log_content += f" " + "-" * 110 + "\n"
                for i, cid in enumerate(client_ids):
                    score_val = scores_cpu[i].item()
                    a_val = a_cpu[i].item()
                    h_val = h_cpu[i].item()
                    c_val = c_cpu[i].item()
                    s_val = s_cpu[i].item()
                    weight_val = weights_cpu[i].item()
                    norm_val = norms_cpu[i].item()
                    is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                    if scales_cpu is not None:
                        scale_val = scales_cpu[i].item()
                        log_content += f" {cid:9d} | {score_val:5.3f} | {a_val:9.4f} | {h_val:8.4f} | {c_val:10.4f} | {s_val:9.4f} | {scale_val:4.3f} | {weight_val:6.4f} | {norm_val:6.4f} | {is_attacker}\n"
                    else:
                        log_content += f" {cid:9d} | {score_val:5.3f} | {a_val:9.4f} | {h_val:8.4f} | {c_val:10.4f} | {s_val:9.4f} | {weight_val:6.4f} | {norm_val:6.4f} | {is_attacker}\n"
                if num_attackers > 0:
                    log_content += f" [Defense Metric] Adaptive (MAD-based): Recall={recall_adaptive:.2%}, Precision={precision_adaptive:.2%}\n"
                    log_content += f" [Defense Metric] Bottom-k (k={num_attackers}): Recall={recall_bottomk:.2%}, Precision={precision_bottomk:.2%}\n"
                else:
                    log_content += f" [Defense Metric] No attackers in this round.\n"
                log_content += "\n"
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_content)
            except Exception as e:
                pass
    def get_detection_history(self):
        return self.detection_recall_history.copy(), self.detection_precision_history.copy()


class C_HRAC(CentraliedAggregation):
    """
    History-Residual Adaptive Clipping (HRAC)
    Only suppresses per-round attack impact, no long-term weight reduction, no labeling.
    Each client only compares with its own history (Non-IID friendly).
    """
    def __init__(self, honest_nodes, byzantine_nodes,
                 rho_b=0.98, rho_mu=0.95, rho_nu=0.95,
                 rho_g=0.95, global_scale_floor=1e-3,
                 c=2.5, c_g=3.0,
                 enable_logging=True, log_interval=100, eps=1e-12, log_file=None,
                 nu_penalty_start_iter=50, nu_penalty_alpha=5.0,
                 rho_nu_penalty=0.90,
                 verbose_nu_log_interval=50,
                 enable_global_cap=True,
                 enable_per_client_residual_clip=True,
                 enable_nu_weighting=True,
                 enable_post_residual_b_cap=True,
                 enable_invariant_checks=False,
                 invariant_check_mode="log_and_raise",
                 invariant_check_tol=1e-8,
                 invariant_log_file=None,
                 invariant_log_interval=1):
        super().__init__(name='HRAC', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        
        self.rho_b = rho_b
        self.rho_mu = rho_mu
        self.rho_nu = rho_nu  # ν EMA: ν <- ρν + (1-ρ)·d
        self.rho_g = rho_g    # Global median-scale EMA: s <- ρg s + (1-ρg)·median_norm
        self.c = c          # Per-client residual clip multiplier
        self.c_g = c_g      # Predictable global EMA-median cap multiplier
        self.global_scale_floor = global_scale_floor
        self.enable_global_cap = enable_global_cap
        self.enable_per_client_residual_clip = enable_per_client_residual_clip  # False: r_bar=r, skip _clip_by_norm
        self.enable_nu_weighting = enable_nu_weighting
        self.enable_post_residual_b_cap = enable_post_residual_b_cap  # False: no min(1,B/‖Δ̃‖) after b+r̄
        self.eps = eps
        
        self.enable_logging = enable_logging
        self.log_interval = log_interval
        self.log_file = log_file
        self.verbose_nu_log_interval = verbose_nu_log_interval  # >0: every N iters print d/nu summary (e.g. 1=every iter)
        self.enable_invariant_checks = enable_invariant_checks
        self.invariant_check_mode = invariant_check_mode
        self.invariant_check_tol = invariant_check_tol
        self.invariant_log_file = invariant_log_file
        self.invariant_log_interval = max(1, int(invariant_log_interval))
        
        # Stats (τ, μ, ν, weights) from normalized vectors (norm = median_norm); aggregation uses clip/original
        # Nu-based penalty parameters (only for weighting; aggregation still uses original updates)
        self.nu_penalty_start_iter = nu_penalty_start_iter
        self.nu_penalty_alpha = nu_penalty_alpha
        self.rho_nu_penalty = rho_nu_penalty
        self.nu_weight_max = 0.30
        
        # Per-client histories (initialized on first run)
        self.b = {}           # EMA mean (bias) for aggregation path (clip/original)
        self.b_norm = {}     # EMA mean for stats path (normalized-vector only)
        self.mu = {}          # from normalized path → τ = c*μ
        self.nu = {}          # from normalized path → weights
        self.nu_penalty_ema = {}  # smoothed excess-nu penalty used for weights
        self.r_prev = {}      # previous clipped residual (normalized path)
        
        self.one = None       # Reusable tensor for clip (device/dtype adaptive)
        self.mu_min = 1e-3    # Minimum for mu/nu to prevent collapse
        self.nu_min = 1e-3
        self.median_norm_prev = None  # Previous median norm for ratio logging
        self.global_scale_ema = None  # Predictable global clipping scale used before current-round update
        self._last_global_scale_for_cap = None
        self._last_global_scale_next = None
        
        self.iteration_count = -1
        self.current_accuracy = None
        self._hrac_log_buffer = []
        self._invariant_log_buffer = []
        self._invariant_checked_rounds = 0

    def set_accuracy(self, accuracy):
        self.current_accuracy = accuracy

    def flush_log_to_file(self):
        """Write buffered log to file once (call after training ends). No-op if log_file is None or buffer empty."""
        if self.log_file is None or not getattr(self, '_hrac_log_buffer', None) or len(self._hrac_log_buffer) == 0:
            self.flush_invariant_log_to_file()
            return
        import os
        try:
            log_dir = os.path.dirname(self.log_file)
            if log_dir and log_dir.strip():
                os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write("[HRAC] Log generated after training (single write).\n")
                f.write("=" * 80 + "\n\n")
                f.write("\n\n".join(self._hrac_log_buffer))
                f.write("\n")
            self._hrac_log_buffer.clear()
        except Exception:
            pass
        self.flush_invariant_log_to_file()

    def flush_invariant_log_to_file(self):
        """Write buffered invariant-check logs once after training ends."""
        if self.invariant_log_file is None or len(self._invariant_log_buffer) == 0:
            return
        import os
        try:
            log_dir = os.path.dirname(self.invariant_log_file)
            if log_dir and log_dir.strip():
                os.makedirs(log_dir, exist_ok=True)
            with open(self.invariant_log_file, 'w', encoding='utf-8') as f:
                f.write("[HRAC Invariants] Log generated after training (single write).\n")
                f.write(f"[HRAC Invariants] Checked rounds={self._invariant_checked_rounds}\n")
                f.write("=" * 80 + "\n\n")
                f.write("\n\n".join(self._invariant_log_buffer))
                f.write("\n")
            self._invariant_log_buffer.clear()
        except Exception:
            pass

    def _append_invariant_log(self, text):
        if self.invariant_log_file is not None:
            self._invariant_log_buffer.append(text)

    @torch.no_grad()
    def _clip_by_norm(self, r, tau):
        """Clip residual r by threshold tau."""
        device, dtype = r.device, r.dtype
        if self.one is None or self.one.device != device or self.one.dtype != dtype:
            self.one = torch.tensor(1.0, device=device, dtype=dtype)
        n = torch.norm(r) + self.eps
        scale = torch.minimum(self.one, tau / n)
        return r * scale

    def _check_close(self, name, value, rhs, violations):
        violation = abs(float(value) - float(rhs))
        if violation > self.invariant_check_tol:
            violations.append({
                'name': name,
                'kind': 'close',
                'value': float(value),
                'rhs': float(rhs),
                'violation': violation,
            })

    def _check_leq(self, name, value, rhs, violations):
        violation = float(value) - float(rhs)
        if violation > self.invariant_check_tol:
            violations.append({
                'name': name,
                'kind': 'leq',
                'value': float(value),
                'rhs': float(rhs),
                'violation': violation,
            })

    def _handle_invariant_failure(self, context):
        import time
        stamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
        lines = [
            stamp + f"[HRAC Invariants] FAILURE at iter={context['iteration']}",
            stamp + f"  checked_rounds={self._invariant_checked_rounds}",
            stamp + f"  median_norm={context['median_norm']:.8f}, B={context['B']:.8f}",
            stamp + f"  mode={self.invariant_check_mode}, tol={self.invariant_check_tol:.1e}",
        ]
        if context.get('accuracy') is not None:
            lines.insert(2, stamp + f"  accuracy={context['accuracy']:.4f}")
        for item in context['violated_checks']:
            lines.append(
                stamp
                + f"  violate[{item['name']}] kind={item['kind']} "
                + f"value={item['value']:.8e} rhs={item['rhs']:.8e} "
                + f"gap={item['violation']:.8e}"
            )
        lines.append(stamp + "  per-client context:")
        for info in context['per_client_context']:
            lines.append(
                stamp
                + "    "
                + f"cid={info['cid']} attacker={info['is_attacker']} new={info['is_new']} "
                + f"||msg||={info['msg_norm']:.8f} ||clip||={info['msg_clip_norm']:.8f} "
                + f"||processed||={info['processed_norm']:.8f} tau={info['tau']:.8f} "
                + f"mu_before={info['mu_before']:.8f} mu_after={info['mu_after']:.8f} "
                + f"nu_before={info['nu_before']:.8f} nu_after={info['nu_after']:.8f} "
                + f"w={info['weight']:.8f}"
            )
        self._append_invariant_log("\n".join(lines))
        if self.invariant_check_mode == "log_only":
            return
        raise AssertionError(
            f"HRAC invariant violation at iter={context['iteration']} "
            + f"(max_gap={context['max_violation']:.8e}, checks={len(context['violated_checks'])})"
        )

    def _check_invariants(self, context):
        violations = []
        iteration = context['iteration']
        norms = context['norms']
        messages_clipped = context['messages_clipped']
        processed_tensor = context['processed_tensor']
        weights_tensor = context['weights_tensor']
        g_t = context['g_t']
        B = context['B']
        median_norm = context['median_norm']
        use_nu_penalty_this_round = context['use_nu_penalty_this_round']
        client_ids = context['client_ids']
        clipped_norms = torch.norm(messages_clipped, dim=1)
        processed_norms = torch.norm(processed_tensor, dim=1)
        weighted_energy = torch.sum(weights_tensor * (processed_norms ** 2))
        g_sq = torch.norm(g_t) ** 2
        if self.enable_global_cap:
            self._check_leq(f"iter{iteration}.global_cap_positive", -B.item(), 0.0, violations)
            for i, cid in enumerate(client_ids):
                self._check_leq(f"iter{iteration}.global_clip[c{cid}]", clipped_norms[i].item(), B.item(), violations)

        self._check_close(f"iter{iteration}.weight_sum", weights_tensor.sum().item(), 1.0, violations)
        for i, cid in enumerate(client_ids):
            self._check_leq(f"iter{iteration}.weight_nonneg[c{cid}]", -weights_tensor[i].item(), 0.0, violations)
            if use_nu_penalty_this_round:
                self._check_leq(f"iter{iteration}.weight_cap[c{cid}]", weights_tensor[i].item(), self.nu_weight_max, violations)
        self._check_leq(f"iter{iteration}.jensen", g_sq.item(), weighted_energy.item(), violations)
        if self.enable_global_cap and self.enable_post_residual_b_cap:
            for i, cid in enumerate(client_ids):
                self._check_leq(f"iter{iteration}.post_cap[c{cid}]", processed_norms[i].item(), B.item(), violations)

        per_client_context = []
        for entry in context['client_debug']:
            cid = entry['cid']
            if not entry['is_new']:
                self._check_close(f"iter{iteration}.tau_eq[c{cid}]", entry['tau_before'], self.c * entry['mu_before'] + self.eps, violations)
                self._check_close(f"iter{iteration}.clip_identity[c{cid}]", entry['clip_gap'], entry['clip_rhs'], violations)
                self._check_close(f"iter{iteration}.delta_identity[c{cid}]", entry['delta_gap'], entry['clip_rhs'], violations)
                self._check_close(f"iter{iteration}.mu_ema[c{cid}]", entry['mu_after'], entry['mu_expected'], violations)
                self._check_close(f"iter{iteration}.nu_ema[c{cid}]", entry['nu_after'], entry['nu_expected'], violations)
            per_client_context.append({
                'cid': cid,
                'is_attacker': cid in self.byzantine_nodes,
                'is_new': entry['is_new'],
                'msg_norm': entry['msg_norm'],
                'msg_clip_norm': entry['msg_clip_norm'],
                'processed_norm': entry['processed_norm'],
                'tau': entry['tau_before'],
                'mu_before': entry['mu_before'],
                'mu_after': entry['mu_after'],
                'nu_before': entry['nu_before'],
                'nu_after': entry['nu_after'],
                'weight': entry['weight'],
            })

        self._invariant_checked_rounds += 1
        if len(violations) == 0:
            if self.invariant_log_file is not None and self._invariant_checked_rounds % self.invariant_log_interval == 0:
                self._append_invariant_log(
                    f"[HRAC Invariants] iter={iteration} OK "
                    + f"median_norm={median_norm.item():.8f} B={B.item():.8f} "
                    + f"g_norm={torch.norm(g_t).item():.8f}"
                )
            return

        self._handle_invariant_failure({
            'iteration': iteration,
            'accuracy': self.current_accuracy,
            'median_norm': median_norm.item(),
            'B': B.item(),
            'violated_checks': violations,
            'max_violation': max(item['violation'] for item in violations),
            'per_client_context': per_client_context,
        })

    def run(self, messages, client_ids=None):
        """
        messages: (N, D) tensor of raw client updates
        client_ids: optional list of client IDs (length N)
        returns: (D,) aggregated update
        """
        N, D = messages.shape
        device, dtype = messages.device, messages.dtype
        
        if client_ids is None:
            client_ids = list(range(N))
        else:
            client_ids = [int(x) for x in client_ids]
        
        # 先 +1 再聚合/打日志，使 [HRAC] Iteration N 与主循环 iter N 一致
        self.iteration_count += 1
        
        # --- Predictable global robust norm cap ---
        # Current-round median updates the next round's EMA scale; the current cap
        # uses the already stored scale, so it is predictable w.r.t. this round.
        norms = torch.norm(messages, dim=1)  # (N,)
        median_norm = torch.median(norms)
        if self.enable_global_cap:
            floor = torch.tensor(
                max(float(self.global_scale_floor), float(self.eps)),
                device=device,
                dtype=dtype,
            )
            if self.global_scale_ema is None:
                # Warm-start only: after this call, B_t is based on previous
                # rounds through the EMA recursion below.
                self.global_scale_ema = median_norm.detach().clone()
            elif self.global_scale_ema.device != device or self.global_scale_ema.dtype != dtype:
                self.global_scale_ema = self.global_scale_ema.to(device=device, dtype=dtype)
            global_scale_for_cap = torch.maximum(self.global_scale_ema.detach(), floor)
            B = (self.c_g * global_scale_for_cap).clamp_min(self.eps)
        else:
            global_scale_for_cap = median_norm.detach().clone()
            B = norms.max().clamp_min(self.eps)
        self._last_global_scale_for_cap = global_scale_for_cap.detach().clone()
        self._last_median_norm_prev_for_log = self.median_norm_prev  # save for log (before we overwrite at end of round)
        if self.enable_global_cap:
            scale_global = torch.minimum(torch.ones_like(norms), B / (norms + self.eps))
        else:
            scale_global = torch.ones_like(norms)
        messages_clipped = messages * scale_global.unsqueeze(1)  # Clip all updates to global cap

        # 仅当存在 norm < 0.5*median 时启用 lift：把这些 client scale 到 median 用于 τ,μ,ν；否则完全按原始流程用真实值
        norm_small = norms < (0.5 * median_norm)
        has_small_norm = norm_small.any().item()
        scale_lift = median_norm / (norms + self.eps)
        messages_eff = torch.where(norm_small.unsqueeze(1), messages * scale_lift.unsqueeze(1), messages_clipped)
        
        # Initialize self.one if needed (for cap operations)
        if self.one is None or self.one.device != device or self.one.dtype != dtype:
            self.one = torch.tensor(1.0, device=device, dtype=dtype)
        
        processed = []
        r_bar_list = []
        r_bar_norm_list = []
        delta_tilde_norm_list = []
        tau_list = []
        new_cids = set()
        client_debug = []
        
        for i, cid in enumerate(client_ids):
            # 聚合路径始终用未 scale 的 update（messages_clipped）；小 norm 时仅 stats 路径用 scale 后的 messages_eff
            delta_t_i = messages_clipped[i]
            if cid not in self.b:
                new_cids.add(cid)
                # Match the proof invariant: the initial residual scale should
                # not exceed the largest possible distance between two
                # globally capped vectors.
                init_norm = torch.minimum(median_norm.clamp_min(self.mu_min), 2.0 * B)
                self.b[cid] = delta_t_i.detach().clone()
                self.b_norm[cid] = (messages_eff[i] if has_small_norm else delta_t_i).detach().clone()
                self.mu[cid] = init_norm.detach()
                self.nu[cid] = init_norm.detach().clone()
                self.r_prev[cid] = torch.zeros(D, device=device, dtype=dtype)
                processed.append(delta_t_i)
                r_bar_list.append(torch.zeros(D, device=device, dtype=dtype))
                r_bar_norm_list.append(torch.zeros(D, device=device, dtype=dtype))
                delta_tilde_norm_list.append((messages_eff[i] if has_small_norm else delta_t_i).detach().clone())
                tau_list.append((self.c * self.mu[cid] + self.eps).item())
                client_debug.append({
                    'cid': cid,
                    'is_new': True,
                    'msg_norm': norms[i].item(),
                    'msg_clip_norm': torch.norm(delta_t_i).item(),
                    'processed_norm': torch.norm(delta_t_i).item(),
                    'tau_before': (self.c * self.mu[cid] + self.eps).item(),
                    'mu_before': self.mu[cid].item(),
                    'mu_after': self.mu[cid].item(),
                    'nu_before': self.nu[cid].item(),
                    'nu_after': self.nu[cid].item(),
                    'clip_gap': 0.0,
                    'clip_rhs': 0.0,
                    'delta_gap': 0.0,
                    'mu_expected': self.mu[cid].item(),
                    'nu_expected': self.nu[cid].item(),
                    'weight': 0.0,
                })
                continue
            
            tau = self.c * self.mu[cid] + self.eps
            mu_before = self.mu[cid].item()
            nu_before = self.nu[cid].item()
            r = delta_t_i - self.b[cid]
            if self.enable_per_client_residual_clip:
                r_bar = self._clip_by_norm(r, tau)
            else:
                r_bar = r
            delta_tilde = self.b[cid] + r_bar
            delta_tilde_pre_cap = delta_tilde.clone()
            if self.enable_global_cap and self.enable_post_residual_b_cap:
                delta_tilde = delta_tilde * torch.minimum(self.one, B / (torch.norm(delta_tilde) + self.eps))
            processed.append(delta_tilde)
            r_bar_list.append(r_bar)
            if has_small_norm:
                # 存在过小范数：用 scale 到 median 的 effective 向量算 τ,μ,ν
                delta_eff_i = messages_eff[i]
                r_norm = delta_eff_i - self.b_norm[cid]
                if self.enable_per_client_residual_clip:
                    r_bar_norm = self._clip_by_norm(r_norm, tau)
                else:
                    r_bar_norm = r_norm
                delta_tilde_norm = self.b_norm[cid] + r_bar_norm
                if self.enable_global_cap and self.enable_post_residual_b_cap:
                    delta_tilde_norm = delta_tilde_norm * torch.minimum(self.one, B / (torch.norm(delta_tilde_norm) + self.eps))
                r_bar_norm_list.append(r_bar_norm)
                delta_tilde_norm_list.append(delta_tilde_norm)
            else:
                # 无过小范数：完全按原始流程，τ,μ,ν 用真实 r_bar / delta_tilde
                r_bar_norm_list.append(r_bar)
                delta_tilde_norm_list.append(delta_tilde)
            tau_list.append(tau.item())
            client_debug.append({
                'cid': cid,
                'is_new': False,
                'msg_norm': norms[i].item(),
                'msg_clip_norm': torch.norm(delta_t_i).item(),
                'processed_norm': torch.norm(delta_tilde).item(),
                'tau_before': tau.item(),
                'mu_before': mu_before,
                'mu_after': mu_before,
                'nu_before': nu_before,
                'nu_after': nu_before,
                'clip_gap': torch.norm(r_bar - r).item(),
                'clip_rhs': max(torch.norm(r).item() - tau.item(), 0.0),
                'delta_gap': torch.norm(delta_tilde_pre_cap - delta_t_i).item(),
                'mu_expected': mu_before,
                'nu_expected': nu_before,
                'weight': 0.0,
            })
        
        # Aggregate: weighted mean (with nu-based penalty after start_iter)
        # Why low attacker weight can still hurt: (1) 3 attackers × 0.003 ≈ 1% weight but same direction
        # → residual in attack direction every round. (2) When many clients pass the ν>0.9 gate,
        # honest clients can also receive a mild penalty, so relative attacker share can rise. (3) Using (ν-med)^+ for outlier
        # keeps penalizing attackers when mean is inflated.
        processed_tensor = torch.stack(processed, dim=0)  # (N, D)

        # Calculate weights: use ν penalty only when iter > start_iter
        use_nu_penalty_this_round = self.enable_nu_weighting and self.iteration_count > self.nu_penalty_start_iter
        if use_nu_penalty_this_round:
            # Collect nu values for all clients (including new ones)
            nu_values = torch.empty(N, device=device, dtype=dtype)
            for i, cid in enumerate(client_ids):
                if cid in self.nu:
                    nu_val = self.nu[cid].item()
                else:
                    # New client: use median_norm as initial nu (same as initialization)
                    nu_val = median_norm.item()
                nu_values[i] = nu_val
            
            # Only penalize when ν > 0.9; penalty is based on all clients' ν: excess = (ν - mean(ν))^+ so that
            # clients only slightly above mean (e.g. honest with ν≈1.6–2) get mild penalty, real outliers (attackers) get large penalty
            nu_mean = nu_values.mean()
            nu_excess = torch.where(
                nu_values > 0.9,
                torch.clamp(nu_values - nu_mean, min=0.0),
                torch.zeros_like(nu_values),
            )
            penalty_values = torch.empty(N, device=device, dtype=dtype)
            rho_penalty = torch.tensor(self.rho_nu_penalty, device=device, dtype=dtype)
            for i, cid in enumerate(client_ids):
                old_penalty = self.nu_penalty_ema.get(
                    cid,
                    torch.zeros((), device=device, dtype=dtype),
                ).to(device=device, dtype=dtype)
                new_penalty = (
                    rho_penalty * old_penalty
                    + (1.0 - rho_penalty) * nu_excess[i]
                ).detach()
                self.nu_penalty_ema[cid] = new_penalty
                penalty_values[i] = new_penalty

            raw_weights = 1.0 / (1.0 + self.nu_penalty_alpha * penalty_values)

            # Normalize weights to sum to 1
            weight_sum = raw_weights.sum() + self.eps
            weights_tensor = raw_weights / weight_sum
            # Anti-mutation: cap max weight per client so one round cannot be dominated by 1–2 clients.
            for _ in range(10):  # at most 10 iterations
                if (weights_tensor > self.nu_weight_max).any().item():
                    weights_tensor = torch.clamp(weights_tensor, max=self.nu_weight_max)
                    weight_sum = weights_tensor.sum() + self.eps
                    weights_tensor = weights_tensor / weight_sum
                else:
                    break
            
            # Weighted aggregation
            g_t = (weights_tensor[:, None] * processed_tensor).sum(dim=0)

            g_norm_after = torch.norm(g_t) + self.eps
            self._last_g_norm = g_norm_after.item()
            self._last_g_capped = False
            self._last_w_max = weights_tensor.max().item()
        else:
            # Uniform weights: before nu_penalty_start_iter
            weights_tensor = torch.ones(N, device=device, dtype=dtype) / N
            g_t = processed_tensor.mean(dim=0)
            g_norm_after = torch.norm(g_t) + self.eps
            self._last_g_norm = g_norm_after.item()
            self._last_g_capped = False
            self._last_w_max = 1.0 / N
        
        # Update histories (skip newly initialized clients)
        mu_min_t = torch.tensor(self.mu_min, device=device, dtype=dtype)
        nu_min_t = torch.tensor(self.nu_min, device=device, dtype=dtype)
        
        d_eff_list = []  # for verbose ν update logging
        for i, cid in enumerate(client_ids):
            if cid in new_cids:
                continue
            
            delta_processed = processed[i]
            r_bar = r_bar_list[i]
            r_bar_norm = r_bar_norm_list[i]
            delta_tilde_norm = delta_tilde_norm_list[i]
            
            # Aggregation path: update b from the reconstructed message used in aggregation.
            self.b[cid] = (self.rho_b * self.b[cid] + (1.0 - self.rho_b) * delta_processed).detach()
            # Stats path: 小 norm client 的 b_norm/μ/ν/r_prev 全部由 scale 后的量计算；仅聚合路径的 b 与 g_t 用未 scale 的数值
            self.b_norm[cid] = (self.rho_b * self.b_norm[cid] + (1.0 - self.rho_b) * delta_tilde_norm).detach()
            # μ: 小 norm 时用 scale 后向量的范数（||delta_tilde_norm||），否则用残差范数 ||r_bar_norm||
            r_bar_norm_scalar = torch.norm(r_bar_norm) + self.eps
            if has_small_norm and norm_small[i].item():
                r_bar_norm_scalar = torch.norm(delta_tilde_norm) + self.eps  # 由 scale 后的数值计算
            self.mu[cid] = (self.rho_mu * self.mu[cid] + (1.0 - self.rho_mu) * r_bar_norm_scalar).detach()
            self.mu[cid] = torch.maximum(self.mu[cid], mu_min_t)
            d = torch.norm(r_bar_norm - self.r_prev[cid])  # ν 的 d 已由 scale 路径的 r_bar_norm 计算
            d_eff = d.clamp(min=self.eps)
            d_eff_list.append(d_eff.item())
            rho_use = self.rho_nu
            nu_new = (rho_use * self.nu[cid] + (1.0 - rho_use) * d_eff).detach()
            # Optional per-round cap: ν change limiter (disabled by default)
            if getattr(self, 'enable_nu_change_cap', False):
                # ν can't drop >25% or rise >33% in one step → avoid 骤降/骤升 (e.g. 1.1→0.35)
                nu_old_val = self.nu[cid]
                nu_new = torch.clamp(nu_new, nu_old_val * 0.75, nu_old_val * 1.33)
            nu_new = torch.maximum(nu_new, nu_min_t)
            self.nu[cid] = nu_new
            self.r_prev[cid] = r_bar_norm.detach().clone()
            client_debug[i]['mu_expected'] = max(
                self.rho_mu * client_debug[i]['mu_before'] + (1.0 - self.rho_mu) * r_bar_norm_scalar.item(),
                self.mu_min,
            )
            client_debug[i]['nu_expected'] = max(
                self.rho_nu * client_debug[i]['nu_before'] + (1.0 - self.rho_nu) * d_eff.item(),
                self.nu_min,
            )
            client_debug[i]['mu_after'] = self.mu[cid].item()
            client_debug[i]['nu_after'] = self.nu[cid].item()
        
        # Verbose ν update detail: every verbose_nu_log_interval iters print d/nu stats + per-client d and ν
        verbose_nu_interval = getattr(self, 'verbose_nu_log_interval', 0)
        do_nu_detail = (verbose_nu_interval > 0 and d_eff_list and self.iteration_count % verbose_nu_interval == 0)
        if do_nu_detail:
            import time
            from ByrdLab.library.tool import log
            d_mean = sum(d_eff_list) / len(d_eff_list)
            d_min = min(d_eff_list)
            d_max = max(d_eff_list)
            ordered_cids = [cid for cid in client_ids if cid not in new_cids and cid in self.nu]
            nu_list = [self.nu[cid].item() for cid in ordered_cids]
            nu_mean = sum(nu_list) / len(nu_list) if nu_list else 0.0
            nu_min = min(nu_list) if nu_list else 0.0
            nu_max = max(nu_list) if nu_list else 0.0
            msg = (f"[HRAC nu detail] iter={self.iteration_count} d_mean={d_mean:.4f} d_min={d_min:.4f} d_max={d_max:.4f} "
                   f"nu_mean={nu_mean:.4f} nu_min={nu_min:.4f} nu_max={nu_max:.4f} (rho_nu={self.rho_nu})")
            # Per-client d and nu (same order as ordered_cids; d_eff_list matches that order)
            for idx, cid in enumerate(ordered_cids):
                d_val = d_eff_list[idx]
                nu_val = self.nu[cid].item()
                msg += f" | c{cid}:d={d_val:.4f} nu={nu_val:.4f}"
            log(msg)
            if self._hrac_log_buffer is not None:
                self._hrac_log_buffer.append(time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime()) + msg)
        # Update predictable global scale for the next round only after all
        # current-round clipping decisions have been made.
        if self.enable_global_cap:
            self.global_scale_ema = (
                self.rho_g * self.global_scale_ema
                + (1.0 - self.rho_g) * median_norm.detach()
            ).detach()
            self._last_global_scale_next = self.global_scale_ema.detach().clone()
        else:
            self._last_global_scale_next = None

        # Update median_norm_prev for next round's ratio logging
        self.median_norm_prev = median_norm.item()

        # Logging
        do_full_log = (self.enable_logging and self.iteration_count % self.log_interval == 0)
        if do_full_log:
            norms_pre = torch.norm(messages, dim=1).cpu()
            norms_post = torch.norm(messages_clipped, dim=1).cpu()
            norms_final = torch.norm(processed_tensor, dim=1).cpu()
            weights_for_log = weights_tensor.cpu().tolist()
            self._log_statistics(client_ids, tau_list,
                                [self.mu[cid].item() if cid in self.mu else 0.0 for cid in client_ids],
                                [self.nu[cid].item() if cid in self.nu else 0.0 for cid in client_ids],
                                median_norm.item(), B.item(), scale_global,
                                norms_pre.tolist(), norms_post.tolist(),
                                weights_for_log, norms_final.tolist(),
                                global_scale_for_cap=self._last_global_scale_for_cap.item() if self._last_global_scale_for_cap is not None else None,
                                global_scale_next=self._last_global_scale_next.item() if self._last_global_scale_next is not None else None)

        if self.enable_invariant_checks:
            for i in range(len(client_debug)):
                client_debug[i]['weight'] = weights_tensor[i].item()
                client_debug[i]['processed_norm'] = torch.norm(processed_tensor[i]).item()
            self._check_invariants({
                'iteration': self.iteration_count,
                'client_ids': client_ids,
                'norms': norms.detach(),
                'median_norm': median_norm.detach(),
                'B': B.detach(),
                'messages_clipped': messages_clipped.detach(),
                'processed_tensor': processed_tensor.detach(),
                'weights_tensor': weights_tensor.detach(),
                'g_t': g_t.detach(),
                'use_nu_penalty_this_round': use_nu_penalty_this_round,
                'client_debug': client_debug,
            })

        return g_t

    def _log_statistics(self, client_ids, tau_list, mu_list, nu_list,
                        median_norm=None, global_cap=None, scale_global=None,
                        norms_pre=None, norms_post=None, weights=None, norms_final=None,
                        global_scale_for_cap=None, global_scale_next=None):
        """Log HRAC statistics; buffer for file, print to console. File written at flush_log_to_file()."""
        from ByrdLab.library.tool import log
        import os
        import time
        timeStamp = time.strftime('[%y-%m-%d %H:%M:%S] ', time.localtime())
        lines = []
        def add(s):
            lines.append(timeStamp + s)
            log(s)

        add(f"[HRAC] Iteration {self.iteration_count} Statistics:")
        if not self.enable_global_cap and self.iteration_count == 0:
            add("  [Global median norm cap: OFF - using raw update norms]")
        if not self.enable_per_client_residual_clip and self.iteration_count == 0:
            add("  [Per-client residual clipping: OFF - using r_bar=r, r_bar_norm=r_norm]")
        if not self.enable_nu_weighting and self.iteration_count == 0:
            add("  [Nu weighting: OFF - using uniform weights for all rounds]")
        if not self.enable_post_residual_b_cap and self.iteration_count == 0:
            add("  [Post-residual B cap: OFF - no min(1,B/||delta_tilde||) after b+r_bar]")
        if self.current_accuracy is not None:
            add(f"  Accuracy: {self.current_accuracy:.4f}")
        if median_norm is not None and global_cap is not None:
            scale_info = ""
            if global_scale_for_cap is not None:
                scale_info += f", scale_used={global_scale_for_cap:.4f}"
            if global_scale_next is not None:
                scale_info += f", scale_next={global_scale_next:.4f}"
            add(f"  Global clip: median_norm={median_norm:.4f}, cap={global_cap:.4f} (c_g={self.c_g}, rho_g={self.rho_g}{scale_info})")
            if scale_global is not None:
                scale_min = scale_global.min().item()
                scale_max = scale_global.max().item()
                add(f"  Global clip scales: min={scale_min:.4f}, max={scale_max:.4f}")
            if norms_pre is not None and norms_post is not None:
                pre_t = torch.tensor(norms_pre)
                post_t = torch.tensor(norms_post)
                add(f"  Norms: pre_mean={pre_t.mean():.4f}, pre_max={pre_t.max():.4f}, "
                    f"post_mean={post_t.mean():.4f}, post_max={post_t.max():.4f}")
                if norms_final is not None:
                    final_t = torch.tensor(norms_final)
                    add(f"  Norms (final aggregated): mean={final_t.mean():.4f}, max={final_t.max():.4f}")
        
        tau_t = torch.tensor(tau_list)
        mu_t = torch.tensor(mu_list)
        nu_t = torch.tensor(nu_list)
        
        tm, tmin, tmax = tau_t.mean().item(), tau_t.min().item(), tau_t.max().item()
        mum, mumin, mumax = mu_t.mean().item(), mu_t.min().item(), mu_t.max().item()
        num, numin, numax = nu_t.mean().item(), nu_t.min().item(), nu_t.max().item()
        
        add(f"  Clip threshold (tau): mean={tm:.4f}, min={tmin:.4f}, max={tmax:.4f}")
        add(f"  Norm baseline (mu): mean={mum:.4f}, min={mumin:.4f}, max={mumax:.4f}")
        add(f"  Change baseline (nu): mean={num:.4f}, min={numin:.4f}, max={numax:.4f}")
        
        if median_norm is not None:
            median_norm_val = median_norm.item() if hasattr(median_norm, 'item') else median_norm
            prev_val = getattr(self, '_last_median_norm_prev_for_log', None)
            if prev_val is None:
                prev_val = "N/A"
            else:
                ratio = (median_norm_val + self.eps) / (prev_val + self.eps)
                add(f"  Norm change ratio: {ratio:.3f} (prev={prev_val}, curr={median_norm_val:.3f})")
        if self.iteration_count > self.nu_penalty_start_iter:
            add(f"  [Nu Penalty] ACTIVE: nu>0.9 -> excess=(nu-mean(nu))^+, EMA-smoothed with rho={self.rho_nu_penalty:.2f}; alpha={self.nu_penalty_alpha:.2f}; tau, mu, nu: real, norm<0.5*med lifted to median for stats")
            if weights is not None:
                weights_t = torch.tensor(weights)
                add(f"  Weights: mean={weights_t.mean():.4f}, min={weights_t.min():.4f}, max={weights_t.max():.4f}, sum={weights_t.sum():.4f}")
                nu_t = torch.tensor(nu_list)
                nu_above_one = (nu_t > 1.0).sum().item()
                add(f"  Clients with nu > 1: {nu_above_one}/{len(client_ids)}")
        else:
            add(f"  [Nu Penalty] INACTIVE (iter <= {self.nu_penalty_start_iter}, using equal weights)")
        if getattr(self, '_last_g_norm', None) is not None:
            g_cap_str = " [g_t CAPPED]" if getattr(self, '_last_g_capped', False) else ""
            add(f"  Aggregate: ||g_t||={self._last_g_norm:.4f}, w_max={getattr(self, '_last_w_max', 0):.4f}{g_cap_str}")
        
        add("  [Per-Client Values]:")
        
        if norms_pre is not None and norms_post is not None:
            if weights is not None:
                if norms_final is not None:
                    add("    (tau, mu, nu: real; norm<0.5*median lifted to median for stats. norm_pre=raw, norm_post=clip, norm_final=aggregation)")
                    add("    Client_ID | tau | mu | nu | norm_pre | norm_post | norm_final | weight | Attacker")
                    add("    " + "-" * 100)
                    for i, cid in enumerate(client_ids):
                        is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                        add(f"    {cid:9d} | {tau_list[i]:.4f} | {mu_list[i]:.4f} | {nu_list[i]:.4f} | "
                            f"{norms_pre[i]:.4f} | {norms_post[i]:.4f} | {norms_final[i]:.4f} | {weights[i]:.4f} | {is_attacker}")
                else:
                    add("    Client_ID | tau | mu | nu | norm_pre | norm_post | weight | Attacker")
                    add("    " + "-" * 85)
                    for i, cid in enumerate(client_ids):
                        is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                        add(f"    {cid:9d} | {tau_list[i]:.4f} | {mu_list[i]:.4f} | {nu_list[i]:.4f} | "
                            f"{norms_pre[i]:.4f} | {norms_post[i]:.4f} | {weights[i]:.4f} | {is_attacker}")
            else:
                if norms_final is not None:
                    add("    Client_ID | tau | mu | nu | norm_pre | norm_post | norm_final | Attacker")
                    add("    " + "-" * 90)
                    for i, cid in enumerate(client_ids):
                        is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                        add(f"    {cid:9d} | {tau_list[i]:.4f} | {mu_list[i]:.4f} | {nu_list[i]:.4f} | "
                            f"{norms_pre[i]:.4f} | {norms_post[i]:.4f} | {norms_final[i]:.4f} | {is_attacker}")
                else:
                    add("    Client_ID | tau | mu | nu | norm_pre | norm_post | Attacker")
                    add("    " + "-" * 75)
                    for i, cid in enumerate(client_ids):
                        is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                        add(f"    {cid:9d} | {tau_list[i]:.4f} | {mu_list[i]:.4f} | {nu_list[i]:.4f} | "
                            f"{norms_pre[i]:.4f} | {norms_post[i]:.4f} | {is_attacker}")
        else:
            add("    Client_ID | tau | mu | nu | Attacker")
            add("    " + "-" * 50)
            for i, cid in enumerate(client_ids):
                is_attacker = "YES" if cid in self.byzantine_nodes else "NO"
                add(f"    {cid:9d} | {tau_list[i]:.4f} | {mu_list[i]:.4f} | {nu_list[i]:.4f} | {is_attacker}")
        add("")
        # One-line summary (included in buffer)
        acc_str = f"{self.current_accuracy:.4f}" if self.current_accuracy is not None else "N/A"
        line = f"iter={self.iteration_count} acc={acc_str} tau_mean={tm:.4f} mu_mean={mum:.4f} nu_mean={num:.4f} "
        if getattr(self, '_last_g_norm', None) is not None:
            line += f"g_norm={self._last_g_norm:.4f} w_max={getattr(self, '_last_w_max', 0):.4f} g_capped={1 if getattr(self, '_last_g_capped', False) else 0} "
        if self.iteration_count > self.nu_penalty_start_iter:
            line += f"[NuPenalty:ON alpha={self.nu_penalty_alpha:.2f} rho_p={self.rho_nu_penalty:.2f} (nu>0.9 -> EMA((nu-mean)^+))] "
            if weights is not None:
                weights_t = torch.tensor(weights)
                line += f"w_mean={weights_t.mean():.4f} w_min={weights_t.min():.4f} w_max={weights_t.max():.4f} "
        else:
            line += "[NuPenalty:OFF] "
        for i, cid in enumerate(client_ids):
            line += f"| c{cid}: tau={tau_list[i]:.4f} mu={mu_list[i]:.4f} nu={nu_list[i]:.4f}"
            if weights is not None and i < len(weights):
                line += f" w={weights[i]:.4f}"
        add(line)
        # Buffer for later single-file write (no per-line file I/O)
        if self.log_file is not None:
            self._hrac_log_buffer.append("\n".join(lines))
