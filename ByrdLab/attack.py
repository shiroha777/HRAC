import math
import random
import copy

import scipy.stats
import torch

from ByrdLab import FEATURE_TYPE, DEVICE
from ByrdLab.library.RandomNumberGenerator import RngPackage
from ByrdLab.library.tool import MH_rule

# Note: HisMSA attack requires algorithm-level support for:
# 1. Calling update_model_history() after each global model update
# 2. Calling get_clipping_factor() to clip malicious gradients before aggregation
# See the attack class documentation for details

def gaussian(messages, honest_nodes, byzantine_nodes, scale, torch_rng=None):
    # with the same mean and larger variance
    mu = torch.zeros(messages.size(1), dtype=FEATURE_TYPE).to(DEVICE)
    for node in honest_nodes:
        mu.add_(messages[node], alpha=1/len(honest_nodes))
    for node in byzantine_nodes:
        messages[node].copy_(mu)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng).to(DEVICE)
        messages[node].add_(noise, alpha=10000)
    
def sign_flipping(messages, honest_nodes, byzantine_nodes, scale,
                  noise_scale=0, torch_rng=None):
    mu = torch.zeros(messages.size(1), dtype=FEATURE_TYPE).to(DEVICE)
    for node in honest_nodes:
        mu.add_(messages[node], alpha=1/len(honest_nodes))
    melicious_message = -scale * mu
    for node in byzantine_nodes:
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng).to(DEVICE)
        messages[node].copy_(melicious_message)
        messages[node].add_(noise, alpha=noise_scale)
             
def get_model_control(messages, honest_nodes, byzantine_nodes, target_message):
    s = torch.zeros(messages.size(1), dtype=FEATURE_TYPE).to(DEVICE)
    for node in honest_nodes:
        s.add_(messages[node])
    melicious_message = (target_message*len(honest_nodes)-s) / len(byzantine_nodes)
    return melicious_message

def get_model_control_weight(messages, honest_nodes, byzantine_nodes, target_message, weights):
    s = torch.zeros(messages.size(1), dtype=FEATURE_TYPE).to(DEVICE)
    for node in honest_nodes:
        s.add_(messages[node], alpha=weights[node])
    byzantine_weight = weights[byzantine_nodes].sum()
    melicious_message = (target_message-s) / byzantine_weight
    return melicious_message

def model_control(messages, honest_nodes, byzantine_nodes, target_message):
    melicious_message = get_model_control(messages, honest_nodes, 
                                          byzantine_nodes, target_message)
    for node in byzantine_nodes:
        messages[node].copy_(melicious_message)
    
def zero_attack(messages, honest_nodes, byzantine_nodes, noise_scale=0,
                torch_rng=None):
    target_message = torch.zeros(messages.size(1))
    melicious_message = get_model_control(messages, honest_nodes, 
                                          byzantine_nodes, target_message)
    for node in byzantine_nodes:
        messages[node].copy_(melicious_message)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE,
                            generator=torch_rng)
        messages[node].add_(noise, alpha=noise_scale)
        
def same_value_attack(messages, honest_nodes, byzantine_nodes, scale=1,
                      noise_scale=0, rng=None):
    c = 0
    for node in honest_nodes:
        # c += messages[node].mean().item()
        c += messages[node].mean().item() / len(honest_nodes)
    model_dim = messages.size(1)
    attack_value = scale*c / math.sqrt(model_dim)
    for node in byzantine_nodes:
        messages[node].copy_(attack_value)
        noise = torch.randn(messages.size(1), dtype=FEATURE_TYPE, generator=rng)
        messages[node].add_(noise, alpha=noise_scale)
    
    
class CentralizedAttack():
    def __init__(self, name, honest_nodes, byzantine_nodes):
        self.name = name
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
    
class CentralizedAttackWrapper(CentralizedAttack):
    def __init__(self, name, honest_nodes, byzantine_nodes, attack_fn, **kw):
        super().__init__(name=name, honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes)
        self.kw = kw
        self.attack_fn = attack_fn
        
    def run(self, messages):
        self.attack_fn(messages, self.honest_nodes, self.byzantine_nodes, **self.kw)
    
class C_gaussian(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=30):
        super().__init__(name='gaussian', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=gaussian, scale=scale)
        self.scale = scale
            
class C_sign_flipping(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=100, noise_scale=0):
        super().__init__(name='sign_flipping', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=sign_flipping, scale=scale,
                         noise_scale=noise_scale)
        self.scale = scale


def bit_flipping(messages, honest_nodes, byzantine_nodes):
    """Bit Flipping (BF): each Byzantine sends the negation of its own message."""
    for node in byzantine_nodes:
        messages[node].neg_()


def ipm_attack(messages, honest_nodes, byzantine_nodes, epsilon=0.1):
    """Inner Product Manipulation (IPM): malicious = -epsilon * mean(honest). From byzantine-robust-optimizer."""
    mu = torch.zeros(messages.size(1), dtype=FEATURE_TYPE).to(DEVICE)
    for node in honest_nodes:
        mu.add_(messages[node], alpha=1.0 / len(honest_nodes))
    malicious = -epsilon * mu
    for node in byzantine_nodes:
        messages[node].copy_(malicious)


def alie_attack(messages, honest_nodes, byzantine_nodes, n=None, m=None):
    """A Little Is Enough (ALIE): malicious = mu - std * z_max. From byzantine-robust-optimizer."""
    n = n if n is not None else (len(honest_nodes) + len(byzantine_nodes))
    m = m if m is not None else len(byzantine_nodes)
    s = math.floor(n / 2 + 1) - m
    cdf_value = (n - m - s) / (n - m)
    z_max = scipy.stats.norm.ppf(cdf_value)
    honest_stack = torch.stack([messages[node] for node in honest_nodes], dim=0)
    mu = torch.mean(honest_stack, dim=0).to(DEVICE)
    std = torch.std(honest_stack, dim=0).to(DEVICE)
    # avoid div by zero
    std = torch.clamp(std, min=1e-8)
    malicious = mu - std * z_max
    for node in byzantine_nodes:
        messages[node].copy_(malicious)


def min_max_attack(messages, honest_nodes, byzantine_nodes,
                   dev_type='unit_vec', initial_lambda=10.0,
                   threshold_diff=1e-5, eps=1e-12,
                   use_honest=True):
    """
    Min-Max model poisoning attack from NDSS21-Model-Poisoning.

    The malicious update is mean(honest) - lambda * deviation, where lambda is
    searched so that its maximum squared distance to any honest update is no
    larger than the maximum pairwise squared distance inside the honest cloud.

    If use_honest=False, this follows the unknown-benign-gradient notebook:
    the attacker estimates the reference cloud using only Byzantine messages.
    """
    reference_nodes = honest_nodes if use_honest else byzantine_nodes
    reference_stack = messages[reference_nodes]
    model_re = torch.mean(reference_stack, dim=0)

    if dev_type == 'unit_vec':
        norm = torch.norm(model_re)
        if norm <= eps:
            deviation = torch.sign(model_re)
        else:
            deviation = model_re / norm
    elif dev_type == 'sign':
        deviation = torch.sign(model_re)
    elif dev_type == 'std':
        deviation = torch.std(reference_stack, dim=0)
    else:
        raise ValueError(f"unknown Min-Max deviation type: {dev_type!r}")

    if torch.norm(deviation) <= eps:
        malicious = model_re
        for node in byzantine_nodes:
            messages[node].copy_(malicious)
        return

    pairwise_dist = torch.cdist(reference_stack, reference_stack, p=2).pow(2)
    max_distance = torch.max(pairwise_dist)

    def is_success(lamda):
        mal_update = model_re - lamda * deviation
        max_d = torch.norm(reference_stack - mal_update, dim=1).pow(2).max()
        return max_d <= max_distance

    lamda = torch.tensor(float(initial_lambda), device=messages.device, dtype=messages.dtype)
    lamda_fail = lamda.clone()
    lamda_succ = torch.zeros((), device=messages.device, dtype=messages.dtype)

    # Match NDSS21-Model-Poisoning: start from lambda=10 and halve lamda_fail
    # until lambda and the last successful lambda are within threshold_diff.
    while torch.abs(lamda_succ - lamda).item() > threshold_diff:
        if is_success(lamda):
            lamda_succ = lamda.clone()
            lamda = lamda + lamda_fail / 2.0
        else:
            lamda = lamda - lamda_fail / 2.0
        lamda_fail = lamda_fail / 2.0

    malicious = model_re - lamda_succ * deviation
    for node in byzantine_nodes:
        messages[node].copy_(malicious)


def mimic_attack(messages, honest_nodes, byzantine_nodes, epsilon=0):
    """
    Mimic attack from ByzFL: all Byzantine workers copy one honest worker.

    ByzFL defines Mimic_epsilon(x_1, ..., x_n) = x_{epsilon+1}. Its Byzantine
    client wrapper then repeats this single vector for all Byzantine clients.
    Here epsilon indexes honest_nodes in the local node ordering.
    """
    if not isinstance(epsilon, int) or epsilon < 0:
        raise ValueError("Mimic epsilon must be a non-negative integer")
    if epsilon >= len(honest_nodes):
        raise ValueError("Mimic epsilon must be smaller than the number of honest nodes")
    source_node = honest_nodes[epsilon]
    malicious = messages[source_node].clone()
    for node in byzantine_nodes:
        messages[node].copy_(malicious)


class C_bit_flipping(CentralizedAttackWrapper):
    """Bit Flipping (BF): Byzantine workers negate their own submitted messages."""
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='bit_flipping', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes, attack_fn=bit_flipping)


class C_IPM(CentralizedAttackWrapper):
    """Inner Product Manipulation (IPM). Paper: byzantine-robust-optimizer. epsilon=0.1 by default."""
    def __init__(self, honest_nodes, byzantine_nodes, epsilon=0.1):
        super().__init__(name='IPM', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes, attack_fn=ipm_attack, epsilon=epsilon)
        self.epsilon = epsilon


class C_ALIE(CentralizedAttack):
    """A Little Is Enough (ALIE). Paper: byzantine-robust-optimizer."""
    def __init__(self, honest_nodes, byzantine_nodes, n=None, m=None):
        super().__init__(name='ALIE', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        self.n = n if n is not None else (len(honest_nodes) + len(byzantine_nodes))
        self.m = m if m is not None else len(byzantine_nodes)

    def run(self, messages):
        alie_attack(messages, self.honest_nodes, self.byzantine_nodes, n=self.n, m=self.m)


class C_MinMaxFull(CentralizedAttackWrapper):
    """Min-Max attack with full knowledge of current honest messages."""
    def __init__(self, honest_nodes, byzantine_nodes, dev_type='unit_vec',
                 initial_lambda=10.0, threshold_diff=1e-5):
        super().__init__(name='min_max_full', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes,
                         attack_fn=min_max_attack, dev_type=dev_type,
                         initial_lambda=initial_lambda,
                         threshold_diff=threshold_diff,
                         use_honest=True)
        self.dev_type = dev_type
        self.initial_lambda = initial_lambda
        self.threshold_diff = threshold_diff


class C_MinMaxUnknown(CentralizedAttackWrapper):
    """Min-Max attack when current honest messages are unknown to attackers."""
    def __init__(self, honest_nodes, byzantine_nodes, dev_type='unit_vec',
                 initial_lambda=30.0, threshold_diff=1e-5):
        super().__init__(name='min_max_unknown', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes,
                         attack_fn=min_max_attack, dev_type=dev_type,
                         initial_lambda=initial_lambda,
                         threshold_diff=threshold_diff,
                         use_honest=False)
        self.dev_type = dev_type
        self.initial_lambda = initial_lambda
        self.threshold_diff = threshold_diff


# Backward-compatible alias. The command-line entry now chooses explicit classes.
C_MinMax = C_MinMaxFull


class C_Mimic(CentralizedAttackWrapper):
    """Mimic attack: Byzantine workers repeat one honest worker's message."""
    def __init__(self, honest_nodes, byzantine_nodes, epsilon=0):
        super().__init__(name='mimic', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes,
                         attack_fn=mimic_attack, epsilon=epsilon)
        self.epsilon = epsilon


class C_PoisonedFL(CentralizedAttack):
    """
    PoisonedFL-style multi-round consistency attack.

    The original MXNet implementation builds a model-delta attack from the
    previous global-model motion, previous malicious update, a fixed random
    sign vector, and a feedback-adjusted scale. This PyTorch/LPA integration
    keeps the same stateful mechanism inside the message-level attack API.

    LPA applies aggregated messages with param -= lr * message, while the
    original PoisonedFL code applies model deltas with param += update. Hence
    this class sends the negative of the PoisonedFL model-delta proxy.
    """
    def __init__(self, honest_nodes, byzantine_nodes, scaling_factor=8.0,
                 feedback_interval=50, min_scaling_factor=0.5, eps=1e-9):
        super().__init__(name='poisonedfl', honest_nodes=honest_nodes,
                         byzantine_nodes=byzantine_nodes)
        self.scaling_factor = float(scaling_factor)
        self.feedback_interval = int(feedback_interval)
        self.min_scaling_factor = float(min_scaling_factor)
        self.eps = eps
        self.iteration = 0
        self.fixed_rand = None
        self.last_model = None
        self.last_50_model = None
        self.history = None
        self.feedback_delta = None
        self.last_malicious_delta = None
        self.pending_last_50_refresh = False
        self.current_lr = 1.0

    def _ensure_state(self, messages):
        dim = messages.size(1)
        if self.fixed_rand is None:
            rand = torch.randn(dim, device=messages.device, dtype=messages.dtype)
            self.fixed_rand = torch.sign(rand)
            self.fixed_rand[self.fixed_rand == 0] = 1
        elif (self.fixed_rand.device != messages.device
                or self.fixed_rand.dtype != messages.dtype or self.fixed_rand.numel() != dim):
            rand = torch.randn(dim, device=messages.device, dtype=messages.dtype)
            self.fixed_rand = torch.sign(rand)
            self.fixed_rand[self.fixed_rand == 0] = 1
            self.history = None
            self.last_model = None
            self.last_50_model = None
            self.feedback_delta = None
            self.last_malicious_delta = None
            self.pending_last_50_refresh = False
            self.iteration = 0

    def update_model_history(self, current_model_flat, lr=None):
        if lr is not None:
            self.current_lr = max(float(lr), self.eps)
        current = current_model_flat.detach().clone()
        if self.last_model is None or self.last_model.numel() != current.numel():
            self.last_model = current
            self.last_50_model = None
            self.history = None
            self.feedback_delta = None
            return
        self.history = current - self.last_model
        self.last_model = current
        if self.pending_last_50_refresh or self.last_50_model is None:
            self.last_50_model = current.clone()
            self.pending_last_50_refresh = False
        self.feedback_delta = current - self.last_50_model

    def _feedback_scale(self):
        if self.feedback_interval <= 0 or self.iteration % self.feedback_interval != 0:
            return self.scaling_factor
        if self.feedback_delta is None:
            return self.scaling_factor
        accumulated_delta = torch.where(
            self.feedback_delta == 0,
            self.last_model,
            self.feedback_delta
        )
        aligned_dim_cnt = (torch.sign(accumulated_delta) == self.fixed_rand).sum().item()
        dim = self.fixed_rand.numel()
        # Normal approximation to the one-sided 99% binomial threshold used by
        # the original implementation's hard-coded k_99 constants.
        k_99 = dim / 2.0 + 2.326347874 * math.sqrt(dim / 4.0)
        if aligned_dim_cnt < k_99 and self.scaling_factor * 0.7 >= self.min_scaling_factor:
            self.scaling_factor = self.scaling_factor * 0.7
        self.pending_last_50_refresh = True
        return self.scaling_factor

    def run(self, messages):
        self._ensure_state(messages)

        model_delta = None
        if self.history is not None and self.last_malicious_delta is not None:
            history_norm = torch.norm(self.history)
            last_grad_norm = torch.norm(self.last_malicious_delta)
            scale = torch.abs(
                self.history
                - self.last_malicious_delta * history_norm / (last_grad_norm + self.eps)
            )
            deviation = scale * self.fixed_rand / (torch.norm(scale) + self.eps)
            sf = self._feedback_scale()
            lamda_succ = sf * history_norm
            model_delta = lamda_succ * deviation
            malicious_message = -model_delta / self.current_lr
            for node in self.byzantine_nodes:
                messages[node].copy_(malicious_message)

        if model_delta is not None:
            self.last_malicious_delta = model_delta.detach().clone()
        elif self.last_malicious_delta is None:
            self.last_malicious_delta = torch.zeros(messages.size(1), device=messages.device,
                                                    dtype=messages.dtype)
        self.iteration += 1


class C_TauBoundary(CentralizedAttack):
    """
    Tau-Boundary Slow Drift: adaptive attack targeting HRAC.
    - Sends update so residual sits near τ (clip barely bites), direction u almost constant → ν stays small → high weight.
    - Poisons client history b via EMA so drift is sustainable.

    Threat model (use_honest=True, default):
      - Round-gradient full-information: sees this round's messages[honest_nodes], can compute median_norm and mean(honest) to choose u = -mean(honest). Does NOT know server state (b, τ, μ, weights, aggregate). Standard strong adversary for pressure testing (e.g. Krum/Bulyan style).
    Threat model (use_honest=False):
      - Cross-device realistic: attacker only uses own gradient (messages[byzantine_nodes[0]]). No honest_nodes, no honest_stack. u = -normalize(own_grad), scale from own norm. Weaker but realistic when clients cannot see others' updates.
    """
    def __init__(self, honest_nodes, byzantine_nodes, c=2.5, rho_b=0.98, c_g=3.0, mu_frac=0.4, eps=1e-12, use_honest=True):
        super().__init__(name='TauBoundary', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes)
        self.c = c
        self.rho_b = rho_b
        self.c_g = c_g
        self.mu_frac = mu_frac  # τ_hat = c * (mu_frac * median_norm)
        self.eps = eps
        self.use_honest = use_honest  # True: round-gradient omniscient; False: only own gradient (realistic)
        self.b_hat = None
        self.last_sent = None

    def run(self, messages):
        device, dtype = messages.device, messages.dtype
        D = messages.size(1)
        if self.use_honest:
            honest_stack = messages[self.honest_nodes]
            norms_honest = torch.norm(honest_stack, dim=1)
            median_norm = torch.median(norms_honest).clamp_min(self.eps).to(device)
            g_mean = honest_stack.mean(dim=0)
            gn = torch.norm(g_mean) + self.eps
            u = (-g_mean / gn).to(device)
        else:
            # Realistic: only own gradient (no honest_nodes / honest_stack)
            own = messages[self.byzantine_nodes[0]].clone()
            median_norm = torch.norm(own).clamp_min(self.eps).to(device)
            gn = median_norm.clone()
            u = (-own / gn).to(device)
        B_est = (self.c_g * median_norm).clamp_min(self.eps)
        mu_proxy = (self.mu_frac * median_norm).to(device)
        tau_hat = (self.c * mu_proxy).to(device)
        if self.b_hat is None or self.b_hat.device != device or self.b_hat.dtype != dtype or self.b_hat.size(0) != D:
            self.b_hat = torch.zeros(D, device=device, dtype=dtype)
        if self.last_sent is not None:
            self.b_hat = (self.rho_b * self.b_hat + (1.0 - self.rho_b) * self.last_sent).detach()
        delta = (self.b_hat + tau_hat * u).detach()
        dn = torch.norm(delta) + self.eps
        if dn > B_est:
            delta = delta * (B_est / dn)
        for node in self.byzantine_nodes:
            messages[node].copy_(delta)
        self.last_sent = delta.clone()


class C_zero_gradient(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, noise_scale=0):
        super().__init__(name='zero_gradient', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, 
                         attack_fn=zero_attack, noise_scale=noise_scale)
        
class C_isolation(CentralizedAttack):
    def __init__(self, honest_nodes, byzantine_nodes):
        super().__init__(name='isolation', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes)
    def run(self, messages):
        melicious_message = get_model_control(messages, self.honest_nodes, 
                                              self.byzantine_nodes, 
                                              messages[-1])
        for node in self.byzantine_nodes:
            messages[node].copy_(melicious_message)

class C_same_value(CentralizedAttackWrapper):
    def __init__(self, honest_nodes, byzantine_nodes, scale=1, noise_scale=0):
        super().__init__(name='same_value', honest_nodes=honest_nodes, 
                         byzantine_nodes=byzantine_nodes, scale=scale,
                         attack_fn=same_value_attack, noise_scale=noise_scale)

class decentralizedAttack():
    def __init__(self, name, graph):
        self.graph = graph
        self.name = name
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        raise NotImplementedError
    
class D_gaussian(decentralizedAttack):
    def __init__(self, graph, scale=30):
        super().__init__(name='gaussian', graph=graph)
        self.scale = scale
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node] 
        mu = torch.mean(local_models[honest_neighbors], dim=0) * 100
        for n in byzantine_neigbors:
            local_models[n].copy_(mu)
            noise = torch.randn(local_models.size(1), 
                                generator=rng_pack.torch,
                                dtype=FEATURE_TYPE).to(DEVICE)
            local_models[n].add_(noise, alpha=self.scale)
            
class D_sign_flipping(decentralizedAttack):
    def __init__(self, graph, scale=None):
        if scale is None:
            scale = 1
            name = 'sign_flipping'
        else:
            name = f'sign_flipping_s={scale}'
        super().__init__(name=name, graph=graph)
        self.scale = scale
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbor = self.graph.byzantine_neighbors[node]
        mu = torch.mean(local_models[honest_neighbors+[node]], dim=0)
        melicious_message = -self.scale * mu * 100
        for n in byzantine_neigbor:
            local_models[n].copy_(melicious_message)
         
class D_zero_sum(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='zero_sum', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control(self.graph, local_models, node, 
                                                  torch.zeros_like(local_models[node]))
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
            
class D_zero_value(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='zero_value', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        for n in byzantine_neigbors:
            local_models[n].copy_(torch.zeros_like(local_models[node]))
            
def get_dec_model_control(graph, messages, node, target_model):
    honest_neighbors = graph.honest_neighbors[node]
    byzantine_neigbors = graph.byzantine_neighbors[node]
    melicious_message = get_model_control(messages, honest_neighbors,
                                          byzantine_neigbors, target_model)
    return melicious_message

def get_dec_model_control_weight(graph, messages, node, target_model, weight):
    honest_neighbors = graph.honest_neighbors_and_itself[node]
    byzantine_neigbors = graph.byzantine_neighbors[node]
    melicious_message = get_model_control_weight(messages, honest_neighbors,
                                                 byzantine_neigbors,
                                                 target_model, weight)
    return melicious_message

class D_isolation(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='isolation', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control(self.graph, local_models, node, 
                                                  local_models[node])
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
            
class D_isolation_weight(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='isolation_w', graph=graph)
        self.W = MH_rule(graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        melicious_message = get_dec_model_control_weight(self.graph, 
                                                         local_models, node, 
                                                         local_models[node],
                                                         self.W[node])
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)
        # avg = local_models[self.graph.neighbors_and_itself[node]].sum(dim=0) / (self.graph.neighbor_sizes[node]+1)

class D_sample_duplicate(decentralizedAttack):
    def __init__(self, graph):
        super().__init__(name='duplicate', graph=graph)
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        # duplicate_index = rng_pack.random.choice(honest_neighbors)
        duplicate_index = self.graph.honest_nodes[0]
        for n in byzantine_neigbors:
            local_models[n].copy_(local_models[duplicate_index])
        

class D_same_value(decentralizedAttack):
    def __init__(self, graph, scale=None, noise_scale=None, value=None):
        name = 'same_value'
        if scale is None:
            scale = 1
        else:
            name += f'_scale={scale:.1f}'
        if noise_scale is None:
            noise_scale = 0
        else:
            name += f'_noise_scale={noise_scale:.1f}'
        if value is not None:
            name += f'_value={value:.1f}'
        super().__init__(name=name, graph=graph)
        self.scale = scale
        self.noise_scale = noise_scale
        self.value = value
    def get_attack_value(self, local_models, node):
        honest_neighbors = self.graph.honest_neighbors[node]
        if self.value is None:
            c = 0
            for node in honest_neighbors:
                c += local_models[node].mean().item() / len(honest_neighbors)
            model_dim = local_models.size(1)
            return self.scale*c / math.sqrt(model_dim)
        else:
            return self.value
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        attack_value = self.get_attack_value(local_models, node)
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        for node in byzantine_neigbors:
            local_models[node] = attack_value
            noise = torch.randn(local_models.size(1), dtype=FEATURE_TYPE, 
                                generator=rng_pack.torch)
            local_models[node].add_(noise, alpha=self.noise_scale)
        
# A Little is Enough
class D_alie(decentralizedAttack):
    def __init__(self, graph, scale=None):
        if scale is None:
            name = 'alie'
        else:
            name = f'alie_scale={scale}'
        super().__init__(name=name, graph=graph)
        if scale is None:
            self.scale_table = [0] * self.graph.node_size
            for node in self.graph.honest_nodes:
                neighbors_size = self.graph.neighbor_sizes[node]
                byzantine_size = self.graph.byzantine_sizes[node]
                s = math.floor((neighbors_size+1)/2)-byzantine_size
                percent_point = (neighbors_size-s)/neighbors_size
                scale = scipy.stats.norm.ppf(percent_point)
                self.scale_table[node] = scale
        else:
            self.scale_table = [scale] * self.graph.node_size
    def run(self, local_models, node, rng_pack: RngPackage=RngPackage()):
        honest_neighbors = self.graph.honest_neighbors[node]
        byzantine_neigbors = self.graph.byzantine_neighbors[node]
        mu = torch.mean(local_models[honest_neighbors], dim=0)
        std = torch.std(local_models[honest_neighbors], dim=0)
        melicious_message = mu + self.scale_table[node]*std
        for n in byzantine_neigbors:
            local_models[n].copy_(melicious_message)

# Data Poisoning Attack
class DataPoisoningAttack():
    def __init__(self, name):
        self.name = name

    def run(self, features, targets, model=None, rng_pack: RngPackage=RngPackage(),):
        raise NotImplementedError

        
class label_flipping(DataPoisoningAttack):

    def __init__(self):
        super().__init__(name='label_flipping')
    
    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        features = features
        targets = 9 - targets
        # for i in range(len(targets)):
        #     if targets[i] == 0:
        #         targets[i] = 2
        #     elif targets[i] == 1:
        #         targets[i] = 9
        #     elif targets[i] == 5:
        #         targets[i] = 3 
        return features, targets
    
class label_random(DataPoisoningAttack):

    def __init__(self):
        super().__init__(name='label_random')

    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        features = features
        targets = torch.randint(0, 9, size=targets.shape, generator=rng_pack.torch)
        return features, targets
    
class feature_label_random(DataPoisoningAttack):

    def __init__(self):
        super().__init__(name='feature_label_random')

    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        features = 2 * torch.rand(size=features.shape, generator=rng_pack.torch, dtype=FEATURE_TYPE) - 1
        targets = torch.randint(0, 9, size=targets.shape, generator=rng_pack.torch)
        return features, targets
    
class furthest_label_flipping(DataPoisoningAttack):

    def __init__(self):
        super().__init__(name='furthest_label_flipping')

    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        data_size = len(targets)
        for i in range(data_size):
            feature = features[i].clone().to(DEVICE)
            # feature = feature.view(feature.size(0), -1).squeeze().clone()
            # distance = torch.mv(model.linear.weight.data, feature) + model.linear.bias.data
            distance = model(feature).squeeze()
            _, prediction_cls = torch.min(distance, dim=0)
            targets[i] = prediction_cls
        return features, targets
    

class adversarial_label_flipping(DataPoisoningAttack):

    def __init__(self):
        super().__init__(name='adversarial_label_flipping')

    def run(self, features, targets, model= None, rng_pack: RngPackage = RngPackage()):
        features = features
        targets = targets
        return features, targets


class MSA(DataPoisoningAttack):
    """
    Model Shuffling Attack (MSA)
    A simplified version of HisMSA that only implements Step 1 (Model Shuffling and Scaling).
    
    This attack performs:
    Step 1: Shuffle model parameters with cross-layer synchronization and apply 
            offset scaling (α and 1/α across adjacent layers) - Formula (4)
    
    Unlike HisMSA, MSA does NOT implement Step 2 (gradient clipping based on history).
    This makes MSA a simpler baseline attack for comparison with HisMSA.
    
    IMPORTANT LIMITATIONS (same as HisMSA):
    - We restrict our reproduction to sequential CNNs (nn.Sequential style),
      where module registration order closely matches forward execution order.
    - For other architectures (ResNet, Inception, multi-branch), Step-1 equivalence
      may fail; we use test_step1_equivalence() to detect such cases.
    - Only BatchNorm1d/2d are supported (LayerNorm/GroupNorm need special handling).
    """
    
    def __init__(self, shuffle_prob=1.0, scaling_factor_range=(0.8, 1.2), 
                 strict_equivalence=False, break_equivalence=True, 
                 equivalence_break_ratio=0.3, rng_pack: RngPackage = RngPackage()):
        """
        Args:
            shuffle_prob: Probability of applying shuffle (default 1.0 for consistent attack)
            scaling_factor_range: Range for scaling factor α (default (0.8, 1.2))
            strict_equivalence: If True, maintain approximately function-preserving transformation
                               (up to numerical noise). If False, allow slight inconsistencies 
                               for stronger attack effect.
            break_equivalence: If True, actively break function equivalence by NOT applying 1/α 
                              to next layer for some channels. This significantly increases attack strength.
            equivalence_break_ratio: Ratio of channels/layers to break equivalence (default 0.3)
            rng_pack: Random number generator package
        """
        super().__init__(name='MSA')
        self.shuffle_prob = shuffle_prob
        self.scaling_factor_range = scaling_factor_range
        self.strict_equivalence = strict_equivalence
        self.break_equivalence = break_equivalence  # New: actively break equivalence
        self.equivalence_break_ratio = equivalence_break_ratio  # Ratio to break
        self.rng_pack = rng_pack
        self.shuffled_models = {}  # Track which models have been shuffled (by id)
    
    def _get_ordered_layers(self, model):
        """Get model layers in forward order (reused from HisMSA)"""
        layers = []
        for name, module in model.named_modules():
            # Check BN FIRST (before weight check, because BN also has weight)
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                layers.append((name, None, 'bn', module))
                continue
            
            # Check conv/linear layers by weight shape and type
            if hasattr(module, 'weight') and module.weight is not None:
                param = module.weight
                if isinstance(module, torch.nn.Conv2d) and len(param.shape) == 4:
                    layers.append((name, param, 'conv', module))
                elif isinstance(module, torch.nn.Conv1d) and len(param.shape) == 3:
                    layers.append((name, param, 'conv', module))
                elif isinstance(module, (torch.nn.Linear)) and len(param.shape) == 2:
                    layers.append((name, param, 'linear', module))
                elif 'conv' in name.lower() and len(param.shape) == 4:
                    layers.append((name, param, 'conv', module))
                elif ('linear' in name.lower() or 'fc' in name.lower()) and len(param.shape) == 2:
                    layers.append((name, param, 'linear', module))
        return layers
    
    def _find_bn_after_layer(self, model, layer_name, layers_list, current_idx, out_channels):
        """Find BatchNorm layer that follows the given layer (reused from HisMSA)"""
        for j in range(current_idx + 1, len(layers_list)):
            next_name, next_param, next_type, next_module = layers_list[j]
            
            if next_type in ['conv', 'linear']:
                break
            
            if next_type == 'bn':
                if isinstance(next_module, torch.nn.BatchNorm2d):
                    if hasattr(next_module, 'num_features') and next_module.num_features == out_channels:
                        return next_module, next_name
                    continue
                elif isinstance(next_module, torch.nn.BatchNorm1d):
                    if hasattr(next_module, 'num_features') and next_module.num_features == out_channels:
                        return next_module, next_name
                    continue
        
        # Fallback: Try to find BN by name pattern (for Sequential models)
        layer_base = layer_name.rsplit('.', 1)[0] if '.' in layer_name else ''
        layer_idx_str = layer_name.rsplit('.', 1)[-1] if '.' in layer_name else layer_name
        
        if layer_base and layer_idx_str.isdigit():
            try:
                next_idx = int(layer_idx_str) + 1
                potential_bn_name = f"{layer_base}.{next_idx}"
                for name, module in model.named_modules():
                    if name == potential_bn_name and isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        if hasattr(module, 'num_features') and module.num_features == out_channels:
                            return module, name
            except (ValueError, AttributeError):
                pass
        
        return None, None
    
    def _shuffle_conv_kernels(self, model, rng_pack):
        """
        Step 1: Shuffle with cross-layer synchronization and per-channel offset scaling
        (Reused from HisMSA - same implementation)
        
        Note: This modifies model in-place. Caller should use a copy.
        """
        with torch.no_grad():
            layers = self._get_ordered_layers(model)
            
            if len(layers) == 0:
                return
            
            i = 0
            while i < len(layers):
                name, param, layer_type, module = layers[i]
                
                # Skip BN layers (they'll be handled when processing their preceding layer)
                if layer_type == 'bn':
                    i += 1
                    continue
                
                # Decide whether to attack this layer
                if rng_pack.random.random() >= self.shuffle_prob:
                    i += 1
                    continue
                
                if layer_type == 'conv':
                    # Conv layer: [out_channels, in_channels, H, W]
                    if param.shape[0] > 1:  # Need at least 2 channels to shuffle
                        out_channels = param.shape[0]
                        # Generate permutation for output channels
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            perm = torch.randperm(out_channels, generator=rng_pack.torch, device='cpu')
                            perm = perm.to(param.device)
                        else:
                            perm = torch.randperm(out_channels, generator=rng_pack.torch, device=param.device)
                        
                        # Shuffle current layer's output channels (weight)
                        param.data = param.data[perm]
                        
                        # Shuffle bias if exists
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data[perm]
                        
                        # Find and shuffle BN layer if exists after this conv
                        bn_module, bn_name = self._find_bn_after_layer(model, name, layers, i, out_channels)
                        if bn_module is not None:
                            if bn_module.weight is not None:
                                bn_module.weight.data = bn_module.weight.data[perm]
                            if bn_module.bias is not None:
                                bn_module.bias.data = bn_module.bias.data[perm]
                            if hasattr(bn_module, 'running_mean') and bn_module.running_mean is not None:
                                bn_module.running_mean.data = bn_module.running_mean.data[perm]
                            if hasattr(bn_module, 'running_var') and bn_module.running_var is not None:
                                bn_module.running_var.data = bn_module.running_var.data[perm]
                        
                        # Synchronize next layer's input channels
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'conv':
                                if next_param.shape[1] == out_channels:
                                    next_param.data = next_param.data[:, perm, :, :]
                            elif next_type == 'linear':
                                if next_param.shape[1] == out_channels:
                                    next_param.data = next_param.data[:, perm]
                        
                        # Per-channel offset scaling
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            alphas = torch.rand(out_channels, generator=rng_pack.torch, device='cpu', dtype=param.dtype)
                            alphas = alphas.to(param.device)
                        else:
                            alphas = torch.rand(out_channels, generator=rng_pack.torch, 
                                               device=param.device, dtype=param.dtype)
                        alphas = alphas * (self.scaling_factor_range[1] - self.scaling_factor_range[0]) + self.scaling_factor_range[0]
                        
                        # If not strict equivalence, only apply to subset of channels
                        if not self.strict_equivalence and rng_pack.random.random() < 0.3:
                            if param.device.type == 'cuda' and rng_pack.torch is not None:
                                mask = torch.rand(out_channels, generator=rng_pack.torch, device='cpu') > 0.2
                                mask = mask.to(param.device)
                            else:
                                mask = torch.rand(out_channels, generator=rng_pack.torch, device=param.device) > 0.2
                            alphas = torch.where(mask, alphas, torch.ones_like(alphas))
                        
                        # Apply per-channel α to current layer (Conv2d)
                        alphas_expanded = alphas.view(out_channels, 1, 1, 1)
                        param.data *= alphas_expanded
                        
                        # Apply per-channel 1/α to next layer (if not breaking equivalence)
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            should_break = self.break_equivalence and rng_pack.random.random() < self.equivalence_break_ratio
                            
                            if not should_break:  # Normal: apply 1/α to maintain equivalence
                                if next_type == 'conv' and next_param.shape[1] == out_channels:
                                    inv_alphas = (1.0 / alphas).view(1, out_channels, 1, 1)
                                    next_param.data *= inv_alphas
                                elif next_type == 'linear' and next_param.shape[1] == out_channels:
                                    inv_alphas = (1.0 / alphas).view(1, out_channels)
                                    next_param.data *= inv_alphas
                            # else: Break equivalence - don't apply 1/α, causing function change
                
                elif layer_type == 'linear':
                    # Linear layer: [out_features, in_features]
                    if param.shape[0] > 1:  # Need at least 2 output features
                        out_features = param.shape[0]
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            perm = torch.randperm(out_features, generator=rng_pack.torch, device='cpu')
                            perm = perm.to(param.device)
                        else:
                            perm = torch.randperm(out_features, generator=rng_pack.torch, device=param.device)
                        
                        # Shuffle current layer's output features (weight)
                        param.data = param.data[perm]
                        
                        # Shuffle bias if exists
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data[perm]
                        
                        # Synchronize next layer's input features
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'linear':
                                if next_param.shape[1] == out_features:
                                    next_param.data = next_param.data[:, perm]
                        
                        # Per-channel offset scaling (Linear layer)
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            alphas = torch.rand(out_features, generator=rng_pack.torch, device='cpu', dtype=param.dtype)
                            alphas = alphas.to(param.device)
                        else:
                            alphas = torch.rand(out_features, generator=rng_pack.torch,
                                               device=param.device, dtype=param.dtype)
                        alphas = alphas * (self.scaling_factor_range[1] - self.scaling_factor_range[0]) + self.scaling_factor_range[0]
                        
                        # If not strict equivalence, allow slight inconsistency
                        if not self.strict_equivalence and rng_pack.random.random() < 0.3:
                            if param.device.type == 'cuda' and rng_pack.torch is not None:
                                mask = torch.rand(out_features, generator=rng_pack.torch, device='cpu') > 0.2
                                mask = mask.to(param.device)
                            else:
                                mask = torch.rand(out_features, generator=rng_pack.torch, device=param.device) > 0.2
                            alphas = torch.where(mask, alphas, torch.ones_like(alphas))
                        
                        # Apply per-channel α to current layer (Linear)
                        alphas_expanded = alphas.view(out_features, 1)
                        param.data *= alphas_expanded
                        
                        # Apply per-channel 1/α to next layer (Linear) (if not breaking equivalence)
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            should_break = self.break_equivalence and rng_pack.random.random() < self.equivalence_break_ratio
                            
                            if not should_break:  # Normal: apply 1/α to maintain equivalence
                                if next_type == 'linear' and next_param.shape[1] == out_features:
                                    inv_alphas = (1.0 / alphas).view(1, out_features)
                                    next_param.data *= inv_alphas
                            # else: Break equivalence - don't apply 1/α, causing function change
                
                i += 1
    
    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        """
        Execute MSA attack - Step 1: Model Shuffling and Scaling only
        
        Note: This is model poisoning, not data poisoning. Features/targets are returned unchanged.
        
        WARNING: This method modifies the model in-place. The caller MUST pass a copy:
            client_model = copy.deepcopy(global_model)
            features, targets = attack.run(features, targets, model=client_model, rng_pack=rng_pack)
        """
        if model is None:
            return features, targets
        
        # Step 1: Apply shuffle and scaling
        model_id = id(model)
        
        # Apply attack if not already shuffled for this model instance
        if model_id not in self.shuffled_models:
            if rng_pack.random.random() < self.shuffle_prob:
                self._shuffle_conv_kernels(model, rng_pack)
                self.shuffled_models[model_id] = True
        
        # Return original data (MSA is model poisoning, not data poisoning)
        return features, targets
    
    def apply_step1_to_model(self, model, rng_pack: RngPackage = RngPackage()):
        """
        Apply Step1 (shuffle + scaling) to a model copy.
        This is a safer interface that makes it clear a copy should be used.
        
        Standard MSA: Shuffle and scale the model ONCE per model instance.
        This maintains functional equivalence (approximately) while creating 
        a different gradient direction.
        
        Args:
            model: The model to attack (will be modified in-place)
            rng_pack: Random number generator package
            
        Returns:
            The same model object (modified in-place)
            
        Usage:
            client_model = copy.deepcopy(global_model)
            attack.apply_step1_to_model(client_model, rng_pack)
            # Now train on client_model
        """
        # Standard MSA: Apply shuffle only once per model instance
        # Track by model_id to ensure each model is shuffled only once
        model_id = id(model)
        if model_id not in self.shuffled_models:
            if rng_pack.random.random() < self.shuffle_prob:
                self._shuffle_conv_kernels(model, rng_pack)
                self.shuffled_models[model_id] = True
        return model
    
    def test_step1_equivalence(self, model, test_input, tolerance=1e-5, verbose=False):
        """
        Unit test for Step1: Verify that shuffle+scaling maintains approximately 
        function-preserving transformation (up to numerical noise)
        
        Args:
            model: Original model
            test_input: Test input tensor
            tolerance: Maximum allowed difference in outputs (default 1e-5)
            verbose: If True, print detailed information
            
        Returns:
            (max_diff, mean_diff, passed): Maximum difference, mean difference, and pass status
        """
        import copy
        
        # Create two copies
        model0 = copy.deepcopy(model)
        model1 = copy.deepcopy(model)
        
        # Get original output
        model0.eval()
        with torch.no_grad():
            output0 = model0(test_input)
        
        # Apply Step1 to model1
        self.apply_step1_to_model(model1, self.rng_pack)
        
        # Get shuffled output
        model1.eval()
        with torch.no_grad():
            output1 = model1(test_input)
        
        # Compute differences
        diff = torch.abs(output1 - output0)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        passed = max_diff < tolerance
        
        if verbose or not passed:
            print(f"MSA Step1 Equivalence Test:")
            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  Tolerance: {tolerance:.6e}")
            print(f"  Status: {'PASSED' if passed else 'FAILED'}")
            if not passed:
                print(f"  WARNING: Step1 may have broken functional equivalence!")
                print(f"  This could indicate:")
                print(f"    - BN synchronization failed (wrong BN matched)")
                print(f"    - Layer order mismatch (not true forward order)")
                print(f"    - Missing layer synchronization")
        
        return max_diff, mean_diff, passed


class HisMSA(DataPoisoningAttack):
    """
    History-based Model Shuffling and Scaling Attack (HisMSA)
    From: "Attacks and countermeasures on federated learning via historical knowledge modeling"
    
    This attack consists of two steps:
    Step 1: Shuffle model parameters with cross-layer synchronization and apply 
            offset scaling (α and 1/α across adjacent layers) - Formula (4)
    Step 2: Dynamically approximate defense thresholds using historical model updates
            and apply γ clipping to update vector - Formula (5)
    
    IMPORTANT REPRODUCTION LIMITATIONS:
    - We restrict our reproduction to sequential CNNs (nn.Sequential style),
      where module registration order closely matches forward execution order.
    - For other architectures (ResNet, Inception, multi-branch), Step-1 equivalence
      may fail; we use test_step1_equivalence() to detect such cases.
    - Only BatchNorm1d/2d are supported (LayerNorm/GroupNorm need special handling).
    
    Key fixes from original implementation:
    1. Cross-layer shuffle synchronization (next layer's in_channels follows current layer's out_channels)
    2. Per-channel offset scaling (α_c and 1/α_c) instead of single-layer scaling
    3. Proper R_min/R_max computation with warmup + percentile-based adaptive defaults
    4. Direct application of γ to update vector
    5. BN identification fix (BN checked before weight-based checks to prevent double-processing)
    6. Vectorized scaling for both Conv and Linear layers (with correct shape handling)
    """
    
    def __init__(self, shuffle_prob=1.0, scaling_factor_range=(0.8, 1.2), 
                 strict_equivalence=True, warmup_rounds=5, rng_pack: RngPackage = RngPackage()):
        """
        Args:
            shuffle_prob: Probability of applying shuffle (default 1.0 for consistent attack)
            scaling_factor_range: Range for scaling factor α (default (0.8, 1.2))
            strict_equivalence: If True, maintain approximately function-preserving transformation
                               (up to numerical noise). If False, allow slight inconsistencies 
                               for stronger attack effect.
            warmup_rounds: Number of early rounds to use for boundary estimation (before attack pollution)
            rng_pack: Random number generator package
        """
        super().__init__(name='HisMSA')
        self.shuffle_prob = shuffle_prob
        self.scaling_factor_range = scaling_factor_range
        self.strict_equivalence = strict_equivalence
        self.warmup_rounds = warmup_rounds
        self.rng_pack = rng_pack
        self.history_global_models = []  # Store historical global models
        self.history_updates = []  # Store historical model updates (L2 norms)
        self.shuffled_models = {}  # Track which models have been shuffled (by id)
        self.init_scale = None  # Initial scale estimate for adaptive R_min/R_max
        self.warmup_complete = False  # Track if warmup period is complete
        
    def _get_ordered_layers(self, model):
        """
        Get model layers in forward order (for cross-layer operations)
        Returns list of (name, param, layer_type, module) tuples
        Also includes BN layers for synchronization
        
        IMPORTANT REPRODUCTION LIMITATION:
        This uses named_modules() which gives module registration order,
        not necessarily true forward execution order. 
        
        ✅ We restrict our reproduction to sequential CNNs (nn.Sequential style),
        where the module registration order closely matches the forward execution order.
        For other architectures (ResNet, Inception, multi-branch), Step-1 equivalence
        may fail; we use test_step1_equivalence() to detect such cases.
        
        Fix A: BN layers must be checked BEFORE weight-based checks, because
        BN also has weight (affine=True), which would cause them to be skipped.
        
        CRITICAL: The 'continue' statement ensures BN modules are NEVER processed
        by the weight-based branch below, preventing double-processing.
        """
        layers = []
        for name, module in model.named_modules():
            # Fix A: Check BN FIRST (before weight check, because BN also has weight)
            # CRITICAL: This continue ensures BN is NEVER processed by weight branch
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Only track BatchNorm1d/2d (LayerNorm/GroupNorm need special handling)
                layers.append((name, None, 'bn', module))
                continue  # BN is handled, skip to next module (prevents double-processing)
            
            # SAFE: At this point, module is NOT a BN (BN branch above has continue)
            # Then check conv/linear layers by weight shape and type
            if hasattr(module, 'weight') and module.weight is not None:
                param = module.weight  # SAFE: Only reached for non-BN modules
                # PRIORITY: Check by isinstance first (more reliable than name matching)
                if isinstance(module, torch.nn.Conv2d) and len(param.shape) == 4:
                    layers.append((name, param, 'conv', module))
                elif isinstance(module, torch.nn.Conv1d) and len(param.shape) == 3:
                    layers.append((name, param, 'conv', module))
                elif isinstance(module, (torch.nn.Linear)) and len(param.shape) == 2:
                    layers.append((name, param, 'linear', module))
                # Fallback: name-based matching (less reliable, but for compatibility)
                # NOTE: This fallback should rarely trigger if isinstance works correctly
                elif 'conv' in name.lower() and len(param.shape) == 4:
                    layers.append((name, param, 'conv', module))
                elif ('linear' in name.lower() or 'fc' in name.lower()) and len(param.shape) == 2:
                    layers.append((name, param, 'linear', module))
        return layers
    
    def _find_bn_after_layer(self, model, layer_name, layers_list, current_idx, out_channels):
        """
        Find BatchNorm layer that follows the given layer in forward order
        
        Fix 1: Use ordered layer list to find the next BN, ensuring it's actually
        the BN that normalizes this layer's output (not a different branch)
        
        Fix B: Actually verify num_features matches out_channels before returning
        
        Args:
            model: The model
            layer_name: Name of the current layer
            layers_list: Ordered list of all layers from _get_ordered_layers
            current_idx: Current index in layers_list
            out_channels: Number of output channels/features of current layer (for verification)
            
        Returns:
            (bn_module, bn_name) if found and verified, (None, None) otherwise
        """
        # Strategy: Look for BN in the layers list immediately after current layer
        # This ensures we get the BN that actually follows in forward order
        for j in range(current_idx + 1, len(layers_list)):
            next_name, next_param, next_type, next_module = layers_list[j]
            
            # If we hit another conv/linear before BN, this BN is not for current layer
            if next_type in ['conv', 'linear']:
                break
            
            # Found a BN - Fix B: Verify channel count matches BEFORE returning
            if next_type == 'bn':
                if isinstance(next_module, torch.nn.BatchNorm2d):
                    # Fix B: Actually check num_features matches out_channels
                    if hasattr(next_module, 'num_features') and next_module.num_features == out_channels:
                        return next_module, next_name
                    # Channel mismatch - this BN is not for current layer, continue searching
                    continue
                elif isinstance(next_module, torch.nn.BatchNorm1d):
                    # Fix B: Check num_features for BatchNorm1d
                    if hasattr(next_module, 'num_features') and next_module.num_features == out_channels:
                        return next_module, next_name
                    continue
        
        # Fallback: Try to find BN by name pattern (for Sequential models)
        # Fix 5: More conservative fallback - only for clear Sequential patterns
        # This is less reliable and should only be used as last resort
        layer_base = layer_name.rsplit('.', 1)[0] if '.' in layer_name else ''
        layer_idx_str = layer_name.rsplit('.', 1)[-1] if '.' in layer_name else layer_name
        
        # Only use fallback for clear Sequential patterns (e.g., "features.0" -> "features.1")
        # This avoids false matches in complex naming schemes
        if layer_base and layer_idx_str.isdigit():
            try:
                next_idx = int(layer_idx_str) + 1
                potential_bn_name = f"{layer_base}.{next_idx}"
                for name, module in model.named_modules():
                    if name == potential_bn_name and isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                        # Fix B: Still verify channel count even in fallback
                        if hasattr(module, 'num_features') and module.num_features == out_channels:
                            return module, name
            except (ValueError, AttributeError):
                pass
        
        # If fallback didn't work, return None (safer than wrong match)
        # This ensures we don't permute the wrong BN, which would break equivalence
        return None, None
    
    def _shuffle_conv_kernels(self, model, rng_pack):
        """
        Step 1: Shuffle with cross-layer synchronization and per-channel offset scaling
        
        Fix 1: Cross-layer shuffle synchronization (including bias and BN)
        - When shuffling layer L's out_channels, also shuffle:
          * Layer L's bias (if exists)
          * Layer L+1's in_channels
          * BN layers following L (weight, bias, running_mean, running_var)
        
        Fix 3: Per-channel offset scaling (Formula 4) - NOT whole-layer scaling
        - For each channel c: apply α_c to current layer, 1/α_c to next layer
        - This maintains functional equivalence while introducing attack
        
        Note: This modifies model in-place. Caller should use a copy.
        """
        with torch.no_grad():
            layers = self._get_ordered_layers(model)
            
            if len(layers) == 0:
                return
            
            i = 0
            while i < len(layers):
                name, param, layer_type, module = layers[i]
                
                # Skip BN layers (they'll be handled when processing their preceding layer)
                if layer_type == 'bn':
                    i += 1
                    continue
                
                # Decide whether to attack this layer
                if rng_pack.random.random() >= self.shuffle_prob:
                    i += 1
                    continue
                
                if layer_type == 'conv':
                    # Conv layer: [out_channels, in_channels, H, W]
                    if param.shape[0] > 1:  # Need at least 2 channels to shuffle
                        out_channels = param.shape[0]
                        # Generate permutation for output channels
                        # Handle device mismatch: if param is on CUDA but generator is on CPU
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            # Create permutation on CPU first, then move to target device
                            perm = torch.randperm(out_channels, generator=rng_pack.torch, device='cpu')
                            perm = perm.to(param.device)
                        else:
                            perm = torch.randperm(out_channels, generator=rng_pack.torch, device=param.device)
                        
                        # Shuffle current layer's output channels (weight)
                        param.data = param.data[perm]
                        
                        # Fix 1: Shuffle bias if exists
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data[perm]
                        
                        # Fix 1: Find and shuffle BN layer if exists after this conv
                        # Fix B: Pass out_channels for verification
                        bn_module, bn_name = self._find_bn_after_layer(model, name, layers, i, out_channels)
                        if bn_module is not None:
                            # Channel count already verified in _find_bn_after_layer
                            # Shuffle BN parameters: weight, bias, running_mean, running_var
                            if bn_module.weight is not None:
                                bn_module.weight.data = bn_module.weight.data[perm]
                            if bn_module.bias is not None:
                                bn_module.bias.data = bn_module.bias.data[perm]
                            if hasattr(bn_module, 'running_mean') and bn_module.running_mean is not None:
                                bn_module.running_mean.data = bn_module.running_mean.data[perm]
                            if hasattr(bn_module, 'running_var') and bn_module.running_var is not None:
                                bn_module.running_var.data = bn_module.running_var.data[perm]
                        
                        # Fix A: Synchronize next layer's input channels
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'conv':
                                # Next conv layer: shuffle input channels (dimension 1)
                                if next_param.shape[1] == out_channels:  # Match channel count
                                    next_param.data = next_param.data[:, perm, :, :]
                            
                            elif next_type == 'linear':
                                # Next linear layer: [out_features, in_features]
                                if next_param.shape[1] == out_channels:
                                    next_param.data = next_param.data[:, perm]
                        
                        # Fix 3: Per-channel offset scaling (NOT whole-layer)
                        # Generate per-channel scaling factors
                        # Fix 3: Ensure device and dtype consistency
                        # Handle device mismatch: if param is on CUDA but generator is on CPU
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            # Create on CPU first, then move to target device
                            alphas = torch.rand(out_channels, generator=rng_pack.torch, device='cpu', dtype=param.dtype)
                            alphas = alphas.to(param.device)
                        else:
                            alphas = torch.rand(out_channels, generator=rng_pack.torch, 
                                               device=param.device, dtype=param.dtype)
                        alphas = alphas * (self.scaling_factor_range[1] - self.scaling_factor_range[0]) + self.scaling_factor_range[0]
                        
                        # Fix 2: If not strict equivalence, only apply to subset of channels
                        # This introduces slight inconsistency for stronger attack
                        if not self.strict_equivalence and rng_pack.random.random() < 0.3:
                            # 30% chance to skip scaling for some channels (introduce inconsistency)
                            if param.device.type == 'cuda' and rng_pack.torch is not None:
                                mask = torch.rand(out_channels, generator=rng_pack.torch, device='cpu') > 0.2
                                mask = mask.to(param.device)
                            else:
                                mask = torch.rand(out_channels, generator=rng_pack.torch, device=param.device) > 0.2
                            alphas = torch.where(mask, alphas, torch.ones_like(alphas))
                        
                        # Apply per-channel α to current layer (Conv2d)
                        # param shape: [out_channels, in_channels, H, W]
                        # Vectorized version for efficiency
                        alphas_expanded = alphas.view(out_channels, 1, 1, 1)  # [C, 1, 1, 1] for broadcasting
                        param.data *= alphas_expanded
                        
                        # Apply per-channel 1/α to next layer
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'conv' and next_param.shape[1] == out_channels:
                                # Next conv: [out_channels_next, in_channels, H, W]
                                # Apply 1/α to input channels (dimension 1)
                                inv_alphas = (1.0 / alphas).view(1, out_channels, 1, 1)  # [1, C, 1, 1]
                                next_param.data *= inv_alphas
                            elif next_type == 'linear' and next_param.shape[1] == out_channels:
                                # Next linear: [out_features, in_features]
                                # Apply 1/α to input features (dimension 1)
                                # CRITICAL: Linear uses [1, C] not [1, C, 1, 1]
                                inv_alphas = (1.0 / alphas).view(1, out_channels)  # [1, C] for [out2, in2]
                                next_param.data *= inv_alphas
                
                elif layer_type == 'linear':
                    # Linear layer: [out_features, in_features]
                    if param.shape[0] > 1:  # Need at least 2 output features
                        out_features = param.shape[0]
                        # Handle device mismatch: if param is on CUDA but generator is on CPU
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            # Create permutation on CPU first, then move to target device
                            perm = torch.randperm(out_features, generator=rng_pack.torch, device='cpu')
                            perm = perm.to(param.device)
                        else:
                            perm = torch.randperm(out_features, generator=rng_pack.torch, device=param.device)
                        
                        # Shuffle current layer's output features (weight)
                        param.data = param.data[perm]
                        
                        # Fix 1: Shuffle bias if exists
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.data = module.bias.data[perm]
                        
                        # Fix A: Synchronize next layer's input features
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'linear':
                                if next_param.shape[1] == out_features:  # Match feature count
                                    next_param.data = next_param.data[:, perm]
                        
                        # Fix 3: Per-channel offset scaling (Linear layer)
                        # Fix 3: Ensure device and dtype consistency
                        # Handle device mismatch: if param is on CUDA but generator is on CPU
                        if param.device.type == 'cuda' and rng_pack.torch is not None:
                            # Create on CPU first, then move to target device
                            alphas = torch.rand(out_features, generator=rng_pack.torch, device='cpu', dtype=param.dtype)
                            alphas = alphas.to(param.device)
                        else:
                            alphas = torch.rand(out_features, generator=rng_pack.torch,
                                               device=param.device, dtype=param.dtype)
                        alphas = alphas * (self.scaling_factor_range[1] - self.scaling_factor_range[0]) + self.scaling_factor_range[0]
                        
                        # Fix 2: If not strict equivalence, allow slight inconsistency
                        if not self.strict_equivalence and rng_pack.random.random() < 0.3:
                            if param.device.type == 'cuda' and rng_pack.torch is not None:
                                mask = torch.rand(out_features, generator=rng_pack.torch, device='cpu') > 0.2
                                mask = mask.to(param.device)
                            else:
                                mask = torch.rand(out_features, generator=rng_pack.torch, device=param.device) > 0.2
                            alphas = torch.where(mask, alphas, torch.ones_like(alphas))
                        
                        # Apply per-channel α to current layer (Linear)
                        # param shape: [out_features, in_features]
                        # Vectorized version: CRITICAL - Linear uses [F, 1] not [F, 1, 1, 1]
                        alphas_expanded = alphas.view(out_features, 1)  # [F, 1] for [out, in]
                        param.data *= alphas_expanded
                        
                        # Apply per-channel 1/α to next layer (Linear)
                        if i + 1 < len(layers):
                            next_name, next_param, next_type, next_module = layers[i + 1]
                            if next_type == 'linear' and next_param.shape[1] == out_features:
                                # Next linear: [out_features_next, in_features]
                                # Apply 1/α to input features (dimension 1)
                                # CRITICAL: Linear uses [1, F] not [1, F, 1, 1]
                                inv_alphas = (1.0 / alphas).view(1, out_features)  # [1, F] for [out2, in2]
                                next_param.data *= inv_alphas
                
                i += 1
    
    def _update_history(self, current_model, previous_model=None, round_num=0):
        """
        Update history of global models and compute update norms
        
        Fix 4: Track warmup period and use early rounds for boundary estimation
        """
        if previous_model is not None:
            # Compute model update (difference)
            update_norm = 0.0
            for (name1, param1), (name2, param2) in zip(
                current_model.named_parameters(), 
                previous_model.named_parameters()
            ):
                if name1 == name2:
                    update_norm += torch.norm(param1.data - param2.data, p=2).item() ** 2
            update_norm = math.sqrt(update_norm)
            self.history_updates.append(update_norm)
            
            # Fix 4: Mark warmup complete after warmup_rounds
            if round_num >= self.warmup_rounds and not self.warmup_complete:
                self.warmup_complete = True
                # Store warmup boundary estimate
                if len(self.history_updates) > 0:
                    warmup_updates = self.history_updates[:self.warmup_rounds]
                    if len(warmup_updates) > 0:
                        self.init_scale = max(warmup_updates)
        
        # Store current model state (deep copy of parameters)
        model_state = {}
        for name, param in current_model.named_parameters():
            model_state[name] = param.data.clone()
        self.history_global_models.append(model_state)
        
        # Keep only recent history (last 100 rounds to save memory)
        if len(self.history_global_models) > 100:
            self.history_global_models.pop(0)
            if len(self.history_updates) > 0:
                self.history_updates.pop(0)
    
    def _compute_clipping_bounds(self, use_percentile=True, lower_percentile=10, upper_percentile=90):
        """
        Step 2: Compute R_min and R_max from historical updates
        Based on Algorithm 2, lines 6-7
        
        Fix 4 & 5: Use warmup-based boundaries + percentile to avoid pollution
        - Use warmup rounds (before attack) to establish baseline
        - Use percentiles to avoid extreme value pollution
        """
        # Fix 4: If warmup complete, prefer warmup-based estimate
        if self.warmup_complete and self.init_scale is not None:
            # Use warmup estimate as baseline, with slow adaptation
            warmup_updates = self.history_updates[:self.warmup_rounds] if len(self.history_updates) > self.warmup_rounds else self.history_updates
            if len(warmup_updates) > 0:
                warmup_min = min(warmup_updates)
                warmup_max = max(warmup_updates)
                
                # Use warmup bounds with slight expansion for robustness
                R_min = warmup_min * 0.8
                R_max = warmup_max * 1.2
                
                # Optionally blend with recent updates (but weight warmup more)
                if len(self.history_updates) > self.warmup_rounds:
                    recent_updates = self.history_updates[self.warmup_rounds:]
                    if use_percentile and len(recent_updates) >= 5:
                        sorted_recent = sorted(recent_updates)
                        lower_idx = int(len(sorted_recent) * lower_percentile / 100)
                        upper_idx = int(len(sorted_recent) * upper_percentile / 100)
                        recent_min = sorted_recent[max(0, lower_idx)]
                        recent_max = sorted_recent[min(len(sorted_recent) - 1, upper_idx)]
                        
                        # Blend: 70% warmup, 30% recent (to avoid being too rigid)
                        R_min = 0.7 * R_min + 0.3 * recent_min
                        R_max = 0.7 * R_max + 0.3 * recent_max
                
                return R_min, R_max
        
        # Fallback: standard computation
        if len(self.history_updates) < 2:
            # Not enough history - use adaptive estimate
            if self.init_scale is not None:
                R_min = self.init_scale * 0.1
                R_max = self.init_scale * 2.0
            elif len(self.history_updates) == 1:
                single_update = self.history_updates[0]
                R_min = single_update * 0.5
                R_max = single_update * 1.5
            else:
                R_min = 0.001
                R_max = 0.1
        else:
            if use_percentile and len(self.history_updates) >= 10:
                # Fix 5: Use percentiles to avoid extreme value pollution
                sorted_updates = sorted(self.history_updates)
                lower_idx = int(len(sorted_updates) * lower_percentile / 100)
                upper_idx = int(len(sorted_updates) * upper_percentile / 100)
                R_min = sorted_updates[max(0, lower_idx)]
                R_max = sorted_updates[min(len(sorted_updates) - 1, upper_idx)]
            else:
                R_min = min(self.history_updates)
                R_max = max(self.history_updates)
            
            # Store initial scale only during warmup
            if self.init_scale is None and not self.warmup_complete and len(self.history_updates) <= self.warmup_rounds:
                self.init_scale = R_max
        
        return R_min, R_max
    
    def _clip_gradient(self, grad_norm, R_min, R_max):
        """
        Apply scaled clipping according to formula (5) in the paper
        γ∇^T = {
            γ∇^T, if R_min ≤ ||∇^T||_2 ≤ R_max, γ = 1
            γ∇^T, if ||∇^T||_2 < R_min, γ = R_min / ||∇^T||_2
            γ∇^T, if ||∇^T||_2 > R_max, γ = R_max / ||∇^T||_2
        }
        """
        if grad_norm < R_min:
            gamma = R_min / (grad_norm + 1e-8)
        elif grad_norm > R_max:
            gamma = R_max / (grad_norm + 1e-8)
        else:
            gamma = 1.0
        
        return gamma
    
    def run(self, features, targets, model=None, rng_pack: RngPackage = RngPackage()):
        """
        Execute HisMSA attack - Step 1: Model Shuffling and Scaling
        
        Fix 2: Do NOT modify global_model in-place. This method should be called
        with a client copy of the model, not the server's global model object.
        
        Note: This is model poisoning, not data poisoning. Features/targets are returned unchanged.
        
        WARNING: This method modifies the model in-place. The caller MUST pass a copy:
            client_model = copy.deepcopy(global_model)
            features, targets = attack.run(features, targets, model=client_model, rng_pack=rng_pack)
        """
        if model is None:
            return features, targets
        
        # Step 1: Apply shuffle and scaling
        # Use model id to track - but apply attack consistently per round
        model_id = id(model)
        
        # Apply attack if not already shuffled for this model instance
        # (In practice, each round gets a fresh model copy, so this ensures consistency)
        if model_id not in self.shuffled_models:
            if rng_pack.random.random() < self.shuffle_prob:
                self._shuffle_conv_kernels(model, rng_pack)
                self.shuffled_models[model_id] = True
        
        # Return original data (HisMSA is model poisoning, not data poisoning)
        return features, targets
    
    def apply_step1_to_model(self, model, rng_pack: RngPackage = RngPackage()):
        """
        Apply Step1 (shuffle + scaling) to a model copy.
        This is a safer interface that makes it clear a copy should be used.
        
        Args:
            model: The model to attack (will be modified in-place)
            rng_pack: Random number generator package
            
        Returns:
            The same model object (modified in-place)
            
        Usage:
            client_model = copy.deepcopy(global_model)
            attack.apply_step1_to_model(client_model, rng_pack)
            # Now train on client_model
        """
        model_id = id(model)
        if model_id not in self.shuffled_models:
            if rng_pack.random.random() < self.shuffle_prob:
                self._shuffle_conv_kernels(model, rng_pack)
                self.shuffled_models[model_id] = True
        return model
    
    def get_clipping_factor(self, grad_update_norm):
        """
        Get the clipping factor γ for a given gradient update norm
        This should be called at the algorithm level after computing gradients
        
        Returns:
            gamma: Scaling factor to apply to the update vector
        """
        R_min, R_max = self._compute_clipping_bounds()
        gamma = self._clip_gradient(grad_update_norm, R_min, R_max)
        return gamma
    
    def apply_clipping_to_update(self, local_model, global_model):
        """
        Apply γ clipping directly to the update vector (Fix C)
        
        This implements: w_attack^{T+1} = W^T + γ(w_attack^{T+1} - W^T)
        According to Formula (5) and Algorithm 2 line 9
        
        Args:
            local_model: The local (potentially malicious) model after training
            global_model: The global model W^T
            
        Returns:
            clipped_model: The model with clipped update applied (modifies local_model in-place)
        """
        # Compute update vector: ∇^T = w_attack^{T+1} - W^T
        # And compute its L2 norm
        update_norm_sq = 0.0
        
        for (name1, param1), (name2, param2) in zip(
            local_model.named_parameters(),
            global_model.named_parameters()
        ):
            if name1 == name2:
                delta = param1.data - param2.data
                update_norm_sq += torch.norm(delta, p=2).item() ** 2
        
        update_norm = math.sqrt(update_norm_sq)
        
        # Get clipping factor γ
        gamma = self.get_clipping_factor(update_norm)
        
        # Apply clipping: w_attack = W^T + γ∇^T
        # This modifies local_model in-place
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                local_model.named_parameters(),
                global_model.named_parameters()
            ):
                if name1 == name2:
                    # Compute delta
                    delta = param1.data - param2.data
                    # Apply: w = W + γ(w - W)
                    param1.data.copy_(param2.data + gamma * delta)
        
        return local_model
    
    def update_model_history(self, current_model, previous_model=None, round_num=0):
        """
        Update history of global models
        This should be called at the algorithm level after each global update
        
        Args:
            current_model: Current global model
            previous_model: Previous global model (None for first round)
            round_num: Current round number (for warmup tracking)
        """
        self._update_history(current_model, previous_model, round_num)
    
    def test_step1_equivalence(self, model, test_input, tolerance=1e-5, verbose=False):
        """
        Unit test for Step1: Verify that shuffle+scaling maintains approximately 
        function-preserving transformation (up to numerical noise)
        
        This is the minimal unit test suggested to catch 80% of bugs.
        If this test fails, it indicates BN synchronization or layer matching issues.
        
        Note: Even with correct implementation, perfect equivalence is impossible due to:
        - Floating-point numerical errors
        - BN running statistics (if model was in training mode)
        - Dropout randomness (if enabled)
        - Other non-deterministic operations
        
        Args:
            model: Original model
            test_input: Test input tensor
            tolerance: Maximum allowed difference in outputs (default 1e-5)
            verbose: If True, print detailed information
            
        Returns:
            (max_diff, mean_diff, passed): Maximum difference, mean difference, and pass status
        """
        import copy
        import torch
        
        # Create two copies
        model0 = copy.deepcopy(model)
        model1 = copy.deepcopy(model)
        
        # Get original output
        model0.eval()
        with torch.no_grad():
            output0 = model0(test_input)
        
        # Apply Step1 to model1
        self.apply_step1_to_model(model1, self.rng_pack)
        
        # Get shuffled output
        model1.eval()
        with torch.no_grad():
            output1 = model1(test_input)
        
        # Compute differences
        diff = torch.abs(output1 - output0)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        passed = max_diff < tolerance
        
        if verbose or not passed:
            print(f"Step1 Equivalence Test:")
            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            print(f"  Tolerance: {tolerance:.6e}")
            print(f"  Status: {'PASSED' if passed else 'FAILED'}")
            if not passed:
                print(f"  WARNING: Step1 may have broken functional equivalence!")
                print(f"  This could indicate:")
                print(f"    - BN synchronization failed (wrong BN matched)")
                print(f"    - Layer order mismatch (not true forward order)")
                print(f"    - Missing layer synchronization")
        
        return max_diff, mean_diff, passed
