import torch
import random
import copy
from ByrdLab import DEVICE
from ByrdLab.environment import  Dist_Dataset_Opt_Env
from ByrdLab.attack import DataPoisoningAttack, C_bit_flipping
from ByrdLab.library.dataset import EmptySet
from ByrdLab.library.partition import EmptyPartition
from ByrdLab.library.measurements import avg_loss_accuracy_dist, consensus_error, one_node_loss_accuracy_dist
from ByrdLab.library.tool import log, flatten_list, unflatten_vector, flatten_vector
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache



# CSGD under model poisoning attacks
class CSGD(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)
            
            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    
class CMomentum_compute_hetero(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        worker_full_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        hetero_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                

            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            # communication and attack
            if self.attack != None and self.byzantine_size!= 0:
                self.attack.run(worker_grad_flat)

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                worker_full_grad_flat = flatten_list(worker_full_grad)
                avg_grad_flat = torch.mean(worker_full_grad_flat[self.honest_nodes], dim=0)
                distances = torch.tensor([torch.norm(worker_full_grad_flat[node] - avg_grad_flat) for node in self.honest_nodes])
                heterogeneity = distances.max()
                print('Heterogeneity:', heterogeneity.item())
                hetero_list.append(heterogeneity.item())

        return server_model, loss_path, acc_path, hetero_list


# CSGD under model poisoning attacks
class CSGD_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    
class CMomentum_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)
        # alpha = 0.01

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        # worker_grad = [
        #     [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
        #     for _ in range(self.node_size)
        # ]

        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]
        # byzantine-robust-optimizer: honest send momentum, Byzantine (BF) send -raw_gradient
        use_bf_raw_grad = (
            self.attack is not None and self.byzantine_size != 0
            and isinstance(self.attack, C_bit_flipping)
        )
        if use_bf_raw_grad:
            worker_raw_grad = [
                [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
                for _ in range(self.node_size)
            ]

        iteration = 0
        while iteration <= self.total_iterations:
            lr = self.lr_ctrl.get_lr(iteration)
            accuracy_before_this_round = None
            if iteration % self.display_interval == 0:
                accuracy_before_this_round = acc_path[-1] if len(acc_path) > 0 else None
                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                if hasattr(self.aggregation, 'set_accuracy'):
                    self.aggregation.set_accuracy(test_accuracy)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            for node in self.nodes:
                features, targets = next(data_iters[node])
                if node in self.byzantine_nodes and isinstance(self.attack, DataPoisoningAttack):
                    features, targets = self.attack.run(features, targets, model=server_model)
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                    if use_bf_raw_grad and node in self.byzantine_nodes:
                        for index, para in enumerate(server_model.parameters()):
                            # Paper TorchWorker: only save p.grad (no weight_decay in optimizer)
                            worker_raw_grad[node][index].data.copy_(para.grad.data)
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                    if use_bf_raw_grad and node in self.byzantine_nodes:
                        for index, para in enumerate(server_model.parameters()):
                            worker_raw_grad[node][index].data.copy_(para.grad.data)

            if use_bf_raw_grad:
                mixed = [
                    worker_raw_grad[node] if node in self.byzantine_nodes else worker_momentum[node]
                    for node in self.nodes
                ]
                worker_grad_flat = flatten_list(mixed)
            else:
                worker_grad_flat = flatten_list(worker_momentum)
            if self.attack is not None and self.byzantine_size != 0 and not isinstance(self.attack, DataPoisoningAttack):
                self.attack.run(worker_grad_flat)

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)
            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha=lr)

            iteration += 1

        if hasattr(self.aggregation, 'flush_log_to_file'):
            self.aggregation.flush_log_to_file()
        return server_model, loss_path, acc_path
    
class CMomentum_under_DPA_compute_bound(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        worker_full_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        Bound_A_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                worker_full_grad_flat = flatten_list(worker_full_grad)
                avg_regular_grad = torch.mean(worker_full_grad_flat[self.honest_nodes], dim=0)
                grad_norms = torch.tensor([torch.norm(worker_full_grad_flat[node] - avg_regular_grad) for node in self.byzantine_nodes]) 
                Bound_A_max = grad_norms.max()
                print(f'{iteration}-iteration Bound_A:', Bound_A_max.item())
                Bound_A_list.append(Bound_A_max.item())  

        return server_model, loss_path, acc_path, Bound_A_list

class CMomentum_under_DPA_compute_variance(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        variances_regular_list = []
        variances_poison_list = []

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                if iteration % self.display_interval == 0:
                    loss.backward(retain_graph=True)
                else: 
                    loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                worker_full_grad = [
                    [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
                        for _ in range(self.node_size)
                ]
                
                variances = torch.zeros(self.node_size)
                variances = variances.to(DEVICE)

                for node in self.nodes:
                    length = len(self.dist_train_set[node])
                    features_2, targets_2 = self.dist_train_set[node][:length] 
                    features_2 = features_2.to(DEVICE)
                    targets_2 = targets_2.to(DEVICE)
                    predictions_2 = server_model(features_2)
                    loss = self.loss_fn(predictions_2, targets_2)
                    server_model.zero_grad()
                    loss.backward(retain_graph=True)

                    for index, para in enumerate(server_model.parameters()):
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                worker_full_grad_flat = flatten_list(worker_full_grad)

                for node in self.nodes:
                    worker_sto_grad = [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
                    length = len(self.dist_train_set[node])
                    for i in range(length):
                        feature_i, target_i = self.dist_train_set[node][i]
                        feature_i = feature_i.to(DEVICE)
                        target_i = target_i.to(DEVICE)
                        predictions_i = server_model(feature_i)
                        loss = self.loss_fn(predictions_i, target_i)
                        server_model.zero_grad()
                        loss.backward(retain_graph=True)

                        for index, para in enumerate(server_model.parameters()):
                            worker_sto_grad[index].data.zero_()
                            worker_sto_grad[index].data.add_(para.grad.data, alpha=1)

                        worker_sto_grad_flat = flatten_vector(worker_sto_grad)

                        grad_norms = torch.norm(worker_sto_grad_flat - worker_full_grad_flat[node]) ** 2
                        variances[node].add_(grad_norms, alpha = 1 / length)

                variance_regular = variances[self.honest_nodes].max()
                variance_poison = variances[self.byzantine_nodes].max()
                print(f'{iteration}-iteration maximum variance of regular stochastic gradients:', variance_regular.item())
                print(f'{iteration}-iteration maximum variance of poisoned stochastic gradients:', variance_poison.item())
                variances_regular_list.append(variance_regular.item())
                variances_poison_list.append(variance_poison.item())
                # Bound_A_list.append(Bound_A_max.item())  

        return server_model, loss_path, acc_path, variances_regular_list, variances_poison_list



class CMomentum_under_DPA_compute_hetero_bound(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        worker_full_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        Bound_A_list = []
        hetero_list = []

        iteration = 0
        while iteration <= self.total_iterations:
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            if iteration % self.display_interval == 0:
                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                if hasattr(self.aggregation, 'set_accuracy'):
                    self.aggregation.set_accuracy(test_accuracy)
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                        worker_full_grad[node][index].data.zero_()
                        worker_full_grad[node][index].data.add_(para.grad.data, alpha=1)
                
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            if iteration % self.display_interval == 0:
                worker_full_grad_flat = flatten_list(worker_full_grad)
                avg_regular_grad = torch.mean(worker_full_grad_flat[self.honest_nodes], dim=0)
                distances_bound_A = torch.tensor([torch.norm(worker_full_grad_flat[node] - avg_regular_grad) for node in self.byzantine_nodes]) 
                distances_hetero = torch.tensor([torch.norm(worker_full_grad_flat[node] - avg_regular_grad) for node in self.honest_nodes])
                Bound_A_max = distances_bound_A.max()
                hetero_max = distances_hetero.max()
                print('Bound_A:', Bound_A_max.item())
                print('Hetero:', hetero_max.item())
                Bound_A_list.append(Bound_A_max.item())
                hetero_list.append(hetero_max.item())  
            iteration += 1

        if hasattr(self.aggregation, 'flush_log_to_file'):
            self.aggregation.flush_log_to_file()
        return server_model, loss_path, acc_path, Bound_A_list, hetero_list
    

class CSGD_with_LFighter_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes,  *args, **kw):
        super().__init__(name='CSGD', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            min_norm_feature = 0
            max_norm_feature = 0
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                norm_feature = torch.norm(features)
                if norm_feature > max_norm_feature:
                    max_norm_feature = norm_feature
                if min_norm_feature == 0:
                    min_norm_feature = norm_feature
                elif norm_feature < min_norm_feature:
                    min_norm_feature = norm_feature
                assert min_norm_feature <= max_norm_feature

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                

                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_grad)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            print('minimum of features norm:', min_norm_feature.item())
            print('maximum of features norm:', max_norm_feature.item())

        return server_model, loss_path, acc_path
    
class CMomentum_with_LFighter_under_DPA(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1,  *args, **kw):
        super().__init__(name='CMomentum', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)
        # alpha = 0.1

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # data poisoning attack
                if node in self.byzantine_nodes:
                    features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_momentum)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    

# CSGD under data poisoning attacks
class CSGD_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob, *args, **kw):
        super().__init__(name=f'CSGD_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            random_number = random.uniform(0, 1)
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_grad)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    

class CSGD_with_LFighter_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob,  *args, **kw):
        super().__init__(name=f'CSGD_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CSGD]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_grad = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            # min_norm_feature = 0
            # max_norm_feature = 0
                
            random_number = random.uniform(0, 1)
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # norm_feature = torch.norm(features)
                # if norm_feature > max_norm_feature:
                #     max_norm_feature = norm_feature
                # if min_norm_feature == 0:
                #     min_norm_feature = norm_feature
                # elif norm_feature < min_norm_feature:
                #     min_norm_feature = norm_feature
                # assert min_norm_feature <= max_norm_feature

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                

                
                # store the workers' gradients
                for index, para in enumerate(server_model.parameters()):
                    worker_grad[node][index].data.zero_()
                    worker_grad[node][index].data.add_(para.grad.data, alpha=1)
                    worker_grad[node][index].data.add_(para, alpha=self.weight_decay)

            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_grad)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

            # print('minimum of features norm:', min_norm_feature.item())
            # print('maximum of features norm:', max_norm_feature.item())

        return server_model, loss_path, acc_path
    
    
# CMomentum under data poisoning attacks
class CMomentum_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob, alpha=0.1, *args, **kw):
        super().__init__(name=f'CMomentum_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

            random_number = random.uniform(0, 1)
                
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                # store the worker's momentums
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
            # the master node aggregate the stochastic gradients under Byzantine attacks
            worker_grad_flat = flatten_list(worker_momentum)    

            aggrGrad_flat = self.aggregation.run(worker_grad_flat)

            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)

        return server_model, loss_path, acc_path
    

class CMomentum_with_LFighter_under_DPA_with_prob(Dist_Dataset_Opt_Env):
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, prob, alpha=0.1, *args, **kw):
        super().__init__(name=f'CMomentum_p={prob}', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes,  *args, **kw)
        self.aggregation = aggregation
        self.prob = prob
        self.alpha = alpha
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        hint = '[CMomentum]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # initialize the stochastic gradients of all workers
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # lastest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times)
            if iteration % self.display_interval == 0:

                test_loss, test_accuracy = one_node_loss_accuracy_dist(
                    server_model, self.get_test_iter,
                    self.loss_fn, self.test_fn,
                    weight_decay=0, node_list=self.honest_nodes)
                
                loss_path.append(test_loss)
                acc_path.append(test_accuracy)
                
                log(hint.format(
                    iteration, self.total_iterations,
                    iteration / self.total_iterations * 100,
                    test_loss, test_accuracy, lr
                ))

                
            random_number = random.uniform(0, 1)
            # gradient descent
            for node in self.nodes:
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)

                # data poisoning attack
                if node in self.byzantine_nodes:
                    if random_number <= self.prob:
                        features, targets = self.attack.run(features, targets, model=server_model)

                predictions = server_model(features)
                loss = self.loss_fn(predictions, targets)
                server_model.zero_grad()
                loss.backward()
                
                # store the workers' gradients
                if iteration == 0:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                
                else:
                    for index, para in enumerate(server_model.parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)

            # the master node aggregate the stochastic gradients under Byzantine attacks  
            aggrGrad = self.aggregation.run(worker_momentum)

            # the master node update the global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha = lr)


        return server_model, loss_path, acc_path


# CMomentum under HisMSA/MSA attack (model poisoning)
class CMomentum_under_HisMSA(Dist_Dataset_Opt_Env):
    """
    CMomentum algorithm under HisMSA (History-based Model Shuffling and Scaling Attack) or MSA (Model Shuffling Attack)
    
    HisMSA is a model poisoning attack that:
    1. Step 1: Shuffles and scales model parameters before training (maintains approximate function-preserving)
    2. Step 2: Clips malicious updates using historical bounds to evade detection
    
    MSA is a simplified version that only implements Step 1 (no Step 2).
    
    This class integrates both HisMSA and MSA into the CMomentum training loop.
    """
    def __init__(self, aggregation, honest_nodes, byzantine_nodes, alpha=0.1, *args, **kw):
        # First call super().__init__ to initialize everything including self.attack
        super().__init__(name='CMomentum_HisMSA', honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, *args, **kw)
        
        self.aggregation = aggregation
        self.alpha = alpha
        
        # Check if attack is HisMSA or MSA (both use Step 1)
        if self.attack is not None and hasattr(self.attack, 'name') and self.attack.name in ['HisMSA', 'MSA']:
            self.model_shuffle_attack = self.attack
            self.has_step2 = (self.attack.name == 'HisMSA')  # Only HisMSA has Step 2
            # Update name based on attack type
            if self.attack.name == 'MSA':
                self.name = 'CMomentum_MSA'
        else:
            self.model_shuffle_attack = None
            self.has_step2 = False
            if self.attack is not None:
                raise ValueError("CMomentum_under_HisMSA requires HisMSA or MSA attack, but got: {}".format(type(self.attack)))
            
    def run(self):
        self.construct_rng_pack()
        # initialize
        server_model = self.model.to(DEVICE)
        previous_server_model = None  # For history tracking

        # initial record
        loss_path = []
        acc_path = []
        
        # log formatter
        num_len = len(str(self.total_iterations))
        num_format = '{:>' + f'{num_len}' + 'd}'
        attack_display_name = self.attack.name if self.attack is not None else 'None'
        hint = f'[CMomentum_{attack_display_name}]' + num_format + '/{} iterations ({:>6.2f}%) ' + \
            'loss={:.3e}, accuracy={:.4f}, lr={:f}'

        data_iters = [self.get_train_iter(dataset=self.dist_train_set[node],
                                          rng_pack=self.rng_pack) 
                      for node in self.nodes]
        
        # Initialize worker models (each node has a copy for local training)
        worker_models = [
            copy.deepcopy(server_model) for _ in range(self.node_size)
        ]
        
        # Initialize worker momentums
        worker_momentum = [
            [torch.zeros_like(para, requires_grad=False) for para in server_model.parameters()]
            for _ in range(self.node_size)
        ]

        for iteration in range(0, self.total_iterations + 1):
            # latest learning rate
            lr = self.lr_ctrl.get_lr(iteration)
            
            # record (totally 'rounds+1' times). For bypass: use last display acc so we detect drop since last display.
            accuracy_before_this_round = None
            if iteration % self.display_interval == 0:
                accuracy_before_this_round = acc_path[-1] if len(acc_path) > 0 else None
                try:
                    test_loss, test_accuracy = one_node_loss_accuracy_dist(
                        server_model, self.get_test_iter,
                        self.loss_fn, self.test_fn,
                        weight_decay=0, node_list=self.honest_nodes)
                    loss_path.append(test_loss)
                    acc_path.append(test_accuracy)
                    acc_for_aggregation = test_accuracy
                except Exception as e:
                    # 评估失败时用上一轮 acc，避免 HRAC 等聚合器日志里 acc 显示 N/A
                    acc_for_aggregation = acc_path[-1] if len(acc_path) > 0 else None
                    loss_path.append(loss_path[-1] if loss_path else 0.0)
                    acc_path.append(acc_for_aggregation if acc_for_aggregation is not None else 0.0)
                    log(f'[CMomentum_{self.attack.name if self.attack else "None"}] iter={iteration} eval failed: {e}, using last acc={acc_for_aggregation}')
                if hasattr(self.aggregation, 'set_accuracy'):
                    self.aggregation.set_accuracy(acc_for_aggregation)
                
                if len(loss_path) > 0 and len(acc_path) > 0:
                    log(hint.format(
                        iteration, self.total_iterations,
                        iteration / self.total_iterations * 100,
                        loss_path[-1], acc_path[-1], lr
                    ))

            # Update history (before training this round) - only for HisMSA (has Step 2)
            # CRITICAL: In CMomentum, we need to track momentum norms, not model update norms
            # because we clip momentum, not model updates
            if self.model_shuffle_attack is not None and self.has_step2 and iteration > 0:
                # Calculate round number (for warmup tracking)
                round_num = iteration // self.display_interval
                
                # For CMomentum, we should track momentum norms instead of model update norms
                # But for now, we'll track model updates and scale them appropriately
                # The model update = lr * aggregated_momentum, so momentum_norm ≈ update_norm / lr
                # Only HisMSA has update_model_history method
                if hasattr(self.model_shuffle_attack, 'update_model_history'):
                    self.model_shuffle_attack.update_model_history(
                        server_model, previous_server_model, round_num=round_num
                    )
                    previous_server_model = copy.deepcopy(server_model)  # for next round's diff
            
            # gradient descent
            for node in self.nodes:
                # Step 1: Apply Step 1 (shuffle + scaling) to malicious nodes (both MSA and HisMSA)
                if node in self.byzantine_nodes and self.model_shuffle_attack is not None:
                    # Create a fresh copy for this node (important: don't modify server_model)
                    worker_models[node] = copy.deepcopy(server_model)
                    # Apply Step 1: shuffle and scale model parameters
                    self.model_shuffle_attack.apply_step1_to_model(worker_models[node], self.rng_pack)
                else:
                    # Honest nodes use server model copy
                    worker_models[node] = copy.deepcopy(server_model)
                
                # Get training data
                features, targets = next(data_iters[node])
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                
                # Local training on worker model
                predictions = worker_models[node](features)
                loss = self.loss_fn(predictions, targets)
                worker_models[node].zero_grad()
                loss.backward()
                
                # Compute momentum from gradients
                if iteration == 0:
                    for index, para in enumerate(worker_models[node].parameters()):
                        worker_momentum[node][index].data.zero_()
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=1)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay)
                else:
                    for index, para in enumerate(worker_models[node].parameters()):
                        worker_momentum[node][index].data.mul_(1 - self.alpha)
                        worker_momentum[node][index].data.add_(para.grad.data, alpha=self.alpha)
                        worker_momentum[node][index].data.add_(para, alpha=self.weight_decay * self.alpha)
                
                # Standard MSA: Only model shuffling and scaling (Step 1)
                # No additional momentum manipulation - pure model poisoning attack
                # MSA relies solely on model shuffling to create gradient divergence
                
                # DEBUG: Compare malicious vs honest momentum (only for first few iterations)
                if node == self.byzantine_nodes[0] and iteration % self.display_interval == 0 and len(self.honest_nodes) > 0:
                    # Compute malicious momentum norm
                    mal_momentum_norm_sq = 0.0
                    for mom in worker_momentum[node]:
                        mal_momentum_norm_sq += torch.norm(mom.data, p=2).item() ** 2
                    mal_momentum_norm = (mal_momentum_norm_sq ** 0.5)
                    
                    # Compute average honest momentum norm
                    honest_momentum_norms = []
                    for h_node in self.honest_nodes[:3]:  # Sample first 3 honest nodes
                        h_norm_sq = 0.0
                        for mom in worker_momentum[h_node]:
                            h_norm_sq += torch.norm(mom.data, p=2).item() ** 2
                        honest_momentum_norms.append((h_norm_sq ** 0.5))
                    avg_honest_norm = sum(honest_momentum_norms) / len(honest_momentum_norms) if honest_momentum_norms else 0.0
                    
                    # Compute momentum direction difference (cosine similarity)
                    if len(worker_momentum[node]) > 0 and len(worker_momentum[self.honest_nodes[0]]) > 0:
                        # Flatten momentums for comparison
                        mal_mom_flat = torch.cat([mom.data.flatten() for mom in worker_momentum[node]])
                        hon_mom_flat = torch.cat([mom.data.flatten() for mom in worker_momentum[self.honest_nodes[0]]])
                        cosine_sim = torch.dot(mal_mom_flat, hon_mom_flat) / (torch.norm(mal_mom_flat) * torch.norm(hon_mom_flat) + 1e-8)
                        
                        log(f'[Step1 Debug] Iter {iteration}: mal_norm={mal_momentum_norm:.6e}, honest_norm={avg_honest_norm:.6e}, cosine_sim={cosine_sim.item():.6f}')
                
                # Step 2: Apply Step 2 (γ clipping) to malicious nodes before aggregation (only for HisMSA)
                # CRITICAL FIX: In CMomentum, the update is: server_model -= lr * aggregated_momentum
                # So momentum IS the update vector. We need to clip the momentum norm, not the model difference.
                # MSA does NOT have Step 2, so skip this for HisMSA
                if node in self.byzantine_nodes and self.model_shuffle_attack is not None and self.has_step2:
                    # Compute momentum norm (this is what will be uploaded)
                    momentum_norm_sq = 0.0
                    for mom in worker_momentum[node]:
                        momentum_norm_sq += torch.norm(mom.data, p=2).item() ** 2
                    momentum_norm = (momentum_norm_sq ** 0.5)
                    
                    # CRITICAL FIX: In CMomentum, model update = lr * momentum
                    # So momentum_norm ≈ (model_update_norm) / lr
                    # The history stores model_update_norms, so we need to scale them
                    # to get momentum_norm bounds
                    # Only HisMSA has _compute_clipping_bounds method
                    if hasattr(self.model_shuffle_attack, '_compute_clipping_bounds'):
                        R_min, R_max = self.model_shuffle_attack._compute_clipping_bounds()
                    else:
                        # Fallback (should not happen for HisMSA)
                        R_min, R_max = 0.001, 0.1
                    
                    # Scale bounds from model_update_norm to momentum_norm
                    # momentum_norm = model_update_norm / lr
                    # So: R_min_momentum = R_min_model / lr, R_max_momentum = R_max_model / lr
                    R_min_momentum = R_min / (lr + 1e-8)
                    R_max_momentum = R_max / (lr + 1e-8)
                    
                    # Apply clipping based on momentum norm bounds (Step 2 for HisMSA)
                    if momentum_norm < R_min_momentum:
                        gamma = R_min_momentum / (momentum_norm + 1e-8)
                    elif momentum_norm > R_max_momentum:
                        gamma = R_max_momentum / (momentum_norm + 1e-8)
                    else:
                        gamma = 1.0
                    
                    # DEBUG: Print gamma for first few iterations (only for first malicious node)
                    if node == self.byzantine_nodes[0] and iteration % self.display_interval == 0:
                        attack_name = self.model_shuffle_attack.name if self.model_shuffle_attack else 'Unknown'
                        log(f'[{attack_name} Debug] Iter {iteration}: Step2 active, momentum_norm={momentum_norm:.6e}, R_min={R_min:.6e}, R_max={R_max:.6e}, R_min_mom={R_min_momentum:.6e}, R_max_mom={R_max_momentum:.6e}, lr={lr:.6e}, gamma={gamma:.6f}')
                    
                    # Apply γ clipping to momentum (scale all momentum components by γ)
                    for index in range(len(worker_momentum[node])):
                        worker_momentum[node][index].data.mul_(gamma)
                
            # Aggregate momentums
            worker_grad_flat = flatten_list(worker_momentum)  
            aggrGrad_flat = self.aggregation.run(worker_grad_flat)
            aggrGrad = unflatten_vector(aggrGrad_flat, server_model)

            # Update global model
            for para, grad in zip(server_model.parameters(), aggrGrad):
                para.data.sub_(grad, alpha=lr)

            # After iter 0, save server model so HisMSA history has previous_model at next display boundary
            if iteration == 0 and self.model_shuffle_attack is not None and self.has_step2:
                previous_server_model = copy.deepcopy(server_model)

        if hasattr(self.aggregation, 'flush_log_to_file'):
            self.aggregation.flush_log_to_file()
        return server_model, loss_path, acc_path
    
    
