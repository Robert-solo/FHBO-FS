import copy
import random
import time
import numpy as np
import torch
import importlib
from flgo.algorithm.fedbase import BasicServer, BasicClient
from flgo.utils import fmodule

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu': 0.2})  #mu为超参数，设定为0.1


class Client(BasicClient):
    @fmodule.with_multi_gpus
    def train(self, model):
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        model.train()

        if self.calculator.optimizer_name.lower() == 'bhbo':
            optimizer = self.calculator.get_optimizer(model=None)  # 创建 BHBO 优化器
            for iter in range(self.num_steps):
                optimizer._optimize(iter)  # 调用 BHBO 优化方法
        else:
            optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                      momentum=self.momentum)
            for iter in range(self.num_steps):
                batch_data = self.get_batch_data()
                model.zero_grad()
                loss = self.calculator.compute_loss(model, batch_data)['loss']
                loss_proximal = sum(
                    torch.sum(torch.pow(pm - ps, 2)) for pm, ps in zip(model.parameters(), src_model.parameters()))
                loss = loss + 0.5 * self.mu * loss_proximal
                loss.backward()
                optimizer.step()

# class Client(BasicClient):
#     @fmodule.with_multi_gpus
#     def train(self, model):
#         src_model = copy.deepcopy(model)
#         src_model.freeze_grad()
#         model.train()
#         optimizer = self.calculator.get_optimizer(model)
#         for iter in range(self.num_steps):
#             batch_data = self.get_batch_data()
#             model.zero_grad()
#             loss = self.calculator.compute_loss(model, batch_data)['loss']
#             loss_proximal = sum(torch.sum(torch.pow(pm - ps, 2)) for pm, ps in zip(model.parameters(), src_model.parameters()))
#             loss = loss + 0.5 * self.mu * loss_proximal
#             loss.backward()
#             optimizer.step()

class BHBO:
    def __init__(self, objf, lb, ub, dim, iters, PopSize):
        self.groupSize = PopSize // 3
        self.max_trial = 10
        self.population = []
        self.Convergence_curve1 = np.zeros(iters)
        self.Convergence_curve2 = np.zeros(iters)
        self.Convergence_curve3 = np.zeros(iters)
        self.objf = objf
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.iters = iters
        self.PopSize = PopSize

        for _ in range(PopSize):
            rice = Individual(lb, ub, dim)
            rice.calculate_fitness(objf)
            self.population.append(rice)

        self.population.sort(key=lambda s: s.fitness)
        self.best_rice = copy.deepcopy(self.population[0])
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        for l in range(iters):
            self._optimize(l)

    def _optimize(self, l):
        groupSize = self.groupSize
        population = self.population
        best_rice = self.best_rice

        for i in range(2 * groupSize, self.PopSize):
            trial_rice = Individual(self.lb, self.ub, self.dim)
            r1 = 2 * np.random.random(self.dim) - 1
            r2 = 2 * np.random.random(self.dim) - 1

            maintainer = population[self._select(0, groupSize, size=1, excludes=[i])]
            sterile = population[self._select(2 * groupSize, self.PopSize, size=1, excludes=[i])]
            trial_rice.vector = (r1 * maintainer.vector + r2 * maintainer.vector) / (r1 + r2)
            trial_rice.exchange_binary()
            trial_rice.fitness = np.mean(self.objf(trial_rice.position))  # 取数组的平均值作为适应度值
            if trial_rice.fitness < population[i].fitness:
                population[i] = copy.deepcopy(trial_rice)
                if trial_rice.fitness < best_rice.fitness:
                    best_rice = copy.deepcopy(trial_rice)

        for i in range(groupSize, 2 * groupSize):
            trial_rice = Individual(self.lb, self.ub, self.dim)
            if population[i].trial < self.max_trial:
                neighbor = population[self._select(groupSize, 2 * groupSize, size=1, excludes=[i])]
                r3 = random.random()
                trial_rice.vector = r3 * (best_rice.vector - neighbor.vector) + population[i].vector
                trial_rice.exchange_binary()
                trial_rice.fitness = np.mean(self.objf(trial_rice.position))  # 取数组的平均值作为适应度值
                if trial_rice.fitness < population[i].fitness:
                    population[i] = copy.deepcopy(trial_rice)
                    if trial_rice.fitness < best_rice.fitness:
                        best_rice = copy.deepcopy(trial_rice)
                else:
                    population[i].trial += 1
            else:
                r4 = random.random()
                for j in range(self.dim):
                    population[i].vector[j] = r4 * (self.ub - self.lb) + population[i].vector[j] + self.lb
                population[i].exchange_binary()
                population[i].fitness = np.mean(self.objf(population[i].position))  # 取数组的平均值作为适应度值
                population[i].trial = 0
                if population[i].fitness < best_rice.fitness:
                    best_rice = copy.deepcopy(population[i])

        population.sort(key=lambda s: s.fitness)
        self.Convergence_curve1[l] = best_rice.fitness
        self.Convergence_curve2[l] = best_rice.test_acc
        self.Convergence_curve3[l] = sum(best_rice.position)

    def _select(self, start, end, size=2, excludes=None):
        l = list(range(start, end))
        if excludes is not None:
            l = [i for i in l if i not in excludes]
        res = random.choices(l, k=size)
        if size == 1:
            return res[0]
        return res

class Individual:
    def __init__(self, lb, ub, dim):
        self.vector = np.random.uniform(lb, ub, size=dim)
        self.position = np.random.randint(0, 2, size=dim)
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.fitness = 0
        self.train_acc = 0
        self.test_acc = 0
        self.kappa = None
        self.trial = 0

    def exchange_binary(self, transfer_func=None, use_random=False):
        self.vector = np.clip(self.vector, self.lb, self.ub)
        transfer_func = transfer_functions_benchmark.s2 if not transfer_func else transfer_func
        if not use_random:
            self.position = np.round(transfer_func(self.vector)).astype(int)
        else:
            self.position = (transfer_func(self.vector) > np.random.random(self.dim)).astype(int)
        self.examine_all_zero()

    def examine_all_zero(self):
        if not np.any(self.position):
            self.position = np.random.randint(0, 2, size=self.dim)

    def calculate_fitness(self, objf):
        fitness_values = objf(self.position)
        if isinstance(fitness_values, (int, float)):
            self.fitness = fitness_values
        else:
            self.fitness = np.mean(fitness_values)

class transfer_functions_benchmark:
    @staticmethod
    def s2(x):  #encoder function
        return 1 / (1 + np.exp(-x))
    def s1(self, x): #decoder function
        return -(np.log((1/x)-1))

class Solution:
    def __init__(self, optimizer_name='adam'):
        self.optimizer_name = optimizer_name

    def get_optimizer(self, model, lr=0.1, weight_decay=0, momentum=0):
        r"""
        Create optimizer of the model parameters

        Args:
            model (torch.nn.Module): model
            lr (float): learning rate
            weight_decay (float): the weight_decay coefficient
            momentum (float): the momentum coefficient

        Returns:
            the optimizer
        """
        if model is None:
            raise ValueError("Model cannot be None.")

        if self.optimizer_name.lower() == 'bhbo':
            # 初始化相关参数
            def objf(x): #定义目标函数
                fitness_value =  0.01 * np.exp(x) + 0.99 * 1 / (1+np.exp(x))
                return fitness_value
            lb = -5
            ub = 5
            dim = 10
            iters = 100
            # ds_name = ...
            PopSize = 30
            return BHBO(objf, lb, ub, dim, iters, PopSize)  # 创建 BHBO 优化器
        else:
            OPTIM = getattr(importlib.import_module('torch.optim'), self.optimizer_name)
            filter_fn = filter(lambda p: p.requires_grad, model.parameters())
            if self.optimizer_name.lower() == 'sgd':
                return OPTIM(filter_fn, lr=lr, momentum=momentum, weight_decay=weight_decay)
            elif self.optimizer_name.lower() in ['adam', 'rmsprop', 'adagrad']:
                return OPTIM(filter_fn, lr=lr, weight_decay=weight_decay)
            else:
                raise RuntimeError("Invalid Optimizer.")
