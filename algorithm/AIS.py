import numpy as np
# 定义目标函数，这里以Rosenbrock函数为例，可以根据实际需要进行更换
def rosenbrock(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
# 定义人工免疫系统算法
def artificial_immune_system(max_iter, pop_size, num_features, mutation_rate):
    # 初始化种群
    population = np.random.uniform(low=-2, high=2, size=(pop_size, num_features))
    fitness = np.zeros(pop_size)
    # 迭代优化
    for iter in range(max_iter):
        # 计算适应度
        for i in range(pop_size):
            fitness[i] = rosenbrock(population[i])
        # 选择操作（锦标赛选择）
        tournament_size = 2
        selected_indices = np.zeros(pop_size, dtype=int)
        for i in range(pop_size):
            tournament_indices = np.random.choice(pop_size, size=tournament_size, replace=False)
            selected_indices[i] = tournament_indices[np.argmin(fitness[tournament_indices])]
        # 免疫进化（变异操作）
        for i in range(pop_size):
            mutant = population[selected_indices[i]].copy()
            for j in range(num_features):
                if np.random.rand() < mutation_rate:
                    mutant[j] += np.random.uniform(-0.1, 0.1)
            # 选择更优的个体作为下一代
            if rosenbrock(mutant) < fitness[selected_indices[i]]:
                population[selected_indices[i]] = mutant
    # 返回最优解
    best_index = np.argmin(fitness)
    best_solution = population[best_index]
    best_fitness = fitness[best_index]
    return best_solution, best_fitness
# 设置参数并运行算法
max_iter = 100
pop_size = 50
num_features = 10
mutation_rate = 0.1
best_solution, best_fitness = artificial_immune_system(max_iter, pop_size, num_features, mutation_rate)
print("最优解：", best_solution)
print("最优适应度：", best_fitness)

# import numpy as np
# import matplotlib.pyplot as plt
# # 生成信号数据
# t = np.linspace(0, 2*np.pi, 1000)
# signal = np.sin(2*t) + np.sin(3*t)
# # 定义人工免疫系统算法
# def artificial_immune_system(signal, max_iter, pop_size, num_antibodies, mutation_rate):
#     # 初始化抗体群体
#     antibodies = np.random.uniform(low=-1, high=1, size=(num_antibodies, len(signal)))
#     # 迭代优化
#     for iter in range(max_iter):
#         # 计算抗体的亲和力（与信号的相似度）
#         affinity = np.zeros(num_antibodies)
#         for i in range(num_antibodies):
#             affinity[i] = np.sum(np.abs(signal - antibodies[i]))
#         # 选择操作（锦标赛选择）
#         tournament_size = 2
#         selected_indices = np.zeros(pop_size, dtype=int)
#         for i in range(pop_size):
#             tournament_indices = np.random.choice(num_antibodies, size=tournament_size, replace=False)
#             selected_indices[i] = tournament_indices[np.argmin(affinity[tournament_indices])]
#         # 免疫进化（变异操作）
#         for i in range(pop_size):
#             mutant = antibodies[selected_indices[i]].copy()
#             for j in range(len(signal)):
#                 if np.random.rand() < mutation_rate:
#                     mutant[j] += np.random.uniform(-0.1, 0.1)
#             # 选择更优的抗体作为下一代
#             if np.sum(np.abs(signal - mutant)) < affinity[selected_indices[i]]:
#                 antibodies[selected_indices[i]] = mutant
#     # 返回最优解（信号重建）
#     best_index = np.argmin(affinity)
#     best_signal = antibodies[best_index]
#     return best_signal
# # 设置参数并运行算法
# max_iter = 1000
# pop_size = 50
# num_antibodies = 100
# mutation_rate = 0.1
# reconstructed_signal = artificial_immune_system(signal, max_iter, pop_size, num_antibodies, mutation_rate)
# # 绘制原始信号和重建信号的对比图
# plt.plot(t, signal, label='Original Signal')
# plt.plot(t, reconstructed_signal, label='Reconstructed Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()