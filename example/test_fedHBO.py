import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
from example import FedHBO
import os

task = '../my_task'
if not os.path.exists(task):
    flgo.gen_task('../gen_config.yml', task_path=task)

runner_fedavg = flgo.init(task, fedavg, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner_fedprox = flgo.init(task, fedavg, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner_FedHBO = flgo.init(task, FedHBO, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner_fedavg.run()
runner_fedprox.run()
runner_FedHBO.run()

flgo.experiment.analyzer.show('./res_fedhbo.yml')

