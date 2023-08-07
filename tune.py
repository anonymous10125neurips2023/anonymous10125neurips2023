"""Tuning hyperparameters for bandit algorithms on classification bandits. 
"""
import os
import subprocess
import time
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--models_per_gpu', type=int, default=6)
parser.add_argument('--max_models_per_gpu', type=int, default=6)
parser.add_argument('--gpus', type=int, default=[0], help='gpus indices used for multi_gpu')
parser.add_argument('--data',  type=str, default='quadratic') 
parser.add_argument('--algo', nargs ='+', type=str, default=['NeuraLMC'])
parser.add_argument('--T', type=int, default=1000)
parser.add_argument('--task', type=str, default='create_and_run_commands', choices=['create_and_run_commands'])

args = parser.parse_args()

def multi_gpu_launcher(commands,gpus,models_per_gpu):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    procs = [None]*len(gpus)*models_per_gpu

    while len(commands) > 0:
        for i,proc in enumerate(procs):
            gpu_idx = gpus[i % len(gpus)]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this index; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs[i] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs:
        if p is not None:
            p.wait()

# for the reported synthetic, lr = [0.0001, 0.001, 0.01]
lr_space = [0.1, 0.05, 0.01, 0.001] # [0.0001, 0.001] # [0.0001]
lambd_space = [0.01] #[0.001, 0.01, 0.1, 1]
wdecay_space = [1.0, 0.1, 0.01, 0.001, 0.0001]
## Best lr and lambda per dataset
# mushroom: lr = ..., lambda = ...

# NeuraLCB 
beta_space = [0.001, 0.01, 0.1, 1, 5, 10]

# NeuralPER 
perturbed_variance_space = [0.001, 0.01, 0.1, 1, 5, 10] 
# M_list = [50,100,200]

# NeuraLMC
inverse_beta_space = [0.000001, 0.00001, 0.0001, 0.001, 0.01] #[0.0001, 0.001, 0.01, 0.1, 1, 5, 10] #
# inverse_beta_space = [1] # unused 
# M_list = [1, 10, 20]
M_list = [30, 50, 100]
#-------------------------------------------------
# NeuralTS 
scaled_variance_space = [0.1, 0.01, 0.001, 0.0001]




# NeuralEpsGreedy 
epsilon_space = [0.001, 0.01, 0.1, 0.2]

def create_and_run_commands(): 
    commands = []
    for algo in args.algo:
        if algo in ['NeuraLCB', 'NeuraLCBDiag']: 
            for lr in lr_space:
                for lambd in lambd_space: 
                    for wdecay in wdecay_space:
                        for beta in beta_space: 
                            commands.append('python main.py --data {} --algo {} --learning_rate {} --reg_factor {} --beta {} --weight_decay {} --T {} --hpo'.format(args.data, algo, lr, lambd, beta, wdecay, args.T))

        elif algo == 'LinLCB': 
            for lambd in lambd_space: 
                for beta in beta_space: 
                    commands.append('python main.py --data {} --algo {} --reg_factor {} --beta {} --T {} --hpo'.format(args.data, algo, lambd, beta, args.T))
        
        elif algo == 'NeuralEpsGreedy':
            for lr in lr_space:
                for lambd in lambd_space: 
                    for epsilon in epsilon_space: 
                        commands.append('python main.py --data {} --algo {} --learning_rate {} --reg_factor {} --epsilon {} --T {} --hpo'.format(args.data, algo, lr, lambd, epsilon, args.T))

        elif algo == 'NeuralGreedy':
            for lr in lr_space:
                for lambd in lambd_space: 
                    for wdecay in wdecay_space:
                        commands.append('python main.py --data {} --algo {} --learning_rate {} --reg_factor {} --weight_decay {} --T {} --hpo'.format(args.data, algo, lr, lambd, wdecay, args.T))

        elif algo == 'NeuralTS': 
            for lr in lr_space:
                for lambd in lambd_space: 
                    for scaled_var in scaled_variance_space: 
                        commands.append('python main.py --data {} --algo {} --learning_rate {} --reg_factor {} --scaled_variance {} --T {} --hpo'.format(args.data, algo, lr, lambd, scaled_var, args.T))


        elif algo in ['NeuraLMC']: 
            for lr in lr_space:
                for lambd in lambd_space: 
                    for wdecay in wdecay_space:
                        for inverse_beta in inverse_beta_space: 
                            for M in M_list:
                                commands.append('python main.py --data {} --algo {} --learning_rate {} --reg_factor {} --inverse_beta {} --M {} --weight_decay {} --T {} --hpo'.format(args.data, algo, lr, lambd, inverse_beta, M, wdecay, args.T))
 
    print(commands)
    if len(commands) < args.max_models_per_gpu:
        args.models_per_gpu = len(commands)
    else: 
        args.models_per_gpu = args.max_models_per_gpu 
    multi_gpu_launcher(commands, args.gpus, args.models_per_gpu)

if __name__ == '__main__':
    eval(args.task)()