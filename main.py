"""This script runs one single algorithm in one single dataset at a single trial. 
"""
from typing import Text
import numpy as np 
import os 
import argparse, json
import random 
import torch 

from algorithms.neuraLCB import NeuraLCB
from algorithms.neuraLCBDiag import NeuraLCBDiag
from algorithms.linLCB import LinLCB 
from algorithms.neuraLMC import NeuraLMC
from algorithms.neuralTS import NeuralTS

from algorithms.neuralGreedy import NeuralGreedy
from algorithms.utils import file_is_empty

from data.synthetic_bandit import SyntheticBandit
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' # JAX pre-allocate 90% of acquired GPU mem by default, this line disables this feature.

parser = argparse.ArgumentParser()

# config = flags.FLAGS 

# Bandit 
# parser.add_argument('--data_type', type=str, default='synthetic', help='synthetic/realworld')
parser.add_argument('--data', type=str, default='quadratic', help='dataset')
parser.add_argument('--algo', type=str, default='NeuraLMC', help='Algorithm to run.')
parser.add_argument('--data_root', type=str, default='/data/bandit-data', help='path where bandit instance data is saved')

# Experiment 
parser.add_argument('--T', type=int, default=5000, help='Number of rounds. Set 5k for synthetic and 10k for realworld')
parser.add_argument('--num_test', type=int, default=1000, help='Number of test samples')
parser.add_argument('--max_T', type=int, default=10000, help='Number of rounds. Set 5k for synthetic and 10k for realworld')
parser.add_argument('--trial', nargs='+', type=int, default=[0], help='Trial number')
parser.add_argument('--hpo', default=True, action=argparse.BooleanOptionalAction, help='If True, tune hyperparams; if False, run with the best hyperparam')
parser.add_argument('--result_dir', type=str, default='results', help='If None, use default "result". ')

# Neural network 
parser.add_argument('--n_layers',  type=int, default=3, help='Number of layers') 
parser.add_argument('--dropout_p', type=float, default= 0.2, help='Dropout probability') #0.2 previously, now set 0
parser.add_argument('--hidden_size', type=int, default = 64, help='Hidden size')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for SGD')


# Train config 
parser.add_argument('--batch_size', type=int, default=64, help='batch size') 
# parser.add_argument('--start_train_after', type=int, default= 500, help='Start training only after that many iterations. Make sure it is larger batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate') 
parser.add_argument('--epochs', type=int, default=1000, help='training epochs') # 500 for synthetic
parser.add_argument('--use_cuda', default=True, action=argparse.BooleanOptionalAction) 
parser.add_argument('--train_every', type=int, default=100, help='Train periodically')
parser.add_argument('--reg_factor', type=float, default=0.01, help='regularization factor')
parser.add_argument('--sample_window', type=int, default=1000, help='Use latest samples within the window to train')

# Offline data 
parser.add_argument('--behavior_pi', type=str, default='unif', help='unif/eps-greedy/partial-space-unif') 
parser.add_argument('--behavior_epsilon', type=float, default=0.5, help='the probability a uniform action is drawn') # set it first before run 'tune_all.py' and 'run_all.py'
parser.add_argument('--partial_space_rate', type=float, default=0.5)
parser.add_argument('--context_dim', type=int, default=16)
parser.add_argument('--num_actions', type=int, default=10)
parser.add_argument('--noise_std', type=float, default=0.01)

# Test 
parser.add_argument('--evaluate_every', type=int, default=100, help='Evaluate periodically')


# NeuraLCB 
parser.add_argument('--beta', type=float, default=0.01, help='Exploration parameter') 


# NeuralPR 
parser.add_argument('--perturbed_variance', type=float, default=0.1, help='Perturbed variance')
parser.add_argument('--M', type=int, default=10, help='Number of bootstraps')


# NeuraLMC 
parser.add_argument('--inverse_beta', type=float, default=1, help='inverse_beta') # unused 
parser.add_argument('--skip', type=int, default=100, help='interval between two consecutive LMC samples')

#NeuraLAVI 

parser.add_argument('--tau', type=float, default=0.0001, help='tau') 
parser.add_argument('--linear_reg', type=float, default=0.01, help='linear_reg') 
parser.add_argument('--linear_lr', type=float, default=0.01, help='linear_lr') 
parser.add_argument('--lin_epochs', type=int, default=1000, help='linear_reg') 


__allowed_algos__ = ['NeuraLCB', 'LinLCB', 'NeuralGreedy', 'NeuraLCBDiag',  'NeuraLMC', 'NeuralTS']
name2ind = {__allowed_algos__[i]:i for i in range(len(__allowed_algos__))}

firstline = {
    'LinLCB': 'lr | lambda | beta | opt-rate | regret\n', 
    'NeuraLMC': 'lr | inverse_beta | M | wdecay | opt-rate| regret\n',
    'NeuraLCB': 'lr | reg | beta | wdecay | opt-rate | regret\n', 
    'NeuraLCBDiag': 'lr | reg | beta | opt-rate | regret\n', 
    'NeuralGreedy': 'lr | wdecay | opt-rate | regret\n',
    'NeuralTS': 'lr | reg | M | wdecay | opt-rate | regret\n'
}

synthetic_group = ['quadratic',  'cosine']

config = parser.parse_args()

def set_randomness(seed): 
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():    

    for trial in config.trial: 
        set_randomness(trial)

        # if config.data in synthetic_group:
        bandit = SyntheticBandit(
            trial=trial, 
            T=config.max_T, 
            num_test=config.num_test, 
            context_dim=config.context_dim, 
            num_actions=config.num_actions, 
            noise_std=config.noise_std, 
            name=config.data, 
            behavior_pi = config.behavior_pi, 
            behavior_epsilon = config.behavior_epsilon,
            partial_space_rate = config.partial_space_rate, 
            data_root=config.data_root
            )
        if config.behavior_pi == 'eps-greedy':
            pi_name = 'mu_eps={}'.format(config.behavior_epsilon) 
        elif config.behavior_pi == 'partial-space-unif':
            pi_name = 'partial-space-unif-{}'.format(config.partial_space_rate)
        elif config.behavior_pi == 'unif':
            pi_name = 'unif'
        else:
            raise NotImplementedError
        result_dir = os.path.join('results' if config.result_dir is None else config.result_dir, pi_name, config.data, config.algo) 

        # for hpo only 
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir,'config.txt'), 'w') as f:
            json.dump(config.__dict__, f, indent=2)
        fname = os.path.join(result_dir, 'tune.txt')
        if config.hpo: 
            if not os.path.exists(fname) or file_is_empty(fname): 
                with open(fname, "a") as fo:
                    fo.write(firstline[config.algo])

        assert config.algo in __allowed_algos__ 

        print(config)
            
        if config.algo == 'NeuraLCB': 
            algo = NeuraLCB(bandit,
                # is_cls_bandit=False, 
                T = config.T, 
                hidden_size= config.hidden_size,
                reg_factor= config.reg_factor,
                n_layers= config.n_layers,
                batch_size = config.batch_size,
                p= config.dropout_p,
                learning_rate= config.learning_rate,
                epochs = config.epochs,
                train_every = config.train_every,
                evaluate_every = config.evaluate_every, 
                sample_window=config.sample_window, 
                use_cuda = config.use_cuda,
                beta = config.beta,
                weight_decay=config.weight_decay)

        elif config.algo == 'NeuraLCBDiag': 
            algo = NeuraLCBDiag(bandit,
                # is_cls_bandit=False, 
                T = config.T, 
                hidden_size= config.hidden_size,
                reg_factor= config.reg_factor,
                n_layers= config.n_layers,
                batch_size = config.batch_size,
                p= config.dropout_p,
                learning_rate= config.learning_rate,
                epochs = config.epochs,
                train_every = config.train_every,
                evaluate_every = config.evaluate_every,
                use_cuda = config.use_cuda,
                sample_window=config.sample_window,
                beta = config.beta)

        elif config.algo == 'LinLCB': 
            algo = LinLCB(bandit,
                # is_cls_bandit=False,
                T = config.T, 
                reg_factor= config.reg_factor,
                evaluate_every = config.evaluate_every,
                beta = config.beta)

        elif config.algo == 'NeuralGreedy': 
            # NeuralGreedy
            algo = NeuralGreedy(bandit,
                T = config.T,
                # is_cls_bandit=False, 
                hidden_size=config.hidden_size,
                reg_factor=config.reg_factor,
                n_layers=config.n_layers,
                batch_size = config.batch_size,
                p=config.dropout_p,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                train_every=config.train_every,
                evaluate_every = config.evaluate_every,
                use_cuda=config.use_cuda,
                sample_window=config.sample_window,
                weight_decay=config.weight_decay
                )
        elif config.algo == 'NeuraLMC':
            algo = NeuraLMC(bandit,
                T = config.T,
                # is_cls_bandit=False, 
                hidden_size=config.hidden_size,
                reg_factor=config.reg_factor,
                n_layers=config.n_layers,
                batch_size = config.batch_size,
                M = config.M, 
                p=config.dropout_p,
                learning_rate=config.learning_rate,
                epochs=config.epochs,
                train_every=config.train_every,
                use_cuda=config.use_cuda,
                evaluate_every = config.evaluate_every,
                sample_window=config.sample_window,
                inverse_beta = config.inverse_beta,
                weight_decay=config.weight_decay)
       
        elif config.algo == 'NeuralTS': 
            algo = NeuralTS(bandit,
                # is_cls_bandit=False, 
                T = config.T, 
                hidden_size= config.hidden_size,
                reg_factor= config.reg_factor,
                n_layers= config.n_layers,
                batch_size = config.batch_size,
                p= config.dropout_p,
                learning_rate= config.learning_rate,
                epochs = config.epochs,
                train_every = config.train_every,
                evaluate_every = config.evaluate_every, 
                sample_window=config.sample_window, 
                use_cuda = config.use_cuda,
                M = config.M,
                weight_decay=config.weight_decay)
        else:
            raise NotImplementedError 
        
        algo.run() 

        if not config.hpo:
            print('Make sure you are using the best hyperparameters as it will overwrite the result file.')
            subopts = np.array(algo.subopts) # (t, sub-opt)
            update_time = np.array(algo.update_times) # (t, time)
            action_selection_time = np.array(algo.action_selection_times) # (t, time)
            fname = os.path.join(result_dir, 'trial={}.npz'.format(trial))
            np.savez(fname, subopts, update_time, action_selection_time)
        else: # HPO 
            last_subopt = np.array(algo.subopts)[-1,1] 
            last_opt_rate = np.array(algo.opt_arm_select_percent_stats)[-1,1] 
            # Construct text to write 
            text = '{} | {}'.format(config.learning_rate, config.reg_factor)
            if config.algo in ['NeuraLCB', 'LinLCB', 'NeuraLCBDiag']:
                text += '| {} | {} | {} | {}\n'.format(config.beta, config.weight_decay, last_opt_rate, last_subopt) 
            elif config.algo in ['NeuraLMC']: 
                text += '| {} | {} | {} | {} | {}\n'.format(config.inverse_beta, config.M, config.weight_decay, last_opt_rate, last_subopt) 
            elif config.algo in ['NeuralGreedy']: 
                text = '{} | {} | {} | {}\n'.format(config.learning_rate, config.weight_decay, last_opt_rate, last_subopt) 
            elif config.algo == 'NeuralTS':
                # 'lr | reg | M | wdecay | opt-rate | regret\n'
                text = '{} | {} | {} | {} | {} | {}\n'.format(config.learning_rate, config.reg_factor, config.M, config.weight_decay, last_opt_rate, last_subopt)
            with open(fname, "a") as fo:
                fo.write(text)

if __name__ == '__main__': 
    # app.run(main)
    main()
