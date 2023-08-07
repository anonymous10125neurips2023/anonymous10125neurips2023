"""Functions to create bandit problems from datasets."""

# %% 
import numpy as np
import pandas as pd
import os 
from easydict import EasyDict as edict
from .utils import sample_offline_actions


class SyntheticBandit(object):
    def __init__(self, 
            trial, 
            T, 
            context_dim, 
            num_actions, 
            noise_std, 
            name, 
            behavior_pi = 'eps-greedy', 
            behavior_epsilon=0.1,
            partial_space_rate=0.5, 
            num_test = 1000,
            data_root='',
            ): # Create or load on trial basis. If saved, be consistent with T
        os.makedirs(data_root, exist_ok=True)
        self.num_test = num_test 
        self.name = name 
        self.T = T 
        self.trial = trial 
        self.n_arms = num_actions
        self.behavior_epsilon = behavior_epsilon # epsilon-greedy with respect to the optimal actions 
        self.behavior_pi = behavior_pi # behavior policy method 
        self.partial_space_rate = partial_space_rate, 
        if self.behavior_pi == 'eps-greedy':
            pi_name = '{}-greedy'.format(behavior_epsilon) 
        elif self.behavior_pi == 'partial-space-unif':
            pi_name = '{}-{}'.format(partial_space_rate, self.behavior_pi)
        elif self.behavior_pi == 'unif':
            pi_name = 'unif'
        else:
            raise NotImplementedError

        # self.context_dim = CLSDataset[name][0] * self.n_arms
        self.data_dir = os.path.join(data_root, name , pi_name)

        os.makedirs(self.data_dir, exist_ok=True)
        fname = os.path.join(self.data_dir, 'trial={}.npz'.format(trial))
        if os.path.exists(fname): 
                print('Loading data from {}'.format(fname))
                arr = np.load(fname)
                contexts = arr['arr_0']
                rewards = arr['arr_1']
                mean_rewards = arr['arr_2']
                opt_rewards = arr['arr_3']
                opt_actions = arr['arr_4']
                off_actions = arr['arr_5']
        else:
            print('Creating new data into {}'.format(fname))
            if name == 'quadratic':
                contexts, rewards, opt_rewards, opt_actions, mean_rewards = sample_quadratic(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'quadratic2':
                contexts, rewards, opt_rewards, opt_actions, mean_rewards = sample_quadratic2(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'cosine':
                contexts, rewards, opt_rewards, opt_actions, mean_rewards = sample_cosine(T + self.num_test, context_dim, num_actions, noise_std)
            elif name == 'exp':
                contexts, rewards, opt_rewards, opt_actions, mean_rewards = sample_exp(T + self.num_test, context_dim, num_actions, noise_std)
            else: 
                raise NotImplementedError 
            
            # off_actions = sample_offline_policy(opt_actions, contexts.shape[0], self.n_arms, behavior_pi='eps-greedy', behavior_epsilon=behavior_epsilon)
            off_actions = sample_offline_actions(opt_actions = opt_actions, 
                                                 num_contexts=contexts.shape[0], 
                                                 num_actions=self.n_arms, 
                                                 behavior_pi=behavior_pi, 
                                                 behavior_epsilon=behavior_epsilon,
                                                 partial_space_rate=partial_space_rate                         )
            np.savez(fname, contexts, rewards, mean_rewards, opt_rewards, opt_actions, off_actions)

        self.features = contexts #(T, na, context_dim)
        self.context_dim = context_dim 
        self.rewards = rewards #(T, n_arms)
        self.mean_rewards = mean_rewards #(T, n_arms)
        self.best_rewards = opt_rewards # (T,)
        self.best_arms = opt_actions # (T,)
        self.offline_arms = off_actions 
        self.real_T = self.features.shape[0] # mushroom contains only ~8k < 10k data points 
        print(self.features.shape, self.rewards.shape)
    
    @property 
    def arms(self): 
        return range(self.n_arms)

def sample_quadratic(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: 10 * np.square(x @ thetas) # x: (n, d), h: (n, a) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim)) # (n,d)
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (n, a)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions, mean_rewards

def sample_quadratic2(num_contexts, context_dim, num_actions, noise_std):
    A = np.random.randn(context_dim, context_dim, num_actions) # (d,d,a)
    B = np.zeros((context_dim, context_dim, num_actions)) # (d,d,a)
    for a in range(num_actions):
        B[:,:,a] = A[:,:,a].T @ A[:,:,a] 
    h = lambda x: np.sum( np.dot(x,B) * x[:,:,None], axis=1) # x: (n, d), h: (n, a) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]

    mean_rewards = h(contexts) # (n, a)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions, mean_rewards

def sample_cosine(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: np.cos(3 * x @ thetas) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (T, na)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions, mean_rewards

def sample_exp(num_contexts, context_dim, num_actions, noise_std):
    thetas = np.random.uniform(-1,1,size = (context_dim, num_actions)) 
    thetas /= np.linalg.norm(thetas, axis=0)[None, :] # (d,a)
    h = lambda x: np.exp(-10 * (x @ thetas)**2 ) 
    contexts = np.random.uniform(-1,1, size=(num_contexts, context_dim))
    contexts /= np.linalg.norm(contexts, axis=1)[:,None]
    mean_rewards = h(contexts) # (T, na)
    opt_rewards = np.max(mean_rewards, axis=1) 
    opt_actions = np.argmax(mean_rewards, axis=1) 
    rewards = mean_rewards + noise_std * np.random.normal(size=mean_rewards.shape)
    return contexts, rewards, opt_rewards, opt_actions, mean_rewards

def sample_offline_policy(opt_mean_rewards, num_contexts, num_actions, behavior_pi='eps-greedy', behavior_epsilon=0.1, subset_r = 0.5, 
                contexts=None, rewards=None): 
    """Sample offline actions 
    Args:
        opt_mean_rewards: (num_contexts,)
        num_contexts: int 
        num_actions: int
        pi: ['eps-greedy', 'subset', 'online']
    """

    if behavior_pi == 'eps-greedy':
        uniform_actions = np.random.randint(low=0, high=num_actions, size=(num_contexts,)) 
        # opt_actions = np.argmax(mean_rewards, axis=1)
        delta = np.random.uniform(size=(num_contexts,))
        selector = np.array(delta <= behavior_epsilon).astype('float32') 
        actions = selector.ravel() * uniform_actions + (1 - selector.ravel()) * opt_mean_rewards 
        actions = actions.astype('int')
        return actions

    