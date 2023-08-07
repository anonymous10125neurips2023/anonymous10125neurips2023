# %%
import numpy as np
import math, os 
import torch.nn as nn
import jax 
import jax.numpy as jnp 
from tqdm import tqdm 
import time
import torch
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List

def file_is_empty(path):
    return os.stat(path).st_size==0

def cls2bandit_context(contexts, actions, num_actions):
    """Compute action-convoluted (one-hot) contexts. 
    
    Args:
        contexts: (None, context_dim)
        actions: (None,)
    Return:
        convoluted_contexts: (None, context_dim * num_actions)
    """
    one_hot_actions = jax.nn.one_hot(actions, num_actions) # (None, num_actions) 
    convoluted_contexts = jax.vmap(jax.jit(jnp.kron))(one_hot_actions, contexts) # (None, context_dim * num_actions) 
    return np.array(convoluted_contexts)


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au)/(1+np.dot(u.T, Au))
    return A_inv

#@TODO: multi-head model is more efficient than one-hot-input model
class Model(nn.Module):
    """Template for fully connected neural network for scalar approximation.
    """
    def __init__(self,
                 input_size=1,
                 hidden_size=2,
                 n_layers=1,
                 activation='ReLU',
                 p=0.0,
                 ):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        if self.n_layers == 1:
            self.layers = [nn.Linear(input_size, 1)]
        else:
            size = [input_size] + [hidden_size, ] * (self.n_layers-1) + [1]
            self.layers = [nn.Linear(size[i], size[i+1]) for i in range(self.n_layers)]
        self.layers = nn.ModuleList(self.layers)

        # dropout layer
        self.dropout = nn.Dropout(p=p)

        # activation function
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise Exception('{} not an available activation'.format(activation))


        print('#params = {}'.format(sum(w.numel() for w in self.parameters() if w.requires_grad)))

    def forward(self, x):
        for i in range(self.n_layers-1):
            x = self.dropout(self.activation(self.layers[i](x)))
        x = self.layers[-1](x)
        return x 

# Bandit prototype
class BanditAlgorithm(object): 
    def __init__(self):
        pass

    def reset(self):
        """Reset the internal estimates.
        """
        self.iteration = 0
        self.update_times = [] # collect the update time 
        self.action_selection_times = [] # collect the action selection time 
        self.opt_arm_select_percent_stats = []
        self.subopts = []

    def train(self):
        pass 

    def evaluate(self):
        pass 

    def evaluate_util(self, opt_pred_sel):
        """
            args: opt_pred_sel: (n,a) - learned policy (to be evaluated)
        """
        subopts = self.bandit.best_rewards[-self.num_test:] - np.sum(self.bandit.mean_rewards[-self.num_test:, :] * opt_pred_sel, axis=1)       
        # assert np.all(subopts >= 0)
        subopt = np.mean(subopts) # estimated expected sub-optimality
        self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret
        # new eval of opt-rate
        best_arms = np.zeros((self.num_test, self.bandit.n_arms)) 
        best_arms[np.arange(self.num_test), self.bandit.best_arms[-self.num_test:].astype('int')] = 1 
        opt_arm_select_percent = np.mean(np.sum(best_arms * opt_pred_sel, axis=1) / np.sum(opt_pred_sel, axis=1))
        self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))

        # predicted_arms = np.argmax(np.min(ensemble_preds, axis=0 ), axis=1).astype('int') #(n,)
        # subopts = self.bandit.best_rewards[-self.num_test:] - self.bandit.rewards[-self.num_test:, :][np.arange(self.num_test), predicted_arms]
        # subopt = np.mean(subopts) # estimated expected sub-optimality
        # self.subopts.append((self.iteration, subopt)) #save the index of offline data and the curresponding regret
        # opt_arm_select_percent = np.mean(predicted_arms == self.bandit.best_arms[-self.num_test:])
        # self.opt_arm_select_percent_stats.append((self.iteration, opt_arm_select_percent))

    def run(self):
        """Run an episode of bandit.
        """
        postfix = {
            'update time': -1,
        }

        with tqdm(total=self.T, postfix=postfix) as pbar:
            for t in range(self.T):
                self.offline_action = self.bandit.offline_arms[t] # Get offline action for updating the internal model 
                
                # update approximator
                if t % self.train_every == 0 and t >= self.bandit.n_arms:
                    train_start = time.time()
                    self.train()
                    train_end = time.time() 
                    self.update_times.append((t, train_end - train_start)) # @REMARK: not (yet) divided by M

                # Evaluate 

                if t % self.evaluate_every == 0: 
                    start_time = time.time()
                    self.evaluate() # include computing grad_out and A_inv in the test data
                    end_time = time.time()
                    elapsed_time_per_arm = (end_time - start_time) / self.bandit.n_arms # @REMARK: not (yet) divided by M 
                    self.action_selection_times.append((t, elapsed_time_per_arm ))
                    # action selection time include computing A_inv, grad_out, and test_grad_out

                    print('\n[{}] t={}, subopt={}, % optimal arms={}, action select time={}'.format(self.name, t, self.subopts[-1][1], \
                        self.opt_arm_select_percent_stats[-1][1], self.action_selection_times[-1][1] ))

                # increment counter
                self.iteration += 1

                # log
                update_time = sum([ item[1] for item in self.update_times]) / len(self.update_times) if len(self.update_times) > 0 else 0 
                postfix['update time'] = '{}'.format(update_time)


                if t % self.throttle == 0:
                    pbar.set_postfix(postfix)
                    pbar.update(self.throttle)

## Langevin dynamics 
class SGLD(Optimizer):
    """Implements SGLD algorithm based on
        https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf
    Built on the PyTorch SGD implementation
    (https://github.com/pytorch/pytorch/blob/v1.4.0/torch/optim/sgd.py)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state[
                            'momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(d_p, alpha=-group['lr'])
                noise_std = torch.Tensor([2*group['lr']])
                noise_std = noise_std.sqrt()
                noise = p.data.new(p.data.size()).normal_(mean=0, std=1)*noise_std
                p.data.add_(noise)

        return 1.0

class pSGLD(Optimizer):
    """Implements pSGLD algorithm based on https://arxiv.org/pdf/1512.07666.pdf
    Built on the PyTorch RMSprop implementation
    (https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html#RMSprop)
    """

    def __init__(self, params, lr=1e-2, beta=0.99, Lambda=1e-15, weight_decay=0,
                 centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= Lambda:
            raise ValueError("Invalid epsilon value: {}".format(Lambda))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".
                             format(weight_decay))
        if not 0.0 <= beta:
            raise ValueError("Invalid beta value: {}".format(beta))

        defaults = dict(lr=lr, beta=beta, Lambda=Lambda, centered=centered,
            weight_decay=weight_decay)
        super(pSGLD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(pSGLD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('pSGLD does not support sparse '
                                       'gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['V'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                V = state['V']
                beta = group['beta']
                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                V.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(beta).add_(1 - beta, grad)
                    G = V.addcmul(
                        grad_avg, grad_avg, value=-1).sqrt_().add_(
                            group['Lambda'])
                else:
                    G = V.sqrt().add_(group['Lambda'])

                p.data.addcdiv_(grad, G, value=-group['lr'])

                noise_std =  2*group['lr']/G
                noise_std = noise_std.sqrt()
                noise = p.data.new(
                    p.data.size()).normal_(mean=0, std=1)*noise_std
                p.data.add_(noise)

        return G
        
def lmc(params: List[Tensor],
        d_p_list: List[Tensor],
        weight_decay: float,
        lr: float):
    r"""Functional API that performs Langevine MC algorithm computation.
    """

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        param.add_(d_p, alpha=-lr)


class LangevinMC(Optimizer):
    def __init__(self,
                 params,              # parameters of the model
                 lr=0.01,             # learning rate
                 beta_inv=0.01,       # inverse temperature parameter
                 sigma=1.0,           # variance of the Gaussian noise
                 weight_decay=1.0,
                 device=None):   # l2 penalty
        if lr < 0:
            raise ValueError('lr must be positive')
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.beta_inv = beta_inv
        self.lr = lr
        self.sigma = sigma
        self.temp = - math.sqrt(2 * beta_inv / lr) * sigma
        self.curr_step = 0
        defaults = dict(weight_decay=weight_decay)
        super(LangevinMC, self).__init__(params, defaults)

    def init_map(self):
        self.mapping = dict()
        index = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    num_param = p.numel()
                    self.mapping[p] = [index, num_param]
                    index += num_param
        self.total_size = index

    @torch.no_grad()
    def step(self):
        self.curr_step += 1
        if self.curr_step == 1:
            self.init_map()

        lr = self.lr
        temp = self.temp
        noise = temp * torch.randn(self.total_size, device=self.device)

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            params_with_grad = []
            d_p_list = []
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)

                    start, length = self.mapping[p]
                    add_noise = noise[start: start + length].reshape(p.shape)
                    delta_p = p.grad
                    delta_p = delta_p.add_(add_noise)
                    d_p_list.append(delta_p)
                    # p.add_(delta_p)
            lmc(params_with_grad, d_p_list, weight_decay, lr)

# x = np.random.randn(2,3) 
# actions = np.array([0,1]) 
# n_arms = 2


# y = cls2bandit_context(x, actions, n_arms)