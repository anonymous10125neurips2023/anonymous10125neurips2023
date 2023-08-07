# %% 
from genericpath import exists, isdir
import numpy as np 
import os 
from numpy.linalg import inv as npinv 
from tqdm import tqdm
import pickle 
import argparse
import math 
import matplotlib.pyplot as plt 
# %matplotlib inline 
# %%
def int2bin(n, width): 
    return np.array([int(c) for c in np.binary_repr(int(n), width)])

class LinearMDP(object): 
    def __init__(self, d=None, n_actions=100, n_states=2, H = 10): 
        self.H = H
        self.dim = d if d is not None else math.ceil(np.log2(n_actions)) + 2
        self.n_actions = n_actions 
        self.n_states = n_states
        self.state_space = np.arange(self.n_states)
        self.action_space = np.arange(self.n_actions)


        delta = np.zeros((self.n_states, self.n_actions))
        delta[0,0] = 1 
        # delta[1,1:] = 1
        self.delta = delta

        self.reset()

    def reset(self):  
        # initialize alpha 
        self.alpha = np.random.binomial(1, 0.5, size=self.H)

        self.theta = np.zeros(self.dim) 
        r = 0.99
        self.theta[-2] = r 
        self.theta[-1] = 1 - r 


    def phi(self, s,a): 
        phi = np.zeros(self.dim) 
        phi[:-2] = int2bin(a, self.dim-2) 
        phi[-2] = self.delta[s,a] 
        phi[-1] = 1 - phi[-2] 
        return phi 

    def nu(self, h, s): 
        nu = np.zeros(self.dim) 
        nu[-2] = int(self.alpha[h]) ^ int(1 - s) 
        nu[-1] = int(self.alpha[h]) ^ int(s)  
        return nu 

    def transition_kernel(self, h, s,a,ss):
        p = self.phi(s,a)[None,:] @ self.nu(h, ss)[:,None]  
        return p.ravel()[0] 

    def reward(self, s,a): 
        r = self.phi(s,a)[None,:] @ self.theta[:,None] 
        return r.ravel()[0] 

    def compute_Q_value(self, pi=None): 
        """
        Args:
            pi: H of [2x100], indexing 0,...,H-1
                If None, return Q^*

        Return:     
            Qs: H-1, ..., 0
        """
        Qs = []
        for h in range(self.H): 
            Q_h = np.zeros((2, self.n_actions)) 
            for s in range(2): 
                for a in range(self.n_actions):
                    y =  0
                    if h > 0:
                        assert len(Qs) > 0 
                        P_h = np.array([self.transition_kernel(self.H - 1 - h,s,a,ss) for ss in range(self.n_states)])
                        if pi is not None:
                            V_hp1 = np.sum(Qs[-1] * pi[self.H - h], axis=1) # (nS,)
                            y = np.sum( V_hp1 * P_h) # (1,)
                        else: # compute Q^*
                            V_hp1 = np.max(Qs[-1], axis=1) # (nS,)
                            y = np.sum( V_hp1 * P_h)
                        
                    Q_h[s,a] = self.reward(s,a) + y
            Qs.append(Q_h)

        Qs.reverse()
        return Qs


    def generate_trajectory(self, pi): 
        """
        Args:
            pi: list of H x [2 x nA], 0,..., H-1

        Return: 
            0, ..., H-1

        """
        states = [] 
        actions = [] 
        rewards = [] 
        for h in range(self.H):
            if h == 0:
                p = None
            else:
                assert len(states) > 0, len(actions) > 0 
                prev_s = states[-1] 
                prev_a = actions[-1] 
                p = [self.transition_kernel(h-1, prev_s, prev_a, ss) for ss in range(self.n_states)] 
            s = np.random.choice(self.n_states, p=p)
            a = np.random.choice(self.n_actions, p = pi[h][s,:]) 
            r = self.reward(s,a) 
            states.append(s) 
            actions.append(a) 
            rewards.append(r)
        return states, actions, rewards


    def compute_feasible_states(self, pi): 
        """A list of 2x100, index 0,...,H-1
        Args: 
            pi: H of [2x100], indexing H-1, ..., 0
        
        Return:
            A list of H lists, indexing 0, ..., H-1
        """
        feasible_states = [[0,1]]
        for h in range(1, self.H): 
            fea_s = []
            for s in feasible_states[-1]: 
                for a in range(self.n_actions): 
                    if pi[self.H-h][s,a] > 0: 
                        for ss in range(self.n_states): 
                            if self.transition_kernel(h-1,s,a, ss) > 0 and ss not in fea_s: 
                                fea_s.append(ss)
            feasible_states.append(fea_s) 
        return feasible_states 

def run_pevi(linmdp, D, V_opt_1, args):
    subopt_list = []
    Sigma = np.zeros((linmdp.H, linmdp.dim, linmdp.dim))
    for h in range(linmdp.H): 
        Sigma[h] = args.lambd * np.eye(linmdp.dim)
    target = np.zeros((linmdp.H, linmdp.dim))
    for i in tqdm(range(1,args.n)):
        # % PEVI 
        pi_hat = []
        upper_Q = None
        for h in range(linmdp.H-1,-1,-1): 
            s_h = int(D[i,0,h]) 
            a_h = int(D[i,1,h]) 
            r_h = D[i,2,h]

            if h < linmdp.H-1:
                assert upper_Q is not None
                s_hp1 = int(D[i,0,h+1])
                y = r_h + np.max(upper_Q[s_hp1, :]) 
            else:
                y = r_h 
            target[h,:] += y * linmdp.phi(s_h,a_h)
            

            Sigma[h,:,:] += linmdp.phi(s_h,a_h)[:,None] @ linmdp.phi(s_h,a_h)[None,:]

            inv_Sigma_h = npinv(Sigma[h,:,:])
            theta_h =  inv_Sigma_h @ target[h,:]

            # Output Q_h and b_h functions 
            b_h = np.zeros((2,linmdp.n_actions)) 
            Q_h = np.zeros((2,linmdp.n_actions)) 
            for s in range(2): 
                for a in range(linmdp.n_actions):
                    b_h[s,a] = args.beta * np.sqrt((linmdp.phi(s,a)[None,:] @ inv_Sigma_h) @ linmdp.phi(s,a)[:,None]).ravel()

                    Q_h[s,a] = max(0, min((linmdp.phi(s,a)[None,:] @ theta_h[:,None]).ravel()[0] - b_h[s,a], args.H-h))

            pi_hat_h = (np.max(Q_h, axis=1)[:,None] - Q_h == 0).astype('float') # (nS,nA)
            pi_hat_h = pi_hat_h / np.sum(pi_hat_h, axis=1)[:,None] 
            pi_hat.append(pi_hat_h)

            upper_Q = Q_h 

        pi_hat.reverse()
        Q_pevi = linmdp.compute_Q_value(pi_hat)

        V_pevi_1 = np.max(Q_pevi[0], axis=1)

        subopt = np.mean(V_opt_1 - V_pevi_1) 

        # print(subopt)

        subopt_list.append(subopt)

    return subopt_list


def run_ts(linmdp, D, V_opt_1, args):
    subopt_list = []
    Sigma = np.zeros((linmdp.H, linmdp.dim, linmdp.dim))
    for h in range(linmdp.H): 
        Sigma[h] = args.lambd * np.eye(linmdp.dim)
    target = np.zeros((linmdp.H, linmdp.dim))
    for i in tqdm(range(1,args.n)):
        # % TS 
        pi_hat = []
        upper_Q = None
        for h in range(linmdp.H-1,-1,-1): 
            s_h = int(D[i,0,h]) 
            a_h = int(D[i,1,h]) 
            r_h = D[i,2,h]

            if h < linmdp.H-1:
                assert upper_Q is not None
                s_hp1 = int(D[i,0,h+1])
                y = r_h + np.max(upper_Q[s_hp1, :]) 
            else:
                y = r_h 
            target[h,:] += y * linmdp.phi(s_h,a_h)
            

            Sigma[h,:,:] += linmdp.phi(s_h,a_h)[:,None] @ linmdp.phi(s_h,a_h)[None,:]

            inv_Sigma_h = npinv(Sigma[h,:,:])
            theta_h =  inv_Sigma_h @ target[h,:]

            noise = np.random.multivariate_normal(mean=np.zeros(theta_h.shape), cov=args.sigma**2 * inv_Sigma_h, size=args.M) 


            # Output Q_h and b_h functions 
            Q_h = np.zeros((args.M, 2,linmdp.n_actions)) 
            for j in range(args.M):
                perturbed_theta_h = theta_h + noise[j,:]
                for s in range(2): 
                    for a in range(linmdp.n_actions):
                        Q_h[j, s,a] = (linmdp.phi(s,a)[None,:] @ perturbed_theta_h[:,None]).ravel()[0]  

            Q_h = np.min(Q_h, axis=0) # (S,A)

            Q_h = np.maximum(0, np.minimum(Q_h, args.H - h) )

            pi_hat_h = (np.max(Q_h, axis=1)[:,None] - Q_h == 0).astype('float') # (nS,nA)
            pi_hat_h = pi_hat_h / np.sum(pi_hat_h, axis=1)[:,None] 
            pi_hat.append(pi_hat_h)

            upper_Q = Q_h 

        pi_hat.reverse()
        Q_pevi = linmdp.compute_Q_value(pi_hat)

        V_pevi_1 = np.max(Q_pevi[0], axis=1)

        subopt = np.mean(V_opt_1 - V_pevi_1) 

        # print(subopt)

        subopt_list.append(subopt)

    return subopt_list

def run_lmc(linmdp, D, V_opt_1, args):
    subopt_list = []
    Sigma = np.zeros((linmdp.H, linmdp.dim, linmdp.dim))
    for h in range(linmdp.H): 
        Sigma[h] = args.lambd * np.eye(linmdp.dim)
    target = np.zeros((linmdp.H, linmdp.dim))
    for i in tqdm(range(1,args.n)):
        # % LMC 
        pi_hat = []
        upper_Q = None
        for h in range(linmdp.H-1,-1,-1): 
            s_h = int(D[i,0,h]) 
            a_h = int(D[i,1,h]) 
            r_h = D[i,2,h]

            if h < linmdp.H-1:
                assert upper_Q is not None
                s_hp1 = int(D[i,0,h+1])
                y = r_h + np.max(upper_Q[s_hp1, :]) 
            else:
                y = r_h 
            target[h,:] += y * linmdp.phi(s_h,a_h)
            

            Sigma[h,:,:] += linmdp.phi(s_h,a_h)[:,None] @ linmdp.phi(s_h,a_h)[None,:]

            inv_Sigma_h = npinv(Sigma[h,:,:])
            theta_h =  inv_Sigma_h @ target[h,:]
            
            eg,_ = np.linalg.eig( Sigma[h,:,:] )
            lambda_max = np.max(eg)
            eta = 1 / (4 * lambda_max)
            Ah = np.eye(linmdp.dim) - 2 *eta * Sigma[h,:,:] 
            # The tricky part is to choose a right eta. If eta is too large, the noise blows up.
            # If eta is too small, the perturbance does not disappear and affect the cov estimate. 
            # print('==========')
            # print(np.linalg.matrix_power(Ah,args.J))
            mu = (np.eye(linmdp.dim) - np.linalg.matrix_power(Ah,args.J)) @ theta_h 
            Cov_mat = args.tau**2 * (np.eye(linmdp.dim) -  np.linalg.matrix_power(Ah,2*args.J)) @ inv_Sigma_h @ npinv(np.eye(linmdp.dim) +Ah) + 1e-6 * np.eye(linmdp.dim)
            # Cov_mat = args.tau**2 * inv_Sigma_h

            noise = np.random.multivariate_normal(mean=np.zeros(theta_h.shape), cov=Cov_mat, size=args.M) 


            # Output Q_h and b_h functions 
            Q_h = np.zeros((args.M, 2,linmdp.n_actions)) 
            for j in range(args.M):
                perturbed_theta_h = theta_h + noise[j,:]
                for s in range(2): 
                    for a in range(linmdp.n_actions):
                        Q_h[j, s,a] = (linmdp.phi(s,a)[None,:] @ perturbed_theta_h[:,None]).ravel()[0]  

            Q_h = np.min(Q_h, axis=0) # (S,A)

            Q_h = np.maximum(0, np.minimum(Q_h, args.H - h) )

            pi_hat_h = (np.max(Q_h, axis=1)[:,None] - Q_h == 0).astype('float') # (nS,nA)
            pi_hat_h = pi_hat_h / np.sum(pi_hat_h, axis=1)[:,None] 
            pi_hat.append(pi_hat_h)

            upper_Q = Q_h 

        pi_hat.reverse()
        Q_pevi = linmdp.compute_Q_value(pi_hat)

        V_pevi_1 = np.max(Q_pevi[0], axis=1)

        subopt = np.mean(V_opt_1 - V_pevi_1) 

        # print(subopt)

        subopt_list.append(subopt)

    return subopt_list
def generate_data_in_advance(mu, args): 
    """Generate and save offline data in advance. 
    """
    data_dir = 'data'
    env_name = 'linmdp-nA={}-H={}-p={}-n={}'.format(args.n_actions, args.H, args.p, args.n)
    res = 'n'
    if os.path.isdir(os.path.join(data_dir, env_name)):
        res = input('You are about to overwrite the existing experiment data. Are you sure to create a new one instead of using the old one? [y/n]:')
    if res in ['y', 'Y', 'Yes', 'Yeah'] or not os.path.isdir(os.path.join(data_dir, env_name)):
        np.random.seed(2022)
        for run in tqdm(range(args.n_trials)):
            run_fname = os.path.join(data_dir, env_name, 'run={}'.format(run))
            os.makedirs(run_fname, exist_ok=True)
            linmdp = LinearMDP(d=None, H=args.H, n_actions=args.n_actions)
            Q_opt = linmdp.compute_Q_value()
            V_opt_1 = np.max(Q_opt[0], axis=1) # (nS,)
            with open(os.path.join(run_fname, 'mdp.pkl'), 'wb') as fo:
                pickle.dump(linmdp, fo, pickle.HIGHEST_PROTOCOL) 

            np.save(os.path.join(run_fname, 'V_opt_1.npy'), V_opt_1) 


            D = []
            for _ in range(args.n):
                ss, acts, rs = linmdp.generate_trajectory([mu] * args.H)
                D.append([ss, acts, rs])

            D = np.array(D) # (n, 3, H) 
            np.save(os.path.join(run_fname, 'D.npy'), D) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000) 
    parser.add_argument('--n_actions', type=int, default=100) 
    parser.add_argument('--H', type=int, default=20) 
    parser.add_argument('--lambd', type=float, default=0.01)
    parser.add_argument('--beta', type=float, default=0.02) # 0.1, 0.2
    parser.add_argument('--p', type=float, default=0.6)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--start_trial', type=int, default = 0)

    # TS 
    parser.add_argument('--M', type=int, default=10) 
    parser.add_argument('--sigma', type=float, default=0.01) # 0.1, 0.2
    
    #LMC
    parser.add_argument('--tau', type=float,default=0.01)
    parser.add_argument('--J',type=int,default=1000)
    parser.add_argument('--eta',type=float,default=1.0)

    parser.add_argument('--algo', type=str, default='PEVI') 

    args = parser.parse_args()

    # exp_dir = 'results/linmdp-nA={}-H={}-p={}-n={}'.format(args.n_actions, args.H, args.p, args.n) 
    # os.makedirs(exp_dir, exist_ok=True)

    # Behavior policy 
    mu = np.zeros((2,args.n_actions)) 
    mu[0,0] = args.p 
    # mu[0,1] = (1 - args.p) / (args.n_actions-1)  
    mu[0,1] = 1-args.p 
    mu[1,0] = args.p 
    mu[1,1:] = (1 - args.p) / (args.n_actions-1) 

    # Generate data 
    # generate_data_in_advance(mu, args) 

    data_dir = 'data'
    env_name = 'linmdp-nA={}-H={}-p={}-n={}'.format(args.n_actions, args.H, args.p, args.n)
    exp_dir = os.path.join('icml-results', env_name) 
    os.makedirs(exp_dir, exist_ok=True)
    for run in range(args.start_trial, args.n_trials):
        print('run = {}'.format(run))
        run_fname = os.path.join(data_dir, env_name, 'run={}'.format(run))

        # PEVI 
        with open(os.path.join(run_fname, 'mdp.pkl'), 'rb') as fo:
            linmdp = pickle.load(fo) 
        V_opt_1 = np.load(os.path.join(run_fname, 'V_opt_1.npy'))
        
        D = np.load(os.path.join(run_fname, 'D.npy')) 

        if args.algo == 'PEVI':
            subopt_list = run_pevi(linmdp, D, V_opt_1, args)
            np.save(os.path.join(exp_dir, 'pevi-subopt-beta={}-lambda={}-run={}.npy'.format(args.beta, args.lambd, run)), np.array(subopt_list))

        elif args.algo == 'TS': 
            subopt_list = run_ts(linmdp, D, V_opt_1, args)
            np.save(os.path.join(exp_dir, 'ts-subopt-sigma={}-M={}-lambda={}-run={}.npy'.format(args.sigma, args.M, args.lambd, run)), np.array(subopt_list))

        elif args.algo == 'LMC': 
            subopt_list = run_lmc(linmdp, D, V_opt_1, args)
            np.save(os.path.join(exp_dir, 'lmc-subopt-tau={}-eta={}-M={}-lambda={}-run={}.npy'.format(args.tau,args.eta, args.M, args.lambd, run)), np.array(subopt_list))
        else: 
            raise NotImplementedError 