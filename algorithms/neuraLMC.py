"""
Langevin Posterior Sampling using Langevin optimizer 

"""
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context, BanditAlgorithm, LangevinMC
from functorch import make_functional_with_buffers, vmap, grad, jacrev

class NeuraLMC(BanditAlgorithm):
    def __init__(self,
                 bandit,
                 is_cls_bandit=True, 
                 T = 10000, 
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 batch_size=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 throttle=1,
                 use_cuda=False,
                 M=1, 
                 evaluate_every=1,
                 sample_window=1000, 
                 inverse_beta = 1., # unused 
                 skip=20, 
                 weight_decay=1
                #  start_train_after = 1
                 ):
        self.name = 'NeuraLMC'
        self.is_cls_bandit = is_cls_bandit
        self.bandit = bandit 
        self.sample_window = sample_window
        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers

        # number of rewards in the training buffer
        self.batch_size = batch_size

        # self.start_train_after = start_train_after
        self.M = M # number of bootstrapps 
        self.skip = skip 

        self.T = T
        self.num_test = self.bandit.num_test
        self.weight_decay = weight_decay
        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_every = train_every 
        self.throttle = throttle
        self.reg_factor = reg_factor

        self.evaluate_every = evaluate_every

        self.inverse_beta = inverse_beta 

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')


        # dropout rate
        self.p = p   

        # create an ensemble 
        self.ensemble = [ Model(input_size=bandit.context_dim * self.bandit.n_arms if self.is_cls_bandit else bandit.context_dim,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device) for _ in range(self.M) ] 

        self.model =  Model(input_size=bandit.context_dim * self.bandit.n_arms if self.is_cls_bandit else bandit.context_dim,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device) # base model 

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) 
        self.optimizer = LangevinMC(self.model.parameters(), lr=self.learning_rate,  beta_inv=self.inverse_beta, weight_decay=self.weight_decay) 
        # self.optimizers = [torch.optim.Adam(self.ensemble[m].parameters(), lr=self.learning_rate) for m in range(self.M) ]

        self.reset()


    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def evaluate(self):
        """
        """
        [self.ensemble[m].eval() for m in range(self.M)] 
        ensemble_preds = np.zeros((self.M, self.num_test, self.bandit.n_arms))
        for a in tqdm(self.bandit.arms):
            if self.is_cls_bandit:
                x_batch = self.bandit.features[-self.num_test:, :] # (n,d)
                a_hot = np.zeros(self.bandit.n_arms) 
                a_hot[a] = 1 
                x_batch = np.kron(a_hot, x_batch) # (n, da)
            else: 
                x_batch = self.bandit.features[-self.num_test:, a] # (n,d)
            # convert cls 
            x_batch = torch.FloatTensor(x_batch).to(self.device)
            for m in range(self.M):
                ensemble_preds[m,:,a] = self.ensemble[m].forward(x_batch).detach().squeeze().cpu().detach().numpy()

        preds = np.min(ensemble_preds, axis=0 )
        opt_pred_sel = np.isclose(preds, np.max(preds, axis=1)[:,None]).astype('float')
        # opt_pred_sel = np.array(preds - np.max(preds, axis=1)[:,None] == 0).astype('float') # (n,a)
        probs = opt_pred_sel / np.sum(opt_pred_sel, axis=1)[:,None]
        self.evaluate_util(probs)

    def train(self):
        """Train neural approximator.
        """
        iterations_so_far = range(self.iteration+1)
        # actions_so_far = self.actions[:self.iteration+1]
        offline_actions_so_far = self.bandit.offline_arms[:self.iteration+1]

        if self.is_cls_bandit:
            x_train = self.bandit.features[iterations_so_far] 
            x_train = cls2bandit_context(x_train, offline_actions_so_far, self.bandit.n_arms)
        else: 
            x_train = self.bandit.features[iterations_so_far, offline_actions_so_far] 

        x_train = torch.FloatTensor(x_train).to(self.device)
        y_train = torch.FloatTensor(self.bandit.rewards[iterations_so_far, offline_actions_so_far]).squeeze().to(self.device)

        sample_pool = np.arange(self.iteration)[-self.sample_window:] # force to the latest samples only 
        pool_size = sample_pool.shape[0] 

        # train mode
        # for m in range(self.M):
        self.model.train()
        for j in range(self.epochs + (self.M -1) * self.skip + 1):
        # for j in range(self.epochs):
            # rand_indices = np.random.choice(self.iteration, size=self.batch_size, replace=True)
            rand_indices = np.random.choice(sample_pool, size=self.batch_size, replace=True if self.batch_size >= pool_size else False)
            x_batch = x_train[rand_indices] 
            y_batch = y_train[rand_indices] 
            y_pred = self.model(x_batch).squeeze()
            
            self.model.zero_grad()
            
            # lin_noise_loss = 0
            # std = np.sqrt(2 * self.learning_rate * self.inverse_beta)
            # for param in self.model.parameters():
            #     noise = torch.randn_like(param) * std
            #     lin_noise_loss += (noise * param).sum()

            loss = nn.MSELoss()(y_batch, y_pred) #+ lin_noise_loss
            loss.backward()
            self.optimizer.step()
            # for p in self.model.parameters():
            #     p.data = p.data + torch.randn_like(p) * std

            if j >= self.epochs:
                jj = int((j - self.epochs) / self.skip ) 
                if jj * self.skip == j - self.epochs:
                    # print(jj)
                    # extract the current weights 
                    _, params, _ = make_functional_with_buffers(self.model)
                    # Restore back to the init weights before inference 
                    for p,w in zip(self.ensemble[jj].parameters(), params):
                        p.data = w.data # TODO: make sure this copy works as intended and maybe independence is sufficient? 

        # _, params, _ = make_functional_with_buffers(self.model)
        # std = np.sqrt(2 * self.learning_rate * self.inverse_beta)

        # for i in range(self.M):                
        #     for p,w in zip(self.ensemble[i].parameters(), params):
        #         p.data = w.data + torch.randn_like(w) * std