import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm 
from .utils import Model, inv_sherman_morrison, cls2bandit_context, BanditAlgorithm

class NeuralGreedy(BanditAlgorithm):
    def __init__(self,
                 bandit,
                 T = 10000, 
                 hidden_size=20,
                 n_layers=2,
                 reg_factor=1.0,
                 beta=1,
                 batch_size=100,
                 p=0.0,
                 learning_rate=0.01,
                 epochs=1,
                 train_every=1,
                 evaluate_every=1, 
                 throttle=1,
                 use_cuda=False,
                 is_cls_bandit = True, 
                 sample_window = 1000,
                 weight_decay=1
                 ):
        self.name = 'NeuralGreedy'
        self.is_cls_bandit = is_cls_bandit
        self.sample_window = sample_window

        self.bandit = bandit 
        self.num_test = self.bandit.num_test
        # self.test_bandit = test_bandit 
        # hidden size of the NN layers
        self.hidden_size = hidden_size
        # number of layers
        self.n_layers = n_layers

        # number of rewards in the training buffer
        self.batch_size = batch_size

        # self.start_train_after = start_train_after

        # NN parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.reg_factor = reg_factor 
        self.train_every = train_every 
        self.evaluate_every = evaluate_every
        self.throttle = throttle 
        self.beta = beta 
        self.weight_decay = weight_decay

        self.T = T 

        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu')

        # dropout rate
        self.p = p

        # neural network
        self.model = Model(input_size=bandit.context_dim * self.bandit.n_arms if self.is_cls_bandit else bandit.context_dim,
                           hidden_size=self.hidden_size,
                           n_layers=self.n_layers,
                           p=self.p
                           ).to(self.device)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.reset()


    @property
    def approximator_dim(self):
        """Sum of the dimensions of all trainable layers in the network.
        """
        return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

    def evaluate(self):
        """
        """
        self.model.eval()
        preds = np.zeros((self.num_test, self.bandit.n_arms))
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
            preds[:,a] = self.model.forward(x_batch).detach().squeeze().cpu().detach().numpy()

        # opt_pred_sel = np.array(preds - np.max(preds, axis=1)[:,None] == 0).astype('float') # (n,a)
        opt_pred_sel = np.isclose(preds, np.max(preds, axis=1)[:,None]).astype('float')
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

        # train mode
        self.model.train()
        # loss_ = 0
        sample_pool = np.arange(self.iteration)[-self.sample_window:] # force to the latest samples only 
        pool_size = sample_pool.shape[0] 
        assert pool_size == min(self.sample_window, self.iteration)
        for _ in range(self.epochs):
            # rand_indices = np.random.choice(self.iteration, size=self.batch_size, replace=True)
            rand_indices = np.random.choice(sample_pool, size=self.batch_size, replace=True if self.batch_size >= pool_size else False)
            x_batch = x_train[rand_indices] 
            y_batch = y_train[rand_indices] 
            y_pred = self.model.forward(x_batch).squeeze()
            loss = nn.MSELoss()(y_batch, y_pred) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()