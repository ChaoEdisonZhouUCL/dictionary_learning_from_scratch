import numpy as np
from sklearn import linear_model
from tqdm import tqdm


class OnlineDictionaryLearning:
    def __init__(self, data, ):
        self.data = data

        self.n_obs = len(self.data)
        self.dim_obs = len(self.data[0])


        self.losses = []
        self.regret = []
        self.offline_loss = []
        self.objective = []

        self.alphas = []
        self.observed = []
        self.cumulative_losses = []


    @staticmethod
    def compute_alpha(x, dic, lam):
        reg = linear_model.LassoLars(alpha=lam)
        reg.fit(X=dic, y=x)
        return reg.coef_

    @staticmethod
    def compute_dic(a, b, d, dict_size):

        # We run only one iteration for the optimization over D
        # converged = False

        # while not converged :
        for j in range(dict_size) :
            u_j = (b[:, j] - np.matmul(d, a[:, j])) / a[j, j] + d[:, j]
            d[:, j] = u_j / max([1, np.linalg.norm(u_j)])

        return d

    def learn(self, it, lam, dict_size):
        self.initialize_logs()

        data_gen = self.sample(self.data)

        a_prev = 0.01 * np.identity(dict_size)
        b_prev = 0
        d_prev = self.initialize_dic(dict_size, data_gen)

        for it_curr in tqdm(range(it)):
            x = next(data_gen)

            alpha = self.compute_alpha(x, d_prev, lam)

            a_curr = a_prev + np.outer(alpha, alpha.T)
            b_curr = b_prev + np.outer(x, alpha.T)

            d_curr = self.compute_dic(a=a_curr, b=b_curr, d=d_prev, dict_size=dict_size)

            a_prev = a_curr
            b_prev = b_curr
            d_prev = d_curr



        return d_curr.T

 
    @staticmethod
    def initialize_dic(dict_size, data_gen):
        return np.array([next(data_gen) for _ in range(dict_size)]).T
