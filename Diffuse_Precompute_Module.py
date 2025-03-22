import torch
import math
import numpy as np

def sign(x):
    return 0 if x == 0 else x/abs(x)

class Diffuse_Precompute_Moudle:
    def __init__(self, diff_mode, noise_schedule, T: int, K: int, N: int, P: int,
                 noise_range) -> None:  # K includes absorbing state. absorbing state is the last state.
        """
        Efficiently compute uniform&absorbing diffusion process in which the matrices are both low rank.
        :param diff_mode: 'uni' or 'abs'
        :param noise_schedule:
        :param T: T>0
        :param K: K includes absorbing state. absorbing state is the last state.
        :param noise_range:
        """
        self.diff_mode, self.noise_schedule, self.T, self.K, self.N, self.P, self.noise_range = diff_mode, noise_schedule, T, K, N, P, noise_range
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # generate alpha_bar & beta_bar for later computation: Q_bar = alpha_bar*I + beta_bar*basic_matrix
        self.betas = None
        if noise_schedule == 'linear':
            step = (noise_range[1] - noise_range[0]) / (self.T)
            self.betas = np.array([noise_range[0] + step * i for i in range(self.T)])
        elif noise_schedule == 'cosine':
            s, half_pi = 0.008, 0.5 * math.pi
            betas = []
            for i in range(self.T):
                degree_t, degree_t_plus_one = half_pi * (i / T + s) / (1 + s), half_pi * ((i + 1) / T + s) / (1 + s)
                betas.append(1 - math.cos(degree_t_plus_one) / (math.cos(degree_t) + 1e-7))
            self.betas = np.array(betas)
        elif noise_schedule == 'mutual-information':
            self.betas = np.array([1 / (self.T - (i + 1) + 1) for i in range(self.T)])
        elif noise_schedule == 'multi-power':
            beta_lists = []
            for pos in range(self.N):
                # dist = pos-math.floor(self.N/2)
                # beta_lists.append([pow((t+1)/self.T, pow(abs(dist), -sign(dist))) for t in range(self.T)])
                # beta(t) = min(1, Nt/T(N-pos))
                beta_lists.append([min(1, pow((t*self.N)/((abs(self.N-pos))*self.T), self.P)) for t in range(self.T)])
            self.betas = np.stack(beta_lists, axis=0)
        # elif noise_schedule == 'tan':
        #     beta_lists = []
        #     for pos in range(self.N):
        #         # dist = pos-math.floor(self.N/2)
        #         # beta_lists.append([pow((t+1)/self.T, pow(abs(dist), -sign(dist))) for t in range(self.T)])
        #         # beta(t) = min(1, Nt/T(N-pos))
        #         beta_lists.append(
        #             [min(math.pi/4, (1/(self.T/self.P + (self.N-pos)*((self.T-self.T/self.P)/self.N)))*t*math.pi/4) for t in range(self.T)])
        #     self.betas = np.tan(np.stack(beta_lists, axis=0))
        else:
            raise Exception("Wrong noise schedule.")
        # if noise_schedule is not 'multi-power':
        #     assert self.betas.ndim == 1 and self.betas.shape[0] == self.T
        self.betas = torch.tensor(self.betas).to(self.device)
        if self.betas.ndim == 1:
            self.betas = self.betas.unsqueeze(0).repeat(self.N, 1)  # the same noise schedule for every positions.

        # fill alpha_bar and beta_bar(Q_bar = alpha_bar*I + beta_bar*basic_matrix).
        self.alpha_bar, self.beta_bar = torch.zeros(self.N, self.T + 1).to(self.device), torch.zeros(self.N, self.T + 1).to(self.device)
        self.alpha_bar[:,0], self.beta_bar[:,0] = 1, 0  # This is for compatibility with diffusing 0 step (identity matrix).
        # self.alpha_bar[0], self.beta_bar[0] = 1 - self.betas[0], self.betas[0]
        for i in range(1, self.T + 1):
            self.alpha_bar[:,i], self.beta_bar[:,i] = (self.alpha_bar[:,i - 1] * (1 - self.betas[:,i - 1]),
                                                       self.beta_bar[:,i - 1] + self.betas[:,i - 1] - self.beta_bar[:,i - 1] * self.betas[:,i - 1])
        print('done')

    def x_mul_Q_bar(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        """
            Given x(must be a distribution) calculate x*Q_bar.
            x: onehot tensor with shape [bs, len, # categories].
            t: diffusion steps(range: [0, T]) with shape [bs].
                !(t can be 0 because alpha_bar and beta_bar are compatible with diffusing 0 step (identity matrix).)!
        """
        assert x.shape[0] == t.shape[0]
        alpha_bar_t = self.alpha_bar[:,t]
        beta_bar_t = self.beta_bar[:,t]
        if self.diff_mode == 'uni':
            return (alpha_bar_t * (x.permute(2,1,0)) + beta_bar_t / self.K).permute(2,1,0).contiguous()
        elif self.diff_mode == 'abs':
            res = alpha_bar_t * (x.permute(2,1,0))
            res[-1, :, :] += beta_bar_t
            return res.permute(2,1,0).contiguous()

    def p_xt_x_pre_t_x0(self, x0, xt, t: torch.tensor):
        """
            Given x0, xt(shape: [bs, len, 1]) calculate P(x{t}|x{t-1},x{0}).
            x0: The result has nothing to do with x0 in mathematics.
            xt: A observation at step t. shape: [bs, len, 1].
            t: diffusion steps(range: [1, T]) with shape [bs].
        """
        assert xt.shape[0] == t.shape[0] and xt.ndim == 3
        res = None
        bs = xt.shape[0]
        # calculate p(xt|x(t-1),x0), where x(t-1) is unknown.
        beta_t = self.betas[:,t - 1]
        if self.diff_mode == 'uni':
            res = (beta_t / self.K).unsqueeze(2).repeat(1, 1, self.K).permute(1,0,2)
        elif self.diff_mode == 'abs':
            res = ((beta_t.permute(1,0))*((xt == (self.K - 1)).squeeze(2))).unsqueeze(2).repeat(1, 1, self.K).double()
            """ another way but inefficient. """
            # res = torch.zeros(xt.shape[1], self.K, bs).to(self.device).double()
            # idx_to_change = (xt.repeat(1,1,self.K) == (self.K - 1)).permute(1,2,0)
            # res[idx_to_change] = 1
            # res *= beta_t
            # res = res.permute(2,0,1)
            """ another way but inefficient. """
            # res = torch.zeros(bs, xt.shape[1], self.K).to(self.device).double()
            # idx_to_change = (xt.repeat(1, 1, self.K) == (self.K - 1))
            # for i in range(bs):
            #     res[i][idx_to_change[i]] = beta_t[i]
        one_minus_beta = 1 - beta_t
        values_after_change = (res.gather(2, xt).permute(2,1,0) + one_minus_beta).permute(2,1,0)
        res.scatter_(2, xt, values_after_change)
        """ another way but inefficient. """
        # for i in range(bs):
        #     res[i, range(xt.shape[1]), xt[i].squeeze()] += one_minus_beta[i]
        return res

    def p_x_pre_t_x0(self, x0, t):
        """
            Given x0 and t, calculate P(x{t-1}|x{0}).
            x0: shape: [bs, len, # categories], must be a distribution
            t: diffusion steps(range: [1, T]) with shape [bs].
        """
        return self.x_mul_Q_bar(x0, t - 1)

    def p_xt_x0(self, x0, xt, t):
        """
            Given x0, xt and t, calculate P(x{t}|x{0}).
            x0: shape: [bs, len, # categories], must be a distribution
            t: diffusion steps(range: [1, T]) with shape [bs].
        """
        x0_mul_Q_bar = self.x_mul_Q_bar(x0, t)
        return torch.gather(x0_mul_Q_bar, 2, xt)

    def p_x_pre_t_xt_x0(self, x0, xt, t, return_logits=True):
        """
            Given x0, xt and t, calculate P(x{t-1}|x{t}, x{0}).
            x0: shape: [bs, len, # categories], must be a distribution
            xt: shape: [bs, len, 1], an observation.
            t: diffusion steps(range: [1, T]) with shape [bs].
        """
        if return_logits:
            a = self.p_xt_x_pre_t_x0(x0, xt, t)
            b = self.p_x_pre_t_x0(x0, t)
            return torch.log(a+1e-7) + torch.log(b+1e-7)
        else:
            numerator = self.p_xt_x_pre_t_x0(x0, xt, t) * self.p_x_pre_t_x0(x0, t)
            denominator = self.p_xt_x0(x0, xt, t)
            return numerator / (denominator + 1e-7)  # 1e-7 is used for numeric stability

    # def p_x_pre_t_xt(self, x0, xt, t):
    #     """
    #     calculate p(x{t-1}|x{t}) = sum{x0} p(x{t-1}|x{t},x0)*p(x0|xt)
    #     :param x0: x0 is predicted distribution
    #     :param xt: shape [bs, len, 1]
    #     :param t:
    #     :return:
    #     """
    #     res_biased = ((self.p_xt_x_pre_t_x0(x0, xt, t) * self.p_x_pre_t_x0(x0, t)).permute(1,2,0) / (self.beta_bar[t]/self.K)).permute(2,0,1).contiguous()
    #     xt_onehot = F.one_hot(xt.squeeze(-1), self.K)
    #     fix_term = ((self.p_xt_x_pre_t_x0(xt_onehot, xt, t) * self.p_x_pre_t_x0(xt_onehot, t)) * x0.gather(2, xt)).permute(1,2,0)
    #     wrong_add = (fix_term / (self.beta_bar[t]/self.K)).permute(2,0,1).contiguous()
    #     true_add = (fix_term / (self.alpha_bar[t] + self.beta_bar[t]/self.K)).permute(2,0,1).contiguous()
    #     return res_biased - wrong_add + true_add

# test
if __name__ == '__main__':
    import torch.nn.functional as F
    from torch.distributions import Categorical
    import time
    np.random.seed(42)
    start_time = time.time()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bs, len, T, K = 64, 200, 100, 2707
    diff_hpc_moudle = Diffuse_Precompute_Moudle('uni', 'multi-power', T, K, len, 8, [0.02, 1])
    x0 = torch.randint(0, K - 1, (bs, len)).to(dev)
    print(x0)
    x0_onehot = F.one_hot(x0, K)
    t = torch.randint(T, T + 1, (bs,)).to(dev)

    xT_prob = diff_hpc_moudle.x_mul_Q_bar(x0_onehot, t)
    cat_dist = Categorical(probs=xT_prob)
    xT = cat_dist.sample()
    print(xT)

    xt = xT
    denoise_step = T  # 7
    for t in reversed(range(T - denoise_step + 1, T)):
        t = torch.randint(t, t + 1, (bs,)).to(dev)
        x_pre_t_prob_logits = diff_hpc_moudle.p_x_pre_t_xt_x0(x0_onehot, xt.unsqueeze(2), t)
        x_pre_t_prob_label = diff_hpc_moudle.x_mul_Q_bar(x0_onehot, t-1)
        # p_x_pre_t_xt = diff_hpc_moudle.p_x_pre_t_xt_x0(x0_onehot, xt.unsqueeze(2), t)
        # print(x_pre_t_prob)
        # print(x_pre_t_prob_label)
        # print('numeric check: ', torch.allclose(x_pre_t_prob, x_pre_t_prob_label, rtol=0.01), 'failure ratio: ', torch.abs(x_pre_t_prob-x_pre_t_prob_label).max())
        # print(xt_prob.sum(dim=-1))
        cat_dist = Categorical(x_pre_t_prob_label, validate_args=False)
        xt = cat_dist.sample()
        print(xt)
        # xt_predict = torch.argmax(x_pre_t_prob, dim=-1)
        # xt = torch.argmax(x_pre_t_prob_label, dim=-1)
        # if torch.allclose(xt_predict, xt, atol=0) == False:
        # print(xt_predict)
        # print(xt)
        # print('numeric check: ', torch.allclose(xt_predict, xt, atol=0), 'failure ratio: ', ((~torch.eq(xt_predict, xt)).sum()) / (bs * len))
    # print(x_pre_t)
    end_time = time.time()
    print('time consumed: ', end_time - start_time)
    # print('numeric check: ', torch.all(torch.eq(x0, xt)), 'failure ratio: ', ((~torch.eq(x0, xt)).sum()) / (bs * len))
