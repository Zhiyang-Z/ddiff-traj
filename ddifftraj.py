import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from data_loader import TrajectoryData
import wandb
from tqdm import tqdm

from Diffuse_Precompute_Module import Diffuse_Precompute_Moudle
from X0_Conditional_Model import X0_Conditional_Model
from unet import Model

class ddifftraj:
    def __init__(self,
                 T,
                 diff_mode,
                 hybrid_loss_coeff,
                 noise_schedule,
                 K,
                 N,
                 P,
                 noise_range,
                 x0_model_config,
                 optim_config) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.diff_mode, self.T, self.K, self.N, self.P = diff_mode, T, K, N, P
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.precomp = Diffuse_Precompute_Moudle(self.diff_mode,
                                                 noise_schedule,
                                                 self.T,
                                                 self.K,
                                                 self.N,
                                                 self.P,
                                                 noise_range)
        self.x0_model = Model(self.K, None).to(self.device)
        # self.x0_model = (X0_Conditional_Model(K=self.K,
        #                                      nlayers=x0_model_config['nlayer'],
        #                                      nhead=x0_model_config['nhead'],
        #                                      model_dim=x0_model_config['ndim'],
        #                                      feedforward_dim=x0_model_config['ndim_feedforward']))
        if x0_model_config['model_compile']: self.x0_model = torch.compile(self.x0_model)
        self.x0_model = self.x0_model.to(self.device)
        self.optim_config = optim_config

    def diffuse(self, x0, T):
        # x0 shape: [bs, length] need to transfer into shape: [bs, length, category]. x0: [[12, 23, 53, ..., <EOT>, <EOT>], [...], [...], ...]
        bs = x0.shape[0]
        x0_onehot = F.one_hot(x0, self.K)
        t = torch.randint(1, T, (bs,), device=self.device)
        x0_diffused_prob = self.precomp.x_mul_Q_bar(x0_onehot, t)
        cat_dist = Categorical(probs=x0_diffused_prob, validate_args=False)
        x0_diffused = cat_dist.sample()
        return x0_onehot, x0_diffused, t

    def KL(self, p1, p2, logits=True):
        # print(p1.sum(dim=-1), p2.sum(dim=-1))
        p1 = p1.flatten(start_dim=0, end_dim=-2)
        p2 = p2.flatten(start_dim=0, end_dim=-2)
        # calculate E[log(p/q)]
        if logits:
            out = torch.softmax(p1 + 1e-7, dim=-1) * (
                    torch.log_softmax(p1 + 1e-7, dim=-1) -
                    torch.log_softmax(p2 + 1e-7, dim=-1))
        else:
            out = p1 * (torch.log(p1 + 1e-7) - torch.log(p2 + 1e-7))
        return torch.nn.ReLU()(out.sum(dim=-1)).mean()

    def sample_x0(self, xt, t, y):
        self.x0_model.clear_memory()
        x0 = torch.zeros(xt.shape[0],1).to(self.device).long()
        for _ in range(xt.shape[1]+1):
            pre_x0 = self.x0_model(xt.unsqueeze(2), t, y, x0.unsqueeze(2), True)[:,-1,:]
            cat_dist = Categorical(probs=torch.softmax(pre_x0, dim=-1))
            x0_next = cat_dist.sample()
            # x0_next = torch.argmax(pre_x0, dim=-1)
            new_x0 = torch.zeros(x0.shape[0], x0.shape[1]+1).to(self.device).long()
            new_x0[:,:-1], new_x0[:, -1] = x0, x0_next
            for i in range(new_x0.shape[0]):
                if new_x0[i][-1] == 0 or new_x0[i][-1] == self.K + 1: new_x0[i][-1] = new_x0[i][-2]
            x0 = new_x0
        return x0[:,1:-1]-1

    def train(self, dataloader):
        wandb.init(project="ddifftraj")
        lr = optim_config['lr']
        optim = torch.optim.AdamW(self.x0_model.parameters(), lr=optim_config['lr'])
        total_params = sum(p.numel() for p in self.x0_model.parameters())
        print(f"Total parameters: {total_params}")
        update_step = 0
        scaler = torch.cuda.amp.GradScaler()
        ce_loss = torch.nn.CrossEntropyLoss()
        for _ in tqdm(range(1000000)):
            for batch, (x0, y) in enumerate(dataloader):
                x0, y = torch.tensor(x0).to(self.device).long(), y.to(self.device).float()
                self.x0_model.train()
                # calculate perturbed data.
                x0_onehot, x0_diffused, t = self.diffuse(x0, self.T)
                # rand_idx = np.random.choice(np.array(range(y.shape[0])), size=np.floor(y.shape[0]*0.1).astype(np.int32), replace=False)
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=True):
                    pre_x0 = self.x0_model(x0_diffused.unsqueeze(2), t, y)#, torch.tensor(rand_idx))
                    # 1. ce loss:
                    loss_ce = ce_loss(pre_x0.flatten(start_dim=0, end_dim=-2), x0.flatten())
                    # 2. vb loss:
                    pre_prob_logits = self.precomp.p_x_pre_t_xt_x0(torch.softmax(pre_x0, dim=-1), x0_diffused.unsqueeze(2), t)
                    label_prob_logits = self.precomp.p_x_pre_t_xt_x0(x0_onehot, x0_diffused.unsqueeze(2), t)
                    loss_vb = self.KL(label_prob_logits, pre_prob_logits)
                    # 3. final loss
                    loss = self.hybrid_loss_coeff * loss_ce + loss_vb
                # update parameters
                optim.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                # optim.step()
                update_step += 1
                wandb.log({"ce_loss": loss_ce.detach(), "vb_loss": loss_vb.detach(), "total_loss": loss.detach()})
                if update_step % 1000 == 0:
                    torch.save({'model_para': self.x0_model.state_dict()},
                                self.optim_config['save_path'] + 'model_' + str(update_step) + '.ckpt')
        wandb.finish()

    def eval(self, animation=False):
        saved_dict = torch.load('/data/programs_data/ddifftraj/model_' + str(7777777) + '1.ckpt')
        self.x0_model.load_state_dict(saved_dict['model_para'])
        self.x0_model.eval()
        if self.diff_mode == 'uni':
            noise = torch.randint(0, self.K, (3, 200)).to(self.device)
        elif self.diff_mode == 'abs':
            noise = torch.full((3, 200), self.K - 1).to(self.device)
        else:
            raise NotImplementedError
        y = np.load('/data/datasets/DIDI_GAIA/chengdu_201611/unzip/extract_from_csv/road_net_trajs/total30_label.npy')
        y[:, 0] = np.floor((y[:, 0] % 86400) / 300)
        means = y[:, 1:6].mean(axis=0)
        stds = y[:, 1:6].std(axis=0)
        y[:, 1:6] = (y[:, 1:6] - means) / stds
        start_from = 1001
        y = torch.tensor(y[start_from:start_from+3]).to(self.device).float()
        y[1,:] = y[0,:]
        y[2, :] = y[0, :]
        # y[:, 1] += 1
        # y[:, 2] += 1
        # max_distance = y[:, 3].max()
        # y[:, 0] = (y[:, 0] % 86400) / 300
        # y[:, 3] = (y[:, 3]) / max_distance
        # print(y_test)
        t = torch.full((3,), self.T).to(self.device)
        for i in tqdm(reversed(range(1, self.T))):
            t = t - 1
            x0_prob_est = self.x0_model(noise.unsqueeze(2), t, y)
            # cat_dist_x0 = Categorical(probs=torch.softmax(x0_prob_est, dim=-1))
            # x0_sample = cat_dist_x0.sample()
            # print(x0_sample[2])
            # x0_cat_dist = Categorical(probs=torch.softmax(x0_prob_est, dim=-1))
            # print(x0_cat_dist.sample())
            # exit()
            # x0_est = self.sample_x0(x0_est, t, y[0:3])
            # cat_dist = Categorical(probs=torch.softmax(x0_prob_est, dim=-1))
            # x0_est = cat_dist.sample()
            pre_prob_logits = self.precomp.p_x_pre_t_xt_x0(torch.softmax(x0_prob_est, dim=-1), noise.unsqueeze(2), t)
            # pre_prob = self.precomp.p_x_pre_t_xt_x0(x0_onehot[0:3], noise.unsqueeze(2), t)
            # print(pre_prob[2])
            cat_dist = Categorical(logits=pre_prob_logits)
            noise = cat_dist.sample()
            # for i in range(noise.shape[0]):
            #     tail = False
            #     for j in range(noise.shape[1]):
            #         if tail:
            #             noise[i][j] = self.K - 1
            #             continue
            #         elif noise[i][j] == self.K - 1: tail = True

            print(noise[2])
        # print(noise)
        np.save('/data/programs_data/ddifftraj/noise.npy', noise.cpu().numpy())
        trajs = np.load(
            '/data/datasets/DIDI_GAIA/chengdu_201611/unzip/extract_from_csv/road_net_trajs/total30.npy')

        print(y)
        print(trajs[start_from:start_from+3])
        # print(noise)
                        
if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    import yaml
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dataset_config = config['dataset']
    diffuse_config = config['diffuse']
    model_config = config['model']
    optim_config = config['optim']

    dataset = TrajectoryData(dataset_config['data_path'], dataset_config['label_path'])
    dataloader = DataLoader(dataset, batch_size=optim_config['batch_size'], shuffle=dataset_config['shuffle'], num_workers=dataset_config['num_workers'])
    diffuser = ddifftraj(diffuse_config['T'],
                         diffuse_config['mode'],
                         optim_config['hybrid_loss_coeff'],
                         diffuse_config['noise_schedule'],
                         diffuse_config['K'],
                         diffuse_config['N'],
                         diffuse_config['P'],
                         [0.02, 1],
                         model_config,
                         optim_config)
    diffuser.train(dataloader)
    # diffuser.eval()




