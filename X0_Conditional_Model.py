import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import utils

from Timestep_Transformer import Timestep_Transformer

class TimestepEmbedder(nn.Module):
    def __init__(self, embedding_dim, frequency_embedding_dim=786):
        super().__init__()
        self.register_buffer('sinusoidalEmbeddings', utils.gen_pos_encoding(5000, frequency_embedding_dim))
        # self.mlp = nn.Sequential(
        #     nn.Linear(frequency_embedding_dim, embedding_dim),
        #     nn.SiLU(),
        #     nn.Linear(embedding_dim, embedding_dim),
        # )

    def forward(self, t):
        t_sin_embedding = self.sinusoidalEmbeddings[t,:]
        # t_emb = self.mlp(t_sin_embedding).unsqueeze(1)
        return t_sin_embedding # t_emb

class X0_Conditional_Model(nn.Module):
    def __init__(self,
                 K,
                 nlayers,
                 nhead,
                 model_dim,
                 feedforward_dim):
        super(X0_Conditional_Model, self).__init__()
        self.K, self.model_dim = K, model_dim
        # condition embedding:
        self.departure_time_embedding = nn.Embedding(288, self.model_dim)
        self.start_region_embedding = nn.Embedding(256, self.model_dim)
        self.end_region_embedding = nn.Embedding(256, self.model_dim)
        self.node_embedding = nn.Embedding(self.K, self.model_dim)
        self.trip_distance_fc, self.trip_time_fc, self.trip_length_fc, self.avg_distance_fc, self.avg_speed_fc = (
                                                                                nn.Linear(1, self.model_dim),
                                                                                nn.Linear(1, self.model_dim),
                                                                                nn.Linear(1, self.model_dim),
                                                                                nn.Linear(1, self.model_dim),
                                                                                nn.Linear(1, self.model_dim))
        # timestep embedding
        self.timestep_embedding = TimestepEmbedder(self.model_dim)
        # cache position encoding.
        self.register_buffer('pos_encoding', utils.gen_pos_encoding(10000, self.model_dim))
        # Transformer
        # decoder_only transformer
        self.model = Timestep_Transformer(vocabulary_size=self.K,
                                          mode='all',
                                          nlayer=nlayers,
                                          nhead=nhead,
                                          ndim=model_dim,
                                          ndim_feedforward=feedforward_dim,
                                          drop_out=0.1,
                                          pre_norm=True)
        self._ini_para()

    def _ini_para(self):
        # for name, module in self.named_modules():
        #     print(f'{name}: {module.__class__.__name__}')
        print('embedding initializing...')
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, noise, t, condition):#, rand_idx):
        trip_distance_embedding, trip_time_embedding, trip_length_embedding, avg_distance_embedding, avg_speed_embedding = (
                   self.trip_distance_fc(condition[:, 1].unsqueeze(1)), self.trip_time_fc(condition[:, 2].unsqueeze(1)),
                   self.trip_length_fc(condition[:, 3].unsqueeze(1)), self.avg_distance_fc(condition[:, 4].unsqueeze(1)),
                   self.avg_speed_fc(condition[:, 5].unsqueeze(1)))
        start_region_embedding, end_region_embedding = (
            self.start_region_embedding(condition[:, 6].long()),
            self.end_region_embedding(condition[:, 7].long()))
        departure_time_embedding, start_region_embedding, end_region_embedding = (
                                                                  self.departure_time_embedding(condition[:, 0].long()),
                                                                  self.start_region_embedding(condition[:, 6].long()),
                                                                  self.end_region_embedding(condition[:, 7].long()))
        condition_embeddings = torch.stack((start_region_embedding, end_region_embedding))
        condition_embeddings = torch.stack((departure_time_embedding, start_region_embedding, end_region_embedding,
                                            trip_distance_embedding, trip_time_embedding, trip_length_embedding,
                                            avg_distance_embedding, avg_speed_embedding))
        tmb = self.timestep_embedding(t)
        condition_embeddings = condition_embeddings.permute(1, 0, 2).contiguous()
        # condition complete.
        # feed into model.
        decoder_in = self.node_embedding(noise.squeeze(2)) + self.pos_encoding[0:noise.shape[1],:]
        # condition_embeddings[rand_idx,:,:] = 0
        out = self.model(encoder_in=condition_embeddings,
                         encoder_out=None,
                         decoder_in=decoder_in,
                         timestep=tmb,
                         encoder_attn_mask=None,
                         decoder_attn_mask=None,
                         encoder_padding_mask=None,
                         decoder_padding_mask=None)
        return out

if __name__ == '__main__':
    x0_model = X0_Conditional_Model(K=2707,
                                    nlayers=[6,6],
                                    nhead=6,
                                    model_dim=768,
                                    feedforward_dim=2048)
    noise, t, condition = torch.randint(0, 2707, (3, 200, 1)), torch.randint(0, 100, (3,)), torch.randint(0, 100, (3, 8)).float()
    out = x0_model(noise, t, condition)
    print(out.shape)
