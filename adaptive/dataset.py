import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetAdapter(Dataset):
    def __init__(self, num_states, num_context, v_size, x_size, q_v_size=None, q_x_size=None, num_queries=1):
        self.num_states = num_states
        self.num_context = num_context
        self.x_size = x_size
        self.v_size = v_size
        self.q_v_size = q_v_size if q_v_size is not None else v_size
        self.q_x_size = q_x_size if q_x_size is not None else x_size
        self.num_queries = num_queries

        self.gen_query()

        self.aux_getter = {}

    def __len__(self):
        return self.num_states

    def __getitem__(self, idx):
        x = self.x[idx]
        v = self.v.expand([*x.shape[:-1], self.v_size])
        q_idx = self.query_idx[idx]
        q_v = v[q_idx]
        q_x = x[q_idx]

        aux = {k: getter(idx) for k, getter in self.aux_getter.items()}

        if aux:
            return (v, x, q_v, q_x, aux)
        else:
            return (v, x, q_v, q_x)

    def get_param(self):
        return self.v_size, self.x_size, self.q_v_size, self.q_x_size, self.num_context, self.num_queries

    def gen_query(self, fixed=False):
        if fixed:
            self.query_idx = torch.arange(self.num_queries).repeat([self.num_states, 1]).int() + 4
        else:
            self.query_idx = torch.randint(self.num_context, size=[self.num_states, self.num_queries])

