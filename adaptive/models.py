import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger("model")


class AttentionGQN(torch.nn.Module):
    def __init__(self, dataset_param, config):
        super().__init__()

        self.v_size, self.x_size, self.q_v_size, self.q_x_size, self.num_context, self.num_queries = dataset_param

        self.h_size = config.h_size
        self.r_size = self.h_size
        self.c_size = self.num_queries * self.h_size + self.r_size

        self.init_repr = torch.nn.Parameter(torch.zeros([self.r_size]))

        self.encoder = Encoder(self.v_size, h_size=self.h_size, num_heads=config.num_heads, num_layers=config.num_layers)
        self.decoder = Decoder(self.c_size, h_size=self.h_size, num_heads=config.num_heads)
        if config.repr_net == "encoder":
            self.repr_net = Encoder(self.h_size + self.x_size, self.r_size, num_heads=config.num_heads, num_layers=config.num_layers)
        else:
            self.repr_net = MLP(self.h_size + self.x_size, self.h_size, [self.h_size, self.h_size * 4, self.h_size * 2])
        self.repr_fn = getattr(config, "repr_fn", None)

    def forward(self, context_v, context_x, query_v, config):

        batch_size = context_x.shape[0]
        num_context = self.num_context

        padded_query = torch.nn.functional.pad(query_v, [0, self.v_size - self.q_v_size])
        h = self.encoder(torch.concat([context_v, padded_query], dim=1))
        context_h, query_h = h.split([num_context, self.num_queries], dim=1)
        long_context_h = context_h.repeat_interleave(config.max_num_samples, dim=0)
        query_h = query_h.view([batch_size, self.num_queries * self.h_size])

        context_hx = torch.concat([context_h, context_x], dim=-1).unsqueeze(1)
        context_hx = context_hx.expand([-1, config.max_num_samples, -1, -1])

        num_samples = 1
        weights = torch.ones([batch_size, num_samples], device=context_h.device)
        visited = torch.zeros([batch_size, num_samples, num_context], dtype=torch.int, device=context_h.device)

        for r in range(config.recurrence):
            if r == 0:
                mean_repr = self.init_repr.expand([batch_size, -1])
            else:
                visited_hx = context_hx[:, :num_samples][visited.bool()]
                visited_hx = visited_hx.view(batch_size * num_samples, r, self.h_size + self.x_size)
                mean_repr = self.repr_net(visited_hx)
                mean_repr = mean_repr.mean(-2)

            if num_samples == config.max_num_samples:
                tmp_long_context_h = long_context_h
            else:
                tmp_long_context_h = context_h.repeat_interleave(num_samples, dim=0)
            weights = self.step(tmp_long_context_h, query_h, mean_repr, visited, weights)

            num_valid_candidates = num_samples * (num_context - r)
            num_samples = min(config.max_num_samples, num_valid_candidates)
            weights, top_indices = weights.topk(num_samples, dim=1)
            weights = F.normalize(weights, p=1, dim=-1)

            batch_indexer = torch.arange(batch_size, device=visited.device).unsqueeze(-1)
            sample_indexer = top_indices // num_context
            context_indexer = top_indices % num_context
            visited = visited[batch_indexer, sample_indexer]
            sample_indexer2 = torch.arange(num_samples, device=visited.device).view([1, -1])
            assert not visited[batch_indexer, sample_indexer2, context_indexer].bool().any(), "duplicate visit"
            visited[batch_indexer, sample_indexer2, context_indexer] = r + 1

        if config.get_repr:
            return visited, weights, mean_repr.view([batch_size, num_samples, self.r_size])

        return visited, weights

    def rollout(self, context_v, context_x, query_v, config, greedy=False):
        """Generate one complete trajectory and its policy log-probability.

        ``greedy=False`` samples one action from the conditional policy at
        every recurrence, which is the trajectory required by REINFORCE.
        ``greedy=True`` performs the deterministic rollout used by the
        frozen baseline policy.
        """
        batch_size = context_x.shape[0]
        num_context = context_v.shape[1]
        recurrence = config.recurrence

        padded_query = F.pad(query_v, [0, self.v_size - self.q_v_size])
        h = self.encoder(torch.concat([context_v, padded_query], dim=1))
        context_h, query_h = h.split([num_context, self.num_queries], dim=1)
        query_h = query_h.flatten(1)

        context_hx = torch.concat([context_h, context_x], dim=-1)
        visited = torch.zeros(
            [batch_size, num_context], dtype=torch.long, device=context_h.device
        )
        log_prob = torch.zeros([batch_size], dtype=context_h.dtype, device=context_h.device)

        for r in range(recurrence):
            if r == 0:
                mean_repr = self.init_repr.expand([batch_size, -1])
            else:
                visited_hx = context_hx[visited.bool()]
                visited_hx = visited_hx.view(batch_size, r, self.h_size + self.x_size)
                mean_repr = self.repr_net(visited_hx).mean(-2)

            c = torch.concat([mean_repr, query_h], dim=-1)
            probs = self.decoder(c, context_h, visited.bool())

            if greedy:
                action = probs.argmax(dim=-1)
            else:
                action = torch.distributions.Categorical(probs=probs).sample()

            selected_prob = probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
            log_prob = log_prob + selected_prob.clamp_min(torch.finfo(probs.dtype).tiny).log()
            visited.scatter_(1, action.unsqueeze(-1), r + 1)

        return visited.unsqueeze(1), log_prob

    def step(self, long_context_h, query_h, mean_repr, visited, weights):
        batch_size, num_samples, num_context = visited.shape

        long_query_h = query_h.unsqueeze(1).expand([-1, num_samples, -1])
        mean_repr = mean_repr.view([batch_size, num_samples, self.r_size])

        c = torch.concat([mean_repr, long_query_h], dim=-1)
        c = c.view([batch_size * num_samples, self.c_size])

        next_prob = self.decoder(c, long_context_h, visited.view([batch_size * num_samples, num_context]).bool())
        next_prob = next_prob.view([batch_size, num_samples, num_context])
        weights = weights.unsqueeze(-1) * next_prob
        weights = weights.masked_fill(visited.bool(), -torch.inf)

        weights = weights.view([batch_size, num_samples * num_context])

        return weights




class Encoder(torch.nn.Module):
    def __init__(self, v_size, h_size, num_layers=1, num_heads=1):
        super().__init__()
        self.projection = torch.nn.Linear(v_size, h_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=h_size, nhead=num_heads, dim_feedforward=h_size * 4, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, enable_nested_tensor=False)

    def forward(self, v):
        h_0 = self.projection(v)
        h_2 = self.transformer_encoder(h_0)
        return h_2


class Decoder(torch.nn.Module):
    def __init__(self, c_size, h_size, num_heads=1):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(c_size, num_heads, kdim=h_size, vdim=h_size, batch_first=True)
        self.prob_layer = torch.nn.MultiheadAttention(c_size, num_heads=1, kdim=h_size, vdim=h_size, batch_first=True)

    def forward(self, c, h, visited_mask):
        c = c.unsqueeze(1)
        c_1, p_1 = self.mha(c, h, h, need_weights=True)
        c_2, p_2 = self.prob_layer(c_1, h, h, need_weights=True, key_padding_mask=visited_mask)
        return p_2.squeeze(1)


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        nn = []
        for i, o in zip([input_size] + hidden_size, hidden_size + [output_size]):
            nn.append(torch.nn.Linear(i, o))
            nn.append(torch.nn.ReLU())
        self.nn = torch.nn.Sequential(*nn)

    def forward(self, x):
        return self.nn(x)
