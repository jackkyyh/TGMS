import copy
import logging
from torch.distributions import Normal
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from tqdm import tqdm

auto_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("utils")

AGGR_FN = {
    "last": lambda x: x[-1],
    "sum": lambda x: x.sum(),
    "decayed sum": lambda x: sum(x * 1.2 ** torch.arange(len(x))),
}


class BaseConfig:
    pass


class ModelConfig(BaseConfig):
    def __init__(self):
        self.h_size = 32


class CommonConfig(BaseConfig):
    def __init__(self):
        self.model = ModelConfig()
        self.device = auto_device
        self.do_test = True
        self.lr = 1e-3
        self.epochs = 10
        self.batch_size = 128
        self.save_path = None
        self.verbose = True


class AdaptiveConfig(BaseConfig):
    def __init__(self):
        self.max_num_samples = 10
        self.num_rollout_samples = 4
        self.parallel = False
        self.get_repr = False
        self.progressive = False
        self.recurrence = 10

class Config(CommonConfig):
    def __init__(self):
        super().__init__()
        self.adaptive = AdaptiveConfig()
        self.model.repr_net = "mlp"


def _repeat_batch(value, repeats):
    """Repeat a collated batch value along its leading batch dimension."""
    if torch.is_tensor(value):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, dict):
        return {key: _repeat_batch(item, repeats) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_repeat_batch(item, repeats) for item in value)
    if isinstance(value, list):
        return [_repeat_batch(item, repeats) for item in value]
    return value


def epoch(model, loader, config, baseline_model=None, optimizer=None):
    log_list = [[], [], []]

    model.train(optimizer is not None)

    for data in loader:
        input = [d.to(config.device) for d in data[:3]]
        data = input + data[3:]
        batch_size = input[0].shape[0]

        if optimizer is None:
            path, _ = model.rollout(*input, config.adaptive, greedy=True)
            loss_data = data
        else:
            if baseline_model is None:
                raise ValueError("baseline_model is required for REINFORCE training")

            num_rollout_samples = config.adaptive.num_rollout_samples
            rollout_input = [
                value.repeat_interleave(num_rollout_samples, dim=0)
                for value in input
            ]
            loss_data = [
                _repeat_batch(value, num_rollout_samples) for value in data
            ]
            path, log_prob = model.rollout(
                *rollout_input, config.adaptive, greedy=False
            )
            with torch.no_grad():
                baseline_path, _ = baseline_model.rollout(
                    *input, config.adaptive, greedy=True
                )

        loss = config.loss_fn(path, loss_data)
        selected_loss = loss.reshape(loss.shape[0], -1).mean(-1).detach()
        if optimizer is not None:
            baseline_loss = config.loss_fn(baseline_path, data)
            baseline_loss = baseline_loss.reshape(baseline_loss.shape[0], -1).mean(-1)
            sample_loss = selected_loss.view(batch_size, num_rollout_samples)
            advantage = sample_loss - baseline_loss.detach().unsqueeze(1)
            policy_loss = (
                advantage * log_prob.view(batch_size, num_rollout_samples)
            ).mean()
            policy_loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            log_list[2].append(policy_loss.detach().view(1))

        selected_reward = config.reward_fn(path, loss_data)
        selected_reward = selected_reward.reshape(selected_reward.shape[0], -1).mean(-1)
        if optimizer is not None:
            selected_reward = selected_reward.view(batch_size, num_rollout_samples).mean(-1)
        log_list[0].append(selected_loss)
        log_list[1].append(selected_reward)

    log_list = [torch.concat(log).mean().cpu() if len(log) > 0 else [] for log in log_list]
    return log_list


def train(model, train_set, test_set, config: Config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    log_list = [[[], [], []], [[], [], []]]

    baseline_model = copy.deepcopy(model)
    baseline_model.eval()
    baseline_model.requires_grad_(False)

    original_recurrence = config.adaptive.recurrence
    pbar = tqdm(range(config.epochs))
    try:
        for e in pbar:
            if config.adaptive.progressive:
                config.adaptive.recurrence = int(original_recurrence * (e / config.epochs)) + 1
            log = epoch(model, train_loader, config, baseline_model, optimizer)
            for lst, log in zip(log_list[0], log):
                lst.append(log)

            if config.do_test:
                with torch.inference_mode():
                    log = epoch(model, test_loader, config, None, None)
                for lst, log1 in zip(log_list[1], log):
                    lst.append(log1)

            baseline_model.load_state_dict(model.state_dict())
            baseline_model.eval()
            baseline_model.requires_grad_(False)

            pbar.set_description(f"Loss: {log_list[0][0][-1]:.3f}")
    except torch.OutOfMemoryError as err:
        logger.exception(f"Out of memory error at epoch {e} with recurrence {config.adaptive.recurrence}")
        torch.cuda.empty_cache()
        
    if config.save_path:
        print("saving model to", config.save_path)
        torch.save(model.state_dict(), config.save_path)
    return log_list


class LossFn:
    def cfidelity(query_x, x_mu):
        x_mu = F.normalize(torch.relu(x_mu), p=1, dim=-1)
        query_x = F.normalize(query_x, p=1, dim=-1)

        fidelity = (query_x.sqrt() * x_mu.sqrt()).sum(dim=-1).mean(-1)
        return fidelity

    def reconstruction_loss(query_x, x_mu, sigma=0.1):
        nll = -Normal(x_mu, sigma).log_prob(query_x).mean(dim=(-1, -2))
        return nll

    def mse_loss(query_x, x_mu):
        return ((query_x - x_mu) ** 2).mean((-1, -2))




def evaluate_backbone(path, data, backbone):
    context_v, context_x, query_v, query_x = data
    batch_size, num_samples, num_context = path.shape
    v_size = context_v.shape[-1]
    x_size = context_x.shape[-1]

    context_x = context_x.unsqueeze(1).expand([-1, num_samples, -1, -1])
    context_v = context_v.unsqueeze(1).expand([-1, num_samples, -1, -1])
    context_x = context_x[path.bool()].view([batch_size * num_samples, -1, x_size])
    context_v = context_v[path.bool()].view([batch_size * num_samples, -1, v_size])
    query_v = query_v.repeat_interleave(num_samples, dim=0)
    with torch.no_grad():
        x_mu, _, r = backbone.sample(context_x, context_v, query_v)

    return x_mu, _, r


def gen_random_path(num_states, num_samples, num_context, depth, device="cpu"):
    assert num_context >= depth, "Depth should be less than or equal to num_context"
    path_rand = torch.argsort(torch.rand([num_states, num_samples, num_context], device=device))
    path_rand[path_rand > depth] = 0
    return path_rand


def gather(data, index):
    """
    Gather data from a tensor using index tensor.

    data: [*batch_dim, N, *feature_dim]

    index: [*batch_dim, 1]

    """
    data_shape = list(data.shape)
    index_shape = list(index.shape)
    select_dim = len(index_shape) - 1
    expanded_shape = data_shape
    expanded_shape[select_dim] = 1



    index = index.view(index_shape + [1] * (len(data_shape) - len(index_shape))).expand(expanded_shape)
    result = torch.gather(data, dim=select_dim, index=index)

    return result


def move_context_to_device(data):
    data = [d.to(auto_device) for d in data[:3]] + data[3:]
    return data
