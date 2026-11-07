from torch import nn
import torch
import numpy as np

from adaptive.utils import evaluate_backbone

keys = ['P2x_exps','P2z_exps','P2y_exps','Px_exps','Pzxz_exps','correlation_xs','correlation_zs','entropies','RMIs','inter_entropies','target_fidelities', 'phase_value']
Heisenberg_keys  = ['P2x_exps','P2z_exps','RMIs','RMI2s', 'phase_value']
twoD_keys = ['expectations','entropies']
cat_keys = ['mpn', 'nprob', 'lw_func','parity']

def get_loss_fn(backbone, by_key=False, Loss=nn.MSELoss):
    LossObj = Loss(reduction="none")
    def loss_fn(path, data):
        batch_size, num_samples, num_context = path.shape
        context_v, context_x, query_v, query_x = data

        x_mu, _, _ = evaluate_backbone(path, data, backbone)

        x_mu = {k: v.view(batch_size, num_samples, -1) for k, v in x_mu.items()}
        query_x = {k: v.to(x_mu[k].device).view([batch_size, 1, -1]).expand(-1, num_samples, -1) for k, v in data[-1].items()}
        loss, key_loss = LossCriterion(x_mu, query_x, Loss=LossObj)
        if by_key:
            loss = {k: v.mean(dim=-1) for k, v in key_loss.items()}
            if "P2z_exps" in loss and "P2x_exps" in loss:
                P2z = loss.pop("P2z_exps")
                P2x = loss.pop("P2x_exps")
                loss["$\\mathcal{C}_{1i}$"] = (P2z + P2x) / 2
            if "entropies" in loss:
                loss["$\\mathcal{S}_A$"] = loss.pop("entropies")
            if "phase_value" in loss:
                loss["$\\mathcal{Z}_{R}$"] = loss.pop("phase_value")
        else:
            loss = loss.mean(dim=-1)
        return loss

    return loss_fn


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        loss = torch.log(torch.cosh(ey_t + 1e-12))
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class CODLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x, y):
        res = ((x - y) ** 2).sum(dim=0, keepdim=True)
        mean = y.mean(dim=0, keepdim=True)
        tot = ((y - mean) ** 2).sum(dim=0, keepdim=True)
        loss = 1 - res / tot
        loss[loss < 0] = 0
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def LossCriterion(outputs, targets, Loss):
    key_loss = {}
    for key in targets.keys():
        target =  targets[key]
        output = outputs[key][..., :target.shape[-1]]
        key_loss[key] = Loss(output,target)
    
    loss = sum(key_loss.values())
    return loss, key_loss


def HeisenbergLossCriterion2(outputs, targets, reduction="mean"):
    loss = 0
    key_loss = {}
    for key in Heisenberg_keys:
        key_loss[key] = 0
    for key in Heisenberg_keys:
        if key in targets.keys() and key in outputs.keys():
            target =  targets[key].reshape(-1)
            output = outputs[key].reshape(-1)
            key_loss[key] += LogCoshLoss(reduction)(output,target)
            loss += LogCoshLoss(reduction)(output,target)
    return loss, key_loss


def TestLossCriterion_ablation(outputs, targets):
    loss = 0
    keys = ['phase_value']
    key_loss = np.zeros(3)

    for ikey in range(len(keys)):
        key = keys[ikey]
        target = targets[key]
        output = outputs[key][..., 0:target.shape[-1]]
        key_loss[ikey] = nn.MSELoss(reduction="mean")(output, target)

    return key_loss