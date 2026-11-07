import logging
import sys
import torch
from torch.utils.data import random_split
from pathlib import Path

from loader import load_backbone_and_dataset

from property.model import get_loss_fn


from adaptive import AttentionGQN
from adaptive.utils import train, Config


logger = logging.getLogger("adaptive")


TASK = sys.argv[1]
logger.info(f"Running TGMS for task {TASK}...")

OUT_DIR = Path("_out") / TASK

backbone, ds = load_backbone_and_dataset(TASK)

loss_fn = get_loss_fn(backbone)
reward_fn = lambda *args: 1 - loss_fn(*args)



train_set, test_set = random_split(ds, [0.8, 0.2])

config = Config()
config.batch_size = 128
config.epochs = 1
config.lr = 1e-3
config.adaptive.recurrence = 8
config.adaptive.progressive = True
config.model.h_size = 24
config.model.num_heads = 4
config.model.num_layers = 2
config.model.repr_net = "mlp"
config.loss_fn = loss_fn
config.reward_fn = reward_fn
config.do_test = False
config.save_path = OUT_DIR / f"adaptive_{config.adaptive.recurrence}.pt"
config.adaptive.max_num_samples = 100
config.adaptive.num_rollout_samples = 12


model = AttentionGQN(ds.get_param(), config.model)
model = model.to(config.device)


result = train(model, train_set, test_set, config)
(train_loss_list, train_fidelity_list, grad_list), (test_loss_list, test_fidelity_list, _) = result

