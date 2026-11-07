from torch import nn
from pathlib import Path
MODEL_DIR = Path("property") / "_data" / "models"


from .model import PreHeisenbergPropertyModel, PrePropertyModel_ablation, Backbone, TanH_Decoder, NoAct_Decoder


from .dataset_adapter import *
from .loss_fn import get_loss_fn, CODLoss


auto_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_state_dict(model, state_path, device=None, *, requires_grad=False, eval=False):
    if device is None:
        device = auto_device
    state_dict = torch.load(state_path, map_location=device)
    store_has_module = all(map(lambda k: k.startswith("module."), state_dict.keys()))
    model_has_module = hasattr(model, "module")
    if store_has_module and not model_has_module:
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    elif not store_has_module and model_has_module:
        state_dict = {"module." + k: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.requires_grad_(requires_grad)
    if eval:
        model.eval()

HEISENBERG_STATE_PATH = MODEL_DIR / "model2.pt"
PROPERTY_STATE_PATH = MODEL_DIR / "ppmodel_num_qubits9num_measure_qubits3_r_dim32_num_known_mbases50_train_size300_P2x_P2z_etp"


def property_nn(num_measure_qubits=3, r_dim=32, model_path=PROPERTY_STATE_PATH):
    backbone = Backbone(x_dim=2**num_measure_qubits, v_dim=2 * 4**num_measure_qubits, r_dim=r_dim)
    exps_decoder = TanH_Decoder(r_dim=r_dim, output_dim=9, task_dim=1)
    entropy_decoder = TanH_Decoder(r_dim=r_dim, output_dim=9, task_dim=1)
    model = PrePropertyModel_ablation(backbone, exps_decoder=exps_decoder, entropy_decoder=entropy_decoder)
    if model_path is not None:
        load_state_dict(model, model_path)
    return model


def heisenberg_nn(num_measure_qubits=3, r_dim=24, model_path=HEISENBERG_STATE_PATH):
    backbone = Backbone(x_dim=2**num_measure_qubits, v_dim=2 * 4**num_measure_qubits, r_dim=r_dim)
    exps_decoder = TanH_Decoder(r_dim=r_dim, output_dim=1, task_dim=1)
    entropy_decoder = None
    model = PreHeisenbergPropertyModel(backbone, exps_decoder=exps_decoder, entropy_decoder=entropy_decoder)
    if model_path is not None:
        load_state_dict(model, model_path)
    return model

