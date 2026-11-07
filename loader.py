
from property.model import property_nn, heisenberg_nn
from property.model import (
    ClusterIsingDatasetAdapter,
    ESSHDatasetAdapter,
)

def load_backbone_and_dataset(TASK):
    match TASK:
        case "cluster_ising":
            backbone = property_nn()
            ds = ClusterIsingDatasetAdapter(4096)

        case "essh":
            nsites = 50
            backbone = heisenberg_nn(model_path=f"property/data/models/essh_{nsites}")
            filter_fn = None

            ds = ESSHDatasetAdapter(441, Nsites=nsites, with_B=False, filter_fn=filter_fn)
        case _:
            raise NotImplementedError(f"Task {TASK} is not implemented")
    return backbone, ds