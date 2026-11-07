import torch
import numpy as np
from .dataset import ESSHData, StateData, StateVariationData, HeisenbergData, State4HData
from adaptive import DatasetAdapter



class BaseDatasetAdapter(DatasetAdapter):
    def __init__(self, num_states, num_qubits, Nsites):
        super().__init__(
            num_states=num_states,
            num_context=3**num_qubits * (Nsites - num_qubits + 1),
            x_size=2**num_qubits,
            v_size=4**num_qubits * 2 + num_qubits,
            q_v_size=0,
            q_x_size=1,
            num_queries=1,
        )

        self.observables = self.observables.astype("float32")
        self.input_data = self.input_data.astype("float32")


        self.keys = ["P2x_exps", "P2z_exps", "entropies"]

    def __getitem__(self, idx):
        v, x, q_x = self.parent.__getitem__(self, idx)
        q_v = torch.zeros([*v.shape[:-2], 1, 0])

        q_x = {k: v for k, v in q_x.items() if k in self.keys}
        aux = {k: getter(idx) for k, getter in self.aux_getter.items()}
        return (v, x, q_v, q_x) + ((aux,) if aux else ())

    def get_property(self, enabled, aux_key, prop_key):
        if enabled:
            self.aux_getter[aux_key] = lambda idx: self.output_properties[idx][prop_key]
        else:
            self.aux_getter.pop(aux_key, None)


class ClusterIsingDatasetAdapter(BaseDatasetAdapter, StateData):
    def __init__(self, num_states=1000, num_qubits=3, Nsites=9, variation=None):
        self.parent = StateData
        self.parent.__init__(self, num_states, Nsites, num_qubits)
        restricted_global_measurement = True if variation == "global" else False
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)

    def get_label(self, enabled):
        self.get_property(enabled, "label", "phase_label")

    def get_phase(self, enabled):
        self.get_property(enabled, "phase", "phase_value")

    def get_h1h2(self, enabled):
        self.get_property(enabled, "h1h2", "h1h2")


class ClusterIsingRandomDatasetAdapter(BaseDatasetAdapter, StateVariationData):
    def __init__(self, num_states=1000, num_qubits=3, Nsites=9, restricted_global_measurement=False):
        self.parent = StateVariationData
        self.parent.__init__(self, num_states, Nsites, num_qubits, variation="random")
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)

    def get_label(self, enabled):
        self.get_property(enabled, "label", "h1h2")

    def get_phase(self, enabled):
        self.get_property(enabled, "phase", "phase_value")


class ClusterIsingXXZZDatasetAdapter(BaseDatasetAdapter, StateVariationData):
    def __init__(self, num_states=1000, num_qubits=3, Nsites=9, restricted_global_measurement=False):
        self.parent = StateVariationData
        self.parent.__init__(self, num_states, Nsites, num_qubits, variation="XXZZ")
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)


class ClusterIsingSampleDatasetAdapter(BaseDatasetAdapter, StateVariationData):
    def __init__(self, num_states=4096, num_qubits=3, Nsites=9, restricted_global_measurement=False):
        self.parent = StateVariationData
        self.parent.__init__(self, num_states, Nsites, num_qubits, variation="sample")
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)


class ClusterIsingRingDatasetAdapter(BaseDatasetAdapter, StateVariationData):
    def __init__(self, num_states=1000, num_qubits=3, Nsites=9, restricted_global_measurement=False):
        self.parent = StateVariationData
        self.parent.__init__(self, num_states, Nsites, num_qubits, variation="ring")
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)

    def __getitem__(self, idx):
        v, *items = super().__getitem__(idx)
        v[..., -3:] /= 1
        return v, *items


class ClusterIsing4HDatasetAdapter(BaseDatasetAdapter, State4HData):
    def __init__(self, num_states=2000, num_qubits=3, Nsites=9, restricted_global_measurement=False, filter_fn=None):
        self.parent = State4HData
        self.parent.__init__(self, num_states, Nsites, num_qubits, filter_fn=filter_fn)
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)

        if filter_fn is not None:
            self.num_states = self.parent.__len__(self)

    def get_label(self, enabled):
        if enabled:
            self.aux_getter["label"] = lambda idx: self.output_properties[idx]["phase_label"]
        else:
            self.aux_getter.pop("label", None)

    def get_type(self, enabled):
        if enabled:
            self.aux_getter["type"] = lambda idx: self.output_properties[idx]["h1h2"][1]
        else:
            self.aux_getter.pop("type", None)


class HeisenbergDatasetAdapter(BaseDatasetAdapter, HeisenbergData):
    def __init__(self, num_states=441, num_qubits=3, Nsites=50, restricted_global_measurement=False):
        self.parent = HeisenbergData
        self.parent.__init__(self, num_states, Nsites, num_qubits)
        super().__init__(num_states, num_qubits, Nsites, restricted_global_measurement)

        self.keys = ["phase_value"]

    def get_phase(self, enabled):
        if enabled:
            self.aux_getter["phase"] = lambda idx: self.output_properties[idx]["phase_value"]
        else:
            self.aux_getter.pop("phase", None)


class ESSHDatasetAdapter(BaseDatasetAdapter, ESSHData):
    def __init__(self, num_states=400, num_qubits=3, Nsites=100, with_B=False, filter_fn=None):
        self.parent = ESSHData
        self.parent.__init__(self, num_states, Nsites, num_qubits, with_B, filter_fn)
        super().__init__(num_states, num_qubits, Nsites, False)

        self.keys = ["phase_value"]
        self.nsites = Nsites

        if filter_fn is not None:
            self.num_states = self.parent.__len__(self)

    def get_phase(self, enabled):
        if enabled:
            self.aux_getter["phase"] = lambda idx: self.output_properties[idx]["phase_value"]
        else:
            self.aux_getter.pop("phase", None)

    def get_label(self, enabled):
        if enabled:
            self.aux_getter["label"] = lambda idx: self.output_properties[idx]["phase_label"]
        else:
            self.aux_getter.pop("label", None)
