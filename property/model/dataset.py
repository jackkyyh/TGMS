from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


DATA_DIR = "property" + "/_data/"

class StateData(Dataset):
    def __init__(self,num_states=30,num_qubits=9, num_measure_qubits = 3, max_num_states=4096, subdir=""):
        data_subdir = DATA_DIR + (subdir + "/" if subdir else "")
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(DATA_DIR+
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(DATA_DIR+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        for i in range(0, num_states):
            value = np.load(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_exact_partial' + str(num_measure_qubits) + '_probs'+str(i)+'.npy')
            value = value.reshape(-1,2**num_measure_qubits)
            values.append(value)
        self.input_data = np.array(values)

        output_properties = []
        h1h2 = np.load(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_h1h2s.npy').reshape(max_num_states,-1)



        P2x_exps = np.load(data_subdir+
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Paulix_expct_values.npy').real
        P2z_exps = np.load(data_subdir+
            str(num_qubits) + 'qubit/' + str(num_qubits) + 'qubit_two_point_Pauliz_expct_values.npy').real

        if Path(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').exists():
            phase_labels = np.load(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_labels.npy').reshape(max_num_states)
        else:
            phase_labels = h1h2[:, 0] > 0.45
        phase_values = np.load(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_phase_values.npy').reshape(max_num_states)

        entropies = np.load(data_subdir+str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_entanglement_entropies.npy').real.reshape((max_num_states,-1))

        for i in range(0, num_states):
            properties = {}

            properties['P2x_exps'] = P2x_exps[i]
            properties['P2z_exps'] = P2z_exps[i]

            properties['entropies'] = entropies[i]

            properties['phase_label'] = phase_labels[i].astype(int)
            properties['phase_value'] = phase_values[i]
            properties['h1h2'] = h1h2[i]


            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)


class StateVariationData(StateData):
    def __init__(self,num_states, num_qubits=9, num_measure_qubits = 3, variation=None):
        if variation not in ["ring", "XXZZ", "random", "sample"]:
            raise NotImplementedError(f"Variation {variation} is not implemented")
        
        subdir = variation
        max_num_states = 4096 if variation != "random" else 2000
        super().__init__(num_states, num_qubits, num_measure_qubits, max_num_states=max_num_states, subdir=subdir)


class State4HData(StateData):
    def __init__(self, num_states, num_qubits=9, num_measure_qubits = 3, filter_fn = None):
        super().__init__(num_states, num_qubits, num_measure_qubits, max_num_states=500*4, subdir="4h")

        if filter_fn is not None:
            idx = [i for i in range(num_states) if filter_fn(self.output_properties[i])]
            self.input_data = self.input_data[idx]
            self.output_properties = self.output_properties[idx]


class HeisenbergData(Dataset):
    def __init__(self,num_states=30,num_qubits=50, num_measure_qubits = 3):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(DATA_DIR +
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(DATA_DIR +str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        values = []
        J_ps = np.linspace(0, 3, 64)
        deltas = np.linspace(0, 4, 64)
        for id in range(1, len(deltas), 3):
            for iJ in range(1, len(J_ps), 3):
                J_p = J_ps[iJ]
                delta = deltas[id]
                value = np.load(DATA_DIR +'Heisenberg/prob3_' + str(num_qubits) + 'qubits_Jp' + str(J_p) + '_delta' + str(delta)+'.npy')
                value = value.reshape(-1, 2 ** num_measure_qubits)
                values.append(value)
        self.input_data = np.array(values)


        output_properties = []

        P2x_exps = np.squeeze(np.load(DATA_DIR +"Heisenberg/two_point_xs_"+str(num_qubits) + 'qubits.npy'))

        P2z_exps = np.squeeze(np.load(DATA_DIR +"Heisenberg/two_point_zs_"+str(num_qubits) + 'qubits.npy'))
        if num_qubits == 50:
            RMIs = np.squeeze(np.load(DATA_DIR +"Heisenberg/RMIs_"+str(num_qubits) + 'qubits.npy'))[:,0].reshape(-1, 1)
        else:
            RMIs = np.squeeze(np.load(DATA_DIR +"Heisenberg/RMIs_" + str(num_qubits) + 'qubits.npy'))
        RMI2s = np.squeeze(np.load(DATA_DIR +"Heisenberg/RMI2s_" + str(num_qubits) + 'qubits.npy')).reshape(-1, 1)


        if num_qubits == 50:
            indices = list(range(1, 64, 3))
            ZR_values = np.load(DATA_DIR +'Heisenberg/'+str(num_qubits)+'qubits_ZR_values.npy').reshape(64, 64)
            phase_values = ZR_values[indices][:, indices].reshape(-1, 1)
        elif num_qubits == 10:
            ZR_values = np.load(DATA_DIR +'Heisenberg/' + str(num_qubits) + 'qubits_ZR_values.npy').reshape(21, 21)
            phase_values = ZR_values.reshape(-1, 1)
        else:
            pass
        J_ps = np.linspace(0, 3, 64)
        deltas = np.linspace(0, 4, 64)
        jpdeltas = []
        for id in range(1, len(deltas), 3):
            for iJ in range(1, len(J_ps), 3):
                J_p = J_ps[iJ]
                delta = deltas[id]
                jpdeltas.append([J_p,delta])
        jpdeltas = np.array(jpdeltas)


        for i in range(0, num_states):
            properties = {}

            properties['P2x_exps'] = P2x_exps[i]
            properties['P2z_exps'] = P2z_exps[i]
            properties['RMIs'] = RMIs[i]
            properties['RMI2s'] = RMI2s[i]
            try:
                properties['phase_value'] = phase_values[i]
            except:
                pass
            properties['jpdelta'] = jpdeltas[i]
            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)


class ESSHData(Dataset):
    def __init__(self,num_states=30,num_qubits=100, num_measure_qubits = 3, with_B=False, filter_fn = None):
        observables = []
        for j in range(0, 3 ** num_measure_qubits):
            observable = np.load(DATA_DIR +
                str(num_measure_qubits) + 'qubit/float_observable' + str(num_measure_qubits) + str(j) + '.npy')
            observables.append(observable)

        index_observables = []
        combination_list = np.load(DATA_DIR +str(num_qubits)+'qubit/' + str(num_qubits) +'qubit_' + str(num_measure_qubits) +'combination_list.npy')
        for j in range(0, len(combination_list)):
            for i in range(0,3 ** num_measure_qubits):
                tmp = np.concatenate((observables[i],combination_list[j]))
                index_observables.append(tmp)
        self.observables = np.array(index_observables)

        data = np.load(DATA_DIR + f"eSSH/data_{num_qubits}{'_HB' if with_B else ''}.npz")

        self.input_data = data["born"].reshape(num_states, (num_qubits - num_measure_qubits + 1) * 3 ** num_measure_qubits, 2 ** num_measure_qubits)
        output_properties = []

        PHASES = ["Topological", "Symmetry broken", "Trivial"]
        phase_labels = data["ZT"].flatten().round().astype(np.int8) + 1
        for i in range(0, num_states):
            properties = {}
            properties['phase_value'] = data["ZT"][i]
            properties['phase_label'] = PHASES[phase_labels[i]]
            properties['meta'] = data["meta"][i]
            output_properties.append(properties)
        self.output_properties = np.array(output_properties)

    
        if filter_fn is not None:
            idx = [i for i in range(num_states) if filter_fn(self.output_properties[i])]
            self.input_data = self.input_data[idx]
            self.output_properties = self.output_properties[idx]
            
    def __getitem__(self, idx):
        assert idx < len(self.input_data)
        return self.observables, self.input_data[idx], self.output_properties[idx]
    def __len__(self):
        return len(self.input_data)

