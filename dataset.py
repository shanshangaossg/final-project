# deal with the unbalanced dataset
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
node_file = "/content/drive/My Drive/final_project/data/raw/HIV_train.csv"

data = pd.read_csv(node_file )
data.index = data["index"]
data["HIV_active"].value_counts()
start_index = data.iloc[0]["index"]

# %% Apply oversampling

# Check how many additional samples we need
neg_class = data["HIV_active"].value_counts()[0]
pos_class = data["HIV_active"].value_counts()[1]
multiplier = int(neg_class/pos_class) - 1
print(neg_class)
print(pos_class)
print(multiplier)

from sklearn.utils import resample
positive_samples = data[data["HIV_active"] == 1]
negative_samples = data[data["HIV_active"] == 0]
positive_oversampled = pd.concat([positive_samples] * 2, ignore_index=True)
negative_downsampled = resample(negative_samples,
                                replace=False,
                                n_samples=len(positive_oversampled),
                                random_state=42)

balanced_data = pd.concat([positive_oversampled, negative_downsampled])
# Shuffle dataset
balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)
# Re-assign index (This is our ID later)
index = range(start_index, start_index + balanced_data.shape[0])
balanced_data.index = index
balanced_data["index"] = balanced_data.index
print(balanced_data["HIV_active"].value_counts(),len(balanced_data))

balanced_data.head()
balanced_data.to_csv("/content/drive/My Drive/final_project/data/raw/HIV_train_balanced.csv")

!pip install torch
!pip install torch_geometric
!pip install deepchem
!pip install dgl
!pip install pytorch-lightning
!pip install dm-haiku
!pip install dgl -f https://data.dgl.ai/wheels/repo.html
!pip install dgl-cu118
!pip uninstall torch torchvision torchaudio dgl -y
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install dgl-cu118

import pandas as pd
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol["smiles"])
            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features_with_angles(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(mol["HIV_active"])

            # Create data object
            data = Data(x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol["smiles"]
                        )
            if self.test:
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data,
                    os.path.join(self.processed_dir,
                                 f'data_{index}.pt'))

    def _get_node_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Feature 3: Bond length
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start_pos = conf.GetAtomPosition(start_idx)
            end_pos = conf.GetAtomPosition(end_idx)
            bond_length = ((start_pos.x - end_pos.x)**2 +
                          (start_pos.y - end_pos.y)**2 +
                          (start_pos.z - end_pos.z)**2)**0.5
            edge_feats.append(bond_length)

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def _get_bond_angle(self, mol, atom1, atom2, atom3):
        """
        Calculate the angle (in degrees) between three atoms.
        """
        pos1 = mol.GetConformer().GetAtomPosition(atom1)
        pos2 = mol.GetConformer().GetAtomPosition(atom2)
        pos3 = mol.GetConformer().GetAtomPosition(atom3)

        vec1 = np.array([pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z])
        vec2 = np.array([pos3.x - pos2.x, pos3.y - pos2.y, pos3.z - pos2.z])

        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to avoid numerical errors
        return np.degrees(angle)

    def _get_edge_features_with_angles(self, mol):
        """
        Add bond angles as an additional edge feature.
        """
        all_edge_feats = []
        conf = mol.GetConformer()

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type
            edge_feats.append(bond.GetBondTypeAsDouble())

            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())

            # Feature 3: Bond length
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            start_pos = conf.GetAtomPosition(start_idx)
            end_pos = conf.GetAtomPosition(end_idx)
            bond_length = ((start_pos.x - end_pos.x)**2 +
                          (start_pos.y - end_pos.y)**2 +
                          (start_pos.z - end_pos.z)**2)**0.5
            edge_feats.append(bond_length)

            # Feature 4: Bond angle (if possible)
            angles = []
            for neighbor in mol.GetAtomWithIdx(start_idx).GetNeighbors():
                if neighbor.GetIdx() != end_idx:  # Avoid the bonded atom
                    angle = self._get_bond_angle(mol, neighbor.GetIdx(), start_idx, end_idx)
                    angles.append(angle)
            for neighbor in mol.GetAtomWithIdx(end_idx).GetNeighbors():
                if neighbor.GetIdx() != start_idx:
                    angle = self._get_bond_angle(mol, start_idx, end_idx, neighbor.GetIdx())
                    angles.append(angle)

            # Take the mean bond angle as a feature (or max/min/variance as needed)
            if angles:
                edge_feats.append(np.mean(angles))
            else:
                edge_feats.append(0.0)  # Default to 0 if no angles are found

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)


    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                 f'data_{idx}.pt'))
        return data

dataset = MoleculeDataset(root="/content/drive/My Drive/final_project/data", filename="HIV_train_balanced.csv")
data = dataset.get(1)
print(data)
print(data.edge_index.t())
print(data.x)
print(data.edge_attr)
print(data.y)
test_dataset = MoleculeDataset(root="/content/drive/My Drive/final_project/data", filename='HIV_test.csv', test=True)
