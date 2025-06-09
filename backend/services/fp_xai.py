import pickle
import sys

import pandas as pd
import numpy as np

import scipy.stats as ss

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Crippen
from rdkit.Chem import Crippen

from copy import deepcopy

from scipy.stats import pearsonr


def mol2fp(mol, radius=2, n_bits=9):
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=bi)
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(int)


def mod_fp(mol, atom_idx, radius=2, n_bits=1024):
    
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)
    
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0: # for radius 0
                bitmap[at1][bit] = 0
            else: # for radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0
    
    new_fp = fps_morgan2 & bitmap[atom_idx]
    return fps_morgan2, new_fp, info
                    

def mod_fp_exchange(mol, idx_atom, radius=2, dummy_atom_no=47, n_bits=1024):
    info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, bitInfo=info)
    
    mol_cpy = deepcopy(mol)
    mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
    
    new_info = {}
    new_fp = AllChem.GetMorganFingerprintAsBitVect(mol_cpy, radius, nBits=n_bits, bitInfo=new_info)
    
    return fp, new_fp, info, new_info
   

def AAU_weights(mol, model, radius=2, n_bits=9, num_samples=10):
    """
    Calculate Atomic Attribution to Uncertainty (AAU) weights for the MEGAN model.
    
    Args:
        mol: RDKit molecule object.
        model: Trained MEGAN model.
        radius: Radius for Morgan fingerprint.
        n_bits: Number of bits for Morgan fingerprint.
        num_samples: Number of Monte Carlo samples for uncertainty estimation.
    
    Returns:
        aau_weights: Array of AAU weights for each atom in the molecule.
    """
    from rdkit.Chem import DataStructs
    import numpy as np
    from torch_geometric.data import Data
    import torch

    aau_weights = []

    # Generate Morgan fingerprint and bit information
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)

    # Convert fingerprint to PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(np.array(fps_morgan2, dtype=np.float32)).unsqueeze(0),  # Node features
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
        edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes
        batch=torch.tensor([0], dtype=torch.long)  # Batch information
    )

    # Get original uncertainty using Monte Carlo Dropout
    def monte_carlo_predictions(model, data, num_samples):
        model.train()  # Enable dropout during inference
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                predictions.append(model.predict(data)[0])
        return np.array(predictions)

    mc_predictions = monte_carlo_predictions(model, data, num_samples)
    orig_uncertainty = np.std(mc_predictions, axis=0)

    # Get bits for each atom
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for _ in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0:  # For radius 0
                bitmap[at1][bit] = 0
            else:  # For radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0

    # Loop over atoms to calculate AAU weights
    for at1 in range(mol.GetNumAtoms()):
        # Modify fingerprint by removing bits associated with the atom
        new_fp = fps_morgan2 & bitmap[at1]

        # Convert modified fingerprint to PyTorch Geometric Data object
        new_data = Data(
            x=torch.tensor(np.array(new_fp, dtype=np.float32)).unsqueeze(0),  # Node features
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
            edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes
            batch=torch.tensor([0], dtype=torch.long)  # Batch information
        )

        # Get new uncertainty using Monte Carlo Dropout
        mc_predictions_new = monte_carlo_predictions(model, new_data, num_samples)
        new_uncertainty = np.std(mc_predictions_new, axis=0)

        # Calculate AAU weight as the reduction in uncertainty
        aau_weights.append(orig_uncertainty - new_uncertainty)

    # Normalize weights
    aau_weights = np.array(aau_weights)
    if np.sum(aau_weights) != 0:
        aau_weights = aau_weights / np.sum(aau_weights)

    return aau_weights
  


def RDKit_normalized_weights(mol):
    contribs_update = []
    mol = Chem.AddHs(mol)
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    contribs = [x for x,y in contribs]
    #molecule = mols
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1: 
            index = atom.GetIdx()
            for nabo in atom.GetNeighbors():
                index_nabo= nabo.GetIdx() 
                contribs[index_nabo] += contribs[index]

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            index_update = atom.GetIdx()
            contribs_update.append(contribs[index_update]) 
    contribs_update = np.array(contribs_update)
    Normalized_weightsRDkit = contribs_update.flatten()/np.linalg.norm(contribs_update.flatten())
    return contribs_update, Normalized_weightsRDkit 



def RDKit_bit_vector(mol, model, radius=2, n_bits=9):
    rdkit_contrib, _ = RDKit_normalized_weights(mol)
    rdkit_bit_contribs = []
    ML_weights = []

    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)
    data = Data(
        x=torch.tensor(fps_morgan2, dtype=torch.float).unsqueeze(0),  # Node features
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
        edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes with edge_dim=3
        batch=torch.tensor([0], dtype=torch.long)  # Batch information
    )

    orig_pp = model.predict(data)[0]

    # Get bits for each atom
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for x in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0:  # For radius 0
                bitmap[at1][bit] = 0
            else:  # For radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0

    # Loop over atoms
    for at1 in range(mol.GetNumAtoms()):
        new_fp = fps_morgan2 & bitmap[at1]

        # Construct a new Data object for new_fp
        new_data = Data(
            x=torch.tensor(np.array([list(new_fp)]), dtype=torch.float),  # Node features
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
            edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes with edge_dim=3
            batch=torch.tensor([0], dtype=torch.long)  # Batch information
        )

        # Predict using the new Data object
        new_pp = model.predict(new_data)[0]
        ML_weights.append(orig_pp - new_pp)

        removed_bits = []
        for x in fps_morgan2.GetOnBits():
            if x not in new_fp.GetOnBits():
                removed_bits.append(x)
        bit_contrib = 0
        for bit in removed_bits:
            for at, rad in info[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bit_contrib += rdkit_contrib[at2]
        rdkit_bit_contribs.append(bit_contrib)

    return np.array(ML_weights), np.array(rdkit_contrib), np.array(rdkit_bit_contribs)


def UAA_weights(mol, m, radius=2, n_bits=1024):

    from rdkit.Chem import DataStructs
    import numpy as np
    from torch_geometric.data import Data
    import torch

    uaa_weights = []

    # Generate Morgan fingerprint and bit information
    info = {}
    fps_morgan2 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=info)

    # Convert fingerprint to PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(np.array(fps_morgan2, dtype=np.float32)).unsqueeze(0),  # Node features
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
        edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes
        batch=torch.tensor([0], dtype=torch.long)  # Batch information
    )

    # Get original prediction from the model
    with torch.no_grad():
        orig_prediction = m.predict(data)[0]

    # Get bits for each atom
    bitmap = [~DataStructs.ExplicitBitVect(n_bits) for _ in range(mol.GetNumAtoms())]
    for bit, es in info.items():
        for at1, rad in es:
            if rad == 0:  # For radius 0
                bitmap[at1][bit] = 0
            else:  # For radii > 0
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, at1)
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                for at2 in amap.keys():
                    bitmap[at2][bit] = 0

    # Loop over atoms to calculate UAA weights
    for at1 in range(mol.GetNumAtoms()):
        # Modify fingerprint by removing bits associated with the atom
        new_fp = fps_morgan2 & bitmap[at1]

        # Convert modified fingerprint to PyTorch Geometric Data object
        new_data = Data(
            x=torch.tensor(np.array(new_fp, dtype=np.float32)).unsqueeze(0),  # Node features
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
            edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes
            batch=torch.tensor([0], dtype=torch.long)  # Batch information
        )

        # Get new prediction from the m
        with torch.no_grad():
            new_prediction = m.predict(new_data)[0]

        # Calculate UAA weight as the absolute change in prediction
        uaa_weights.append(abs(orig_prediction - new_prediction))

    return np.array(uaa_weights).flatten()
import torch
from torch_geometric.data import Data


def get_weights_for_visualization(mol, model, radius=2, n_bits=9):
    fp = mol2fp(mol, radius=2, n_bits=9)
    data = Data(
        x=torch.tensor(fp, dtype=torch.float).unsqueeze(0),  # Node features
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),  # Dummy edge index
        edge_attr=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float),  # Edge attributes with edge_dim=3
        batch=torch.tensor([0], dtype=torch.long)  # Batch information
    )

    logp_pred = model.predict(data)[0]
    print(logp_pred)
    ml_weights, atom_weights, FPA_weights = RDKit_bit_vector(mol, model, radius=2, n_bits=9)
    
    #atom_weights, _ = RDKit_normalized_weights(mol)
    clogp = Crippen.MolLogP(mol)
    
    #print(clogp, logp_pred)
    if np.sign(np.sum(ml_weights)) != np.sign(logp_pred):
        print("ml weights: sign problem detected")
        #ml_weights = -ml_weights
    if np.sign(np.sum(FPA_weights)) != np.sign(clogp):
        print("FPA weights: sign problem detected") 
    
    ml_weights = ml_weights*abs(logp_pred/np.sum(ml_weights))
    FPA_weights = FPA_weights*abs(clogp/np.sum(FPA_weights))

    return ml_weights, atom_weights, FPA_weights

from rdkit.Chem.Draw import rdMolDraw2D

def get_contour_image(mol, weights, contour_step=0.06):
    """
    Generate a contour image for the molecule based on weights.
    Args:
        mol: RDKit molecule object.
        weights: Atomic weights (list or NumPy array).
        contour_step: Step size for contour levels.
    Returns:
        fig: RDKit similarity map figure.
    """
    # Ensure weights are a NumPy array
    weights = np.array(weights).flatten()
    AllChem.Compute2DCoords(mol)  # Ensure the molecule has 2D coordinates
    print(len(weights))
    print(mol.GetNumAtoms())
    draw2d = rdMolDraw2D.MolDraw2DSVG(500, 500)
    # Calculate the number of contours as a scalar
    N_contours = (max(weights) - min(weights)) / contour_step
    N_contours = round(float(N_contours))  # Ensure it's a scalar

    # Generate the similarity map
    SimilarityMaps.GetSimilarityMapFromWeights(mol, weights.tolist(),draw2d=draw2d, contourLines=N_contours)
    draw2d.FinishDrawing()
    svg = draw2d.GetDrawingText()
    return svg

if __name__ == "__main__":
    model_filename = sys.argv[1]
    m = pickle.load(open(model_filename, 'rb'))
    smiles = "CC[NH+](CC)[C@@H]1CCN(C(=O)N[C@@H]2CCCC2)C1"
    mol = Chem.MolFromSmiles(smiles)
    ml_weights, atom_weights, fpa_weights = get_weights_for_visualization(mol, m, radius=2, n_bits=2048)
    fig = get_contour_image(mol, ml_weights)
    fig.savefig("ml_weights_example.pdf", bbox_inches='tight')
