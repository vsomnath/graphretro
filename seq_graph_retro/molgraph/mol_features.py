import numpy as np
from rdkit import Chem
from typing import Set, Any, List, Union

from seq_graph_retro.utils.chem import get_mol

idxfunc = lambda a : a.GetAtomMapNum() - 1
bond_idx_fn = lambda a, b, mol: mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx()).GetIdx()

# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', \
    'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', \
    'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', \
    'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', \
    'Ce','Gd','Ga','Cs', '*', 'unk']

MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2]

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
VALENCE = [0, 1, 2, 3, 4, 5, 6]
NUM_Hs = [0, 1, 3, 4, 5]

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2:9, 3:10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]
RXN_CLASSES = list(range(10))

ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
            + len(VALENCE) + len(NUM_Hs) + 1
BOND_FDIM = 6
BINARY_FDIM = 5 + BOND_FDIM
INVALID_BOND = -1
PATTERN_DIM = 389

def sanitize(mol, kekulize: bool = True) -> Chem.Mol:
    """Sanitize mol.
    Parameters
    ----------
    mol: Chem.Mol
        Molecule to sanitize
    kekulize: bool
        Whether to kekulize the molecule
    """
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def onek_encoding_unk(x: Any, allowable_set: Union[List, Set]) -> List:
    """Converts x to one hot encoding.

    Parameters
    ----------
    x: Any,
        An element of any type
    allowable_set: Union[List, Set]
        Allowable element collection
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def get_atom_features(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> np.ndarray:
    """Get atom features.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    if atom.GetSymbol() == '*':
        symbol = onek_encoding_unk(atom.GetSymbol(), ATOM_LIST)
        if use_rxn_class:
            padding = [0] * (ATOM_FDIM + len(RXN_CLASSES)- len(symbol))
        else:
            padding = [0] * (ATOM_FDIM - len(symbol))
        feature_array = symbol + padding
        return feature_array

    if use_rxn_class:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())] + onek_encoding_unk(rxn_class, RXN_CLASSES)).tolist()

    else:
        return np.array(onek_encoding_unk(atom.GetSymbol(), ATOM_LIST) +
                        onek_encoding_unk(atom.GetDegree(), DEGREES) +
                        onek_encoding_unk(atom.GetFormalCharge(), FORMAL_CHARGE) +
                        onek_encoding_unk(atom.GetHybridization(), HYBRIDIZATION) +
                        onek_encoding_unk(atom.GetTotalValence(), VALENCE) +
                        onek_encoding_unk(atom.GetTotalNumHs(), NUM_Hs) +
                        [float(atom.GetIsAromatic())]).tolist()

def get_binary_features(mol: Chem.Mol) -> np.ndarray:
    """
    This function is used to generate descriptions of atom-atom relationships, including
    the bond type between the atoms (if any) and whether they belong to the same molecule.
    It is used in the global attention mechanism.

    Parameters
    ----------
    mol: Chem.Mol,
        Molecule for which we want to compute binary features.
    """
    comp = {}
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    for atom in mol.GetAtoms():
        comp[amap_idx[atom.GetAtomMapNum()]] = 0

    n_comp = 1
    n_atoms = mol.GetNumAtoms()

    bond_map = {}
    for bond in mol.GetBonds():
        a1 = amap_idx[bond.GetBeginAtom().GetAtomMapNum()]
        a2 = amap_idx[bond.GetEndAtom().GetAtomMapNum()]
        bond_map[(a1, a2)] = bond_map[(a2, a1)] = bond

    features = []
    for i in range(n_atoms):
        for j in range(n_atoms):
            f = np.zeros((BINARY_FDIM,))
            if (i, j) in bond_map:
                bond = bond_map[(i, j)]
                f[1:1 + BOND_FDIM] = get_bond_features(bond)
            else:
                f[0] = 1.0
            f[-4] = 1.0 if comp[i] != comp[j] else 0.0
            f[-3] = 1.0 if comp[i] == comp[j] else 0.0
            f[-2] = 1.0 if n_comp == 1 else 0.0
            f[-1] = 1.0 if n_comp > 1 else 0.0
            features.append(f)
    return np.vstack(features).reshape((n_atoms, n_atoms, BINARY_FDIM))

def get_bond_features(bond: Chem.Bond) -> np.ndarray:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [float(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bond_features.extend([float(bond.GetIsConjugated()), float(bond.IsInRing())])
    bond_features = np.array(bond_features, dtype=np.float32)
    return bond_features

def get_atom_graph(mol: Chem.Mol) -> np.ndarray:
    """Atom graph is the adjacency list of atoms and its neighbors.

    Parameters
    ----------
    mol: Chem.Mol,
        Molecule for which we want to compute atom graph.
    """
    agraph = np.zeros((mol.GetNumAtoms(), MAX_NB), dtype=np.int32)
    for idx, atom in enumerate(mol.GetAtoms()):
        nei_indices = [nei.GetIdx() + 1 for nei in atom.GetNeighbors()]
        agraph[idx, :len(nei_indices)] = nei_indices
    return agraph

def get_bond_graph(mol: Chem.Mol) -> np.ndarray:
    """Bond graph is the adjacency list of bond indices for each atom and its neighbors.

    Parameters
    ----------
    mol: Chem.Mol,
        Molecule for which we want to compute bond graph
    """
    bgraph = np.zeros((mol.GetNumAtoms(), MAX_NB), dtype=np.int32)
    for idx, atom in enumerate(mol.GetAtoms()):
        bond_indices = [bond_idx_fn(atom, nei, mol) + 1 for nei in atom.GetNeighbors()]
        bgraph[idx, :len(bond_indices)] = bond_indices
    return bgraph
