from rdkit import Chem
from typing import Iterable, List

# Redfined from seq_graph_retro/molgraph/mol_features.py
BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

def get_mol(smiles: str, kekulize: bool = False) -> Chem.Mol:
    """SMILES string to Mol.

    Parameters
    ----------
    smiles: str,
        SMILES string for molecule
    kekulize: bool,
        Whether to kekulize the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None and kekulize:
        Chem.Kekulize(mol)
    return mol

def apply_edits_to_mol(mol: Chem.Mol, edits: Iterable[str]) -> Chem.Mol:
    """Apply edits to molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object
    edits: Iterable[str],
        Iterable of edits to apply. An edit is structured as a1:a2:b1:b2, where
        a1, a2 are atom maps of participating atoms and b1, b2 are previous and
        new bond orders. When  a2 = 0, we update the hydrogen count.
    """
    new_mol = Chem.RWMol(mol)
    amap = {atom.GetAtomMapNum(): atom.GetIdx() for atom in new_mol.GetAtoms()}

    # Keep track of aromatic nitrogens, might cause explicit hydrogen issues
    aromatic_nitrogen_idx = set()
    aromatic_carbonyl_adj_to_aromatic_nH = {}
    aromatic_carbondeg3_adj_to_aromatic_nH0 = {}
    for a in new_mol.GetAtoms():
        if a.GetIsAromatic() and a.GetSymbol() == 'N':
            aromatic_nitrogen_idx.add(a.GetIdx())
            for nbr in a.GetNeighbors():
                nbr_is_carbon = (nbr.GetSymbol() == 'C')
                nbr_is_aromatic = nbr.GetIsAromatic()
                nbr_has_double_bond = any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds())
                nbr_has_3_bonds = (len(nbr.GetBonds()) == 3)

                if (a.GetNumExplicitHs() ==1 and nbr_is_carbon and nbr_is_aromatic
                    and nbr_has_double_bond):
                    aromatic_carbonyl_adj_to_aromatic_nH[nbr.GetIdx()] = a.GetIdx()
                elif (a.GetNumExplicitHs() == 0 and nbr_is_carbon and nbr_is_aromatic
                    and nbr_has_3_bonds):
                    aromatic_carbondeg3_adj_to_aromatic_nH0[nbr.GetIdx()] = a.GetIdx()
        else:
            a.SetNumExplicitHs(0)
    new_mol.UpdatePropertyCache()

    # Apply the edits as predicted
    for edit in edits:
        x, y, prev_bo, new_bo = edit.split(":")
        x, y = int(x), int(y)
        new_bo = float(new_bo)

        if y == 0:
            continue

        bond = new_mol.GetBondBetweenAtoms(amap[x],amap[y])
        a1 = new_mol.GetAtomWithIdx(amap[x])
        a2 = new_mol.GetAtomWithIdx(amap[y])

        if bond is not None:
            new_mol.RemoveBond(amap[x],amap[y])

            # Are we losing a bond on an aromatic nitrogen?
            if bond.GetBondTypeAsDouble() == 1.0:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 0:
                        a1.SetNumExplicitHs(1)
                    elif a1.GetFormalCharge() == 1:
                        a1.SetFormalCharge(0)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 0:
                        a2.SetNumExplicitHs(1)
                    elif a2.GetFormalCharge() == 1:
                        a2.SetFormalCharge(0)

            # Are we losing a c=O bond on an aromatic ring? If so, remove H from adjacent nH if appropriate
            if bond.GetBondTypeAsDouble() == 2.0:
                if amap[x] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[amap[x]]).SetNumExplicitHs(0)
                elif amap[y] in aromatic_carbonyl_adj_to_aromatic_nH:
                    new_mol.GetAtomWithIdx(aromatic_carbonyl_adj_to_aromatic_nH[amap[y]]).SetNumExplicitHs(0)

        if new_bo > 0:
            new_mol.AddBond(amap[x],amap[y],BOND_FLOAT_TO_TYPE[new_bo])

            # Special alkylation case?
            if new_bo == 1:
                if amap[x] in aromatic_nitrogen_idx:
                    if a1.GetTotalNumHs() == 1:
                        a1.SetNumExplicitHs(0)
                    else:
                        a1.SetFormalCharge(1)
                elif amap[y] in aromatic_nitrogen_idx:
                    if a2.GetTotalNumHs() == 1:
                        a2.SetNumExplicitHs(0)
                    else:
                        a2.SetFormalCharge(1)

            # Are we getting a c=O bond on an aromatic ring? If so, add H to adjacent nH0 if appropriate
            if new_bo == 2:
                if amap[x] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[amap[x]]).SetNumExplicitHs(1)
                elif amap[y] in aromatic_carbondeg3_adj_to_aromatic_nH0:
                    new_mol.GetAtomWithIdx(aromatic_carbondeg3_adj_to_aromatic_nH0[amap[y]]).SetNumExplicitHs(1)

    pred_mol = new_mol.GetMol()

    # Clear formal charges to make molecules valid
    # Note: because S and P (among others) can change valence, be more flexible
    for atom in pred_mol.GetAtoms():
        if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1: # exclude negatively-charged azide
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals <= 3:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N' and atom.GetFormalCharge() == -1: # handle negatively-charged azide addition
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 3 and any([nbr.GetSymbol() == 'N' for nbr in atom.GetNeighbors()]):
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'N':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 4 and not atom.GetIsAromatic(): # and atom.IsInRingSize(5)):
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'C' and atom.GetFormalCharge() != 0:
            atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'O' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]) + atom.GetNumExplicitHs()
            if bond_vals == 2:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() in ['Cl', 'Br', 'I', 'F'] and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals == 1:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'S' and atom.GetFormalCharge() != 0:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'P': # quartenary phosphorous should be pos. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(1)
                atom.SetNumExplicitHs(0)
            elif sum(bond_vals) == 3 and len(bond_vals) == 3: # make sure neutral
                atom.SetFormalCharge(0)
        elif atom.GetSymbol() == 'B': # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)
        elif atom.GetSymbol() in ['Mg', 'Zn']:
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 1 and len(bond_vals) == 1:
                atom.SetFormalCharge(1)
        elif atom.GetSymbol() == 'Si':
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == len(bond_vals):
                atom.SetNumExplicitHs(max(0, 4 - len(bond_vals)))

    return pred_mol

def get_sub_mol(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
    """Extract subgraph from molecular graph.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object,
    sub_atoms: List[int],
        List of atom indices in the subgraph.
    """
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def get_sub_mol_stereo(mol: Chem.Mol, sub_atoms: List[int]) -> Chem.Mol:
    """Extract subgraph from molecular graph, while preserving stereochemistry.

    Parameters
    ----------
    mol: Chem.Mol,
        RDKit mol object,
    sub_atoms: List[int],
        List of atom indices in the subgraph.
    """
    # This version retains stereochemistry, as opposed to the other version
    new_mol = Chem.RWMol(Chem.Mol(mol))
    atoms_to_remove = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in sub_atoms:
            atoms_to_remove.append(atom.GetIdx())

    for atom_idx in sorted(atoms_to_remove, reverse=True):
        new_mol.RemoveAtom(atom_idx)

    return new_mol.GetMol()
