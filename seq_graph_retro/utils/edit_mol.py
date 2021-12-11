from rdkit import Chem
from rdkit.Chem import AllChem
from seq_graph_retro.utils.chem import apply_edits_to_mol, BOND_FLOAT_TO_TYPE
from seq_graph_retro.molgraph.rxn_graphs import MultiElement

MAX_VALENCE = {'N': 3, 'C': 4, 'O': 2, 'Br': 1, 'Cl': 1, 'F': 1, 'I': 1}

SINGLE_BOND_ATTACH_GROUPS = ['Br[C:1](Br)Br','Br[Zn:1]', '[Cl:1]',
'[Cu:1]', '[F:1]', '[I:1]', '[Br:1]', 'O[B:1]O', 'Br[CH:1]Br', 'Br[Mg:1]', 'C1CC[CH:1]OC1',
'C1CO[B:1]O1', 'C1CO[B:1]OC1', 'C=C(C)C(=O)[O:1]', 'C=C1CNC([O:1])O1','C=CB1OB(C=C)O[B:1]O1',
'C=CCO[C:1]=O', 'CB1OB(C)O[B:1]O1', 'CC(=O)C(C)[O:1]','CC(=O)O[C@@H](c1ccccc1)[C:1]=O',
'CC(=O)[O:1]', 'CC(C)(C)C(=O)[O:1]', 'CC(C)(C)OC(=O)[O:1]', 'CC(C)(C)O[C:1]=O', 'CC(C)(C)[C:1]=O',
'CC(C)(C)[O:1]', 'CC(C)(C)[Si:1](C)C', 'CC(C)(C[O:1])NS(=O)(=O)C(F)(F)F', 'CC(C)C(=O)[O:1]',
'CC(C)C(C)(C)[Si:1](C)C', 'CC(C)C[O:1]', 'CC(C)O[B:1]OC(C)C', 'CC(C)[C:1]=O', 'CC(C)[O:1]',
'CC(C)[Si:1](C(C)C)C(C)C', 'CC(Cl)[O:1]', 'CC1(C)CO[B:1]OC1', 'CC1(C)O[B:1]OC1(C)C',
'CC1([C:1]=O)CCCCC1', 'CC1CC(C)(C)O[B:1]O1', 'CC1O[B:1]OC1C', 'CCC(=O)[O:1]', 'CCCC(=O)[O:1]',
'CCCCC(=O)[O:1]', 'CCCCCC(=O)[O:1]', 'CCCCCCCCC(=O)[O:1]', 'CCCCCCCCCCCCCCCCCCCCC(=O)[O:1]',
'CCCCCCCCCCCCCCC[C:1]=O', 'CCCCCCCCCCCC[O:1]','CCCCCCCOc1ccc([C:1]=O)cc1', 'CCCC[C:1]=O',
'CCCC[C:1]Cl','CCCC[O:1]','CCCC[Sn:1](CCCC)CCCC', 'CCC[C:1]=O', 'CCC[O:1]','CCOC(=O)[O:1]',
'CCO[C:1]=O','CCO[P:1](=O)OCC','CC[C:1]=O', 'CC[O:1]', 'CC[Si:1](CC)CC', 'CC[Sn:1](CC)CC',
'CN(C)[C:1]=O','CN1CC(=O)O[B:1]OC(=O)C1','COC(=O)CC[C:1]=O', 'CO[C:1]=O', 'CO[P:1](=O)OC'
'CON=C(C(=O)[O:1])c1csc(NC(c2ccccc2)(c2ccccc2)c2ccccc2)n1', 'CO[CH2:1]', 'CO[N:1]C',
'COc1cc(-c2ncn(C=CC(=O)[O:1])n2)cc(C(F)(F)F)c1', 'COc1ccc([C:1]=O)cc1', 'COc1ccc([CH2:1])cc1',
'COc1ccc2cc([C@@H](C)[C:1]=O)ccc2c1', 'CS(=O)(=O)[O:1]', 'C[C:1](C)C', 'C[C:1]=O', 'C[CH2:1]',
'C[CH:1]C', 'C[N+:1](C)C', 'C[N:1]C', 'C[O:1]', 'C[P+:1](C)C', 'C[S+:1](C)=O', 'C[S+:1]C',
'C[S:1](=O)=O', 'C[Si:1](C)C', 'C[Si](C)(C)[N:1]', 'C[Sn:1](C)C', 'ClC(Cl)(Cl)C[O:1]',
'ClC(Cl)(Cl)[O:1]', 'Cl[C:1](Cl)Cl', 'FC(F)(F)C[O:1]','Fc1ccc(B2OB(c3ccc(F)cc3)O[B:1]O2)cc1',
'I[CH2:1]', 'I[Zn:1]', 'O=C(C(F)(F)Cl)[O:1]', 'O=C(C(F)(F)F)[O:1]', 'O=C(C(F)F)[O:1]',
'O=C(C1CC1)[O:1]', 'O=C(CBr)[O:1]', 'O=C(CCl)[O:1]', 'O=C(CI)[O:1]','O=C(O)C=CC=CC=CC=C[C:1]=O',
'O=C(OCc1ccccc1)[O:1]', 'O=C([O-])C1C=CO[B:1]O1', 'O=C(c1cccc(Cl)c1)[O:1]', 'O=C(c1ccccc1)[O:1]',
'O=C(c1ccccc1Cl)[O:1]', 'O=C1CCC(=O)[N:1]1', 'O=C[O:1]', 'O=S(=O)(C(F)(F)F)[N:1]',
'O=S(=O)(C(F)(F)F)[O:1]', 'O=S([O-])[O:1]', 'O=[C:1]C(F)(F)F', 'O=[C:1]CCCCBr', 'O=[C:1]CCl',
'O=[C:1]OCC1c2ccccc2-c2ccccc21', 'O=[C:1]OCc1ccccc1', 'O=[C:1]c1cccc(Cl)c1', 'O=[C:1]c1ccccc1',
'O=[N+]([O-])c1ccc(C[O:1])cc1', 'O=[N+]([O-])c1ccc(O)c([CH2:1])c1', 'O=[N+]([O-])c1ccc([C:1]=O)cc1',
'O=[P:1](OCC(F)(F)F)OCC(F)(F)F', '[CH3:1]', '[Mg+:1]', '[NH2:1]', '[O-:1]', '[OH:1]', '[Zn+:1]',
'c1ccc(C[O:1])cc1', 'c1ccc(N2CCO[B:1]OCC2)cc1', 'c1ccc([CH2:1])cc1', 'c1ccc([CH:1]c2ccccc2)cc1',
'c1ccc([P+:1](c2ccccc2)c2ccccc2)cc1', "COC[C:1]=O", "O=[C:1]C(F)(F)C(F)(F)C(F)(F)F", "C1CC[N:1]CC1",
"CC(C)(C)[Si](C)(C)Oc1ccc(B2OB(c3ccc(O[Si](C)(C)C(C)(C)C)cc3)O[B:1]O2)cc1", "CO[P:1](=O)OC"]

DOUBLE_BOND_ATTACH_GROUPS = ['[N-]=[N+:1]', '[CH2:1]', '[O:1]', '[S:1]',
'c1ccc([P:1](c2ccccc2)c2ccccc2)cc1']
SINGLE_ATTACH = SINGLE_BOND_ATTACH_GROUPS + DOUBLE_BOND_ATTACH_GROUPS

CARBON_ATTACH = ['[OH:1]', 'CC[O:1]', 'C[O:1]', 'CO[N:1]C', 'CCO[C:1]=O', "[O:1]"]
CYCLIC_ATTACH = ['C([CH2:1])[O:1]', "C(C[O:1])[CH2:1]", 'CC[O:1].C[CH2:1]']

SPECIAL_MULTI_ATTACH = ['C[O:1].[O:1]', 'C[N:1]C.C[O:1].C[O:1]', '[O:1].[O:1]', '[O:1].[OH:1]',
'[O-:1].[O:1]', '[OH:1].[OH:1].[OH:1]', "[Cl:1].[O:1].[O:1]", "[CH2:1].[CH2:1]", 'c1c[n:1]cn1.c1c[n:1]cn1'
'C[O:1].[O:1]'] + CYCLIC_ATTACH

def get_oc_idx(atom_a, atom_b):
    sym_a, sym_b = atom_a.GetSymbol(), atom_b.GetSymbol()

    if sym_a == 'O' and sym_b == 'C':
        return atom_a.GetIdx(), atom_b.GetIdx()

    elif sym_a == 'C' and sym_b == 'O':
        return atom_b.GetIdx(), atom_a.GetIdx()

    else:
        return atom_a.GetIdx(), atom_b.GetIdx()

def single_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol):
    """Adds a leaving group with single attachment point."""
    assert lg_group in SINGLE_ATTACH, "Function meant only for single attach"

    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                if atom.GetAtomMapNum() != 0}

    bt = Chem.BondType.SINGLE if lg_group in SINGLE_BOND_ATTACH_GROUPS else Chem.BondType.DOUBLE
    for atom in lg_mol.GetAtoms():
        if atom.GetAtomMapNum() >= 1000:
            lg_attach_idx = amap_idx[atom.GetAtomMapNum()]

    if len(frag_attach_idxs) > 1 or len(frag_attach_idxs) == 0:
        print("Cannot attach single_attach_lg to multiple or zero attachment points")
        return rw_mol
    else:
        frag_attach_idx = frag_attach_idxs[0]
        rw_mol.AddBond(frag_attach_idx, lg_attach_idx, bt)

    return rw_mol

def special_multi_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol):

    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                if atom.GetAtomMapNum() != 0}

    assert len(frag_attach_idxs) <= 2, print(frag_attach_idxs)

    if lg_group == '[O:1].[O:1]':
        lg_attach_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                lg_attach_idxs.append(amap_idx[atom.GetAtomMapNum()])

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            rw_mol.AddBond(frag_attach_idx, lg_attach_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idx, lg_attach_idxs[1], Chem.BondType.DOUBLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_idxs[1], Chem.BondType.DOUBLE)

        else:
            pass

    elif lg_group == '[O:1].[OH:1]':
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetNumExplicitHs() == 0:
                lg_attach_o_idx = amap_idx[atom.GetAtomMapNum()]
            elif atom.GetAtomMapNum() >= 1000 and atom.GetNumExplicitHs() == 1:
                lg_attach_oh_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            rw_mol.AddBond(frag_attach_idx, lg_attach_o_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idx, lg_attach_oh_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_oh_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == '[O-:1].[O:1]':
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetFormalCharge() == -1:
                lg_attach_o_minus_idx = amap_idx[atom.GetAtomMapNum()]
            elif atom.GetAtomMapNum() >= 1000 and atom.GetFormalCharge() == 0:
                lg_attach_o_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            rw_mol.AddBond(frag_attach_idx, lg_attach_o_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idx, lg_attach_o_minus_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_o_minus_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == '[OH:1].[OH:1].[OH:1]':
        lg_attach_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                lg_attach_idxs.append(amap_idx[atom.GetAtomMapNum()])

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            for index in lg_attach_idxs:
                rw_mol.AddBond(frag_attach_idx, index, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[0], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_idxs[1], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[2], Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == 'c1c[n:1]cn1.c1c[n:1]cn1':
        lg_attach_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                lg_attach_idxs.append(amap_idx[atom.GetAtomMapNum()])

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            for index in lg_attach_idxs:
                rw_mol.AddBond(frag_attach_idx, index, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[0], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_idxs[1], Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == "[Cl:1].[O:1].[O:1]":
        lg_attach_o_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'O':
                lg_attach_o_idxs.append(amap_idx[atom.GetAtomMapNum()])

            elif atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'Cl':
                lg_attach_cl_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            frag_attach_idx = frag_attach_idxs[0]
            rw_mol.AddBond(frag_attach_idx, lg_attach_o_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idx, lg_attach_o_idxs[1], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idx, lg_attach_cl_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_o_idxs[1], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_cl_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == "[CH2:1].[CH2:1]":
        lg_attach_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                lg_attach_idxs.append(amap_idx[atom.GetAtomMapNum()])

        if len(frag_attach_idxs) == 1:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[1], Chem.BondType.DOUBLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_idxs[0], Chem.BondType.DOUBLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_idxs[1], Chem.BondType.DOUBLE)

    elif lg_group == 'C[O:1].[O:1]':
        pass

    elif lg_group == 'C([CH2:1])[O:1]':
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'C':
                c_attach_idx = amap_idx[atom.GetAtomMapNum()]
            elif atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'O':
                o_attach_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], o_attach_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], o_attach_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == 'C(C[O:1])[CH2:1]':
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'C':
                c_attach_idx = amap_idx[atom.GetAtomMapNum()]
            elif atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'O':
                o_attach_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], o_attach_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], o_attach_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == 'CC[O:1].C[CH2:1]':
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'C':
                c_attach_idx = amap_idx[atom.GetAtomMapNum()]
            elif atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'O':
                o_attach_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], o_attach_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], c_attach_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], o_attach_idx, Chem.BondType.SINGLE)

        else:
            pass

    elif lg_group == 'C[N:1]C.C[O:1].C[O:1]':
        lg_attach_o_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'O':
                lg_attach_o_idxs.append(amap_idx[atom.GetAtomMapNum()])

            elif atom.GetAtomMapNum() >= 1000 and atom.GetSymbol() == 'N':
                lg_attach_n_idx = amap_idx[atom.GetAtomMapNum()]

        if len(frag_attach_idxs) == 1:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idxs[0], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idxs[1], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_n_idx, Chem.BondType.SINGLE)

        elif len(frag_attach_idxs) == 2:
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_o_idxs[0], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[1], lg_attach_o_idxs[1], Chem.BondType.SINGLE)
            rw_mol.AddBond(frag_attach_idxs[0], lg_attach_n_idx, Chem.BondType.SINGLE)

        else:
            pass

    return rw_mol

def multi_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol):
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                if atom.GetAtomMapNum() != 0}

    if lg_group in SPECIAL_MULTI_ATTACH:
        rw_mol = special_multi_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol)
        return rw_mol

    else:
        lg_attach_idxs = []
        for atom in lg_mol.GetAtoms():
            if atom.GetAtomMapNum() >= 1000:
                lg_attach_idxs.append(amap_idx[atom.GetAtomMapNum()])

    if len(frag_attach_idxs) == 0:
        print("Cannot attach with no attachment points")
        return rw_mol

    elif len(frag_attach_idxs) == 1:
        frag_attach_idx = frag_attach_idxs[0]
        for lg_attach_idx in lg_attach_idxs:
            rw_mol.AddBond(frag_attach_idx, lg_attach_idx, Chem.BondType.SINGLE)

    elif len(frag_attach_idxs) == 2:
        for frag_idx, lg_idx in zip(*(frag_attach_idxs, lg_attach_idxs)):
            rw_mol.AddBond(frag_idx, lg_idx, Chem.BondType.SINGLE)

    return rw_mol

def attach_lg_to_mol(rw_mol, frag_attach_idxs, lg_group, lg_mol):
    if lg_group == "<eos>":
        return rw_mol

    elif lg_group in SINGLE_ATTACH:
        return single_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol)

    else:
        return multi_attach_lg(rw_mol, frag_attach_idxs, lg_group, lg_mol)

def canonicalize(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
    except:
        print('no mol', flush=True)
        return smiles
    if tmp is None:
        return smiles
    tmp = Chem.RemoveHs(tmp)
    [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    return Chem.MolToSmiles(tmp)

def edit_mol(prod_smi, edit, lg_groups):
    new_mol = Chem.MolFromSmiles(prod_smi)
    combined_mol = Chem.Mol(new_mol)

    if isinstance(edit, list):
        edit = edit[0]

    lg_mols = []
    lg_start = 1000

    for lg_group in lg_groups:
        if lg_group != "<eos>" and lg_group != 'c1c[n:1]cn1.c1c[n:1]cn1':
            lg_mol = Chem.MolFromSmiles(lg_group)
            for atom in lg_mol.GetAtoms():
                if atom.GetAtomMapNum() == 1:
                    atom.SetAtomMapNum(lg_start)
                    lg_start += 1

            combined_mol = Chem.CombineMols(combined_mol, lg_mol)
            lg_mols.append(lg_mol)

        elif lg_group == 'c1c[n:1]cn1.c1c[n:1]cn1':
            lg_mol = Chem.MolFromSmiles('c1c[N:1]cn1.c1c[N:1]cn1')
            for atom in lg_mol.GetAtoms():
                if atom.GetAtomMapNum() == 1:
                    atom.SetAtomMapNum(lg_start)
                    lg_start += 1

            combined_mol = Chem.CombineMols(combined_mol, lg_mol)
            lg_mols.append(lg_mol)

        else:
            lg_mols.append("<eos>")

    rw_mol = Chem.RWMol(Chem.Mol(combined_mol))

    if not isinstance(edit, list):
        fragments = apply_edits_to_mol(Chem.Mol(new_mol), [edit])
    else:
        fragments = apply_edits_to_mol(Chem.Mol(new_mol), edit)

    frag_mols = MultiElement(fragments).mols
    num_frag = len(frag_mols)
    assert len(lg_groups) == num_frag

    frag_amaps = [set([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
                 for mol in frag_mols]

    a1, a2, b1, b2 = edit.split(":")
    a1, a2, b1, b2 = int(a1), int(a2), float(b1), float(b2)

    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in rw_mol.GetAtoms()
                if atom.GetAtomMapNum() != 0}

    if a2 == 0:
        atom = rw_mol.GetAtomWithIdx(amap_idx[a1])

        if atom.GetSymbol() == 'N':
            if atom.GetFormalCharge() == 1:
                # is there a corresponding ring case?
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(1)

            elif atom.GetFormalCharge() == 0 and atom.GetNumExplicitHs() >= 1:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)

        else:
            if atom.GetNumExplicitHs() >= 1:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)

        frag_attach_idxs = [amap_idx[a1]]
        rw_mol = attach_lg_to_mol(rw_mol, frag_attach_idxs, lg_groups[0], lg_mols[0])
        return rw_mol

    else:
        atom_a = rw_mol.GetAtomWithIdx(amap_idx[a1])
        atom_b = rw_mol.GetAtomWithIdx(amap_idx[a2])

        if b2 == 0 and b1 > 0:
            if rw_mol.GetBondBetweenAtoms(atom_a.GetIdx(), atom_b.GetIdx()).GetBondTypeAsDouble() == 1.5:
                b1 = 1.5

            rw_mol.RemoveBond(amap_idx[a1], amap_idx[a2])

            if b1 == 1.0:
                if atom_a.GetSymbol() == 'N' and atom_b.GetSymbol() == 'O':
                    if atom_a.GetFormalCharge() == 1:
                        atom_a.SetFormalCharge(0)
                        atom_a.SetNumExplicitHs(0)

                    if atom_b.GetFormalCharge() == -1:
                        atom_b.SetFormalCharge(0)


                elif atom_b.GetSymbol() == 'N' and atom_a.GetSymbol() == 'O':
                    if atom_a.GetFormalCharge() == 1:
                        atom_b.SetFormalCharge(0)
                        atom_b.SetNumExplicitHs(0)

                    if atom_a.GetFormalCharge() == -1:
                        atom_a.SetFormalCharge(0)

                else:
                    atom_a.SetNumExplicitHs(atom_a.GetNumExplicitHs() + 1)
                    atom_b.SetNumExplicitHs(atom_b.GetNumExplicitHs() + 1)

            elif b1 == 2.0:
                atom_a.SetNumExplicitHs(atom_a.GetNumExplicitHs() + 2)
                atom_b.SetNumExplicitHs(atom_b.GetNumExplicitHs() + 2)

            elif b1 == 1.5:
                pass

        elif b1 > 0 and b2 > 0:
            bond = rw_mol.GetBondBetweenAtoms(atom_a.GetIdx(), atom_b.GetIdx())
            bond.SetBondType(BOND_FLOAT_TO_TYPE[b2])

            if atom_a.GetSymbol() == 'S' and atom_b.GetSymbol() == 'O':
                if b1 == 1.0 and b2 == 2.0 and atom_b.GetFormalCharge() == -1:
                    atom_b.SetFormalCharge(0)

            elif atom_b.GetSymbol() == 'S' and atom_a.GetSymbol() == 'O':
                if b1 == 1.0 and b2 == 2.0 and atom_a.GetFormalCharge() == -1:
                    atom_a.SetFormalCharge(0)

            delta = b1 - b2
            if delta > 0:
                atom_a.SetNumExplicitHs(int(atom_a.GetNumExplicitHs() + delta))
                atom_b.SetNumExplicitHs(int(atom_b.GetNumExplicitHs() + delta))

            elif delta < 0:
                atom_a.SetNumExplicitHs(int(max(0, atom_a.GetNumExplicitHs()-delta)))
                atom_b.SetNumExplicitHs(int(max(0, atom_b.GetNumExplicitHs()-delta)))

        else:
            pass

    if num_frag == 1:
        if lg_groups[0] == "<eos>":
            frag_attach_idxs = [[]]

        elif lg_groups[0] in SINGLE_ATTACH and lg_groups[0] in CARBON_ATTACH:
            # pick one between a1 and a2
            if atom_a.GetSymbol() == 'C' and atom_b.GetSymbol() == 'C':
                frag_attach_idxs = [[amap_idx[a1]]]

            elif atom_b.GetSymbol() == 'C' and atom_a.GetSymbol() != 'C':
                frag_attach_idxs = [[amap_idx[a2]]]

            elif atom_b.GetSymbol() != 'C' and atom_a.GetSymbol() == 'C':
                frag_attach_idxs = [[amap_idx[a1]]]

            else:
                frag_attach_idxs = [[amap_idx[a1]]]

        elif lg_groups[0] in CYCLIC_ATTACH:
            idx1, idx2 = get_oc_idx(atom_a, atom_b)
            frag_attach_idxs = [[idx1, idx2]]

        elif lg_groups[0] in SINGLE_ATTACH:
            frag_attach_idxs = [[amap_idx[a1]]]

        elif lg_groups[0] not in SINGLE_ATTACH:
            frag_attach_idxs = [[amap_idx[a1], amap_idx[a2]]]

    elif num_frag == 2:
        assert len(frag_amaps) == 2
        if a1 in frag_amaps[0] and a2 in frag_amaps[1]:
            frag_attach_idxs = [[amap_idx[a1]], [amap_idx[a2]]]
        elif a1 in frag_amaps[1] and a2 in frag_amaps[0]:
            frag_attach_idxs = [[amap_idx[a2]], [amap_idx[a1]]]

    for frag_id in range(num_frag):
        rw_mol = attach_lg_to_mol(rw_mol, frag_attach_idxs[frag_id],
                                  lg_groups[frag_id], lg_mols[frag_id])

    return rw_mol

def generate_reac_set(prod_smi, edit, lg_groups, verbose=False):
    azide_rule = AllChem.ReactionFromSmarts('[NH:2]=[N+:3]=[N-:4]>>[NH0-:2]=[N+:3]=[N-:4]')
    tmp_mol = Chem.MolFromSmiles(prod_smi)
    aromatic_co_adj_n = set()
    aromatic_co = set()
    aromatic_cs = set()
    aromatic_cn = set()

    for atom in tmp_mol.GetAtoms():
        if atom.GetSymbol() == 'C':
            nei_symbols = [nei.GetSymbol() for nei in atom.GetNeighbors()]
            if 'O' in nei_symbols and 'N' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                aromatic_co_adj_n.add(atom.GetIdx())

            elif 'O' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                aromatic_co.add(atom.GetIdx())

            elif 'N' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                aromatic_cn.add(atom.GetIdx())

            elif 'S' in nei_symbols and len(atom.GetBonds()) == 3 and atom.GetIsAromatic():
                aromatic_cs.add(atom.GetIdx())

    rw_mol = edit_mol(prod_smi, edit, lg_groups)
    reac_mol = rw_mol.GetMol()

    for atom in reac_mol.GetAtoms():
        if atom.GetSymbol() == 'N':
            if not atom.GetIsAromatic():
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= MAX_VALENCE['N']:
                    atom.SetNumExplicitHs(0)
                    atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['N']))

            elif atom.GetIsAromatic() and atom.GetFormalCharge() == 1:
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals == MAX_VALENCE['N']:
                    atom.SetNumExplicitHs(0)
                    atom.SetFormalCharge(0)

        elif atom.GetSymbol() == 'C':
            check1 = atom.GetIdx() in aromatic_co_adj_n
            check2 = atom.GetIdx() in aromatic_co
            check3 = atom.GetIdx() in aromatic_cs
            check4 = atom.GetIdx() in aromatic_cn

            if check1 or check2 or check3 or check4:
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= MAX_VALENCE['C']:
                    atom.SetNumExplicitHs(0)

            else:
                bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
                if bond_vals >= MAX_VALENCE['C']:
                    atom.SetNumExplicitHs(0)
                    atom.SetFormalCharge(int(bond_vals - MAX_VALENCE['C']))

                elif bond_vals < MAX_VALENCE['C']:
                    atom.SetNumExplicitHs(int(MAX_VALENCE['C'] - bond_vals))
                    atom.SetFormalCharge(0)

        elif atom.GetSymbol() == 'S':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals in [2, 4, 6]:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(0)

        elif atom.GetSymbol() == 'Sn':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals >= 4:
                atom.SetNumExplicitHs(0)

        elif atom.GetSymbol() == 'O':
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals >= MAX_VALENCE['O']:
                atom.SetNumExplicitHs(0)

            elif bond_vals < MAX_VALENCE['O'] and atom.GetFormalCharge() != -1:
                atom.SetNumExplicitHs(int(MAX_VALENCE['O'] - bond_vals))
                atom.SetFormalCharge(0)

        elif atom.GetSymbol() == 'B': # quartenary boron should be neg. charge with 0 H
            bond_vals = [bond.GetBondTypeAsDouble() for bond in atom.GetBonds()]
            if sum(bond_vals) == 4 and len(bond_vals) == 4:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)

            elif sum(bond_vals) >= 3:
                atom.SetNumExplicitHs(0)

        elif atom.GetSymbol() in ['Br', 'Cl', 'I', 'F']:
            bond_vals = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
            if bond_vals >= MAX_VALENCE[atom.GetSymbol()]:
                atom.SetNumExplicitHs(0)

    reac_smi = Chem.MolToSmiles(reac_mol)
    reac_smi = canonicalize(reac_smi)

    reac_mols = [Chem.MolFromSmiles(smi) for smi in reac_smi.split(".")]
    for index, mol in enumerate(reac_mols):
        if mol is None:
            continue
        out = azide_rule.RunReactants((mol, ))
        if len(out):
            reac_mols[index] = out[0][0]

    reac_set = set([Chem.MolToSmiles(mol) for mol in reac_mols if mol is not None])
    if verbose:
        print("Generated Reactants: ", reac_set)
    return reac_set
