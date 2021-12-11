from seq_graph_retro.molgraph.mol_features import ATOM_LIST

class Vocab:
    """Vocab class to deal with atom vocabularies and other attributes."""

    def __init__(self, elem_list=ATOM_LIST[:-1]) -> None:
        """
        Parameters
        ----------
        elem_list: List, default ATOM_LIST
            Element list used for setting up the vocab
        """
        self.elem_list = elem_list
        if isinstance(elem_list, dict):
            self.elem_list = list(elem_list.keys())
        self.elem_to_idx = {a: idx for idx, a in enumerate(self.elem_list)}
        self.idx_to_elem = {idx: a for idx, a in enumerate(self.elem_list)}

    def __getitem__(self, a_type: str) -> int:
        return self.elem_to_idx[a_type]

    def get(self, elem: str, idx: int = None) -> int:
        """Returns the index of the element, else a None for missing element.

        Parameters
        ----------
        elem: str,
            Element to query
        idx: int, default None
            Index to return if element not in vocab
        """
        return self.elem_to_idx.get(elem, idx)

    def get_elem(self, idx: int) -> str:
        """Returns the element at given index.

        Parameters
        ----------
        idx: int,
            Index to return if element not in vocab
        """
        return self.idx_to_elem[idx]

    def __len__(self) -> int:
        return len(self.elem_list)

    def index(self, elem: str) -> int:
        """Returns the index of the element.

        Parameters
        ----------
        elem: str,
            Element to query
        """
        return self.elem_to_idx[elem]

    def size(self) -> int:
        """Returns length of Vocab."""
        return len(self.elem_list)

COMMON_ATOMS = [('B', 0), ('Br', 0), ('C', -1), ('C', 0), ('Cl', 0), ('Cu', 0), ('F', 0),
('I', 0), ('Mg', 0), ('Mg', 1), ('N', -1), ('N', 0), ('N', 1), ('O', -1), ('O', 0),
('P', 0), ('P', 1), ('S', -1), ('S', 0), ('S', 1), ('Se', 0), ('Si', 0), ('Sn', 0),
('Zn', 0), ('Zn', 1)]

common_atom_vocab = Vocab(COMMON_ATOMS)
