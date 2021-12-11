import torch
from typing import Any, Tuple, List
import os
import joblib

from seq_graph_retro.utils.parse import ReactionInfo

class BaseDataset(torch.utils.data.Dataset):
    """BaseDataset is an abstract class that loads the saved tensor batches and
    passes them to the model for training."""

    def __init__(self, data_dir: str, mpnn: str = 'graph_feat', **kwargs):
        """
        Parameters
        ----------
        data_dir: str,
            Data directory to load batches from
        mpnn: str, default graph_feat
            MPNN to load batches for
        num_batches: int, default None,
            Number of batches to load in the directory
        """
        if mpnn == 'gtrans':
            mpnn = 'graph_feat'
        self.data_dir = data_dir
        self.data_files = [
            os.path.join(self.data_dir, mpnn, file)
            for file in os.listdir(os.path.join(self.data_dir, mpnn))
            if "batch-" in file
        ]
        self.__dict__.update(**kwargs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Retrieves a particular batch of tensors.

        Parameters
        ----------
        idx: int,
            Batch index
        """
        batch_tensors = torch.load(self.data_files[idx], map_location='cpu')
        return batch_tensors

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        return len(self.data_files)

    def create_loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Creates a DataLoader from given batches.

        Parameters
        ----------
        batch_size: int,
            Batch size of outputs
        num_workers: int, default 6
            Number of workers to use
        shuffle: bool, default True
            Whether to shuffle batches
        """
        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers,
                                           collate_fn=self.collater)

    def collater(self, attributes: List[Any]):
        """Processes the batch of tensors to yield corresponding inputs."""
        raise NotImplementedError("Subclasses must implement for themselves")

class EvalDataset(torch.utils.data.Dataset):

    """EvalDataset is an abstract class that handles evaluation during training."""

    def __init__(self, data_dir: str, data_file: str, labels_file: str = None,
                 num_shards: int = None, use_rxn_class: bool  = False) -> None:
        """
        Parameters
        ----------
        data_dir: str,
            Data directory to load batches from
        data_file: str,
            Info file to load
        labels_file: str, default None,
            Labels file. If None, nothing to load
        num_shards: int, default None,
            Number of info file shards present
        use_rxn_class: bool, default False,
            Whether to use reaction class as additional feature.
        """
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, data_file)
        self.use_rxn_class = use_rxn_class

        if num_shards is not None:
            self.dataset = []
            for shard_num in range(num_shards):
                shard_file = self.data_file + f"-shard-{shard_num}"
                self.dataset.extend(joblib.load(shard_file))

        else:
            self.dataset = joblib.load(self.data_file)

        self.labels = None
        if labels_file is not None:
            self.labels = joblib.load(os.path.join(data_dir, labels_file))
            assert len(self.labels) == len(self.dataset)

    def __len__(self) -> int:
        """Returns length of the Dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> ReactionInfo:
        """Retrieves the corresponding ReactionInfo

        Parameters
        ----------
        idx: int,
        Index of particular element
        """
        if self.labels is not None:
            return self.dataset[idx], self.labels[idx]
        return self.dataset[idx]

    def create_loader(self, batch_size: int, num_workers: int = 6, shuffle: bool = False) -> torch.utils.data.DataLoader:
        """Creates a DataLoader from given batches.

        Parameters
        ----------
        batch_size: int,
            Batch size of outputs
        num_workers: int, default 6
            Number of workers to use
        shuffle: bool, default True
            Whether to shuffle batches
        """
        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size,
                                           shuffle=shuffle, num_workers=num_workers,
                                           collate_fn=self.collater)

    def collater(self, attributes: List[Any]):
        """Processes the batch of tensors to yield corresponding inputs."""
        raise NotImplementedError("Subclasses must implement for themselves")
