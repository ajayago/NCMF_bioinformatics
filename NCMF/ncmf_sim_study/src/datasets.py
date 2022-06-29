import math
import torch
import numpy as np
import scipy as sp
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.sparse import coo_matrix, csr_matrix
# ----------------------------------- Dataset V2 ---------------------------------------- #


class MatrixDatasetV2(Dataset):
    """A matrix dataset."""

    def __init__(self, matrix, mask, batch_size, transform, mask_transform):
        self.matrix = torch.from_numpy(matrix).type(torch.float32)
        self.mask = torch.from_numpy(mask)
        self.batch_size = batch_size
        self.transform = transform
        self.mask_transform = mask_transform
        self.batch_indices = self.__init_batch_indices()

    def __len__(self):
        return len(self.batch_indices)

    def __getitem__(self, idx):
        ind_slice = self.batch_indices[idx]
        batch_matrix = self.matrix[ind_slice]
        batch_mask = self.mask[ind_slice]

        if self.transform:
            batch_matrix = self.transform(batch_matrix)
        if self.mask_transform:
            batch_mask = self.mask_transform(batch_mask)

        return batch_matrix, batch_mask

    def __init_batch_indices(self):
        """Initialises the matrix indices within each batch."""
        nrows, _ = self.matrix.shape
        num_batches = math.ceil(nrows / self.batch_size)

        batch_ind = []
        for i in range(num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            if i == num_batches - 1:  # last batch
                end = nrows
                print('Warning: Last batch has {} rows, while other batch sizes are {}. '.format(
                    end-start, self.batch_size))
            batch_ind.append(slice(start, end))

        return batch_ind


class CellInteractionDataset(TensorDataset):
    def __init__(self, rows, columns, values, transform=None, target_transform=None):
        super(CellInteractionDataset, self).__init__(
            torch.from_numpy(rows), 
            torch.from_numpy(columns), 
            torch.from_numpy(values).type(torch.float32)
        )
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        row = self.tensors[0][idx]
        col = self.tensors[1][idx]
        val = self.tensors[2][idx]

        if self.transform:
            row = self.transform(self.tensors[0][idx])
            col = self.transform(self.tensors[1][idx])
        if self.target_transform:
            val = self.target_transform(self.tensors[2][idx])

        return (row, col, val)
