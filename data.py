"""
Copyright 2020 Michael Yang

The MIT License(MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class FixedDataset(Dataset):
    """
    Dataset with fixed length inputs x and targets y in any data type that can
    be constructed into torch.tensor such as Python list or numpy.ndarray.
    If no y data is given (for test data), each item returns a dummy 0 value.
    """

    def __init__(self, x, y=None, xtype=torch.float, ytype=torch.long, x_transforms=None, y_transforms=None):
        self.xtype = xtype
        self.ytype = ytype

        self.x = torch.tensor(x, dtype=xtype)
        if x_transforms:
            self.x = x_transforms(self.x)

        if y:
            assert len(x) == len(y)
            self.y = torch.tensor(y, dtype=ytype)
            if y_transforms:
                self.y = y_transforms(self.y)
        else:
            self.y = torch.zeros(len(x), dtype=ytype)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class VariableDataset(Dataset):
    """
    Dataset with variable length inputs x and targets y in any data type that can
    be constructed into torch.tensor such as Python list or numpy.ndarray.
    If no y data is given (for test data), each item returns a dummy 0 value.
    torch.tensor constructor is called in __getitem__ rather than __init__ to
    allow for variable length inputs and outputs
    """

    def __init__(self, x, y=None, xtype=torch.float, ytype=torch.long, x_transforms=None, y_transforms=None):
        self.xtype = xtype
        self.ytype = ytype

        self.x = [torch.tensor(seq, dtype=xtype) for seq in x]
        if x_transforms:
            self.x = [x_transforms(seq) for seq in self.x]

        if y:
            assert len(x) == len(y)
            self.y = [torch.tensor(seq, dtype=ytype) for seq in y]
            if y_transforms:
                self.y = [y_transforms(seq) for seq in self.y]
        else:
            self.y = [torch.tensor([0], dtype=ytype) for _ in range(len(x))]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def pad_collate(batch, batch_first=False, padding_value=0):
    """
    Standard collate function for batching variable length input and output
    using torch.nn.utils.rnn.pad_sequence
    """
    X, Y = zip(*batch)  # Collect single (x, y) pairs in batch into X and Y
    X_lens = torch.IntTensor([len(x) for x in X])  # (batch_size, )
    Y_lens = torch.IntTensor([len(y) for y in Y])  # (batch_size, )

    X_padded = pad_sequence(X, batch_first, padding_value)
    Y_padded = pad_sequence(Y, batch_first, padding_value)

    return X_padded, Y_padded, X_lens, Y_lens
    # source: https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html


def unpad_padded_sequence(input, lengths, batch_first=False):
    """
    :param input: (max_len, N, *) padded batch of variable length sequences
    :param lengths: (N, ) list of sequences lengths of each batch element
    :param batch_first: if True, the input is expected in N x T x * format
    :return output: [torch.tensor] list of (variably sized) sequences
    """
    output = []
    N = input.shape[0] if batch_first else input.shape[1]
    for i in range(N):
        seq = input[i, :lengths[i]] if batch_first else input[:lengths[i], i]
        output.append(seq)

    return output
