import os
import sys
from itertools import islice
from itertools import repeat, product
from math import sqrt
from PIL import Image
import numpy as np
import torch
from scipy import stats
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset, DataLoader
from torchvision import transforms

from creat_data_img import creat_data


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='_drug1',
                 xd=None, xt=None, y=None, xt_featrue=None, transform=None,
                 pre_transform=None, smile_graph=None, smile_imageidx=None,
                 img_root=None, img_transform=None, img_normalize=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_featrue, y, smile_graph, smile_imageidx)
            self.data, self.slices = torch.load(self.processed_paths[0])
        # image
        self.img_root, self.img_transform, self.img_normalize = img_root, img_transform, img_normalize
        if self.img_root is not None:
            if self.img_normalize is not None:
                self.img_normalize = img_normalize
            else:
                print("using default img_normalize: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
                self.img_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            if self.img_transform is not None:
                self.img_transform = img_transform
            else:
                print("using default img_transform: transforms.Compose([transforms.ToTensor(), self.img_normalize])")
                self.img_transform = transforms.Compose([transforms.ToTensor(), self.img_normalize])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def get_cell_feature(self, cellId, cell_features):
        for row in islice(cell_features, 0, None):
            if cellId in row[0]:
                return row[1:]
        return False

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_featrue, y, smile_graph, smile_imageidx):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print('number of data', data_len)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            img_idx = smile_imageidx[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.Tensor([labels]),
                                img_idx=img_idx)
            cell = self.get_cell_feature(target, xt_featrue)

            if cell is False : 
                print('cell', cell)
                sys.exit()

            new_cell = []
            # print('cell_feature', cell_feature)
            for n in cell:
                new_cell.append(float(n))
            GCNData.cell = torch.FloatTensor([new_cell])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def get_image(self, path):
        img = Image.open(path).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img

    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            start, end = slices[idx].item(), slices[idx + 1].item()
            # print(slices[idx], slices[idx + 1])
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(start, end)
            elif start + 1 == end:
                s = slices[start]
            else:
                s = slice(start, end)
            data[key] = item[s]

        if hasattr(data, "img_idx") and self.img_root is not None:
            image = self.get_image(f"{self.img_root}/{data.img_idx.item()}.png")
            data.image = image

        return data


def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def save_AUCs(AUCs, filename):
    with open(filename, 'a') as f:
        f.write('\t'.join(map(str, AUCs)) + '\n')
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci
