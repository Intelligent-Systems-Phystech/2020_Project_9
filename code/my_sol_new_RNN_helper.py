import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import MSELoss, Sequential, Linear, Sigmoid, Tanh, ReLU, ELU, Softmax
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils import clip_grad_norm_
import random
import matplotlib.ticker as ticker


def get_neighbors(graph, node, from_node = None):
    neighbors = {'this' : node, 'next' : []}
    for edge in graph[0]:
        if (edge[0] == node) and (edge[1] != from_node):
            neighbors['next'].append(edge[1])
        elif (edge[1] == node) and (edge[0] != from_node):
            neighbors['next'].append(edge[0])
    return neighbors

def get_p_neighbors(graph, node, p, from_node = None):
    neighbors = get_neighbors(graph, node, from_node)
    if p == 0:
        now = [[neighbors['this']]]
    else:
        now = []
        for nei in neighbors['next']:
            nextnei = get_p_neighbors(graph, nei, p - 1, node)
            for neinei in nextnei:
                now.append([neighbors['this']] + neinei)
    return now

def graph2tensor(graph, num_neighbors):
    t = []
    for i in range(len(graph[1])):
        t+=get_p_neighbors(graph, i, num_neighbors)
    ten =  np.array(t)
    if len(ten) == 0:
        return ten
    ten[:, :, 0] = ten[:, :, 0] / 28
    ten[:, :, 1] = ten[:, :, 1] / 28
    return ten

def get_graph_matrix(eges, num_nodes):
    matrix = torch.zeros(num_nodes, num_nodes)
    for i in eges:
        matrix[i[0], i[1]] = 1
        matrix[i[1], i[0]] = 1
    for indx, i in enumerate(matrix):
        matrix[indx] = matrix[indx] / i.sum()
    matrix.requires_grad_(requires_grad = False)
    return matrix

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center) 
    components  = v.t()[:k]#v[:k]#.t()
    explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return { 'X':X, 'k':k, 'components':components,
            'explained_variance':explained_variance }

class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
#         x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.softmax(self.fc2(x))
        return x

class GraphEmbDataset(Dataset):
    """Graph embending of sceletonized MNIST dataset."""

    def __init__(self, graph_file):
        with open (graph_file, 'rb') as fp:
            data_frame = pickle.load(fp)
        
        self.data = []
        for i in data_frame:
            nodes = Tensor(i[1])
            nodes.requires_grad_(requires_grad = False)
            nodes[:, 0] = nodes[:, 0] / 28
            nodes[:, 1] = nodes[:, 1] / 28
            neighbors = []
            for n in range(len(i[1])):
                neighbors.append(get_neighbors(i[:2], n)['next'])
            matrix = get_graph_matrix(i[0], len(i[1]))
            self.data.append([nodes, neighbors, i[2]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return self.data[idx]

def train(model, train_dataset, batch_size, epochs, optimizer, loss):
    losses = []
    with tqdm(total=len(train_dataset)) as pbar:
        mapp = np.arange(len(train_dataset))
        for epoch in range(epochs):
            pbar.desc = str(epoch + 1) + ' from ' + str(epochs)
            pbar.reset()
            np.random.shuffle(mapp)

            for batch_idx in range(0, len(train_dataset), batch_size):
                optimizer.zero_grad()
                labels = []
                out = None
                for indx in mapp[batch_idx:batch_idx+batch_size]:
                    out_1 = train_dataset.__getitem__(indx)
                    if len(out_1[0]) <= 5:
                        continue
                    labels.append(out_1[2])
                    a = model.forward(*out_1[:-1])
                    if out is None:
                        out = a
                    else :
                        out = torch.cat([out, a], 0)
                target = torch.LongTensor(labels)
                output = loss(out, target)
                output.backward()
                optimizer.step()
                pbar.update(batch_size)
                
                if batch_idx % 384 == 0:
                    losses.append(output.item())
    return losses

def showHeatMap(model, train_dataset):
    all_categories = [str(x) for x in range(10)]

    confusion = torch.zeros(10, 10)

    def categoryFromOutput(output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return category_i

    pr = 0
    al = 0

    with tqdm(total=len(train_dataset)) as pbar:
        # Go through a bunch of examples and record which are correctly guessed
        for i in range(len(train_dataset)):
            nodes, matrix, clas = train_dataset.__getitem__(i)
            pbar.update(1)
            if len(nodes) <= 5:
                continue
            output = g2v.test(nodes, matrix)
            guess_i = categoryFromOutput(output)
            confusion[clas][guess_i] += 1
            al+=1
            if clas == guess_i:
                pr+=1
        print('accuracy:',pr/al)
    
    # Normalize by dividing every row by its sum
    for i in range(10):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

