from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pinet.model import PointNetDenseCls12, feature_transform_regularizer
import sys
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))

def get_classifier(weights_path = '../models/dbd_aug.pth', device='cpu'):
    num_classes = 2
    classifier = PointNetDenseCls12(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    classifier.to(device)
    classifier.load_state_dict(torch.load(weights_path))
    classifier.eval()

def get_input(pts_file, device='cpu'):
    points = np.loadtxt(pts_file).astype(np.float32)
    coordset = points[:, 0:3]
    featset = points[:, 3:]
    featset = featset / np.sqrt(np.max(featset ** 2, axis=0))
    coordset = coordset - np.expand_dims(np.mean(coordset, axis=0), 0)  # center
    points[:, 0:5] = np.concatenate((coordset, featset), axis=1)
    points = torch.from_numpy(points).unsqueeze(0)

    memlim = 120000
    if points.size()[1] + points.size()[1] > memlim:
        subset_size = points.size()[1] * memlim / (points.size()[1] + points.size()[1])
        subset = np.random.choice(points.size()[1], subset_size, replace=False)
        points = points[:, subset, :]
    points = points.transpose(2, 1)
    points = points.to(device)
    return points


device = 'cuda' if torch.cuda.is_available() else 'cpu'
pts_r_file = '../data/2I25/2I25-r.pts'
pts_l_file = '../data/2I25/2I25-l.pts'
outfile = '../data/2I25/2I25_prob_r_l.seg'

classifier = get_classifier(device=device)
points_l = get_input(pts_file=pts_l_file,device=device)
points_r = get_input(pts_file=pts_r_file,device=device)

pred, _, _ = classifier(points_l, points_r)
pred = pred.view(-1, 1)
np.savetxt(outfile, torch.sigmoid(pred).view(1, -1).data.cpu())
