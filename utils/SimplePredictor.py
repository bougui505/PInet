import argparse
import os
import sys

import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

script_dir = os.path.curdir
sys.path.append(os.path.join(script_dir, '..'))

from pinet.model import PointNetDenseCls12, feature_transform_regularizer

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))


def get_classifier(weights_path='../models/dbd_aug.pth', device='cpu'):
    num_classes = 2
    classifier = PointNetDenseCls12(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    classifier.load_state_dict(torch.load(weights_path, map_location=device))
    classifier.eval()
    return classifier


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


def get_preds(model, points_r, points_l, dump_r, dump_l):
    # Make inference and add activation
    pred, _, _ = model(points_r, points_l)
    pred = pred.view(-1, 1)
    pred = torch.sigmoid(pred).view(1, -1).squeeze().data.cpu().numpy()

    # split prediction
    size_r = points_r.squeeze().shape[1]
    pred_r, pred_l = pred[:size_r], pred[size_r:]
    if dump_r is not None:
        np.savetxt(fname=dump_r, X=pred_r)
    if dump_l is not None:
        np.savetxt(fname=dump_r, X=pred_l)
    return pred_r, pred_l


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pts_r_file = '../data/2I25/2I25-r.pts'
    pts_l_file = '../data/2I25/2I25-l.pts'
    dump_r = '../data/2I25/2I25_prob_l.seg'
    dump_l = '../data/2I25/2I25_prob_r.seg'

    classifier = get_classifier(device=device)
    points_l = get_input(pts_file=pts_l_file, device=device)
    points_r = get_input(pts_file=pts_r_file, device=device)

    pred_r, pred_l = get_preds(model=classifier, points_r=points_r, points_l=points_l,
                               dump_r=dump_r, dump_l=dump_l)
