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

from pinet.model import PointNetDenseCls12, feature_transform_regularizer, PointNetDenseCls12Double

random.seed(random.randint(1, 10000))
torch.manual_seed(random.randint(1, 10000))


def get_classifier(weights_path='../models/split_0.pth', device='cpu', double=False):
    num_classes = 2
    if double:
        classifier = PointNetDenseCls12Double(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    else:
        classifier = PointNetDenseCls12(k=num_classes, feature_transform=False, pdrop=0.0, id=5)
    classifier.load_state_dict(torch.load(weights_path, map_location='cpu'))
    classifier.eval()
    classifier = classifier.to(device)
    return classifier


def get_input(pts_file, device='cpu'):
    points = np.loadtxt(pts_file).astype(np.float32)
    coordset = points[:, 0:3]
    featset = points[:, 3:]
    featset = featset / np.sqrt(np.max(featset ** 2, axis=0))
    coordset = coordset - np.expand_dims(np.mean(coordset, axis=0), 0)  # center
    points[:, 0:5] = np.concatenate((coordset, featset), axis=1)
    points = torch.from_numpy(points).unsqueeze(0)

    # This precaution does not seem too important for inference...
    # memlim = 120000
    # if points.size()[1] + points.size()[1] > memlim:
    #     subset_size = points.size()[1] * memlim / (points.size()[1] + points.size()[1])
    #     subset_size = int(subset_size)
    #     subset = np.random.choice(points.size()[1], subset_size, replace=False)
    #     points = points[:, subset, :]
    points = points.transpose(2, 1)
    points = points.to(device)
    return points


def get_double_preds(model, points_r, points_l, dump_r, dump_l):
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
        np.savetxt(fname=dump_l, X=pred_l)
    return pred_r, pred_l


def get_preds(model, points, dump):
    # Make inference and add activation
    pred = model(points)
    pred = pred.view(-1, 1)
    pred = torch.sigmoid(pred).view(1, -1).squeeze().data.cpu().numpy()
    np.savetxt(fname=dump, X=pred)
    return pred


def process_all_double(dataset, pts_name_r='receptor.pts', pts_name_l='ligand.pts',
                       device='cpu', overwrite=False, basename_dump='_prob_double.seg'):
    classifier = get_classifier(device=device, double=True)
    for system in tqdm(os.listdir(dataset)):
        basename_r = pts_name_r[:-4]
        basename_l = pts_name_l[:-4]
        pts_r_file = os.path.join(dataset, system, pts_name_r)
        pts_l_file = os.path.join(dataset, system, pts_name_l)

        # Don't try to predict if we failed to dump a .pts file
        if not (os.path.exists(pts_r_file) and os.path.exists(pts_l_file)):
            continue

        # Don't predict if prediction already exist
        dump_r = os.path.join(dataset, system, basename_r + basename_dump)
        dump_l = os.path.join(dataset, system, basename_l + basename_dump)
        # print(dump_r)
        if not overwrite and os.path.exists(dump_l) and os.path.exists(dump_r):
            continue

        points_r = get_input(pts_file=pts_r_file, device=device)
        points_l = get_input(pts_file=pts_l_file, device=device)
        get_double_preds(model=classifier, points_r=points_r, points_l=points_l, dump_r=dump_r, dump_l=dump_l)


def process_one(dataset, system, pts_name, model, device, dump_name, overwrite=False):
    pts_file = os.path.join(dataset, system, pts_name)
    dump_file = os.path.join(dataset, system, dump_name)
    if not os.path.exists(pts_file):
        return
    if not overwrite and os.path.exists(dump_file):
        return
    points = get_input(pts_file=pts_file, device=device)
    get_preds(model=model, points=points, dump=dump_file)


def process_all_dbd5(dataset, device='cpu', overwrite=False, basename_dump='prob.seg'):
    classifier = get_classifier(device=device, double=False)
    for system in tqdm(os.listdir(dataset)):
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='receptor_b_dropbox.pts', dump_name=f'receptor_b_{basename_dump}')
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='receptor_u_dropbox.pts', dump_name=f'receptor_u_{basename_dump}')
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='ligand_b_dropbox.pts', dump_name=f'ligand_b_{basename_dump}')
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='ligand_u_dropbox.pts', dump_name=f'ligand_u_{basename_dump}')


def process_all_epipred(dataset, device='cpu', overwrite=False, basename_dump='prob.seg'):
    classifier = get_classifier(device=device, double=False)
    for system in tqdm(os.listdir(dataset)):
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='receptor.pts', dump_name=f'receptor_{basename_dump}')
        process_one(dataset=dataset, model=classifier, device=device, system=system, overwrite=overwrite,
                    pts_name='ligand.pts', dump_name=f'ligand_{basename_dump}')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # pts_r_file = '../data/2I25/2I25-r.pts'
    # pts_l_file = '../data/2I25/2I25-l.pts'
    # dump_r = '../data/2I25/2I25_prob_l.seg'
    # dump_l = '../data/2I25/2I25_prob_r.seg'

    # points_r = get_input(pts_file=pts_r_file, device=device)
    # points_l = get_input(pts_file=pts_l_file, device=device)
    # points_l = None
    # classifier = get_classifier(device=device)
    # pred_r, pred_l = get_double_preds(model=classifier, points_r=points_r, points_l=points_l,
    #                                   dump_r=dump_r, dump_l=dump_l)

    # Do the double version
    # For Epipred
    # dataset = '../../dl_atomic_density_hd/data/epipred/'
    # pts_name_r = 'receptor.pts'
    # pts_name_l = 'ligand.pts'
    # process_all_double(dataset=dataset, pts_name_r=pts_name_r, pts_name_l=pts_name_l, device=device, overwrite=True)

    # For dbd5
    dataset = '../../dl_atomic_density_hd/data/dbd5/'
    pts_name_r_b = 'receptor_b_dropbox.pts'
    pts_name_r_u = 'receptor_u_dropbox.pts'
    pts_name_l_b = 'ligand_b_dropbox.pts'
    pts_name_l_u = 'ligand_u_dropbox.pts'
    # To get a 'double' version for the apo forms, we pair them with the bound version
    # and then overwrite with the bound couple
    # process_all_double(dataset=dataset, pts_name_r=pts_name_r_u, pts_name_l=pts_name_l_b,
    #                    device=device, overwrite=True, basename_dump='_prob_double_final.seg')
    # process_all_double(dataset=dataset, pts_name_r=pts_name_r_b, pts_name_l=pts_name_l_u,
    #                    device=device, overwrite=True, basename_dump='_prob_double_final.seg')
    process_all_double(dataset=dataset, pts_name_r=pts_name_r_b, pts_name_l=pts_name_l_b,
                       device=device, overwrite=True, basename_dump='_prob_double_final.seg')

    # Do the more honest simple version
    # For Epipred
    dataset = '../../dl_atomic_density_hd/data/epipred/'
    process_all_epipred(dataset=dataset, device=device, overwrite=True, basename_dump='prob_final.seg')

    # For dbd5
    # dataset = '../../DeepInterface/data/dbd5/'
    dataset = '../../dl_atomic_density_hd/data/dbd5/'
    process_all_dbd5(dataset=dataset, device=device, overwrite=True, basename_dump='prob_final.seg')
