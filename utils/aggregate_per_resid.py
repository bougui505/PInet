#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################


import glob
import pickle
import sys

import numpy as np
from getcontactEpipred import getsppider2
from sklearn.neighbors import NearestNeighbors
import os
from Bio.PDB import *
from tqdm import tqdm
from pymol import cmd


def readpdb(file):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', file)

    atoms_ids_perresidue = []
    # tempo=[]
    atom_coords = []
    labeldic = []
    bdic = []
    all_residue_centroids = []
    mark = 0
    resid_keys = []
    for c in structure[0]:
        for resi in c:
            _, _, chain, (_, resid, _) = resi.get_full_id()
            resid_keys.append((chain, resid))
            # residic.append(resi._id[1])
            residue_centroid = [0, 0, 0]
            count = 0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                # if 'H' in atom.get_name():
                #     continue
                residue_centroid[0] += atom.get_coord()[0]
                residue_centroid[1] += atom.get_coord()[1]
                residue_centroid[2] += atom.get_coord()[2]
                count += 1

                # residic.append(resi._id[1])
                atoms_ids_perresidue.append(mark)
                atom_coords.append([atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]])
            residue_centroid = [coor * 1.0 / count for coor in residue_centroid]
            mark += 1
            all_residue_centroids.append(residue_centroid)
            # labeldic.append(1)

    atoms_ids_perresidue = np.asarray(atoms_ids_perresidue)
    atom_coords = np.asarray(atom_coords)
    return atoms_ids_perresidue, atom_coords, all_residue_centroids, resid_keys


def get_protein_coords_and_residues(pdbfilename):
    cmd.reinitialize()
    cmd.load(pdbfilename, 'inpdb')
    pymolsel = 'inpdb and polymer.protein'
    coords_in = cmd.get_coords(pymolsel)
    pymolspace = {'resids_in': [], 'chains_in': []}
    cmd.iterate(pymolsel,
                'resids_in.append(resi); chains_in.append(chain)',
                space=pymolspace)
    # resids_in = np.int_(pymolspace['resids_in'])
    resids_in = np.asarray(pymolspace['resids_in'])
    chains_in = np.asarray(pymolspace['chains_in'])
    return coords_in, resids_in, chains_in


def get_resid_seg(pdbfile, ptsfile, segfile):
    """
    ptsfile: .pts file
    segfile: .seg file
    RL: r or l for receptor or ligand
    """
    pts_coords = np.transpose(np.loadtxt(ptsfile))[0:3, :]
    pts_coords = np.transpose(pts_coords)

    pro = np.loadtxt(segfile)
    if np.min(pro) == 1:
        pro = pro - 1
    print(pro)

    n_neighbors = 3
    dt = 2
    cutoff = 0.5
    tol = [6, 6, 6]

    # atoms_ids_perresidue, atom_coords, _, resid_keys = readpdb(pdbfile)

    coords, resids, chains = get_protein_coords_and_residues(pdbfile)
    atom_resid_keys = [(chain, resid) for chain, resid in zip(chains, resids)]
    # print(len(set(atom_resid_keys)))
    # print(atom_resid_keys)

    prev = atom_resid_keys[0]
    resid_keys = [prev]
    prev_id = 0
    atom_ids = [0]
    for elt in atom_resid_keys[1:]:
        if elt == prev:
            atom_ids.append(prev_id)
        else:
            prev = elt
            prev_id += 1
            atom_ids.append(prev_id)
            resid_keys.append(elt)
    # print(atom_ids - atoms_ids_perresidue)
    # print((atom_coords - coords).sum())
    # print(set(resid_keys)- set(resid_keys))
    # print(set(resid_keys)- set(resid_keys))

    clfr = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(pts_coords)
    distances, indices = clfr.kneighbors(coords)

    # indices is for each atom the 3 nearest neighbors in the pts. shape (n_atom, 3)
    # Then extract relevant values, filtered by distance thresholding
    # Finally, aggregate the atomic values per resid.
    neigh_values = pro[indices]
    filter_values = neigh_values * (distances < dt)
    atom_values = filter_values.max(axis=1)
    res_values = [atom_values[atom_ids == i].max() for i in np.unique(atom_ids)]

    # prob = [0] * len(all_residue_centroids)
    # for ii, ind in enumerate(indices):
    #     for jj, sind in enumerate(ind):
    #         if distances[ii][jj] > dt:
    #             continue
    #         prob[atoms_ids_perresidue[ii]] = max(prob[atoms_ids_perresidue[ii]], pro[sind])
    # print(res_values-prob)
    probs = dict(zip(resid_keys, res_values))
    return probs


def do_dbd5(indirs=glob.glob('/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5/????'), overwrite=False):
    for indir in tqdm(indirs):
        # Aggregate supervision
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate honest prediction
        for infile in ['receptor_b', 'ligand_b', 'receptor_u', 'ligand_u']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}_prob.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction
        for infile in ['receptor_b', 'ligand_b', 'receptor_u', 'ligand_u']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}_prob_double.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))

        ########################## DROPBOX ##########################
        # Aggregate supervision with dropbox
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_dropbox_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate honest prediction with dropbox
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_prob_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_dropbox_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction with dropbox
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts',
                      f'{indir}/{infile}_prob_double_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_dropbox_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))

        ########################## Final  ##########################
        # Aggregate honest prediction with final
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_prob_final.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_final_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction with final
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts',
                      f'{indir}/{infile}_prob_double_final.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_final_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))


def do_epipred(indirs=glob.glob('/c7/scratch2/vmallet/indeep_data/epipred/????'), overwrite=False):
    for indir in tqdm(indirs):
        # Aggregate supervision
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate honest prediction
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}_prob.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}_prob_double.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))

        ########################## DROPBOX ##########################
        # Aggregate supervision with dropbox
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_dropbox_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate honest prediction with dropbox
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_prob_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_dropbox_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction with dropbox
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts',
                      f'{indir}/{infile}_prob_double_dropbox.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_dropbox_prob_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))

        ########################## Final ##########################
        # Aggregate honest prediction with final
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts', f'{indir}/{infile}_prob_final.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_prob_final_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))
        # Aggregate native prediction with final
        for infile in ['receptor', 'ligand']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}_dropbox.pts',
                      f'{indir}/{infile}_prob_double_final.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                outfile = f'{indir}/{infile}_final_prob_double_patch.p'
                if overwrite or not os.path.exists(outfile):
                    probs = get_resid_seg(*inputs)
                    pickle.dump(probs, open(outfile, 'wb'))


if __name__ == '__main__':
    pass
    # pdb = "../../DeepInterface/benchmark/misc/1A2K/receptor_b.pdb"
    # pts = "../../DeepInterface/benchmark/misc/1A2K/receptor_b.pts"
    # seg = "../../DeepInterface/benchmark/misc/1A2K/receptor_b.seg"
    # probs = get_resid_seg(pdb, ptsfile=pts, segfile=seg)
    # print(len(probs))
    # print(set(probs.values()))
    # pickle.dump(probs, open('mine.p', 'wb'))
    #
    # pts = "../../DeepInterface/benchmark/misc/1A2K/dropbox_1A2K-r.pts"
    # seg = "../../DeepInterface/benchmark/misc/1A2K/dropbox_1A2K-r.seg"
    # probs_b = get_resid_seg(pdb, ptsfile=pts, segfile=seg)
    # print(set(probs_b.values()))
    # print(len(probs_b))
    # pickle.dump(probs_b, open('theirs.p', 'wb'))
    #
    # for key, value in probs.items():
    #     if(value != probs_b[key]):
    #         print(value, probs_b[key])
    # print(probs==probs_b)

    indirs = glob.glob('/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5/????')
    # indirs = glob.glob('/home/vmallet/projects/DeepInterface/data/dbd5/????')
    do_dbd5(indirs=indirs, overwrite=False)
    do_epipred(overwrite=False)
