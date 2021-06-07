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
import numpy as np
from getcontactEpipred import getsppider2
from sklearn.neighbors import NearestNeighbors
import os
from Bio.PDB import *


def readpdb(file):
    parser = PDBParser()
    # structure = parser.get_structure('C', '3ogo-bg.pdb')
    structure = parser.get_structure('C', file)
    residic = []
    # tempo=[]
    newdic =[]
    labeldic= []
    bdic = []
    cd=[]
    mark=0
    resid_keys = []
    for c in structure[0]:
        for resi in c:
            _, _, chain, (_, resid, _) = resi.get_full_id()
            resid_keys.append((chain, resid))
            # residic.append(resi._id[1])
            cen = [0, 0, 0]
            count = 0
            for atom in resi:
                # print atom.get_coord()
                # print list(atom.get_vector())
                # if 'H' in atom.get_name():
                #     continue
                cen[0] += atom.get_coord()[0]
                cen[1] += atom.get_coord()[1]
                cen[2] += atom.get_coord()[2]
                count += 1

                # residic.append(resi._id[1])
                residic.append(mark)
                newdic.append([atom.get_coord()[0],atom.get_coord()[1],atom.get_coord()[2]])
            cen = [coor * 1.0 / count for coor in cen]
            mark+=1
            cd.append(cen)
            # labeldic.append(1)

    # print len(residic)
    # print len(bdic)
    return residic, np.asarray(newdic),cd, resid_keys


def get_resid_seg(pdbfile, ptsfile, segfile):
    """
    ptsfile: .pts file
    segfile: .seg file
    RL: r or l for receptor or ligand
    """
    coord = np.transpose(np.loadtxt(ptsfile))[0:3,:]

    pro = np.loadtxt(segfile)

    coord = np.transpose(coord)
    nn = 3
    dt = 2
    cutoff = 0.5
    tol = [6, 6, 6]

    r, n, c, resid_keys = readpdb(pdbfile)

    cencoord = np.asarray(n)

    clfr = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(coord)
    distances, indices = clfr.kneighbors(cencoord)

    prob = [0] * len(c)
    for ii,ind in enumerate(indices):
        for jj,sind in enumerate(ind):
            if distances[ii][jj]>dt:
                continue
            prob[r[ii]] = max(prob[r[ii]], pro[sind])
    prob = np.asarray(prob)
    probs = dict(zip(resid_keys, prob))
    return probs
    

if __name__ == '__main__':
    # INDIR = '/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5/1A2K'
    # indirs = os.listdir('/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5')
    indirs = glob.glob('/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5/????')
    for indir in indirs:
        for infile in ['receptor_b', 'ligand_b']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                probs = get_resid_seg(*inputs)
                pickle.dump(probs, open(f'{indir}/{infile}_patch.p', 'wb'))
        for infile in ['receptor_b', 'ligand_b', 'receptor_u', 'ligand_u']:
            inputs = [f'{indir}/{infile}.pdb', f'{indir}/{infile}.pts', f'{indir}/{infile}_prob.seg']
            if np.all([os.path.exists(e) for e in inputs]):
                probs = get_resid_seg(*inputs)
                pickle.dump(probs, open(f'{indir}/{infile}_prob_patch.p', 'wb'))
