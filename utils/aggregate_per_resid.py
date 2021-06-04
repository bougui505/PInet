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


import numpy as np
from getcontactEpipred import getsppider2
from sklearn.neighbors import NearestNeighbors


def get_resid_seg(pdbfile, ptsfile, segfile, RL='r'):
    """
    ptsfile: .pts file
    segfile: .seg file
    RL: r or l for receptor or ligand
    """
    coord = np.transpose(np.loadtxt(ptsfile))[0:3,:]

    prolabel = np.loadtxt(segfile)
    if RL == 'r':
        pro = prolabel[0:coord.shape[1]]
    else:  # 'l'
        pro = prolabel[-coord.shape[1]:]

    coord = np.transpose(coord)
    nn = 3
    dt = 2
    cutoff = 0.5
    tol = [6, 6, 6]

    r, n, c = getsppider2(pdbfile)

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
    return prob
    

if __name__ == '__main__':
    INDIR = '/c7/scratch2/bougui/dbd5/benchmark5.5/dbd5/1A2K'
    prob = get_resid_seg(f'{INDIR}/receptor_b.pdb', f'{INDIR}/receptor_b.pts',
                         f'{INDIR}/receptor_b.seg', RL='r')
    print(prob.shape)
    print(prob)
