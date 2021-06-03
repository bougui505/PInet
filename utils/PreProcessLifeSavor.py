import pymol
from pymol import cmd, stored
import os
import sys
import re
import numpy as np
from dx2feature import *
from getResiLabel import *
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

if len(sys.argv) == 4 and sys.argv[3] == 'train':
    train_flag = 1


def pdb_to_wrl(pdbfile, dump_name=None):
    """
    Uses pymol to go from a pdb file to a wrl file (points on the surface format)
    """
    cmd.reinitialize()
    if dump_name is None:
        dump_name = pdbfile[0:-4] + '.wrl'
    cmd.load(pdbfile)
    cmd.set('surface_quality', '0')
    cmd.show_as('surface', 'all')
    cmd.set_view('1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,300,1')
    cmd.save(dump_name)
    cmd.delete('all')


# wrl to pts, normholder does not seem to be used...
def wrl_to_coords(wrlfile):
    """
    Parse a wrl file to produce coordinates.
    """
    holder = []
    normholder = []
    cf = 0
    nf = 0
    with open(wrlfile, "r") as vrml:
        for lines in vrml:
            if 'point [' in lines:
                cf = 1
            if cf == 1:
                a = re.findall("[-0-9]{1,3}.[0-9]{6}", lines)
                if len(a) == 3:
                    holder.append(tuple(map(float, a)))
            if 'vector [' in lines:
                nf = 1
            if nf == 1:
                a = re.findall("[-0-9]{1}.[0-9]{4}", lines)
                if len(a) == 3:
                    normholder.append(tuple(map(float, a)))
    coords = np.array(holder)
    coords = np.unique(coords, axis=0)
    return coords


def add_binary_contact(coord_r, coord_l, dump_r, dump_l):
    """
    Based on the pairwise distances, with a thresholding of 2A (tol), annotate two coordset with
    a binary annotation that denotes whether the coords are in contact.
    """
    tol = np.array([2, 2, 2])

    contact = (np.abs(np.asarray(coord_l[:, None]) - np.asarray(coord_r)) < tol).all(2).astype(np.int)

    label_l = np.max(contact, axis=1)
    label_r = np.max(contact, axis=0)

    np.savetxt(fname=dump_r, X=label_r)
    np.savetxt(fname=dump_l, X=label_l)


def add_apbs(basename_pdb):
    """
    This dumps a bunch of files that are used to annotate the pts files produced above.
    """
    pqr_file = basename_pdb + '.pqr'
    apbsin_file = basename_pdb + '.in'
    log_file = basename_pdb + '.log'
    cmd_pdb2pqr = f'pdb2pqr30 --whitespace --ff=AMBER --apbs-input {apbsin_file} {basename_pdb}.pdb {pqr_file}'
    try:
        os.system(cmd_pdb2pqr)
    except:
        print('error for: ' + basename_pdb)

    cmd_apbs = f'apbs {apbsin_file}'
    try:
        os.system(cmd_apbs)
    except:
        print('error for abps: ' + basename_pdb)

    return pqr_file, apbsin_file, log_file


def expand_coords(coords, pdbfile, pqrdxfile, outfile):
    """
    Go from (n,3) to (n,5) matrix using the apbs output.
    """
    # I don't know what happens here, especially the hlabel * 10.
    # I think they use the CA annotation to annotate all atoms.
    centroid, labels = gethydro(pdbfile)
    centroid = np.array(centroid)
    hlabel = np.transpose(np.asarray(labels[0]))
    clf = neighbors.KNeighborsClassifier(3)
    clf.fit(centroid, hlabel * 10)
    distl, indl = clf.kneighbors(coords)
    pred = np.sum(hlabel[indl] * distl, 1) / np.sum(distl, 1)

    # Parse apbs output
    apbsl = open(pqrdxfile, 'r')
    gl, orl, dl, vl = parsefile(apbsl)
    avl = findvalue(coords, gl, orl, dl, vl)

    np.savetxt(outfile,
               np.concatenate((coords, np.expand_dims(avl, 1), np.expand_dims(pred, 1)), axis=1))


def process_one_pdb(pdbfile_r):
    """
    In case we have no ligand, we dont dump a supervision, just a pts file with hydrostatic and geometric features.
    :return : a 'failed' boolean
    """
    basename_r = pdbfile_r[0:-4]
    wrl_r = basename_r + '.wrl'
    pdb_to_wrl(pdbfile_r, dump_name=wrl_r)
    coords_r = wrl_to_coords(wrlfile=wrl_r)
    os.remove(wrl_r)

    # use apbs to get .dx files
    pqr_file, apbsin_file, log_file = add_apbs(basename_r)

    pqrdx_r_file = basename_r + '.pqr.dx'
    outfile_r = basename_r + '.pts'
    if os.path.exists(pqrdx_r_file):
        expand_coords(pdbfile=pdbfile_r, coords=coords_r, pqrdxfile=pqrdx_r_file, outfile=outfile_r)
        os.remove(pqr_file)
        os.remove(pqr_file)
        os.remove(pqrdx_r_file)
        os.remove(log_file)
        return 0
    else:
        return 1


def process_pdbs(pdbfile_r, pdbfile_l):
    """
    In case we have a ligand, we also dump a supervision
    """
    basename_r = pdbfile_r[0:-4]
    basename_l = pdbfile_l[0:-4]
    wrl_r = basename_r + '.wrl'
    wrl_l = basename_l + '.wrl'
    pdb_to_wrl(pdbfile_r, dump_name=wrl_r)
    pdb_to_wrl(pdbfile_l, dump_name=wrl_l)
    coords_r = wrl_to_coords(wrlfile=wrl_r)
    coords_l = wrl_to_coords(wrlfile=wrl_l)
    os.remove(wrl_r)
    os.remove(wrl_l)

    dump_r = basename_r + '.seg'
    dump_l = basename_l + '.seg'
    add_binary_contact(coord_r=coords_r, coord_l=coords_l,
                       dump_r=dump_r, dump_l=dump_l)

    # use apbs to get .dx files and then turn it into .pts files
    pqr_file_r, apbsin_file_r, log_file_r = add_apbs(basename_r)
    failed = False
    pqrdx_file_r = basename_r + '.pqr.dx'
    outfile_r = basename_r + '.pts'
    if os.path.exists(pqrdx_file_r):
        expand_coords(pdbfile=pdbfile_r, coords=coords_r, pqrdxfile=pqrdx_file_r, outfile=outfile_r)
        os.remove(pqr_file_r)
        os.remove(pqr_file_r)
        os.remove(pqrdx_file_r)
        os.remove(log_file_r)
    else:
        failed = True

    pqr_file_l, apbsin_file_l, log_file_l = add_apbs(basename_l)
    pqrdx_file_l = basename_l + '.pqr.dx'
    outfile_l = basename_l + '.pts'
    if os.path.exists(pqrdx_file_l):
        expand_coords(pdbfile=pdbfile_l, coords=coords_l, pqrdxfile=pqrdx_file_l, outfile=outfile_l)
        os.remove(pqr_file_l)
        os.remove(pqr_file_l)
        os.remove(pqrdx_file_l)
        os.remove(log_file_l)
    else:
        failed = True
    return failed


def process_all(dataset, pdbname_r='receptor.pdb', pdbname_l=None):
    total_failed = 0
    for system in os.listdir(dataset):
        pdb_path_r = os.path.join(dataset, system, pdbname_r)
        if pdbname_l is not None:
            pdb_path_l = os.path.join(dataset, system, pdbname_l)
            failed = process_pdbs(pdbfile_r=pdb_path_r, pdbfile_l=pdb_path_l)
        else:
            failed = process_one_pdb(pdb_path_r)
        if failed:
            print('We failed for system : ', system)
        total_failed += failed
    print(f'We failed on {total_failed} on a total of {len(os.listdir(dataset))} systems.')


if __name__ == '__main__':
    # pdb to wrl

    # pdbfile_l = sys.argv[1]
    # pdbfile_r = sys.argv[2]
    # wrl_l = pdbfile_l[0:-4] + '.wrl'
    # wrl_r = pdbfile_r[0:-4] + '.wrl'
    #
    # pdb_to_wrl(pdbfile_l, dump_name=wrl_l)
    # pdb_to_wrl(pdbfile_r, dump_name=wrl_r)
    # lcoords = wrl_to_coords(wrlfile=wrl_l)
    # rcoords = wrl_to_coords(wrlfile=wrl_r)
    #
    # add_binary_contact(coord_l=lcoords, coord_r=rcoords, dump_l=pdbfile_l[0:4] + '-l.seg',
    #                    dump_r=pdbfile_r[0:4] + '-r.seg')
    #
    # # use apbs to get .dx files
    # basename_l = pdbfile_l[0:4] + '-l'
    # basename_r = pdbfile_r[0:4] + '-r'
    # add_apbs(basename_l)
    # add_apbs(basename_r)

    # pdb_r = '../data/2I25/2I25-r.pdb'
    # pdb_l = '../data/2I25/2I25-l.pdb'
    # process_pdbs(pdbfile_r=pdb_r, pdbfile_l=pdb_l)

    # For Epipred
    dataset = '../../dl_atomic_density/data/epipred/'
    pdbname_r = 'receptor.pdb'
    pdbname_l = 'ligand.pdb'
    process_all(dataset=dataset, pdbname_r=pdbname_r, pdbname_l=pdbname_l)

    # For dbd5
    dataset = '../../dl_atomic_density/data/dbd5/'
    pdbname_r_b = 'receptor_b.pdb'
    pdbname_r_u = 'receptor_u.pdb'
    pdbname_l = 'ligand.pdb'
    process_all(dataset=dataset, pdbname_r=pdbname_r_b, pdbname_l=pdbname_l)
    process_all(dataset=dataset, pdbname_r=pdbname_r_u)
