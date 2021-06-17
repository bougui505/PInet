import os
import shutil
from tqdm import tqdm


def move_dbd5():
    # move ligands
    dbd5_db = '/c7/home/vmallet/projects/dl_atomic_density_hd/data/dbd5'
    dropbox_lf_points = '/c7/home/vmallet/projects/dl_atomic_density_hd/data/dbd5_dropbox/lf/points'
    for pts in tqdm(os.listdir(dropbox_lf_points)):
        pdb, _ = pts.split('-')
        old_file_loc = os.path.join(dropbox_lf_points, pts)
        new_file_loc = os.path.join(dbd5_db, pdb, 'ligand_b_dropbox.pts')
        shutil.move(old_file_loc, new_file_loc)

    dropbox_rf_points = '/c7/home/vmallet/projects/dl_atomic_density_hd/data/dbd5_dropbox/rf/points'
    for pts in tqdm(os.listdir(dropbox_rf_points)):
        pdb, _ = pts.split('-')
        old_file_loc = os.path.join(dropbox_rf_points, pts)
        new_file_loc = os.path.join(dbd5_db, pdb, 'receptor_b_dropbox.pts')
        shutil.move(old_file_loc, new_file_loc)

    dropbox_lf_seg = '/c7/home/vmallet/projects/dl_atomic_density_hd/data/dbd5_dropbox/lf/points_label'
    for seg in tqdm(os.listdir(dropbox_lf_seg)):
        pdb, _ = seg.split('-')
        old_file_loc = os.path.join(dropbox_lf_seg, seg)
        new_file_loc = os.path.join(dbd5_db, pdb, 'ligand_b_dropbox.seg')
        shutil.move(old_file_loc, new_file_loc)

    dropbox_rf_seg = '/c7/home/vmallet/projects/dl_atomic_density_hd/data/dbd5_dropbox/rf/points_label'
    for seg in tqdm(os.listdir(dropbox_rf_seg)):
        pdb, _ = seg.split('-')
        old_file_loc = os.path.join(dropbox_rf_seg, seg)
        new_file_loc = os.path.join(dbd5_db, pdb, 'receptor_b_dropbox.seg')
        shutil.move(old_file_loc, new_file_loc)


if __name__ == '__main__':
    pass
    move_dbd5()
