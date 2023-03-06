import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np
import torch
import shutil
import torch.nn as nn
import sys

sys.path.append('../../')
import script.common as cm
from ThreeDMatch.Test.tools import get_pcd, get_keypts, read_log_file, loadlog
from sklearn.neighbors import KDTree
import importlib
import open3d

def noise_Gaussian_proportion(points, std, proportion):
    noise = np.random.normal(0, std, points.shape)
    # noise[noise>0.05]=0.05
    # noise[noise < -0.05] = -0.05
    select=np.random.rand(points.shape[0])
    out = points
    out[select>proportion]=points[select>proportion]+noise[select>proportion]
    return out

def noise_Gaussian(points, std):  #0.02
    noise = np.random.normal(0, std, points.shape)
    noise[noise>0.05]=0.05
    noise[noise < -0.05] = -0.05
    out = points + noise
    return out

def noise_ramdom(points):
    noise = np.random.rand(points.shape[0],points.shape[1])*0.1-0.05
    out = points + noise
    return out


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def build_patch_input(pcd, keypts, vicinity=0.3, num_points_per_patch=2048):

    pts= np.array(pcd.points).astype(np.float32)
    refer_pts = np.array(keypts).astype(np.float32)
    num_patches = refer_pts.shape[0]
    tree = KDTree(pts[:, 0:3])
    ind_local = tree.query_radius(refer_pts[:, 0:3], r=vicinity)
    local_patches = np.zeros([num_patches, num_points_per_patch, 3], dtype=float)
    for i in range(num_patches):
        local_neighbors = pts[ind_local[i], :]
        if local_neighbors.shape[0] >= num_points_per_patch:
            temp = np.random.choice(range(local_neighbors.shape[0]), num_points_per_patch, replace=False)
            local_neighbors = local_neighbors[temp]
            local_neighbors[-1, :] = refer_pts[i, :]
        else:
            fix_idx = np.asarray(range(local_neighbors.shape[0]))
            while local_neighbors.shape[0] + fix_idx.shape[0] < num_points_per_patch:
                fix_idx = np.concatenate((fix_idx, np.asarray(range(local_neighbors.shape[0]))), axis=0)
            random_idx = np.random.choice(local_neighbors.shape[0], num_points_per_patch - fix_idx.shape[0],
                                          replace=False)
            choice_idx = np.concatenate((fix_idx, random_idx), axis=0)
            local_neighbors = local_neighbors[choice_idx]
            local_neighbors[-1, :] = refer_pts[i, :]
        local_patches[i] = local_neighbors

    return local_patches

def prepare_patch(pcdpath, filename, keypts, trans_matrix):

    pcd = get_pcd(pcdpath, filename)
    # load D3Feat keypts

    if is_rotate_dataset:
        # Add arbitrary rotation
        # rotate terminal frament with an arbitrary angle around the z-axis
        angles_3d = np.random.rand(3) * np.pi * 2
        R = cm.angles2rotation_matrix(angles_3d)
        T = np.identity(4)
        T[:3, :3] = R
        pcd.transform(T)
        keypts_pcd = make_open3d_point_cloud(keypts)
        keypts_pcd.transform(T)
        keypts = np.array(keypts_pcd.points)
        trans_matrix.append(T)

    local_patches = build_patch_input(pcd, keypts)  # [num_keypts, 1024, 4]
    return local_patches


def generate_descriptor(model, desc_name, pcdpath, keypts, descpath, test_pairs):
    model.eval()
    with torch.no_grad():
        trans_matrix=[]
        for idx,i in enumerate(test_pairs):
            start_time = time.time()
            j=i['test_pair'][0]
            m = i['test_pair'][1]
            keypoints_src = np.load(os.path.join(keypts, f'tgt_{idx}.npy'))
            keypoints_tgt = np.load(os.path.join(keypts, f'src_{idx}.npy'))
            keypt=[keypoints_src,keypoints_tgt]
            local_patches = prepare_patch(pcdpath, 'cloud_bin_' + str(j), keypt[0], trans_matrix)
            input_ = torch.tensor(local_patches.astype(np.float32))
            B = input_.shape[0]
            input_ = input_.cuda()
            model = model.cuda()
            # calculate descriptors
            desc_list = []
            desc_len = 64
            step_size = 4
            iter_num = np.int32(np.ceil(B / step_size))
            for k in range(iter_num):
                if k == iter_num - 1:
                    desc = model(input_[k * step_size:, :, :])
                else:
                    desc = model(input_[k * step_size: (k + 1) * step_size, :, :])
                desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
                del desc
            desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
            np.save(descpath + 'cloud_bin_' + str(j)+'_'+str(m)+'_0' + f".desc.{desc_name}.bin", desc.astype(np.float32))

            local_patches = prepare_patch(pcdpath, 'cloud_bin_' + str(m), keypt[1], trans_matrix)
            input_ = torch.tensor(local_patches.astype(np.float32))
            B = input_.shape[0]
            input_ = input_.cuda()
            model = model.cuda()
            # calculate descriptors
            desc_list = []
            desc_len = 64
            step_size = 4
            iter_num = np.int32(np.ceil(B / step_size))
            for k in range(iter_num):
                if k == iter_num - 1:
                    desc = model(input_[k * step_size:, :, :])
                else:
                    desc = model(input_[k * step_size: (k + 1) * step_size, :, :])
                desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
                del desc
            desc = np.concatenate(desc_list, 0).reshape([B, desc_len])
            np.save(descpath + 'cloud_bin_' + str(j)+'_'+str(m)+'_1' + f".desc.{desc_name}.bin", desc.astype(np.float32))
            step_time = time.time() - start_time
            print(f'Finish {B} descriptors spend {step_time:.4f}s')

        if is_rotate_dataset:
            scene_name = pcdpath.split('/')[-2]
            all_trans_matrix[scene_name] = trans_matrix



if __name__ == '__main__':
    scene_list = [
        '7-scenes-redkitchen',
        'sun3d-home_at-home_at_scan1_2013_jan_1',
        'sun3d-home_md-home_md_scan9_2012_sep_30',
        'sun3d-hotel_uc-scan3',
        'sun3d-hotel_umd-maryland_hotel1',
        'sun3d-hotel_umd-maryland_hotel3',
        'sun3d-mit_76_studyroom-76-1studyroom2',
        'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    num_keypoints = int(sys.argv[1])
    experiment_id = time.strftime('%m%d%H%M')
    model_str = experiment_id  # sys.argv[1]
    if not os.path.exists(f"SphereNet_desc_{model_str}/"):
        os.mkdir(f"SphereNet_desc_{model_str}")

    # dynamically load the model
    module_file_path = '../model.py'
    shutil.copy2(os.path.join('.', '../../network/SphereNet.py'), module_file_path)
    module_name = ''
    module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.SphereNet(0.3, 15, 40, 20, '3DMatch')
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load('../../pretrain/3DMatch_best.pkl'))

    all_trans_matrix = {}
    is_rotate_dataset = False

    for scene in scene_list:
        pcdpath = f"../../data/3DMatch/fragments/{scene}/"
        interpath = f"../../data/3DMatch/intermediate-files-real/{scene}/"
        keyptspath = f"../../data/3DMatch/keypoints/3DLoMatch_{num_keypoints}_prob/{scene}/"
        gtpath = f'../../data/3DMatch/fragments/3DLoMatch/{scene}/'
        test_pairs=read_log_file(os.path.join(gtpath, f"gt.log"))
        descpath = os.path.join('.', f"SphereNet_desc_{model_str}/{scene}/")
        if not os.path.exists(descpath):
            os.makedirs(descpath)
        start_time = time.time()
        print(f"Begin Processing {scene}")
        generate_descriptor(model, desc_name='SphereNet', pcdpath=pcdpath, keypts=keyptspath, descpath=descpath, test_pairs=test_pairs)

        print(f"Finish in {time.time() - start_time}s")

    if is_rotate_dataset:
        np.save(f"trans_matrix", all_trans_matrix)
