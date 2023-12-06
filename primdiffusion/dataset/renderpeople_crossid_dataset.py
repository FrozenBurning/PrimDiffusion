import os
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2
import json
import random

import logging

logger = logging.getLogger(f"primdiffusion.dataset.{__name__}")

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

def lerp(a, b, t):
    return a + (b - a) * t

def interpolate_matrices(rot1, rot2, trans1, trans2, t):
    # SLERP for rotations
    slerp = Slerp([0, 1], R.from_matrix([rot1, rot2]))
    slerped_rot = slerp([t])

    # LERP for translations
    lerped_trans = lerp(trans1, trans2, t)

    return slerped_rot.as_matrix()[0], lerped_trans

def interpolate_poses(poses, num_interpolations):
    interpolated_poses = []

    for i in range(len(poses)):
        rot1, trans1 = poses[i]
        rot2, trans2 = poses[(i + 1) % len(poses)]

        for t in np.linspace(0, 1, num_interpolations + 1, endpoint=False):
            slerped_rot, lerped_trans = interpolate_matrices(rot1, rot2, trans1, trans2, t)
            interpolated_poses.append((slerped_rot, lerped_trans))

    return interpolated_poses

def get_KRTD(camera, view_index = 0):
    camera = camera['camera{:04d}'.format(view_index)]
    K = np.array(camera['K'])
    R = np.array(camera['R'])
    R_flip = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R = R @ R_flip
    T = np.array(camera['T'])
    D = None
    return K, R, T, D

def load_smpl_params(all_smpl, frame_id):
    # poses,
    # expression,
    # shapes,
    # Rh,
    # Th,
    return {
        k: np.array(v[frame_id], dtype=np.float32) for k, v in all_smpl.items() if k != "id"
    }

class RenderPeopleSViewDataset(Dataset):
    def __init__(
        self,
        root_dir,
        subject_ids,
        smpl_poses,
        image,
        image_mask,
        image_part_mask,
        cam_path,
        frame_list=None,
        cameras=None,
        cond_cameras=None,
        sample_cameras=True,
        camera_id=None,
        image_height=1024,
        image_width=1024,
        is_train=True,
        **kwargs,
    ):
        super().__init__()
        # subject ids is a text file contains list of subject ids
        self.image_height = image_height
        self.image_width = image_width
        self.ref_frame = 0

        with open(subject_ids, 'r') as f:
            human_list = f.read().splitlines()
        self.subject_ids = human_list
        self.root_dir = root_dir

        if frame_list is None:
            n_frames = len(os.listdir(os.path.join(self.root_dir, self.subject_ids[0], 'img', 'camera0000')))
            self.frame_list = [str(fid) for fid in range(n_frames)]

        self.image_path = image
        self.image_mask_path = image_mask
        self.image_part_mask_path = image_part_mask

        self.is_train = is_train
        all_cameras = self.load_all_cameras(cam_path)

        # TODO: inference logics
        if not self.is_train:
            assert not sample_cameras
            assert camera_id is not None

        self.cameras = all_cameras

        self.cond_cameras = cond_cameras

        self.sample_cameras = sample_cameras
        self.camera_id = camera_id

        self.all_smpl = self.load_all_smpl(smpl_poses)

    def load_all_smpl(self, smpl_poses):
        all_smpl = {}
        for people_id in self.subject_ids:
            current_smpl_path = smpl_poses.format(people_id=people_id)
            smpl_param = dict(np.load(current_smpl_path, allow_pickle=True))['smpl'].item()
            poses = np.zeros((smpl_param['body_pose'].shape[0], 72)).astype(np.float32)
            poses[:, :3] = np.array(smpl_param['global_orient']).astype(np.float32)
            poses[:, 3:] = np.array(smpl_param['body_pose']).astype(np.float32)

            shapes = np.array(smpl_param['betas']).astype(np.float32)
            shapes = np.repeat(shapes[:], poses.shape[0], axis=0)
            Rh = smpl_param['global_orient'].astype(np.float32)
            Th = smpl_param['transl'].astype(np.float32)
            current_smpl = {
                'shapes': shapes,
                'Rh': Rh * 0, #FIXME: hack
                'Th': Th,
                'poses': poses,
            }
            all_smpl[people_id] = current_smpl

        return all_smpl

    def load_all_cameras(self, camera_path):
        # input path to camera.json under synbody sequence
        # all_cameras is dict of dict
        all_cameras = {}
        for people_id in self.subject_ids:
            current_camera_path = camera_path.format(people_id=people_id)
            current_camera = {}
            with open(current_camera_path) as f:
                camera = json.load(f)
            for view_index in range(len(camera.keys())):
                K, R, T, _ = get_KRTD(camera, view_index)
                current_camera['camera{:04d}'.format(view_index)] = {
                    "Rt": np.concatenate([R, T[..., None]], axis=1).astype(np.float32),
                    "K": K.astype(np.float32),
                }
            for c in current_camera.values():
                c["cam_pos"] = -np.dot(c["Rt"][:3, :3].T, c["Rt"][:3, 3])
                c["Rt"][:, -1] *= 1000.0
            all_cameras[people_id] = current_camera
        return all_cameras

    def __len__(self):
        return len(self.subject_ids) * 200

    def __getitem__(self, idx):
        # idx is subject_id wise index
        people_id = self.subject_ids[idx % len(self.subject_ids)]

        # random sample frames
        frame = (
            random.choice(self.frame_list)
        )

        # random sample cameras
        camera_id = (
            random.choice(list(self.cameras[people_id].keys()))
            if self.sample_cameras
            else self.camera_id
        )
        fmts = dict(people_id=people_id, frame=int(frame), camera=camera_id)

        sample = {"index": idx, **fmts}

        sample.update(load_smpl_params(self.all_smpl[people_id], int(frame)))

        ref_frame_smpl = {'ref_' + k: v for k, v in load_smpl_params(self.all_smpl[people_id], int(self.ref_frame)).items()}
        sample.update(ref_frame_smpl)

        sample["image"] = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1].astype(np.float32),
            axes=(2, 0, 1),
        )

        # reading all the cond images
        if self.cond_cameras:
            sample["cond_image"] = []
            sample["cond_Rt"] = []
            sample["cond_K"] = []
            # for cond_camera_id in self.cond_cameras:
            # FIXME: hack for random condition views
            cond_camera_id = random.choice(list(self.cameras[people_id].keys()))
            if True:
                cond_image = np.transpose(
                    cv2.imread(
                        self.image_path.format(
                            people_id=people_id, frame=int(self.ref_frame), camera=cond_camera_id
                        )
                    )[..., ::-1].astype(np.float32),
                    axes=(2, 0, 1),
                )
                sample["cond_image"].append(cond_image)
                sample["cond_Rt"].append(self.cameras[people_id][cond_camera_id]["Rt"])
                sample["cond_K"].append(self.cameras[people_id][cond_camera_id]["K"])

            for key in ["image", "K", "Rt"]:
                sample[f"cond_{key}"] = np.stack(sample[f"cond_{key}"], axis=0)

            sample["cond_cameras"] = self.cond_cameras[:]

        sample["image"] = np.transpose(
            cv2.imread(self.image_path.format(**fmts))[..., ::-1].astype(np.float32),
            axes=(2, 0, 1),
        )

        image_mask = cv2.imread(self.image_mask_path.format(**fmts))
        border = 3
        kernel = np.ones((border, border), np.uint8)
        msk_erode = cv2.erode(image_mask.copy(), kernel)[np.newaxis, ..., 0]
        sample["image_mask"] = (msk_erode != 0).astype(np.float32)

        image_part_mask = cv2.imread(self.image_part_mask_path.format(**fmts))
        part_msk_erode = cv2.erode(image_part_mask.copy(), kernel)[np.newaxis, ..., 0]
        sample["image_part_mask"] = part_msk_erode

        sample["image_bg"] = sample["image"] * ~(sample["image_part_mask"] != 0)

        sample.update(self.cameras[people_id][camera_id])

        return sample
    
    def gen_inf_cameras(self, num_views = 5):
        training_views = self.cameras[self.subject_ids[0]]
        self.training_views = training_views
        num_training_views = len(training_views.keys())
        interpolation_anchors = []
        for view_index in range(num_training_views):
            Rt = training_views['camera{:04d}'.format(view_index)]['Rt']
            K = training_views['camera{:04d}'.format(view_index)]['K']
            rot = Rt[:, :3]
            trans = Rt[:, 3]
            interpolation_anchors.append((rot, trans))
        interpolated_poses = interpolate_poses(interpolation_anchors, num_views)

        inf_camera = {}
        for people_id in self.subject_ids:
            current_camera = {}
            for view_index in range(len(interpolated_poses)):
                R, T = interpolated_poses[view_index]
                current_camera['camera{:04d}'.format(view_index)] = {
                    "Rt": np.concatenate([R, T[..., None]], axis=1).astype(np.float32),
                    "K": K.astype(np.float32),
                }
            for c in current_camera.values():
                c["cam_pos"] = -np.dot(c["Rt"][:3, :3].T, c["Rt"][:3, 3])
                # c["Rt"][:, -1] *= 1000.0
            inf_camera[people_id] = current_camera
        self.inf_cameras = inf_camera


    def inf_sample(self, people_id, camera_id, frame_id, cond_sample):
        fmts = dict(people_id=people_id, frame=int(frame_id), camera=camera_id)
        sample = {}
        sample.update({**fmts})

        sample.update(load_smpl_params(self.all_smpl[people_id], int(frame_id)))

        sample.update(self.inf_cameras[people_id][camera_id])

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]

        sample.update(cond_sample)
        return sample

    def cond_sample(self, people_id):
        sample = {}
        # reading all the cond images
        if self.cond_cameras:
            sample["cond_image"] = []
            sample["cond_Rt"] = []
            sample["cond_K"] = []
            cond_camera_id = random.choice(list(self.cameras[people_id].keys()))
            if True:
                cond_image = np.transpose(
                    cv2.imread(
                        self.image_path.format(
                            people_id=people_id, frame=int(self.ref_frame), camera=cond_camera_id
                        )
                    )[..., ::-1].astype(np.float32),
                    axes=(2, 0, 1),
                )
                sample["cond_image"].append(cond_image)
                sample["cond_Rt"].append(self.cameras[people_id][cond_camera_id]["Rt"])
                sample["cond_K"].append(self.cameras[people_id][cond_camera_id]["K"])

            for key in ["image", "K", "Rt"]:
                sample[f"cond_{key}"] = np.stack(sample[f"cond_{key}"], axis=0)

            sample["cond_cameras"] = self.cond_cameras[:]
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]
        return sample
    

    def inf_sample_wsmpl(self, people_id, camera_id, frame_id, cond_sample, smpl_param):
        fmts = dict(people_id=people_id, frame=int(frame_id), camera=camera_id)
        sample = {}
        sample.update({**fmts})

        sample.update(load_smpl_params(smpl_param, int(frame_id)))

        sample.update(self.inf_cameras[people_id][camera_id])

        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]

        sample.update(cond_sample)
        return sample

    def sample_cam_smpl(self):
        people_id = random.choice(self.subject_ids)
        frame_id = random.choice(self.frame_list)
        camera_id = random.choice(list(self.cameras[people_id].keys()))
        fmts = dict(people_id=people_id, frame=int(frame_id), camera=camera_id)
        sample = {}
        sample.update({**fmts})
        sample.update(load_smpl_params(self.all_smpl[people_id], int(frame_id)))
        sample.update(self.cameras[people_id][camera_id])
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = v[None, ...]
        return sample