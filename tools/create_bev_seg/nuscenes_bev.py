import argparse
import os
from os import path as osp
import h5py
import numpy as np
from pyquaternion import Quaternion
import cv2
import mmengine
from tqdm import tqdm
import multiprocessing as mp

from nuscenes.utils import splits
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.map_api import locations as LOCATIONS

map_classes = [
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line',
    'carpark_area', 'divider'
]
xbound = [-50.0, 50.0, 0.5]
ybound = [-50.0, 50.0, 0.5]

patch_h = ybound[1] - ybound[0]
patch_w = xbound[1] - xbound[0]

canvas_h = int(patch_h / ybound[2])
canvas_w = int(patch_w / xbound[2])

patch_size = (patch_h, patch_w)
canvas_size = (canvas_h, canvas_w)

def bev_seg_extractor(dataset_root, ego2global, lidar2ego,
                      location, out_dir, bev_file_name):
    lidar2global = ego2global @ lidar2ego
    map_pose = lidar2global[:2, 3]
    patch_box = (map_pose[0], map_pose[1], patch_size[0], patch_size[1])

    rotation = lidar2global[:3, :3]
    v = np.dot(rotation, np.array([1, 0, 0]))
    yaw = np.arctan2(v[1], v[0])
    patch_angle = yaw / np.pi * 180

    mappings = {}
    for name in map_classes:
        if name == "drivable_area*":
            mappings[name] = ["road_segment", "lane"]
        elif name == "divider":
            mappings[name] = ["road_divider", "lane_divider"]
        else:
            mappings[name] = [name]

    layer_names = []
    for name in mappings:
        layer_names.extend(mappings[name])
    layer_names = list(set(layer_names))
    
    maps = {}
    for location in LOCATIONS:
        maps[location] = NuScenesMap(dataset_root, location)

    masks = maps[location].get_map_mask(
        patch_box=patch_box,
        patch_angle=patch_angle,
        layer_names=layer_names,
        canvas_size=canvas_size,
    )
    masks = masks.transpose(0, 2, 1)
    masks = masks.astype(bool)

    num_classes = len(map_classes)
    labels = np.zeros((num_classes, *canvas_size), dtype=int)

    for k, name in enumerate(map_classes):
        for layer_name in mappings[name]:
            index = layer_names.index(layer_name)
            labels[k, masks[index]] = 1

    out_path = osp.join(out_dir, bev_file_name + '.h5')
    mmengine.mkdir_or_exist(osp.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('labels',
                          data=labels,
                          compression='gzip',
                          compression_opts=9)


def process_frame(frame_data):
    dataset_root, ego2global, lidar2ego, location, out_dir, bev_file_name = frame_data
    bev_seg_extractor(dataset_root, ego2global, lidar2ego,
                      location, out_dir, bev_file_name)


def main():
    parser = argparse.ArgumentParser(
        description="BEV segmentation creator arg parser")
    parser.add_argument("--data-path",
                        type=str,
                        default="./data/nuscenes",
                        help="Root path of dataset")
    parser.add_argument("--version",
                        type=str,
                        default="v1.0-trainval",
                        help="Dataset version")
    parser.add_argument("--pano-dir",
                        type=str,
                        default="./data/nuscenes",
                        required=False,
                        help="Directory name for panorama data")
    

    args = parser.parse_args()
    data_path = args.data_path
    pano_dir = args.pano_dir
    version = args.version
    test = 'test' in version

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])

    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print("train scene: {}, val scene: {}".format(len(train_scenes),
                                                      len(val_scenes)))

    output_dir = osp.join(pano_dir, 'bev')
    mmengine.mkdir_or_exist(output_dir)
    existed = set(os.listdir(output_dir))
    print(f"Already processed: {len(existed)}")

    frames_to_process = []
    for sample in mmengine.track_iter_progress(nusc.sample):
        bev_file_name = str(sample["timestamp"])
        if not test and (sample["scene_token"] in train_scenes
                         or sample["scene_token"] in val_scenes
                         ) and bev_file_name + '.h5' not in existed:
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", lidar_token)
            cs_record = nusc.get("calibrated_sensor",
                                 sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            location = nusc.get(
                "log",
                nusc.get("scene",
                         sample["scene_token"])["log_token"])["location"]

            lidar2ego_translation = cs_record["translation"]
            lidar2ego_rotation = cs_record["rotation"]
            ego2global_translation = pose_record["translation"]
            ego2global_rotation = pose_record["rotation"]

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(
                ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = ego2global_translation

            lidar2ego = np.eye(4).astype(np.float32)
            lidar2ego[:3, :3] = Quaternion(lidar2ego_rotation).rotation_matrix
            lidar2ego[:3, 3] = lidar2ego_translation

            frames_to_process.append(
                (nusc.dataroot, ego2global, lidar2ego, location,
                 output_dir, bev_file_name))

        elif test and sample[
                "scene_token"] in train_scenes and bev_file_name + '.h5' not in existed:
            lidar_token = sample["data"]["LIDAR_TOP"]
            sd_rec = nusc.get("sample_data", lidar_token)
            cs_record = nusc.get("calibrated_sensor",
                                 sd_rec["calibrated_sensor_token"])
            pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
            location = nusc.get(
                "log",
                nusc.get("scene",
                         sample["scene_token"])["log_token"])["location"]

            lidar2ego_translation = cs_record["translation"]
            lidar2ego_rotation = cs_record["rotation"]
            ego2global_translation = pose_record["translation"]
            ego2global_rotation = pose_record["rotation"]

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(
                ego2global_rotation).rotation_matrix
            ego2global[:3, 3] = ego2global_translation

            lidar2ego = np.eye(4).astype(np.float32)
            lidar2ego[:3, :3] = Quaternion(lidar2ego_rotation).rotation_matrix
            lidar2ego[:3, 3] = lidar2ego_translation

            frames_to_process.append(
                (nusc.dataroot, ego2global, lidar2ego, location,
                 output_dir, bev_file_name))
    print(f"Total frames to process: {len(frames_to_process)}")

    with mp.Pool(processes=30) as pool:
        result = list(tqdm(pool.imap_unordered(process_frame, frames_to_process), total=len(frames_to_process)))

def get_available_scenes(nusc):
    available_scenes = []
    print("total scene num: {}".format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get("scene", scene_token)
        sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
        sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f"{os.getcwd()}/")[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num: {}".format(len(available_scenes)))
    return available_scenes


if __name__ == '__main__':
    main()
