import argparse
import os
from os import path as osp
import glob
import numpy as np
import cv2
import mmengine
from tqdm import tqdm
import multiprocessing as mp
import time
import h5py

def bev_seg_extractor(bev_path, out_dir, bev_file_name):
    file = np.load(bev_path)
    data_raw = file['data']
    width = data_raw.shape[0]
    start_idx, end_idx = np.int32(np.linspace(width * 0.1, width * 0.9, 2))
    data_raw = data_raw[start_idx:end_idx, start_idx:end_idx]
    data_raw = cv2.rotate(data_raw, cv2.ROTATE_90_CLOCKWISE)
    valid_class = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17, 19, 20, 21, 22])
    labels = (data_raw[:, :, 2][:, :, None] == valid_class).astype(int)
    labels = cv2.resize(labels, (200, 200), interpolation=cv2.INTER_NEAREST)
    labels = labels.transpose((2, 0, 1))
    out_path = osp.join(out_dir, 'bev', bev_file_name)
    mmengine.mkdir_or_exist(osp.dirname(out_path))
    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('labels', data=labels, compression='gzip', compression_opts=9)


def process_frame(frame_data):
    bev_path, pano_dir, bev_file_name = frame_data
    bev_seg_extractor(bev_path, pano_dir, bev_file_name)

def main():
    parser = argparse.ArgumentParser(description="BEV segmentation creator arg parser")
    parser.add_argument("--data-path", type=str, default="./data/nuscenes", help="Root path of dataset")
    parser.add_argument("--version", type=str, default="trainval", help="Dataset version")
    parser.add_argument("--pano-dir", type=str, default="./data/deepaccident", required=False, help="Directory name for panorama data")
    
    args = parser.parse_args()
    data_path = args.data_path
    pano_dir = args.pano_dir
    version = args.version
    test = 'test' in version

    train_list, val_list = [], []
    if not test:
        with open(osp.join(data_path, 'train.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
        with open(osp.join(data_path, 'val.txt'), 'r') as f:
            val_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
    else:
        with open(osp.join(data_path, 'test.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]
            val_list = []

    total_frames = [osp.join(data_path, scenario_type, 'ego_vehicle', 'label', file_prefix) + '/*'
                    for scenario_type, file_prefix in train_list + val_list]
    total_frames = [item for sublist in total_frames for item in glob.glob(sublist)]
    total_frames = [frame.split('.')[0] for frame in total_frames]

    output_dir = osp.join(pano_dir, 'bev')
    mmengine.mkdir_or_exist(output_dir)
    existed = set(os.listdir(output_dir))
    print(f"Already processed: {len(existed)}")
    frames_to_process = [(osp.join(data_path, frame.split('/')[-5], 'ego_vehicle', 'BEV_instance_camera', frame.split('/')[-2], frame.split('/')[-1] + '.npz'),
                          pano_dir,
                          frame.split('/')[-5] + '-' + frame.split('/')[-1] + '.h5') for frame in total_frames if frame.split('/')[-5] + '-' + frame.split('/')[-1] + '.h5' not in existed]

    print(f"Total frames to process: {len(frames_to_process)}")

    # Process in parallel
    with mp.Pool(processes=50) as pool:
        result = list(tqdm(pool.imap_unordered(process_frame, frames_to_process), total=len(frames_to_process)))


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
