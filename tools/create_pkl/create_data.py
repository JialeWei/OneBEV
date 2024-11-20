import argparse
import os
from os import path as osp
import mmengine

metadata_nuscenes = dict(map_classes=[
    'drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area',
    'divider'
])

metadata_deepaccident = dict(map_classes=[
    'building', 'fence', 'pedestrian', 'pole', 'road_line', 'road',
    'side_walk', 'vegetation', 'vehicles', 'wall', 'ground', 'bridge',
    'guard_rail', 'static', 'dynamic', 'water', 'terrain'
])


def data_prep(root_path, info_prefix, version):
    if info_prefix == "nusc":
        metadata = metadata_nuscenes
    elif info_prefix == "deep":
        metadata = metadata_deepaccident

    bev_path = osp.join(root_path, 'bev')

    if version == "trainval":
        pano_path_train = osp.join(root_path, "train")
        pano_path_val = osp.join(root_path, "val")
        pano_list_train = os.listdir(pano_path_train)
        pano_list_val = os.listdir(pano_path_val)

        train_infos = []
        val_infos = []

        for file_name in pano_list_train:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                frame_name = file_name.split(".")[0]
                pano_file_path = osp.join(pano_path_train, file_name)
                bev_file_path = osp.join(bev_path, frame_name + ".h5")
                info = {
                    "img_path": pano_file_path,
                    "frame_name": frame_name,
                    "bev_path": bev_file_path
                }
                train_infos.append(info)

        for file_name in pano_list_val:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                frame_name = file_name.split(".")[0]
                pano_file_path = osp.join(pano_path_val, file_name)
                bev_file_path = osp.join(bev_path, frame_name + ".h5")
                info = {
                    "img_path": pano_file_path,
                    "frame_name": frame_name,
                    "bev_path": bev_file_path
                }
                val_infos.append(info)

        print("dataset: {}, train sample: {}, val sample: {}".format(
            info_prefix, len(train_infos), len(val_infos)))

        data = dict(data_list=train_infos, metainfo=metadata)
        info_path = osp.join(root_path,
                             "{}_infos_train_mmengine.pkl".format(info_prefix))
        mmengine.dump(data, info_path)

        data = dict(data_list=val_infos, metainfo=metadata)
        info_val_path = osp.join(
            root_path, "{}_infos_val_mmengine.pkl".format(info_prefix))

        mmengine.dump(data, info_val_path)

    elif version == "test":
        pano_path = osp.join(root_path, "test")
        pano_list = os.listdir(pano_path)
        test_infos = []

        for file_name in pano_list:
            if file_name.endswith(".png") or file_name.endswith(".jpg"):
                frame_name = file_name.split(".")[0]
                pano_file_path = osp.join(pano_path, file_name)
                bev_file_path = osp.join(bev_path, frame_name + ".h5")
                info = {
                    "img_path": pano_file_path,
                    "frame_name": frame_name,
                    "bev_path": bev_file_path
                }
                test_infos.append(info)

        print("dataset: {}, test sample: {}".format(
            info_prefix, len(test_infos)))
        data = dict(data_list=test_infos, metainfo=metadata)
        info_path = osp.join(root_path,
                             "{}_infos_test_mmengine.pkl".format(info_prefix))
        mmengine.dump(data, info_path)


parser = argparse.ArgumentParser(description="Data converter arg parser")
parser.add_argument("dataset", type=str, choices=['nusc', 'deep'], help="name of the dataset")
parser.add_argument(
    "--root-path",
    type=str,
    help="specify the root path of dataset",
)
parser.add_argument(
    "--version",
    type=str,
    choices = ['trainval', 'test'],
    help="specify the dataset version",
)
args = parser.parse_args()

if __name__ == "__main__":
    data_prep(root_path=args.root_path,
              info_prefix=args.dataset,
              version=args.version)
