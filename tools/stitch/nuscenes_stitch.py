import os
from os import path as osp
import cv2
import math
import mmengine
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion


refview = 3
fov1 = 70 * math.pi / 180
fov2 = 110 * math.pi / 180

# get hor/vert angles for each view, starting from Matterport matrices (inverse of extrinsic)
def get_angles(matrixList):
    v = np.zeros((6, 3))

    euler = np.array([Rot.from_matrix(matrixList[0]).as_euler('xyz', degrees=False),
                      Rot.from_matrix(matrixList[1]).as_euler('xyz', degrees=False),
                      Rot.from_matrix(matrixList[2]).as_euler('xyz', degrees=False),
                      Rot.from_matrix(matrixList[3]).as_euler('xyz', degrees=False),
                      Rot.from_matrix(matrixList[4]).as_euler('xyz', degrees=False),
                      Rot.from_matrix(matrixList[5]).as_euler('xyz', degrees=False)])

    for i in range(6):
        diff_x = euler[refview][2] - euler[i][2]
        diff_y = euler[refview][1] - euler[i][1]
        diff_z = euler[refview][0] - euler[i][0]
        if diff_x > math.pi:
            diff_x -= 2 * math.pi

        elif diff_x < -math.pi:
            diff_x += 2 * math.pi

        offset_x = math.degrees(diff_x)
        offset_x = offset_x * math.pi / 180
        v[i, 0] = offset_x
        v[i, 1] = -diff_y
        v[i, 2] = -diff_z
    return v


# set blending false for label maps
def combine_views(images, angles, r_offset_list, outsize, blending=True, depth=False):
    nchannels = images[0].shape[2]
    pano = np.zeros((outsize[1], outsize[0], nchannels))
    pano_w = np.zeros((outsize[1], outsize[0], nchannels))
    for i in range(len(images)):
        fov = 0
        if i != 0:
            fov = fov1
        else:
            fov = fov2
        sphere_img, validMap = im2sphere(i,
                                         images[i],
                                         fov,
                                         outsize[0],
                                         outsize[1],
                                         angles[i, 0],
                                         angles[i, 1],
                                         angles[i, 2],
                                         r_offset_list[i],
                                         blending,
                                         i,
                                         depth
                                         )
        sphere_img[validMap < 0.00000001] = 0
        if blending:
            pano = pano + sphere_img
        else:
            if depth:
                sphere_img[:, :, 0] = sphere_img[:, :, 0] * validMap
                pano = pano + sphere_img
            else:
                pano[np.any(sphere_img > 0, axis=2)] = sphere_img[np.any(sphere_img > 0, axis=2)]
        pano_w[:, :, 0] = pano_w[:, :, 0] + validMap
        if nchannels > 1:
            pano_w[:, :, 1] = pano_w[:, :, 1] + validMap
            pano_w[:, :, 2] = pano_w[:, :, 2] + validMap
    pano[pano_w == 0] = 0
    pano_w[pano_w == 0] = 1
    if blending or depth:
        pano = np.divide(pano, pano_w)
        pano = pano[150: 750,:]
    return pano


def im2sphere(i, im, imHoriFOV, sphereW, sphereH, x, y, z, r_offset, interpolate, nr, weightByCenterDist=True):
    # map pixel in panorama to viewing direction
    TX, TY = np.meshgrid(np.array(range(sphereW)), np.array(range(sphereH)))
    TX = TX.flatten('F')
    TY = TY.flatten('F')
    ANGx = ((TX - (sphereW / 2) - 0.5) / sphereW) * math.pi * 2.0  # [-pi, pi]
    ANGy = (-(TY - (sphereH / 2) - 0.5) / sphereH) * math.pi * 50 / 180
    # compute the radius of ball
    imH = im.shape[0]
    imW = im.shape[1]
    R = (imW / 2) / math.tan(imHoriFOV / 2)

    R = R + r_offset
    # im is the tangent plane, contacting with ball at [x0 y0 z0]
    x0 = R * math.cos(y) * math.sin(x)
    y0 = R * math.cos(y) * math.cos(x)
    z0 = R * math.sin(y)
    # plane function: x0(x-x0)+y0(y-y0)+z0(z-z0)=0
    # view line: x/alpha=y/belta=z/gamma
    # alpha=cos(phi)sin(theta);  belta=cos(phi)cos(theta);  gamma=sin(phi)
    alpha = np.multiply(np.cos(ANGy), np.sin(ANGx))
    beta = np.multiply(np.cos(ANGy), np.cos(ANGx))
    gamma = np.sin(ANGy)
    # solve for intersection of plane and viewing line: [x1 y1 z1]
    division = x0 * alpha + y0 * beta + z0 * gamma
    x1 = R * R * np.divide(alpha, division)
    y1 = R * R * np.divide(beta, division)
    z1 = R * R * np.divide(gamma, division)
    # vector in plane: [x1-x0 y1-y0 z1-z0]
    # positive x vector: vecposX = [cos(x) -sin(x) 0]
    # positive y vector: vecposY = [x0 y0 z0] x vecposX
    vec = np.transpose(np.array([x1 - x0, y1 - y0, z1 - z0]))
    vecposX = np.transpose(np.array([math.cos(x), -math.sin(x), 0]))
    deltaX = np.dot(vecposX, np.transpose(vec)) / np.sqrt(np.dot(vecposX, np.transpose(vecposX)))
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    deltaY = np.dot(vecposY, np.transpose(vec)) / np.sqrt(np.dot(vecposY, np.transpose(vecposY)))
    # convert to im coordinates
    Px = np.reshape(deltaX, (sphereH, sphereW), 'F') + (imW + 1) / 2
    Py = np.reshape(deltaY, (sphereH, sphereW), 'F') + (imH + 1) / 2
    # warp image
    sphere_img = warp_image_fast(im, Px, Py, z, interpolate, (sphereW, sphereH))
    validMap = np.zeros((sphere_img.shape[0], sphere_img.shape[1]))
    validMap[:, :] = np.logical_not(np.isnan(sphere_img[:, :, 0])).astype(float)

    if weightByCenterDist:
        weightIm = np.zeros((im.shape[0], im.shape[1], 1))
        c0 = im.shape[0] / 2
        c1 = im.shape[1] / 2
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                weightIm[i, j, 0] = (1 - abs(c0 - i) / c0) * (1 - abs(c1 - j) / c1)

        weightImWarped = warp_image_fast(weightIm, Px, Py, z, False, (sphereW, sphereH))

        validMap = weightImWarped[:, :]
        validMap[sphere_img[:, :] < 1] = 0

    else:
        validMap[sphere_img[:, :, 0] < 0] = 0
    # view direction: [alpha belta gamma]
    # contacting point direction: [x0 y0 z0]
    # so division>0 are valid region
    validMap[np.reshape(division, (sphereH, sphereW), 'F') < 0] = 0

    return sphere_img, validMap


def warp_image_fast(im, XXdense, YYdense, z, interpolate, outsize):
    nchannels = im.shape[2]
    minX = max(1, math.floor(XXdense.min()) - 1)
    minY = max(1, math.floor(YYdense.min()) - 1)
    maxX = min(im.shape[1], math.ceil(XXdense.max()) + 1)
    maxY = min(im.shape[0], math.ceil(YYdense.max()) + 1)
    im = im[minY:maxY, minX:maxX, :]
    center = (im.shape[1] / 2, im.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, z, 1.0)
    im = cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))
    im_warp = np.zeros((outsize[1], outsize[0], nchannels))
    intermode = cv2.INTER_NEAREST
    if interpolate:
        intermode = cv2.INTER_LINEAR
    for i in range(nchannels):
        mapx = XXdense - minX + 1
        mapy = YYdense - minY + 1
        im_warp[:, :, i] = cv2.remap(im[:, :, i].astype(np.float32), mapx.astype(np.float32),
                                     mapy.astype(np.float32),
                                     interpolation=intermode, borderMode=cv2.BORDER_CONSTANT, borderValue=(-1, -1, -1))
    return im_warp

def stitch(img_list, cam_info_list, pano_outsize=[9600, 900]):
    cam_intrinsic_list = [np.array(cam['camera_intrinsic']) for cam in cam_info_list]
    rotation_matrix_list = [Quaternion(cam['rotation']).rotation_matrix for cam in cam_info_list]
    translation_list = [np.array(cam['translation']) for cam in cam_info_list]
    average_translation = np.mean(np.stack(translation_list), axis=0)

    r_offset_list = []
    for i in range(len(img_list)):
        t = translation_list[i]
        offset = t - average_translation
        delta_u = cam_intrinsic_list[i][0, 0] * (offset[0] / 5.5)
        delta_v = cam_intrinsic_list[i][1, 1] * (offset[1] / 5.5)
        r_offset_list.append(math.sqrt(delta_u**2 + delta_v**2))

    angle_list = get_angles(rotation_matrix_list)
    pano_img = combine_views(img_list, angle_list, r_offset_list, pano_outsize)
    return pano_img


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

def timestamp_exists(timestamp, folder):
    file_name = f"{timestamp}.jpg"
    return file_name in os.listdir(folder)

def save_pano(sample, pano_image, split, out_dir):
    timestamp = sample["timestamp"]
    save_path = osp.join(out_dir, split, str(timestamp) + ".jpg")
    cv2.imwrite(save_path, pano_image)


def main():
    data_path = "/lsdf/users/jwei/datasets/nuscenes"
    out_dir = "/cvhci/temp/jwei/nuscenes_full"

    #version = "v1.0-mini"
    #version = "v1.0-trainval"
    version = "v1.0-test"

    nusc = NuScenes(version, data_path, verbose=True)

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

    test = "test" in version
    if test:
        print("test scene: {}".format(len(train_scenes)))
    else:
        print("train scene: {}, val scene: {}".format(len(train_scenes),
                                                        len(val_scenes)))
    
    if version == "v1.0-trainval":    
        train_folder = osp.join(out_dir, "train")
    elif version == "v1.0-test":
        train_folder = osp.join(out_dir, "test")    
    val_folder = osp.join(out_dir, "val")

    train_file_names = os.listdir(train_folder)
    val_file_names = os.listdir(val_folder)
    
    print(f"stitched train file num: {len(train_file_names)}")
    print(f"stitched val file num: {len(val_file_names)}")

    new_samples = []
    for sample in mmengine.track_iter_progress(nusc.sample):
        timestamp = sample["timestamp"]
        scene_token = sample["scene_token"]

        if scene_token in train_scenes:
            folder = train_folder
        else:
            folder = val_folder

        if not timestamp_exists(timestamp, folder):
            new_samples.append(sample)

    print("new sample num: {}".format(len(new_samples)))
    
    for sample in mmengine.track_iter_progress(new_samples):
        camera_types = [
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT',
            'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT'
        ]
        img_list = []
        cam_info_list = []
        for cam in camera_types:
            cam_token = sample["data"][cam]
            sd_rec = nusc.get("sample_data", cam_token)
            data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
            img_list.append(cv2.imread(data_path))

            cs_record = nusc.get('calibrated_sensor',
                                    sd_rec['calibrated_sensor_token'])

            cam_info_list.append({
                "translation":
                cs_record['translation'],
                "rotation":
                cs_record['rotation'],
                "camera_intrinsic":
                cs_record['camera_intrinsic']
            })

        location = nusc.get(
            "log",
            nusc.get("scene", sample["scene_token"])["log_token"])["location"]

        # if sample["scene_token"] in train_scenes and not test:
        #     pano_image = stitch(img_list, cam_info_list)
        #     save_pano(sample, pano_image, "train", out_dir)
        # elif sample["scene_token"] in val_scenes and test:
        #     pano_image = stitch(img_list, cam_info_list)
        #     save_pano(sample, pano_image, "test", out_dir)
        # else:
        #     pano_image = stitch(img_list, cam_info_list)
        #     save_pano(sample, pano_image, "val", out_dir)


        if sample["scene_token"] in train_scenes and not test:
            if f"{sample['timestamp']}.jpg" in train_file_names:
                continue
            pano_image = stitch(img_list, cam_info_list)
            save_pano(sample, pano_image, "train", out_dir)
        elif sample["scene_token"] in train_scenes and test:
            if f"{sample['timestamp']}.jpg" in train_file_names:
                continue
            pano_image = stitch(img_list, cam_info_list)
            save_pano(sample, pano_image, "test", out_dir)    
        else:
            if f"{sample['timestamp']}.jpg" in val_file_names:
                continue
            pano_image = stitch(img_list, cam_info_list)
            save_pano(sample, pano_image, "val", out_dir)

if __name__ == '__main__':
    main()
