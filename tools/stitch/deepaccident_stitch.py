import os
from os import path as osp
import multiprocessing
import glob
import cv2
import math
import mmengine
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import multiprocessing as mp
from tqdm import tqdm

refview = 3
fov1 = 70 * math.pi / 180
fov2 = 110 * math.pi / 180


# get hor/vert angles for each view, starting from Matterport matrices (inverse of extrinsic)
def get_angles(matrixList):
    v = np.zeros((6, 3))

    euler = np.array([
        Rot.from_matrix(matrixList[0]).as_euler('xyz', degrees=False),
        Rot.from_matrix(matrixList[1]).as_euler('xyz', degrees=False),
        Rot.from_matrix(matrixList[2]).as_euler('xyz', degrees=False),
        Rot.from_matrix(matrixList[3]).as_euler('xyz', degrees=False),
        Rot.from_matrix(matrixList[4]).as_euler('xyz', degrees=False),
        Rot.from_matrix(matrixList[5]).as_euler('xyz', degrees=False)
    ])

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

def combine_views(images,
                  angles,
                  r_offset_list,
                  outsize,
                  blending=True,
                  depth=False):
    nchannels = images[0].shape[2]
    pano = np.zeros((outsize[1], outsize[0], nchannels))
    pano_w = np.zeros((outsize[1], outsize[0], nchannels))
    for i in range(len(images)):
        fov = 0
        if i != 0:
            fov = fov1
        else:
            fov = fov2
        sphere_img, validMap = im2sphere(i, images[i], fov, outsize[0],
                                         outsize[1], angles[i, 0],
                                         angles[i, 1], angles[i, 2],
                                         r_offset_list[i], blending, i, depth)
        sphere_img[validMap < 0.00000001] = 0
        if blending:
            pano = pano + sphere_img
        else:
            if depth:
                sphere_img[:, :, 0] = sphere_img[:, :, 0] * validMap
                pano = pano + sphere_img
            else:
                pano[np.any(sphere_img > 0,
                            axis=2)] = sphere_img[np.any(sphere_img > 0,
                                                         axis=2)]
        pano_w[:, :, 0] = pano_w[:, :, 0] + validMap
        if nchannels > 1:
            pano_w[:, :, 1] = pano_w[:, :, 1] + validMap
            pano_w[:, :, 2] = pano_w[:, :, 2] + validMap
    pano[pano_w == 0] = 0
    pano_w[pano_w == 0] = 1
    if blending or depth:
        pano = np.divide(pano, pano_w)
        pano = pano[150:750, :]
    return pano

def im2sphere(i,
              im,
              imHoriFOV,
              sphereW,
              sphereH,
              x,
              y,
              z,
              r_offset,
              interpolate,
              nr,
              weightByCenterDist=True):
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
    deltaX = np.dot(vecposX, np.transpose(vec)) / np.sqrt(
        np.dot(vecposX, np.transpose(vecposX)))
    vecposY = np.cross(np.array([x0, y0, z0]), vecposX)
    deltaY = np.dot(vecposY, np.transpose(vec)) / np.sqrt(
        np.dot(vecposY, np.transpose(vecposY)))
    # convert to im coordinates
    Px = np.reshape(deltaX, (sphereH, sphereW), 'F') + (imW + 1) / 2
    Py = np.reshape(deltaY, (sphereH, sphereW), 'F') + (imH + 1) / 2
    # warp image
    sphere_img = warp_image_fast(im, Px, Py, z, interpolate,
                                 (sphereW, sphereH))
    validMap = np.zeros((sphere_img.shape[0], sphere_img.shape[1]))
    validMap[:, :] = np.logical_not(np.isnan(sphere_img[:, :,
                                                        0])).astype(float)

    if weightByCenterDist:
        weightIm = np.zeros((im.shape[0], im.shape[1], 1))
        c0 = im.shape[0] / 2
        c1 = im.shape[1] / 2
        for i in range(im.shape[0]):
            for j in range(im.shape[1]):
                weightIm[i, j,
                         0] = (1 - abs(c0 - i) / c0) * (1 - abs(c1 - j) / c1)

        weightImWarped = warp_image_fast(weightIm, Px, Py, z, False,
                                         (sphereW, sphereH))

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
    # out_dir = '/Users/garywei/Documents/cvhci/stitching_images_out/'
    # out_path = os.path.join(out_dir, f'im_ro_{nr}.jpg')
    # cv2.imwrite(out_path, im)
    im_warp = np.zeros((outsize[1], outsize[0], nchannels))
    intermode = cv2.INTER_NEAREST
    if interpolate:
        intermode = cv2.INTER_LINEAR
    for i in range(nchannels):
        mapx = XXdense - minX + 1
        mapy = YYdense - minY + 1
        im_warp[:, :, i] = cv2.remap(im[:, :, i].astype(np.float32),
                                     mapx.astype(np.float32),
                                     mapy.astype(np.float32),
                                     interpolation=intermode,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(-1, -1, -1))
    return im_warp


def stitch(img_list, cam_info_list, pano_outsize=[9600, 900]):
    cam_intrinsic_list = [
        cam['camera_intrinsic'] for cam in cam_info_list
    ]
    rotation_matrix_list = [
        cam['rotation'] for cam in cam_info_list
    ]
    translation_list = [cam['translation'] for cam in cam_info_list]
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

def frame_exists(frame, folder):
    scenario_type = frame.split('/')[-5]
    frame_name = frame.split('/')[-1]
    pano_name = scenario_type + '-' + frame_name
    file_name = f"{pano_name}.jpg"
    return file_name in os.listdir(folder)

def save_pano(pano_name, pano_image, split, out_dir):
    out_dir = osp.join(out_dir, split)
    mmengine.mkdir_or_exist(out_dir)
    save_path = osp.join(out_dir, pano_name + ".jpg")
    cv2.imwrite(save_path, pano_image)


def process_frame(frame_data):
    frame, data_path, out_dir, train_file_names, val_file_names, train_frames, val_frames, test = frame_data
    key = 'ego_vehicle'
    scenario_type = frame.split('/')[-5]
    file_prefix = frame.split('/')[-2]
    frame_name = frame.split('/')[-1] 

    calib_path = osp.join(data_path, scenario_type, key, 'calib',
                            file_prefix, frame_name + '.pkl')
    calib_dict = mmengine.load(calib_path)
    camera_types = [
        'Camera_Back', 'Camera_BackLeft', 'Camera_FrontLeft',
        'Camera_Front', 'Camera_FrontRight', 'Camera_BackRight'
    ]
    img_list = []
    cam_info_list = []
    for cam in camera_types:
        cam_path = osp.join(data_path, scenario_type, key, cam,
                            file_prefix, frame_name + '.jpg')
        img_list.append(cv2.imread(cam_path))

        cam_info_list.append({
            "translation":
            calib_dict['lidar_to_' + cam][:3, 3],
            "rotation":
            calib_dict['lidar_to_' + cam][:3, :3],
            "camera_intrinsic":
            calib_dict['intrinsic_' + cam]
        })
        
    pano_name = scenario_type + '-' + frame_name    
    if frame in train_frames and not test:
        if f"{pano_name}.jpg" in train_file_names:
            return
        pano_image = stitch(img_list, cam_info_list)
        save_pano(pano_name, pano_image, "train", out_dir)
    elif frame in train_frames and test:
        if f"{pano_name}.jpg" in train_file_names:
            return
        pano_image = stitch(img_list, cam_info_list)
        save_pano(pano_name, pano_image, "test", out_dir)
    else:
        if f"{pano_name}.jpg" in val_file_names:
            return
        pano_image = stitch(img_list, cam_info_list)
        save_pano(pano_name, pano_image, "val", out_dir)

def main():
    data_path = "/lsdf/users/jwei/datasets/deep_accident"
    out_dir = "/lsdf/users/jwei/deepaccident_full"

    version = "trainval"
    #version = "test"

    test = 'test' in version

    if not test:
        with open(osp.join(data_path, 'train.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0],
                           line.rstrip().split(' ')[1]) for line in f]
        with open(osp.join(data_path, 'val.txt'), 'r') as f:
            # val_list = [line.rstrip() for line in f]
            val_list = [(line.rstrip().split(' ')[0],
                         line.rstrip().split(' ')[1]) for line in f]
        print('train scene: {}, val scene: {}'.format(len(train_list),
                                                      len(val_list)))
    else:
        with open(osp.join(data_path, 'test.txt'), 'r') as f:
            train_list = [(line.rstrip().split(' ')[0],
                           line.rstrip().split(' ')[1]) for line in f]
        print('test scene: {}'.format(len(train_list)))

    if version == "trainval":
        train_folder = osp.join(out_dir, "train")
        val_folder = osp.join(out_dir, "val")
        mmengine.mkdir_or_exist(train_folder)
        mmengine.mkdir_or_exist(val_folder)
    elif version == "test":
        train_folder = osp.join(out_dir, "test")
        mmengine.mkdir_or_exist(train_folder)

    train_file_names = os.listdir(train_folder)
    val_file_names = os.listdir(val_folder)

    print(f"stitched train file num: {len(train_file_names)}")
    print(f"stitched val file num: {len(val_file_names)}")

    train_frames = []
    for (scenario_type, file_prefix) in train_list:
        file_list = glob.glob(
            osp.join(data_path, scenario_type, 'ego_vehicle', 'label',
                     file_prefix) + '/*')
        
        for file_path in file_list:
            train_frames.append(file_path.split('.')[0])

    val_frames = []
    for (scenario_type, file_prefix) in val_list:
        file_list = glob.glob(
            osp.join(data_path, scenario_type, 'ego_vehicle', 'label',
                     file_prefix) + '/*')
        
        for file_path in file_list:
            val_frames.append(file_path.split('.')[0]) 

    total_frames = train_frames + val_frames

    total_frames_filtered = []

    for frame in mmengine.track_iter_progress(total_frames):
        if frame in train_frames:
            folder = train_folder
        else:
            folder = val_folder

        if not frame_exists(frame, folder):
            total_frames_filtered.append(frame)    

    print("total frames num: {}".format(len(total_frames))) 
    print("filtered frames num: {}".format(len(total_frames_filtered)))


    frames_to_process = [
        (frame, data_path, out_dir, train_file_names, val_file_names, train_frames, val_frames, test)
        for frame in mmengine.track_iter_progress(total_frames_filtered)
    ]
    pool_size = multiprocessing.cpu_count()

    with mp.Pool(processes=pool_size) as pool:
        result = list(tqdm(pool.imap_unordered(process_frame, frames_to_process), total=len(frames_to_process)))
        
if __name__ == '__main__':
    main()
