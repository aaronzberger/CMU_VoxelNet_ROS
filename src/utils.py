from __future__ import division
import os
import json
import math
import errno
import time

import numpy as np


def load_config():
    '''
    Load the configuration json file
    '''
    with open('/home/aaron/catkin_ws/src/CMU_VoxelNet_ROS/cfg/config.json') as file:
        config_dict = json.load(file)

    return config_dict


def get_num_voxels():
    '''
    Get the number of voxels in every dimension

    Returns:
        voxel_W (float): number of X voxels
        voxel_H (float): number of Y voxels
        voxel_D (float): number of Z voxels
    '''
    config = load_config()

    # Calculate the size of the voxel grid in every dimensions
    voxel_W = math.ceil(
        (config['pcl_range']['X2'] - config['pcl_range']['X1'])
        / config['voxel_size']['W'])
    voxel_H = math.ceil(
        (config['pcl_range']['Y2'] - config['pcl_range']['Y1'])
        / config['voxel_size']['H'])
    voxel_D = math.ceil(
        (config['pcl_range']['Z2'] - config['pcl_range']['Z1'])
        / config['voxel_size']['D'])

    return voxel_W, voxel_H, voxel_D


def get_anchors():
    '''
    Generate the anchors

    Returns:
        arr: list of anchors in the form [x,y,z,h,w,l,r]
    '''
    config = load_config()
    voxel_W, voxel_H, _ = get_num_voxels()

    # Make the anchor grid (center notation)
    x = np.linspace(config['pcl_range']['X1'] + config['voxel_size']['W'],
                    config['pcl_range']['X2'] - config['voxel_size']['W'],
                    voxel_W // 2)
    y = np.linspace(config['pcl_range']['Y1'] + config['voxel_size']['H'],
                    config['pcl_range']['Y2'] - config['voxel_size']['H'],
                    voxel_H // 2)

    # Get the xs and ys for the grid
    cx, cy = np.meshgrid(x, y)

    # Anchors only move in X and Y, not Z (BEV)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)

    # We only use one anchor size (See 3.1)
    cz = np.ones_like(cx) * -1.0
    width = np.ones_like(cx) * 1.6
    length = np.ones_like(cx) * 3.9
    height = np.ones_like(cx) * 1.56

    # We use two rotations: 0 and 90 deg (See 3.1)
    rotation = np.ones_like(cx)
    rotation[..., 0] = 0
    rotation[..., 1] = np.pi / 2

    return np.stack([cx, cy, cz, height, width, length, rotation], axis=-1)


def filter_pointcloud(lidar):
    '''
    Crop a lidar pointcloud to the dimensions specified in config json

    Parameters:
        lidar (arr): the point cloud

    Returns:
        arr: cropped point cloud
    '''
    config = load_config()

    x_pts = lidar[:, 0]
    y_pts = lidar[:, 1]
    z_pts = lidar[:, 2]

    # Determine indexes of valid, in-bound points
    lidar_x = np.where((x_pts >= config['pcl_range']['X1'])
                       & (x_pts < config['pcl_range']['X2']))[0]
    lidar_y = np.where((y_pts >= config['pcl_range']['Y1'])
                       & (y_pts < config['pcl_range']['Y2']))[0]
    lidar_z = np.where((z_pts >= config['pcl_range']['Z1'])
                       & (z_pts < config['pcl_range']['Z2']))[0]

    # Combine the index arrays
    lidar_valid_xyz = np.intersect1d(lidar_z, np.intersect1d(lidar_x, lidar_y))

    return lidar[lidar_valid_xyz]


def box3d_cam_to_velo(box3d, tr_velo_to_cam, R0_rect):
    '''
    Transform bounding boxes from center to corner notation
    and transform to velodyne frame

    Parameters:
        box3d (arr): the bouning box in center notation
        Tr (arr): the transform from camera to velodyne

    Returns:
        arr: bounding box in corner notation
    '''

    def camera_to_lidar_box(coord, tr_velo_to_cam, R0_rect):
        R0_formatted = np.eye(4)
        R0_formatted[:3, :3] = R0_rect
        tr_formatted = np.concatenate(
            [tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        coord = np.matmul(np.linalg.inv(R0_formatted), coord)
        coord = np.matmul(np.linalg.inv(tr_formatted), coord)
        return coord[:3].reshape(1, 3)

    def ry_to_rz(ry):
        rz = -ry - np.pi / 2
        limit_degree = 5
        while rz >= np.pi / 2:
            rz -= np.pi
        while rz < -np.pi / 2:
            rz += np.pi

        # So we don't have -pi/2 and pi/2
        if abs(rz + np.pi / 2) < limit_degree / 180 * np.pi:
            rz = np.pi / 2
        return rz

    # KITTI labels are formatted [hwlxyzr]
    h, w, l, tx, ty, tz, ry = [float(i) for i in box3d]

    # Position in labels are in cam coordinates. Transform to lidar coords
    cam = np.expand_dims(np.array([tx, ty, tz, 1]), 1)
    translation = camera_to_lidar_box(cam, tr_velo_to_cam, R0_rect)

    rotation = ry_to_rz(ry)

    # Very similar code as in box3d_center_to_corner in conversions.py
    # Create the bounding box outline (to be transposed)
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0, 0, 0, 0, h, h, h, h]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation), 0.0],
        [np.sin(rotation), np.cos(rotation), 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    corner_box = corner_box.transpose()

    return corner_box.astype(np.float32)


def load_kitti_label(label_file, tr_velo_to_cam, R0_rect):
    '''
    Load the labels for a specific image

    Parameters:
        label_file (str): label file full path
        Tr (arr): velodyne to camera transform

    Returns:
        arr: array containing GT boxes in the correct format
    '''
    config = load_config()

    with open(label_file, 'r') as f:
        lines = f.readlines()

    gt_boxes3d_corner = []

    for j in range(len(lines)):
        obj = lines[j].strip().split(' ')

        # Ensure the GT class is one we're using
        if obj[0].strip() not in config['class_list']:
            continue

        # Transform label into coordinates of 8 points that make up the bbox
        box3d_corner = box3d_cam_to_velo(obj[8:], tr_velo_to_cam, R0_rect)

        gt_boxes3d_corner.append(box3d_corner)

    gt_boxes3d_corner = np.array(gt_boxes3d_corner).reshape(-1, 8, 3)

    return gt_boxes3d_corner