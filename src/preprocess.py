import numpy as np
from ros_numpy.point_cloud2 import pointcloud2_to_xyz_array

from utils import load_config, get_num_voxels, get_anchors, filter_pointcloud


class Pre_Process:
    def __init__(self):
        self.config = load_config()

        # Calculate the size of the voxel grid in every dimensions
        self.voxel_W, self.voxel_H, self.voxel_D = get_num_voxels()

        self.anchors = get_anchors().reshape(-1, 7)

        # Because of the network architecure (deconvs), output feature map
        # shape is half that of the input
        self.feature_map_shape = (self.voxel_H // 2, self.voxel_W // 2)

    def voxelize(self, lidar):
        '''
        Convert a point cloud to a voxelized grid

        Parameters:
            lidar (arr): point cloud

        Returns:
            voxel_features (arr): (N, X, 7),
                where N = number of non-empty voxels,
                X = max points per voxel (See T in 2.1.1), and
                7 encodes [x,y,z,reflectance,Δx,Δy,Δz],
                    where Δ is from the mean of all points in the voxel

            voxel_coords (arr): (N, 3),
                where N = number of non-empty voxels and
                3 encodes [Z voxel, Y voxel, X voxel]
        '''
        # Shuffle the points
        np.random.shuffle(lidar)

        voxel_coords = ((lidar[:, :3] - np.array(
            [self.config['pcl_range']['X1'], self.config['pcl_range']['Y1'],
             self.config['pcl_range']['Z1']])) / (
            self.config['voxel_size']['W'], self.config['voxel_size']['H'],
            self.config['voxel_size']['D'])).astype(np.int32)

        # Convert to (D, H, W)
        voxel_coords = voxel_coords[:, [2, 1, 0]]

        # Get info on the non-empty voxels
        voxel_coords, inv_ind, voxel_counts = np.unique(
            voxel_coords, axis=0, return_inverse=True, return_counts=True)

        voxel_features = []

        for i in range(len(voxel_coords)):
            voxel = np.zeros(
                (self.config['max_pts_per_voxel'], 7), dtype=np.float32)

            # inv_ind gives the indices of the elements in the original array
            pts = lidar[inv_ind == i]

            if voxel_counts[i] > self.config['max_pts_per_voxel']:
                pts = pts[:self.config['max_pts_per_voxel'], :]
                voxel_counts[i] = self.config['max_pts_per_voxel']

            # Augment each point with its relative offset
            # w.r.t. the centroid of this voxel (See 2.1.1)
            voxel[:pts.shape[0], :] = np.concatenate(
                (pts, pts[:, :3] - np.mean(pts[:, :3], axis=0)), axis=1)
            voxel_features.append(voxel)

        return np.array(voxel_features), voxel_coords

    def pre_process(self, pointcloud_msg):
        '''
        Convert a received PointCloud2 message into input format for VoxelNet

        Parameters:
            pointcloud_msg (sensor_msgs/PointCloud2): the source point cloud

        Returns:
            voxel_features (arr): (N, X, 7),
                where N = number of non-empty voxels,
                X = max points per voxel (See T in 2.1.1), and
                7 encodes [x,y,z,reflectance,Δx,Δy,Δz],
                    where Δ is from the mean of all points in the voxel

            voxel_coords (arr): (N, 3),
                where N = number of non-empty voxels and
                3 encodes [Z voxel, Y voxel, X voxel]

            pointcloud (arr): (Z, 4), pointcloud
        '''
        pointcloud = pointcloud2_to_xyz_array(pointcloud_msg)

        # Pad with reflectances as 1
        if pointcloud.shape[1] == 3:
            pointcloud = np.concatenate(
                (pointcloud, np.ones((pointcloud.shape[0], 1))), axis=1)

        # Crop the lidar
        pointcloud = filter_pointcloud(pointcloud)

        # Voxelize
        voxel_features, voxel_coords = self.voxelize(pointcloud)

        return voxel_features, voxel_coords, pointcloud
