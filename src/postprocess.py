from geometry_msgs.msg import Pose, Point, Vector3
import numpy as np
import rospy
from std_msgs.msg import Header
import torch
from vision_msgs.msg import BoundingBox3D

from CMU_VoxelNet_ROS.msg import predictions
from nms import nms
from ros_numpy.point_cloud2 import array_to_pointcloud2
from utils import load_config, get_anchors


class Post_Process:
    def delta_to_boxes3d(self, deltas, anchors):
        '''
        Convert regression map deltas to bounding boxes

        Parameters:
            deltas (arr): (N, W, L, 14): the regression output,
                where N = batch size
            anchors (arr): (X, 7): the anchors, where X = number of anchors

        Returns:
            arr: the bounding boxes
        '''
        batch_size = deltas.shape[0]

        # View in the same form as a list of all anchors
        deltas = deltas.view(batch_size, -1, 7)

        anchors = torch.FloatTensor(anchors)
        boxes3d = torch.zeros_like(deltas)

        if deltas.is_cuda:
            anchors = anchors.cuda()
            boxes3d = boxes3d.cuda()

        anchors_reshaped = anchors.view(-1, 7)
        #       _______________
        # dᵃ = √ (lᵃ)² + (wᵃ)²      is the diagonal of the base
        #                           of the anchor box (See 2.2)
        anchors_diagonal = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)

        # Repeat across the batch and across [X, Y]
        anchors_diagonal = anchors_diagonal.repeat(
            batch_size, 2, 1).transpose(1, 2)

        # Repeat over the batch size
        anchors_reshaped = anchors_reshaped.repeat(
            batch_size, 1, 1)

        # Δx = (xᵍ - xᵃ) / dᵃ and Δy = (yᵍ - yᵃ) / dᵃ
        # so x = (Δx * dᵃ) + xᵃ and y = (Δy * dᵃ) + yᵃ
        boxes3d[..., [0, 1]] = torch.mul(
            deltas[..., [0, 1]], anchors_diagonal) \
            + anchors_reshaped[..., [0, 1]]

        # Δz = (zᵍ - zᵃ) / hᵃ so z = (Δz * hᵃ) + zᵃ
        boxes3d[..., [2]] = torch.mul(
            deltas[..., [2]], anchors_reshaped[..., [3]]) \
            + anchors_reshaped[..., [2]]

        # Δw = log(wᵍ / wᵃ) so w = e^(Δw) * wᵃ
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]

        # Δ𝜃 = 𝜃ᵍ - 𝜃ᵃ, so 𝜃 = Δ𝜃 + 𝜃ᵃ
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    def box3d_center_to_corner(self, boxes_center, z_middle=False):
        '''
        Transform bounding boxes from center to corner notation

        Parameters:
            boxes_center (arr): (X, 7):
                boxes in center notation [xyzhwlr]
            z_middle (bool): whether the z in boxes_center is at the middle of the object

        Returns:
            arr: bounding box in corner notation
        '''
        if torch.is_tensor(boxes_center):
            if boxes_center.is_cuda:
                boxes_center = boxes_center.cpu().numpy()
        num_boxes = boxes_center.shape[0]

        # To return
        corner_boxes = np.zeros((num_boxes, 8, 3))

        for box_num, box in enumerate(boxes_center):
            translation = box[0:3]
            h, w, l = box[3], box[4], box[5]
            rotation = box[6]

            # Create the bounding box outline (to be transposed)
            bounding_box = np.array([
                [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
                [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
                [0, 0, 0, 0, h, h, h, h]])
            if z_middle:
                bounding_box[2] = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

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

            corner_boxes[box_num] = corner_box

        return corner_boxes

    def center_to_corner_box2d(self, boxes_center):
        # (N, 5) -> (N, X, 4, 2)
        N = boxes_center.shape[0]
        boxes3d_center = np.zeros((N, 7))
        boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
        boxes3d_corner = self.box3d_center_to_corner(
            boxes3d_center,
        )

        return boxes3d_corner[:, 0:4, 0:2]

    def corner_to_standup_box2d(self, boxes_corner):
        '''
        Convert corner coordinates to xyxy form boxes
        (N, 4, 2) → (N, 4): [x1, y1, x2, y2]

        Parameters:
            boxes_corner (arr): the boxes in corner notation

        Returns:
            arr: the boxes with the max and min x and y values from the input
        '''
        N = boxes_corner.shape[0]
        standup_boxes2d = np.zeros((N, 4))

        standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)  # X1
        standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)  # Y1
        standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)  # X2
        standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)  # Y2

        return standup_boxes2d

    def output_to_boxes(self, prob_score_map, reg_map):
        '''
        Convert VoxelNet output to bounding boxes for visualization

        Parameters:
            prob_score_map (arr): (BS, 2, H, W), probability score map
            reg_map (arr): (BS, 14, H, W), regression map (deltas)

        Returns:
            arr: boxes in center notation
        '''
        config = load_config()
        batch_size, _, _, _ = prob_score_map.shape
        device = prob_score_map.device

        # Convert regression map back to bounding boxes (center notation)
        batch_boxes3d = self.delta_to_boxes3d(reg_map, get_anchors())

        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = prob_score_map.reshape((batch_size, -1))

        batch_boxes3d = batch_boxes3d.cpu().numpy()
        batch_boxes2d = batch_boxes2d.cpu().numpy()
        batch_probs = batch_probs.cpu().numpy()

        return_box3d = []
        return_score = []
        for batch_id in range(batch_size):
            # Remove boxes under the threshold
            ind = np.where(
                batch_probs[batch_id, :] >= config['nms_score_threshold'])
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...].squeeze()
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...].squeeze()
            tmp_scores = batch_probs[batch_id, ind].squeeze()

            # Convert center notation 3d boxes to corner notation 2d boxes
            corner_box2d = self.center_to_corner_box2d(tmp_boxes2d)

            # Convert from xxyy to xyxy
            boxes2d = self.corner_to_standup_box2d(corner_box2d)

            # Apply NMS to get rid of duplicates
            ind, cnt = nms(
                torch.from_numpy(boxes2d).to(device),
                torch.from_numpy(tmp_scores).to(device),
                config['nms_threshold'],
                100,
            )
            try:
                ind = ind[:cnt].cpu().detach().numpy()
            except IndexError:
                print('Unable to select NMS-detected boxes, returning None')
                return None, None

            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            return_box3d.append(tmp_boxes3d)
            return_score.append(tmp_scores)

        return np.array(return_box3d)

    def boxes_to_ros_msg(self, boxes, pointcloud):
        '''
        Convert center notation 3D bounding boxes to ROS message format

        Parameters:
            boxes (arr): (N, 7), where each row is a bounding box in the format [x, y, z, h, w, l, r]
            pointcloud (arr): (X, 4): pointcloud

        Returns:
            predictions_msg (CMU_VoxelNet_ROS/predictions.msg): the ROS message to publish
        '''
        ros_bboxes = []

        for bbox in boxes:
            ros_bboxes.append(BoundingBox3D(
                center=Pose(position=Point(x=bbox[0], y=bbox[1], z=bbox[2])),
                size=Vector3(x=bbox[3], y=bbox[4], z=bbox[5])))

        predictions_msg = predictions()
        predictions_msg.header = Header(stamp=rospy.Time.now(), frame_id='camera_init')
        predictions_msg.bboxes = ros_bboxes
        
        # Do not include a stamp or frame_id, since this message will be published
        # within another message containing a std_msgs/Header
        predictions_msg.source_cloud = array_to_pointcloud2(pointcloud)

        return predictions_msg

