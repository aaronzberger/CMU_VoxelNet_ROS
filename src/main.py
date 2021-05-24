import rospy
from sensor_msgs.msg import PointCloud2
from CMU_VoxelNet_ROS.msg import predictions
from preprocess import Pre_Process
from postprocess import Post_Process
from voxelnet import VoxelNet
import torch
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Pose, Vector3, Point
from CMU_VoxelNet_ROS.msg import predictions
from std_msgs.msg import Header

class VoxelNet_Node:
    def __init__(self):
        self.sub_aggregated = rospy.Subscriber(
            '/velodyne_aggregated', PointCloud2, self.aggregated_callback)
        self.pub_predictions = rospy.Publisher(
            '/voxelnet_predictions', predictions, queue_size=1)

        self.pre_process = Pre_Process()
        self.post_process = Post_Process()

        # Choose a device for the model
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.net = VoxelNet(self.device).to(self.device)
        self.net.eval()

    def aggregated_callback(self, data):
        voxel_features, voxel_coords, pointcloud = \
            self.pre_process.pre_process(data.data)

        voxel_features = torch.Tensor(voxel_features).to(self.device)
        voxel_coords = torch.Tensor(voxel_coords).to(self.device)

        with torch.no_grad():
            prob_score_map, reg_map = self.net(voxel_features, voxel_coords)

        bounding_boxes = self.post_process.output_to_boxes(prob_score_map, reg_map)

        ros_bboxes = []

        for bbox in bounding_boxes:
            ros_bboxes.append(BoundingBox3D(
                center=Pose(position=Point(x=bbox[0], y=bbox[1], z=bbox[2])),
                size=Vector3(x=bbox[3], y=bbox[4], z=bbox[5])))

        predictions_msg = predictions()
        predictions_msg.header = Header(stamp=rospy.Time.now(), frame_id='camera_init')
        predictions_msg.bboxes = ros_bboxes

        # TODO: We should be publishing the cropped and processed pointcloud as the source cloud,
        # not the original, since that is the one that will be used for combining bounding boxes, etc
        predictions_msg.source_cloud = data

        self.pub_predictions.publish(predictions_msg)


if __name__ == '__main__':
    rospy.init_node('voxelnet', log_level=rospy.INFO)

    voxelnet_node = VoxelNet_Node()

    rospy.loginfo('started voxelnet node')

    rospy.spin()
