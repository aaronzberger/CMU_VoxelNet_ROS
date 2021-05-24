import rospy
from sensor_msgs.msg import PointCloud2
import torch

from CMU_VoxelNet_ROS.msg import predictions
from preprocess import Pre_Process
from postprocess import Post_Process
from voxelnet import VoxelNet


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

        predictions_msg = self.post_process.boxes_to_ros_msg(bounding_boxes, pointcloud)

        self.pub_predictions.publish(predictions_msg)


if __name__ == '__main__':
    rospy.init_node('voxelnet', log_level=rospy.INFO)

    voxelnet_node = VoxelNet_Node()

    rospy.loginfo('started voxelnet node')

    rospy.spin()
