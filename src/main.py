import rospy
from sensor_msgs.msg import PointCloud2
from CMU_VoxelNet_ROS.msg import predictions

class VoxelNet_Node:
    def __init__(self):
        self.sub_aggregated = rospy.Subscriber(
            '/velodyne_aggregated', PointCloud2, self.aggregated_callback)
        self.pub_predictions = rospy.Publisher(
            '/voxelnet_predictions', predictions, queue_size=1)

    def aggregated_callback(self, data):
        # TODO Implement VoxelNet
        rospy.loginfo('unimplemented')


if __name__ == '__main__':
    rospy.init_node('voxelnet', log_level=rospy.INFO)

    voxelnet_node = VoxelNet_Node()

    rospy.loginfo('started voxelnet node')

    rospy.spin()
