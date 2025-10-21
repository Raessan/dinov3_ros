from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

# DEFAULT PARAMETERS (change them as arguments if others are needed)
# The real name of the argument is the same as the following but removing "DEFAULT_" and using lowercase
DEFAULT_DEBUG = "true"
DEFAULT_PERFORM_DETECTION = "true"
DEFAULT_PERFORM_SEGMENTATION = "true"
DEFAULT_PERFORM_DEPTH = "true"
DEFAULT_PERFORM_OPTICAL_FLOW = "true"
DEFAULT_PARAMS_FILE =  os.path.join(get_package_share_directory("dinov3_bringup"), "config", "params.yaml")
DEFAULT_TOPIC_IMAGE = "topic_image"
DEFAULT_IMAGE_RELIABILITY = "2" # 0 corresponds to "SYSTEM_DEFAULT", 1 corresponds to "RELIABLE", 2 corresponds to "BEST_EFFORT"

def generate_launch_description():
    # Launch configurations (associated to launch arguments)
    debug = LaunchConfiguration('debug')
    debug_arg = DeclareLaunchArgument('debug', default_value=DEFAULT_DEBUG)

    perform_detection = LaunchConfiguration('perform_detection')
    perform_detection_arg = DeclareLaunchArgument('perform_detection', default_value=DEFAULT_PERFORM_DETECTION)

    perform_segmentation = LaunchConfiguration('perform_segmentation')
    perform_segmentation_arg = DeclareLaunchArgument('perform_segmentation', default_value=DEFAULT_PERFORM_SEGMENTATION)

    perform_depth = LaunchConfiguration('perform_depth')
    perform_depth_arg = DeclareLaunchArgument('perform_depth', default_value=DEFAULT_PERFORM_DEPTH)

    perform_optical_flow = LaunchConfiguration('perform_optical_flow')
    perform_optical_flow_arg = DeclareLaunchArgument('perform_optical_flow', default_value=DEFAULT_PERFORM_OPTICAL_FLOW)

    params_file = LaunchConfiguration('params_file')
    params_file_arg = DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS_FILE)

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false')

    topic_image = LaunchConfiguration('topic_image')
    topic_image_arg = DeclareLaunchArgument('topic_image', default_value=DEFAULT_TOPIC_IMAGE)

    image_reliability = LaunchConfiguration('image_reliability')
    image_reliability_arg = DeclareLaunchArgument('image_reliability', default_value=DEFAULT_IMAGE_RELIABILITY)

    return LaunchDescription([
        perform_detection_arg,
        perform_segmentation_arg,
        perform_depth_arg,
        perform_optical_flow_arg,
        params_file_arg,
        debug_arg,
        use_sim_time_arg,
        topic_image_arg,
        image_reliability_arg,
        Node(
            package='dinov3_ros',
            executable="dinov3_node",
            name="dinov3_node",
            output='screen',
            emulate_tty=True,
            parameters=[{'perform_detection': perform_detection},
                        {'perform_segmentation': perform_segmentation},
                        {'perform_depth': perform_depth},
                        {'perform_optical_flow': perform_optical_flow},
                        {'debug': debug},
                        {'use_sim_time': use_sim_time},
                        {'image_reliability': image_reliability},
                        params_file],
            remappings=[('topic_image', topic_image)]
        ),
    ])