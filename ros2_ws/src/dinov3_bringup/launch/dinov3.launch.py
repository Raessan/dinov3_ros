from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.descriptions import ComposableNode
import os

from ament_index_python.packages import get_package_share_directory

# DEFAULT PARAMETERS (change them as arguments if others are needed)
# The real name of the argument is the same as the following but removing "DEFAULT_" and using lowercase
# The filtered map has less points than the original, and the floor/ceiling is eliminated
DEFAULT_DEBUG = "true"
DEFAULT_PERFORM_DETECTION = "true"
DEFAULT_PERFORM_SEGMENTATION = "true"
DEFAULT_PERFORM_DEPTH = "true"
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
                        {'debug': debug},
                        {'use_sim_time': use_sim_time},
                        {'image_reliability': image_reliability},
                        params_file],
            remappings=[('topic_image', topic_image)]
        ),
    ])