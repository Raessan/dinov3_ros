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
DEFAULT_USE_OBJECT_DETECTION = "true"
DEFAULT_PARAMS_FILE =  os.path.join(get_package_share_directory("dinov3_bringup"), "config", "params.yaml")


def generate_launch_description():
    # Launch configurations (associated to launch arguments)
    debug = LaunchConfiguration('debug')
    debug_arg = DeclareLaunchArgument('debug', default_value=DEFAULT_DEBUG)

    use_object_detection = LaunchConfiguration('use_object_detection')
    use_object_detection_arg = DeclareLaunchArgument('use_object_detection', default_value=DEFAULT_USE_OBJECT_DETECTION)

    params_file = LaunchConfiguration('params_file')
    params_file_arg = DeclareLaunchArgument('params_file', default_value=DEFAULT_PARAMS_FILE)

    return LaunchDescription([
        use_object_detection_arg,
        params_file_arg,
        debug_arg,
        Node(
            package='dinov3_ros',
            executable="dinov3_node",
            name="dinov3_node",
            output='screen',
            emulate_tty=True,
            #arguments=[use_object_detection],
            parameters=[{'use_object_detection': use_object_detection},
                        {'debug': debug},
                        params_file],
        ),
    ])