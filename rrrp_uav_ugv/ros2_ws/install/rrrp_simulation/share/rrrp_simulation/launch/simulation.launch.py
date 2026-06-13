import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():
    pkg_dir = get_package_share_directory('rrrp_simulation')
    world_file = os.path.join(pkg_dir, 'worlds', 'arena_100x100.world')
    
    # Export the resource path so Gazebo can find local models in ~/.gazebo/models
    # This is crucial for pine_tree and oak_tree to load correctly if present locally
    os.environ['IGN_GAZEBO_RESOURCE_PATH'] = os.path.expanduser('~/.gazebo/models')

    from launch_ros.actions import Node

    # Force OGRE engine due to Intel HD Graphics 630 crashing with Ogre2
    return LaunchDescription([
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', world_file, '--render-engine', 'ogre'],
            output='screen'
        ),
        
        # Bridge to send commands to UGV and UAV, and receive UAV odometry
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/model/ugv_1/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                '/model/ugv_1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                '/model/uav_1/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                '/model/uav_1/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
                '/model/uav_1/joint/X3/rotor_0_joint/cmd_vel@std_msgs/msg/Float64]ignition.msgs.Double',
                '/model/uav_1/joint/X3/rotor_1_joint/cmd_vel@std_msgs/msg/Float64]ignition.msgs.Double',
                '/model/uav_1/joint/X3/rotor_2_joint/cmd_vel@std_msgs/msg/Float64]ignition.msgs.Double',
                '/model/uav_1/joint/X3/rotor_3_joint/cmd_vel@std_msgs/msg/Float64]ignition.msgs.Double',
            ],
            remappings=[
                ('/model/ugv_1/cmd_vel', '/ugv/cmd_vel'),
                ('/model/ugv_1/odometry', '/ugv/odom'),
                ('/model/uav_1/cmd_vel', '/uav/cmd_vel'),
                ('/model/uav_1/odometry', '/uav/odom'),
            ],
            output='screen'
        ),
        
        # Our RRRP Decision Node
        Node(
            package='rrrp_simulation',
            executable='rrrp_node',
            name='rrrp_decision_node',
            output='screen'
        ),

        # A* UAV Flight Controller
        Node(
            package='rrrp_simulation',
            executable='uav_a_star_flyer',
            name='uav_a_star_flyer_node',
            output='screen'
        )
    ])
