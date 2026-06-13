import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/iiitd/Desktop/ros2/BATTERY/rrrp_uav_ugv/ros2_ws/install/rrrp_simulation'
