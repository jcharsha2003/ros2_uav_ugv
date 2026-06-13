#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import math
import heapq
import time

class AStarPlanner:
    def __init__(self, start, goal, obstacles, resolution=1.0, grid_min=-50, grid_max=50):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles # List of (x, y, radius)
        self.res = resolution
        self.g_min = grid_min
        self.g_max = grid_max

    def is_valid(self, x, y):
        if x < self.g_min or x > self.g_max or y < self.g_min or y > self.g_max:
            return False
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r:
                return False
        return True

    def heuristic(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = node[0] + dx * self.res, node[1] + dy * self.res
            # Rounding to exactly 1 decimal place prevents floating point state explosion!
            nx, ny = round(nx, 1), round(ny, 1)
            if self.is_valid(nx, ny):
                cost = math.hypot(dx, dy)
                neighbors.append(((nx, ny), cost))
        return neighbors

    def plan(self):
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if math.hypot(current[0] - self.goal[0], current[1] - self.goal[1]) < self.res:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                path.reverse()
                return path

            for neighbor, cost in self.get_neighbors(current):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

class RealisticPropellerPhysics:
    def __init__(self, base_hover_rpm=150.0):
        self.HOVER_RPM = base_hover_rpm
        self.MAX_RPM = 250.0
        self.current_rpm = 0.0
        self.target_rpm = 0.0
        
    def update_physics(self, dt, vertical_velocity_cmd, is_landed, is_flying=False):
        if is_landed:
            self.target_rpm = 0.0
        elif is_flying:
            self.target_rpm = 200.0
        else:
            # Hover or descend
            rpm_offset = vertical_velocity_cmd * 100.0
            self.target_rpm = self.HOVER_RPM + rpm_offset
            self.target_rpm = max(50.0, min(self.target_rpm, self.MAX_RPM))

        inertia_factor = 2.0
        self.current_rpm += (self.target_rpm - self.current_rpm) * inertia_factor * dt
        return self.current_rpm

class UAVAStarFlyer(Node):
    def __init__(self):
        super().__init__('uav_a_star_flyer_node')
        
        self.propeller_physics = RealisticPropellerPhysics()
        self.get_logger().info("UAV A* Flight Controller Initialized.")

        # Publisher for flight control
        self.cmd_pub = self.create_publisher(Twist, '/uav/cmd_vel', 10)
        
        # Flight Controller PID and Limits
        self.Kp_z = 1.5
        self.Ki_z = 0.05
        self.Kd_z = 0.8
        self.MAX_HORIZONTAL_VEL = 5.0
        self.MAX_VERTICAL_VEL = 2.5
        
        self.prev_error_z = 0
        self.integral_z = 0
        self.last_time = time.time()
        
        # Landing State Machine
        self.STATE_FLYING = 0
        self.STATE_HOVER_ABOVE_TARGET = 1
        self.STATE_DESCENDING = 2
        self.STATE_LANDED = 3
        self.current_state = self.STATE_FLYING
        self.landing_precision_radius = 0.3
        
        # Publishers for rotor visual spin
        self.rotor_pubs = [
            self.create_publisher(Float64, f'/model/uav_1/joint/X3/rotor_{i}_joint/cmd_vel', 10)
            for i in range(4)
        ]
        
        # Subscribe to Odometry
        # Using a reliable QoS profile for Gazebo Odometry
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.odom_sub = self.create_subscription(Odometry, '/uav/odom', self.odom_callback, qos)

        self.current_pos = None
        self.current_yaw = 0.0

        # Define map knowledge
        self.start_pos = (0.0, 0.0)
        self.target_pos = (31.0, 31.0) # Safe distance from pine_tree_1 (obstacle at 35,35 with radius 3)
        self.obstacles = [
            (35.0, 35.0, 3.0),   # pine_tree_1
            (-35.0, 15.0, 3.0),  # pine_tree_2
            (15.0, -35.0, 3.0),  # oak_tree_1
            (-30.0, 30.0, 3.0)   # oak_tree_2
        ]

        # Compute A* Path
        self.get_logger().info("Computing A* Path...")
        planner = AStarPlanner(self.start_pos, self.target_pos, self.obstacles)
        self.path = planner.plan()
        
        if not self.path:
            self.get_logger().error("A* could not find a path!")
            self.waypoint_index = -1
        else:
            self.get_logger().info(f"A* Path computed with {len(self.path)} waypoints.")
            self.waypoint_index = 0

        self.timer = self.create_timer(0.1, self.control_loop)

    def euler_from_quaternion(self, q):
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(t3, t4)

    def odom_callback(self, msg):
        self.current_pos = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        )
        q = msg.pose.pose.orientation
        self.current_yaw = self.euler_from_quaternion(q)

    def publish_rotors(self, speed):
        """Publish spin speed to all 4 rotors."""
        msg = Float64()
        # Alternate rotation directions for visual accuracy
        for i, pub in enumerate(self.rotor_pubs):
            if i % 2 == 0:
                msg.data = float(speed)
            else:
                msg.data = float(-speed)
            pub.publish(msg)

    def control_loop(self):
        if self.current_pos is None:
            self.current_pos = (self.start_pos[0], self.start_pos[1], 0.5) # Kickstart if odom is delayed
            
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0: dt = 0.01
        self.last_time = current_time
            
        msg = Twist()
        is_landed = False
        is_flying = False
            
        if self.waypoint_index >= len(self.path) and len(self.path) > 0:
            target_x, target_y = self.path[-1]
            curr_x, curr_y, curr_z = self.current_pos
            
            if self.current_state == self.STATE_FLYING:
                horizontal_dist = math.hypot(curr_x - target_x, curr_y - target_y)
                if horizontal_dist < self.landing_precision_radius:
                    self.current_state = self.STATE_HOVER_ABOVE_TARGET
                    self.get_logger().info("Target tree cleared horizontally. Initializing hover.", once=True)
                else:
                    # Final correction to center over target
                    yaw_err = math.atan2(target_y - curr_y, target_x - curr_x) - self.current_yaw
                    while yaw_err > math.pi: yaw_err -= 2 * math.pi
                    while yaw_err < -math.pi: yaw_err += 2 * math.pi
                    msg.angular.z = max(-1.5, min(1.5, 2.0 * yaw_err))
                    msg.linear.x = max(0.0, min(1.0, horizontal_dist))
                    
                    # Maintain altitude during final alignment
                    error_z = 5.0 - curr_z
                    self.integral_z += error_z * dt
                    derivative_z = (error_z - self.prev_error_z) / dt
                    vel_z = (self.Kp_z * error_z) + (self.Ki_z * self.integral_z) + (self.Kd_z * derivative_z)
                    msg.linear.z = max(min(vel_z, self.MAX_VERTICAL_VEL), -self.MAX_VERTICAL_VEL)
                    self.prev_error_z = error_z
            
            elif self.current_state == self.STATE_HOVER_ABOVE_TARGET:
                self.current_state = self.STATE_DESCENDING
                self.get_logger().info("Settle window complete. Commencing vertical downforce landing.", once=True)
                
            elif self.current_state == self.STATE_DESCENDING:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = -0.5
                
                if self.current_pos[2] <= 0.2:
                    self.current_state = self.STATE_LANDED
                    self.get_logger().info("Ground touch detected. Killing propeller rotations.", once=True)
            
            elif self.current_state == self.STATE_LANDED:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                is_landed = True
                
        else:
            # Still flying to waypoints
            is_flying = True
            target_x, target_y = self.path[self.waypoint_index]
            curr_x, curr_y, curr_z = self.current_pos

            dx = target_x - curr_x
            dy = target_y - curr_y
            distance = math.hypot(dx, dy)

            if distance < 1.0: # Waypoint reached threshold
                self.waypoint_index += 1
                self.get_logger().info(f"Reached waypoint {self.waypoint_index}/{len(self.path)}")
            else:
                target_yaw = math.atan2(dy, dx)
                yaw_error = target_yaw - self.current_yaw

                # Normalize yaw error
                while yaw_error > math.pi: yaw_error -= 2 * math.pi
                while yaw_error < -math.pi: yaw_error += 2 * math.pi

                # Smooth Flight Controller Logic
                error_z = 5.0 - curr_z
                
                # 1. Vertical axis dampening (Fixes Shaking)
                self.integral_z += error_z * dt
                derivative_z = (error_z - self.prev_error_z) / dt
                
                vel_z = (self.Kp_z * error_z) + (self.Ki_z * self.integral_z) + (self.Kd_z * derivative_z)
                msg.linear.z = max(min(vel_z, self.MAX_VERTICAL_VEL), -self.MAX_VERTICAL_VEL)
                
                # 2. Horizontal speed limits (Fixes Slow Movement)
                speed_2d = distance * 1.2
                if speed_2d > self.MAX_HORIZONTAL_VEL:
                    speed_2d = self.MAX_HORIZONTAL_VEL
                    
                Kp_yaw = 2.0
                msg.angular.z = max(-1.5, min(1.5, Kp_yaw * yaw_error))
                
                forward_effort = math.cos(yaw_error)
                if forward_effort > 0.0:
                    msg.linear.x = speed_2d * forward_effort
                else:
                    msg.linear.x = 0.0
                    
                self.prev_error_z = error_z

        # Apply realistic propeller physics
        current_rpm = self.propeller_physics.update_physics(dt, msg.linear.z, is_landed, is_flying)
        self.publish_rotors(current_rpm)
        
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = UAVAStarFlyer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
