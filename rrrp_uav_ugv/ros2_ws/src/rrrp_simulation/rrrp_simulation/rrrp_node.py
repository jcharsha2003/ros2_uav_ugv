#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import heapq

class AStarPlanner:
    def __init__(self, start, goal, obstacles, resolution=1.0, grid_min=-50, grid_max=50, inflation=1.5):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.res = resolution
        self.g_min = grid_min
        self.g_max = grid_max
        self.inflation = inflation

    def is_valid(self, x, y):
        if x < self.g_min or x > self.g_max or y < self.g_min or y > self.g_max:
            return False
        for ox, oy, r in self.obstacles:
            if math.hypot(x - ox, y - oy) <= r + self.inflation:
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
                return self.smooth_path(path)

            for neighbor, cost in self.get_neighbors(current):
                tentative_g = g_score[current] + cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, self.goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def is_line_of_sight_clear(self, p1, p2):
        dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        steps = int(dist / (self.res / 2)) # Check every half-meter
        if steps == 0: return True
        for i in range(1, steps):
            t = i / steps
            x = p1[0] + t*(p2[0]-p1[0])
            y = p1[1] + t*(p2[1]-p1[1])
            if not self.is_valid(x, y):
                return False
        return True

    def smooth_path(self, path):
        if not path: return []
        smoothed = [path[0]]
        current = path[0]
        
        while current != path[-1]:
            # Look as far ahead as possible
            for i in range(len(path)-1, path.index(current), -1):
                target = path[i]
                if self.is_line_of_sight_clear(current, target):
                    smoothed.append(target)
                    current = target
                    break
        return smoothed

class RRRPDecisionNode(Node):
    def __init__(self):
        super().__init__('rrrp_decision_node')
        self.get_logger().info("RRRP Decision Node Initialized.")
        
        # Stochastic Battery Model Parameters
        self.battery_level = 100.0
        self.hover_power = 0.5   # Base discharge rate
        self.flight_power = 1.0  # Extra discharge rate when moving
        self.noise_std_dev = 0.2 # Gaussian noise standard deviation
        
        # RRRP Constraints
        self.risk_tolerance = 0.90 # Require 90% success probability for rendezvous
        self.battery_threshold = 30.0 # Trigger RRRP below 30%
        
        # State
        self.state = "MISSION" # MISSION or RENDEZVOUS
        
        # UGV Pathfinding
        self.ugv_pos = (5.0, 0.0) # Match spawn point
        self.ugv_yaw = 0.0
        self.ugv_target = (-30.0, 28.5) # Target right next to oak_tree_2 trunk (-30, 30)
        
        # Adjust obstacles so A* doesn't block the destination!
        self.obstacles = [
            (35.0, 35.0, 2.0),   # pine_tree_1
            (-35.0, 15.0, 2.0),  # pine_tree_2
            (15.0, -35.0, 2.0),  # oak_tree_1
            # oak_tree_2 removed because it is the target destination
        ]
        
        self.get_logger().info("Computing A* Path for UGV...")
        self.inflation = 1.5
        planner = AStarPlanner(self.ugv_pos, self.ugv_target, self.obstacles)
        self.ugv_path = planner.plan()
        
        if not self.ugv_path:
            self.get_logger().error("A* could not find a path for UGV!")
            self.ugv_waypoint_index = -1
        else:
            self.get_logger().info(f"UGV A* Path computed with {len(self.ugv_path)} waypoints.")
            self.ugv_waypoint_index = 0
        
        # Timer for RHC loop (runs at 10Hz for smoother control)
        self.timer = self.create_timer(0.1, self.rhc_loop)
        
        # Timer for Battery (runs at 1Hz)
        self.battery_timer = self.create_timer(1.0, self.battery_loop)
        
        # Publishers & Subscribers
        self.uav_cmd_pub = self.create_publisher(Twist, '/uav/cmd_vel', 10)
        self.ugv_cmd_pub = self.create_publisher(Twist, '/ugv/cmd_vel', 10)
        self.ugv_odom_sub = self.create_subscription(Odometry, '/ugv/odom', self.ugv_odom_cb, 10)

    def ugv_odom_cb(self, msg):
        self.ugv_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.ugv_yaw = math.atan2(siny_cosp, cosy_cosp)

    def battery_loop(self):
        # Stochastic Update: E_t+1 = E_t - P * dt + w_t
        is_uav_moving = (self.state == "RENDEZVOUS")
        power_draw = self.hover_power
        if is_uav_moving:
            power_draw += self.flight_power
        
        noise = np.random.normal(0, self.noise_std_dev)
        total_draw = power_draw + noise
        
        self.battery_level -= total_draw
        self.battery_level = max(0.0, self.battery_level)
        self.get_logger().info(f"State: {self.state} | UAV Battery: {self.battery_level:.1f}%")

    def build_bipartite_graph(self):
        ugv_future_nodes = [
            {'id': 1, 'time_to_reach': 10.0, 'distance': 50.0},
        ]
        
        edges = []
        for ugv_node in ugv_future_nodes:
            cost = ugv_node['distance'] * 0.5 + ugv_node['time_to_reach']
            expected_draw = (self.hover_power + self.flight_power) * (ugv_node['distance'] / 5.0)
            remaining_after_flight = self.battery_level - expected_draw
            
            if remaining_after_flight > 10.0:
                prob = 0.99
            elif remaining_after_flight > 0.0:
                prob = 0.85
            else:
                prob = 0.10
                
            edges.append({'ugv_node': ugv_node, 'cost': cost, 'prob': prob})
        return edges

    def solve_rrrp(self, edges):
        safe_edges = [e for e in edges if e['prob'] >= self.risk_tolerance]
        if not safe_edges:
            return None
        return min(safe_edges, key=lambda x: x['cost'])

    def rhc_loop(self):
        # 1. Update UGV continuous movement towards oak_tree_1
        ugv_cmd = Twist()
        if self.ugv_pos is not None and getattr(self, 'ugv_path', None):
            if self.ugv_waypoint_index >= 0 and self.ugv_waypoint_index < len(self.ugv_path):
                target_x, target_y = self.ugv_path[self.ugv_waypoint_index]
                dx = target_x - self.ugv_pos[0]
                dy = target_y - self.ugv_pos[1]
                dist = math.hypot(dx, dy)
                
                # Check arrival at waypoint
                if dist < 0.5: # Reduced to 0.5 so it drives right up to the tree!
                    self.ugv_waypoint_index += 1
                    self.get_logger().info(f"UGV reached waypoint {self.ugv_waypoint_index}/{len(self.ugv_path)}")
                else:
                    target_yaw = math.atan2(dy, dx)
                    err_yaw = target_yaw - self.ugv_yaw
                    while err_yaw > math.pi: err_yaw -= 2*math.pi
                    while err_yaw < -math.pi: err_yaw += 2*math.pi
                    
                    # --- THE FIX: Regulated Pure Pursuit ---
                    # Extremely high speeds with sharp exponential turns cause the UGV to lose traction
                    # and fishtail (biasing left/right). We must use gentle, regulated steering!
                    
                    if abs(err_yaw) > 0.5: # If off by more than 28 degrees
                        # Turn tightly to get on track
                        ugv_cmd.angular.z = max(-1.0, min(1.0, err_yaw * 1.5))
                        ugv_cmd.linear.x = 0.5 # Creep forward to prevent friction lock
                    else:
                        # Smooth, gentle adjustments when driving fast
                        angular = err_yaw * 1.2
                        ugv_cmd.angular.z = max(-0.5, min(0.5, angular)) 
                        
                        # Smoothly reduce speed slightly when micro-correcting to prevent wobble
                        max_speed = 3.5 # Fast, but stable
                        ugv_cmd.linear.x = max(1.5, max_speed - abs(ugv_cmd.angular.z) * 2.0)
            else:
                # Finished path
                ugv_cmd.linear.x = 0.0
                ugv_cmd.angular.z = 0.0
        self.ugv_cmd_pub.publish(ugv_cmd)
        
        # 3. Decision Logic
        if self.state == "MISSION":
            if self.battery_level < self.battery_threshold:
                self.get_logger().warn(f"Battery below threshold ({self.battery_threshold}%)! Triggering RRRP...")
                edges = self.build_bipartite_graph()
                assignment = self.solve_rrrp(edges)
                
                if assignment:
                    self.get_logger().info(f"Assigned to UGV Node | Prob: {assignment['prob']}")
                    self.state = "RENDEZVOUS"
        
        elif self.state == "RENDEZVOUS":
            if self.battery_level < 15.0: # Mock trigger for reaching UGV
                self.get_logger().info("UAV has landed on UGV. Recharging...")
                self.battery_level = 100.0
                self.state = "MISSION"

def main(args=None):
    rclpy.init(args=args)
    node = RRRPDecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
