import numpy as np
import math
import random 
import matplotlib.pyplot as plt

class Node():
    """ RRT node 
    """
    def __init__(self, x, y, parent = None, cost = 0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost
        self.tuple = (x, y)

class Search(object):
    """Search methods for discrete motion planner
    """
    def __init__(self, world_state, robot_pose, goal_pose, expand_length, iteration, obs_list, robot_size):
        self.world_state = world_state
        self.robot_pose = Node(robot_pose[0], robot_pose[1])
        self.goal_pose = Node(goal_pose[0], goal_pose[1])
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        
        self.expand_length = expand_length     # Length of tree expansion in every step
        self.iteration = iteration             # Maximum iterations to search for path
        self.obs_list = obs_list               # Obstacles 
        self.robot_size = robot_size           # Size of our robot
        self.tree = []                          
        self.tree.append(self.robot_pose)
        self.store = {}                        # Store distance calculation and collision checking
    
    def random_sample(self):
        # Randomly sample new node
        rand_node = Node(random.uniform(0, self.x_range), random.uniform(0, self.y_range))
        return rand_node
    
    def cal_dis(self, node_1, node_2):
        # Calculate distance between two nodes
        return math.sqrt((node_1.x - node_2.x)**2 + (node_1.y - node_2.y)**2)
    
    def nearest(self, n_rand):
        # Find the index of node in tree that is nearest to the randomly sampled node
        min_dis = float('inf')
        min_index = None
        for i, node in enumerate(self.tree):
            dis = self.cal_dis(n_rand, node)
            if dis < min_dis:
                min_index = i
                min_dis = dis      
        return min_index
    
    def steer(self, nearest_index, n_nearest, n_rand):
        # Steer to get new node 
        theta = math.atan2(n_rand.y - n_nearest.y, n_rand.x - n_nearest.x)
        new_x = n_nearest.x + math.cos(theta) * self.expand_length
        new_y = n_nearest.y + math.sin(theta) * self.expand_length
        x_new = Node(new_x, new_y, nearest_index)
        return x_new
    
    def check_collision(self, n_new):
        # Check if node get collision with obstacles
        for obs in self.obs_list:
            if self.obs_check(n_new, obs):
                return True
        return False
    
    def obs_check(self, n_new, obs):
        # Check if node get collision with obstacle, take into consideration about robot size
        if n_new.x >= obs[0][0] - self.robot_size and n_new.x < obs[0][1] + self.robot_size and n_new.y >= obs[1][0] - self.robot_size and n_new.y < obs[1][1] + self.robot_size:
            return True
        else:
            return False
    
    def reach_goal(self, n_new):
        # Check whether new node could reach goal pose or not
        if self.cal_dis(n_new, self.goal_pose) <= self.expand_length:
            return True
        else:
            return False
    
    def generate_path(self, node):
        # generate path from robot pose to goal pose
        path = []
        path.append(self.goal_pose.tuple)
        path.append(node.tuple)
        node = self.tree[node.parent]
        while node != self.robot_pose:
            path.append(node.tuple)
            node = self.tree[node.parent]
        path.append(self.robot_pose.tuple)
        return path[::-1]
    
    def RRT(self):
        # Sub-optimal planner achieved by RRT
        path = []
        for i in range(self.iteration):
            n_rand = self.random_sample()               
            nearest_index = self.nearest(n_rand)
            n_nearest = self.tree[nearest_index]
            n_new = self.steer(nearest_index, n_nearest, n_rand)
            
            if not self.check_collision(n_new):
                self.tree.append(n_new)
            else:
                continue
            
            if self.reach_goal(n_new):
                path = self.generate_path(n_new)
                break
                
        return path
    
    def find_nears(self, n_new):
        # Find nodes in tree that are near to new node
        nears_index = []
        n = len(self.tree)
        r = 50.0 * math.sqrt((math.log(n + 1) / n))
        for i, node in enumerate(self.tree):
            if self.cal_dis(n_new, node) <= r:
                nears_index.append(i)
        return nears_index
    
    def choose_best_parent(self, n_new, nears_index):
        # Choose the best node as parent of new node to achieve min cost
        min_cost = float('inf')
        min_index = None
        for i in nears_index:
            dis = self.cal_dis(n_new, self.tree[i])
            cost = self.tree[i].cost + dis
            collision = self.check_collision_extend(n_new, self.tree[i], dis)
            self.store[i] = (dis, collision)
            if not collision and cost < min_cost:
                min_cost = cost
                min_index = i
        n_new.cost = min_cost
        n_new.parent = min_index
        return n_new
    
    def check_collision_extend(self, node_1, node_2, d):
        # Check whether connecting line of two nodes is in collision with obstacles or not
        theta = math.atan2(node_1.y - node_2.y, node_1.x - node_2.x)
        for i in range(1, int(d / self.expand_length)):
            x = node_1.x + self.expand_length * i * math.cos(theta)
            y = node_1.y + self.expand_length * i * math.sin(theta)
            if self.check_collision(Node(x, y)):
                return True
        return False
    
    def rewire(self, n_new, nears_index):
        # Rewire the nodes which are near to new node, in which nodes could be rewired if their cost could be reduced by connecting with new node as their parents
        n = len(self.tree)
        for i in nears_index:
            new_cost = n_new.cost + self.store[i][0]
            if i != n_new.parent and new_cost < self.tree[i].cost and not self.store[i][1]:
                self.tree[i].parent = n - 1
                self.tree[i].cost = new_cost
                
    def get_best_index(self):
        # Get node in current tree with least distance to goal pose
        best_dis = float('inf')
        best_index = None
        for i, node in enumerate(self.tree):
            dis = self.cal_dis(node, self.goal_pose)
            if dis < best_dis:
                best_dis = dis
                best_index = i
        # If this distance is less or equal to expand length, we found the path successfully
        if best_dis <= self.expand_length:
            return best_index
        else:
            return None
    
    def RRT_star(self):
        # Optimal planner achieved by RRT*
        path = []
        for i in range(self.iteration):
            n_rand = self.random_sample()               
            nearest_index = self.nearest(n_rand)
            n_nearest = self.tree[nearest_index]
            n_new = self.steer(nearest_index, n_nearest, n_rand)
            
            if not self.check_collision(n_new):
                nears_index = self.find_nears(n_new)
                if not nears_index:
                    continue
                n_new = self.choose_best_parent(n_new, nears_index)
                self.tree.append(n_new)
                self.rewire(n_new, nears_index)
            
        best_last_index = self.get_best_index()
        if best_last_index:
            path = self.generate_path(self.tree[best_last_index])
        return path
    
def generate_obstacle(xr, yr):
    # Obstacle generation
    obs_list = [[(0, xr), (0, 1)], 
                [(0, xr), (yr - 1, yr)], 
                [(0, 1), (0, yr)], 
                [(xr - 1, xr), (0, yr)]]
    obs_list.append([(15, 16), (0, 30)])
    obs_list.append([(35, 36), (20, 50)])
    return obs_list

def generate_graph(xr, yr, obs_list):
    # Generate world graph with obstacles 
    ws = [[0] * yr for i in range(xr)]
    for obs in obs_list:
        for i in range(obs[0][0], obs[0][1]):
            for j in range(obs[1][0], obs[1][1]):
                ws[i][j] = 1
    return ws

def show_result(op_path, so_path, world_state, robot_pose, goal_pose, obs_list):
    # Plot to show result if we found a path
    OPX = []            
    OPY = []
    SPX = []
    SPY = []
    WX = []            
    WY = []
    SX = robot_pose[0]  
    SY = robot_pose[1]
    GX = goal_pose[0]   
    GY = goal_pose[1]
    
    if op_path:
        for node in op_path:
            OPX.append(node[0])
            OPY.append(node[1])
    
    if so_path:
        for node in so_path:
            SPX.append(node[0])
            SPY.append(node[1])

    xr = len(world_state)
    yr = len(world_state[0])
    plt.figure(figsize=(8*(xr // yr), 8))
    plt.plot(SX, SY, "bo", label = "Robot_Pose", markersize = 8)
    plt.plot(GX, GY, "ro", label = "Goal_Pose", markersize = 8)
    
    for obs in obs_list:
        OX = []
        OY = []
        if obs[0][1] - obs[0][0] == 1:
            for i in range(obs[1][0], obs[1][1]):
                OX.append(obs[0][0])
                OY.append(i)
        else:
            for i in range(obs[0][0], obs[0][1]):
                OX.append(i)
                OY.append(obs[1][0])
        plt.plot(OX, OY, "k", linewidth = 5)
    
    if op_path:
        plt.plot(OPX, OPY, alpha = 0.9, label = "Optimal_Path", linewidth = 3)
        
    if so_path:
        plt.plot(SPX, SPY, alpha = 0.9, label = "Sub-optimal_Path", linewidth = 3)
    
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 6)
    plt.show()
    plt.close()

def main():
    x_range = 50
    y_range = 50
    robot_size = 3
    expand_length = 0.5
    iteration = 10000
    obs_list = generate_obstacle(x_range, y_range)
    world_state = generate_graph(x_range, y_range, obs_list)
    robot_pose, goal_pose = (5, 5),  (x_range - 5, y_range - 5)
    
    # Implement optimal planner
    search = Search(world_state, robot_pose, goal_pose, expand_length, iteration, obs_list, robot_size)
    optimal_path = search.RRT_star()
    
    if optimal_path:
        print("Optimal search succeed!")
    else:
        print("No optimal path is found!")
        
    # Implement sub-optimal planner
    search = Search(world_state, robot_pose, goal_pose, expand_length, iteration, obs_list, robot_size)
    sub_path = search.RRT()

    if sub_path:
        print("Sub-optimal search succeed!")
    else:
        print("No sub-optimal path is found!")
        
    # Plot result for comparison
    show_result(optimal_path, sub_path, world_state, robot_pose, goal_pose, obs_list)
            
if __name__ == '__main__':
    main()
