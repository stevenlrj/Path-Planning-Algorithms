import numpy as np
import heapq
import math
import matplotlib.pyplot as plt

class PriorityQueue(object):
    """Priority queue implemented by min heap used to hold nodes in exploring queue, which could achieve extracting node 
    with least priority and inserting new node in O(log(n)) of time complexity, where n is the number of nodes in queue
    """
    def __init__(self):
        self.queue = []

    def push(self, node, cost):
        heapq.heappush(self.queue, (cost, node)) 

    def pop(self):
        return heapq.heappop(self.queue)[1]

    def empty(self):
        return len(self.queue) == 0

class Search(object):
    """Search methods for path planner
    """
    def __init__(self, world_state, robot_pose, goal_pose, obs_list, robot_size):
        self.world_state = world_state
        self.robot_pose = robot_pose
        self.goal_pose = goal_pose
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        
        self.robot_size = robot_size        # Robot size
        self.obs_list = obs_list            # Obstacles
        self.frontier = PriorityQueue()     # Exploring queue
        self.cost = {}                      # Record nodes and their costs from start pose
        self.parent = {}                    # Record visitted nodes and their parents
        
        self.frontier.push(robot_pose, 0)
        self.cost[robot_pose] = 0
        self.parent[robot_pose] = None
        
    def A_star(self):
        # Optimal planner achieved by A* Algorithm
        while not self.frontier.empty():
            # Extract and visit nodes with least priority
            cur = self.frontier.pop()   
            
            # If we reach goal pose, track back to get path 
            if cur == self.goal_pose:            
                return self.generate_path(cur)
            
            # Get possible next step movements of current node
            motions = self.get_robot_motion(cur)
            for motion in motions:
                node = motion[0]
                cost = motion[1]
                new_cost = self.cost[cur] + cost
                # No need to explore node that has been visited or its cost doesn't need to be updated
                if node not in self.parent or new_cost < self.cost[node]:
                    self.cost[node] = new_cost
                    priority = new_cost + self.cal_heuristic(node)
                    self.frontier.push(node, priority)
                    self.parent[node] = cur
                    
        return None

    def cal_heuristic(self, node):
        # Calculate distance between node and goal_pose as heuristic
        return (node[0] - self.goal_pose[0])**2 + (node[1] - self.goal_pose[1])**2
    
    def get_robot_motion(self, node):
        # Robot motion model
        xr = self.x_range
        yr = self.y_range
        next_step = []
        robot_motion = [[(1, 0), 1], [(0, 1), 1], [(-1, 0), 1], [(0, -1), 1], 
                       [(-1, -1), math.sqrt(2)], [(-1, 1), math.sqrt(2)],
                       [(1, -1), math.sqrt(2)], [(1, 1), math.sqrt(2)]]
        for motion in robot_motion:
            x = node[0] + motion[0][0]
            y = node[1] + motion[0][1]
            if x in range(xr) and y in range(yr) and not self.check_collision(x, y):
                next_step.append([(x, y), motion[1]])
        return next_step
    
    def check_collision(self, x, y):
        # Check if node get collision with obstacles
        for obs in self.obs_list:
            if self.obs_check(x, y, obs):
                return True
        return False
    
    def obs_check(self, x, y, obs):
        # Check if node get collision with obstacle, take into consideration about robot size
        if x >= obs[0][0] - self.robot_size and x < obs[0][1] + self.robot_size and y >= obs[1][0] - self.robot_size and y < obs[1][1] + self.robot_size:
            return True
        else:
            return False
    
    def generate_path(self, goal):
        # Track back to get path from robot pose to goal pose
        path = [goal]
        node = self.parent[goal]
        while node != self.robot_pose:
            path.append(node)
            node = self.parent[node]
        path.append(self.robot_pose)
        return path[::-1]
        
        
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

def show_result(op_path, world_state, robot_pose, goal_pose, obs_list):
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
    
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 6)
    plt.show()
    plt.close()

def main():
    # Parameter initialization
    x_range = 50
    y_range = 50
    robot_size = 3
    obs_list = generate_obstacle(x_range, y_range)
    world_state = generate_graph(x_range, y_range, obs_list)
    robot_pose, goal_pose = (5, 5),  (x_range - 5, y_range - 5)
    
    # Run optimal planner
    search = Search(world_state, robot_pose, goal_pose, obs_list, robot_size)
    optimal_path = search.A_star()
    if optimal_path:
        print("Optimal search succeed!")
    else:
        print("No optimal path is found!")
        
    show_result(optimal_path, world_state, robot_pose, goal_pose, obs_list)

if __name__ == '__main__':
    main()
