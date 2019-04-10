import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
import timeit

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
    """Search methods for discrete motion planner
    """
    def __init__(self, world_state, robot_pose, goal_pose, obs_dis):
        self.world_state = world_state
        self.robot_pose = robot_pose
        self.goal_pose = goal_pose
        self.obs_dis = obs_dis
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        
        self.frontier = PriorityQueue()     # Exploring queue
        self.cost = {}                      # Record nodes and their costs from start pose
        self.parent = {}                    # Record visitted nodes and their parents
        
        self.frontier.push(robot_pose, 0)
        self.cost[robot_pose] = 0
        self.parent[robot_pose] = None

    def cal_heuristic(self, node):
        # Calculate distance between node and goal_pose
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
        # Check whether node is in collision with obstacle or stay too near to obstacle
        n = self.obs_dis
        xr = self.x_range
        yr = self.y_range
        if n == 0 and self.world_state[x][y]:
            return True
        else:
            for i in range(-n, n+1):
                for j in range(-n, n+1):
                    x_new = x + i
                    y_new = y + j
                    if x_new in range(xr) and y_new in range(yr) and self.world_state[x_new][y_new]:
                        return True
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


def generate_graph(x_range, y_range):
    # Generate world graph with obstacles 
    world_state = [[0] * y_range for i in range(x_range)]
    for i in range(y_range):
        world_state[0][i] = 1

    for i in range(x_range):
        world_state[i][0] = 1

    for i in range(y_range):
        world_state[-1][i] = 1

    for i in range(x_range):
        world_state[i][-1] = 1

    for i in range(y_range // 5 * 4):
        world_state[x_range // 10 * 4 - 1][i] = 1

    for i in range(y_range // 5 , y_range):
        world_state[x_range // 10 * 6 - 1][i] = 1
     
    return world_state


def generate_robot_goal(x_range, y_range, world_state):
    # Generate robot pose and goal pose
    i, j = 0, 0
    p, q = 0, 0
    
    while world_state[i][j] or world_state[p][q] or (i, j) == (p, q):
        i, j = np.random.randint(x_range), np.random.randint(x_range)
        p, q = np.random.randint(y_range), np.random.randint(y_range)
    
    return (i, j), (p, q)

    
def show_result(op_path, world_state, robot_pose, goal_pose):
    # Plot to show result if we found a path
    OPX = []            
    OPY = []
    WX = []             
    WY = []
    OX = []             
    OY = []
    SX = robot_pose[0]  
    SY = robot_pose[1]
    GX = goal_pose[0]   
    GY = goal_pose[1]
    
    if op_path:
        for node in op_path:
            OPX.append(node[0])
            OPY.append(node[1])

    for i in range(len(world_state)):
        for j in range(len(world_state[0])):
            if world_state[i][j] == 0:
                WX.append(i)
                WY.append(j)
            else:
                OX.append(i)
                OY.append(j)

    xr = len(world_state)
    yr = len(world_state[0])
    plt.figure(figsize = (8 * (xr // yr), 8))
    plt.plot(WX, WY, "w.", label = "World")
    plt.plot(OX, OY, "ko", label = "Obstacle", markersize = 5)
    plt.plot(SX, SY, "bo", label = "Robot_Pose", markersize = 8)
    plt.plot(GX, GY, "ro", label = "Goal_Pose", markersize = 8)
    
    if op_path:
        plt.plot(OPX, OPY, alpha = 0.9, label = "Optimal_Path", linewidth = 3)
        
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 6)
    plt.show()
    plt.close()

def main():
    # Parameter initialization
    xr = 50
    yr = 50 
    obs_dis = 4   # obs_dis is the min distance between robot and obstacle for safety consideration
    #obs_dis = 0  # If obs_dis = 0, we will get the optimal cost path, but it will also stay as near as possbile to the obstacle
    world_state = generate_graph(xr, yr)
    #robot_pose, goal_pose = generate_robot_goal(xr, yr, world_state)
    robot_pose = (8 , 8) 
    goal_pose = (xr - 8, yr - 8)
    
    # Run optimal planner
    search = Search(world_state, robot_pose, goal_pose, obs_dis)
    optimal_path = search.A_star()
    if optimal_path:
        print("Optimal search succeed!")
    else:
        print("No optimal path is found!")
        
    show_result(optimal_path, world_state, robot_pose, goal_pose)

if __name__ == '__main__':
    main()
