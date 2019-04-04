import numpy as np
import heapq
import matplotlib.pyplot as plt
import timeit

class PriorityQueue(object):
    """A priority queue achieved by min heap used to hold nodes in exploring queue, which could achieve extracking node 
    with least priority in O(log(n)) of time complexity, where n is the number of nodes in queue.
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
    """Search methods for discrete motion planner, optimal planner achieved by A* algorithm
    """
    
    def __init__(self, world_state, robot_pose, goal_pose):
    	# Initializa variables and data container
        self.world_state = world_state
        self.robot_pose = robot_pose
        self.goal_pose = goal_pose
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        
        self.frontier = PriorityQueue()   	# Hold and extract node with min priority
        self.cost = {}              		# Hold and record nodes and their distances from start pose
        self.parent = {}            		# Hold nodes that has already been visited and their parent nodes
        
        self.frontier.push(robot_pose, 0)
        self.cost[robot_pose] = 0
        self.parent[robot_pose] = None
   
        
    def cal_heuristic(self, node):
        # Calculate Manhatten distance between node and goal_pose
        return abs(node[0] - self.goal_pose[0]) + abs(node[1] - self.goal_pose[1])
    
    def find_neighbor(self, node):
        # Find possbile neightbor nodes
        xr = self.x_range
        yr = self.y_range
        NS = []
        neighbors = [(node[0] + 1, node[1]), (node[0] - 1, node[1]), (node[0], node[1] + 1), (node[0], node[1] - 1)]
        for neighbor in neighbors:
            if (neighbor[0] in range(xr)) and (neighbor[1] in range(yr)) and not self.world_state[neighbor[0]][neighbor[1]]:
                NS.append(neighbor)
        return NS
    
    def generate_path(self, goal):
        # Track back to get path from robot pose to goal pose
        path = [goal]
        node = self.parent[goal]
        while node != self.robot_pose:
        	path.append(node)
        	node = self.parent[node]
        path.append(self.robot_pose)
        return path[::-1]
        
    def optimal_planner(self):
        # Optimal planner achieved by A* Algorithm
        while not self.frontier.empty():
            # Get and visit node with least priority
            cur = self.frontier.pop()   
            
            # If reach goal pose, track back to get path 
            if cur == self.goal_pose:            
                return self.generate_path(cur)
            
            # Find neighbor nodes of current nodes
            neighbors = self.find_neighbor(cur)
            for neighbor in neighbors:
            	new_cost = self.cost[cur] + 1
                # No need to explore node that has been visited
                if neighbor not in self.parent or new_cost < self.cost[neighbor]:
                	self.cost[neighbor] = new_cost
                	priority = new_cost + self.cal_heuristic(neighbor)
                	self.frontier.push(neighbor, priority)
                	self.parent[neighbor] = cur
                    
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

    for i in range(y_range // 3 * 2):
    	world_state[x_range // 3][i] = 1

    for i in range(y_range // 3, y_range):
    	world_state[x_range // 3 * 2][i] = 1
     
    return world_state


def generate_robot_goal(x_range, y_range, world_state):
    # Generate robot pose and goal pose
    i = np.random.randint(x_range)
    j = np.random.randint(y_range)
    p = np.random.randint(x_range)
    q = np.random.randint(y_range)
    
    while world_state[i][j] or world_state[p][q] or (i, j) == (p, q):
        i = np.random.randint(x_range)
        j = np.random.randint(y_range)
        p = np.random.randint(x_range)
        q = np.random.randint(y_range)
        
    robot = (i, j)
    goal = (p, q)
    
    return robot, goal

    
def show_result(op_path, world_state, robot_pose, goal_pose):
    # Plot to show result if we found a path
    OPX = []   # Optimal path
    OPY = []
    WX = []  # World
    WY = []
    OX = []  # Obstacle
    OY = []
    SX = robot_pose[0]  # Robot pose
    SY = robot_pose[1]
    GX = goal_pose[0]   # Goal pose
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
    plt.figure(figsize=(8*(xr // yr), 8))
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
    # Input generation for performance test
    xr = 50
    yr = 50
    world_state = generate_graph(xr, yr)
    # robot_pose, goal_pose = generate_robot_goal(xr, yr, world_state)
    robot_pose, goal_pose = (5, 5),  (45, 45)
    search = Search(world_state, robot_pose, goal_pose)
    
    # Implement optimal planner
    start = timeit.default_timer()
    optimal_path = search.optimal_planner()
    stop = timeit.default_timer()
    print('Optimal Planner Time cost: ', stop - start)
    if optimal_path:
        print("Optimal search succeed!")
        #print("Optimal path is:", optimal_path)
    else:
        print("No optimal path is found!")
        
    # Plot result for comparison
    show_result(optimal_path, world_state, robot_pose, goal_pose)

            
if __name__ == '__main__':
    main()
