import numpy as np
import math
import matplotlib.pyplot as plt
import timeit

class Node():
    """ Structure to hold position, parent and cost information for every node
    """
    def __init__(self, x, y, parent = None, cost = 0):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = cost

class Search(object):
    """Search methods for discrete motion planner, optimal planner achieved by RRT*, sub-optimal planner achieved by RRT
    """
    def __init__(self, world_state, robot_pose, goal_pose, expand_length = 2, iteration = 1000):
        self.world_state = world_state
        self.robot_pose = Node(robot_pose[0], robot_pose[1])
        self.goal_pose = Node(goal_pose[0], goal_pose[1])
        self.x_range = len(world_state)
        self.y_range = len(world_state[0])
        self.expand_length = expand_length     # Expand length of RRT in every step
        self.iteration = iteration             # Maximum iterations to search for path
        self.tree = []                            
        self.tree.append(self.robot_pose)
    
    def verify_node(self, node):
        # Verify whether a node is available, node that is out of range of graph or in collision with obstacles is not available
        if node.x in range(self.x_range) and node.y in range(self.y_range) and not self.world_state[int(node.x)][int(node.y)]:
            return True
        else:
            return False
    
    def random_sample(self):
        # Random sample new node
        rand_node = Node(np.random.randint(0, self.x_range), np.random.randint(0, self.y_range))
        return rand_node
    
    def nearest(self, x_new):
        # Find the node in tree that is nearest to the new node
        min_dis = float('inf')
        min_index = 0
        for i, n in enumerate(self.tree):
            dis = self.cal_dis(x_new, n)
            if dis < min_dis:
                min_index = i
                min_dis = dis
        return min_index
    
    def steer(self, nearest_index, x_nearest, x_rand):
        # Steer function to get new node 
        theta = math.atan2(x_rand.y - x_nearest.y, x_rand.x - x_nearest.x)
        new_x = x_nearest.x + math.cos(theta) * self.expand_length
        new_y = x_nearest.y + math.sin(theta) * self.expand_length
        x_new = Node(new_x, new_y, nearest_index)
        return x_new
    
    def check_collision(self, x1, x2):
        # Check whether connecting of two nodes is in collision with obstacles or not
        theta = math.atan2(x1.y - x1.y, x2.x - x2.x)
        for i in range(11):
            x = x1.x + self.expand_length / 10 * i * math.cos(theta)
            y = x1.y + self.expand_length / 10 * i * math.sin(theta)
            if self.world_state[int(x)][int(y)]:
                return True
        return False
    
    def reach_goal(self, x_new):
        # Check whether new node could reach goal pose or not
        if self.cal_dis(x_new, self.goal_pose) <= self.expand_length:
            return True
        else:
            return False
    
    def generate_path(self, node):
        # generate path from robot pose to goal pose
        path = []
        path.append((self.goal_pose.x, self.goal_pose.y))
        path.append((node.x, node.y))
        node = self.tree[node.parent]
        while node != self.robot_pose:
            path.append((node.x, node.y))
            node = self.tree[node.parent]
        path.append((self.robot_pose.x, self.robot_pose.y))
        return path[::-1]
    
    def RRT(self):
        # Sub-optimal planner achieved by RRT
        path = []
        for i in range(self.iteration):
            x_rand = self.random_sample()               
            nearest_index = self.nearest(x_rand)
            x_nearest = self.tree[nearest_index]
            x_new = self.steer(nearest_index, x_nearest, x_rand)
            
            if self.verify_node(x_new) and not self.check_collision(x_nearest, x_new):
                self.tree.append(x_new)
            
            if self.reach_goal(x_new):
                path = self.generate_path(x_new)
                break
                
        return path
    
    def find_nears(self, x_new):
        # Find nodes in tree that are near to new node
        nears_index = []
        n = len(self.tree)
        r = 50.0 * math.sqrt((math.log(n+1) / n))
        #r = 5.0 * self.expand_length
        for i, n in enumerate(self.tree):
            if self.cal_dis(x_new, n) <= r:
                nears_index.append(i)
        return nears_index
    
    def cal_dis(self, x1, x2):
        # Calculate distance between two nodes
        return math.sqrt((x1.x - x2.x)**2 + (x1.y - x2.y)**2)
    
    def choose_best_parent(self, x_new, nears_index):
        # Choose node as parent of new node to achieve min cost
        if not nears_index:
            print "No near nodes!"
            return x_new
        min_cost = float('inf')
        min_index = None
        for i in nears_index:
            node = self.tree[i]
            cost = node.cost + self.cal_dis(x_new, node)
            if not self.check_collision(x_new, node) and cost < min_cost:
                min_cost = cost
                min_index = i
        x_new.cost = min_cost
        x_new.parent = min_index
        return x_new
    
    def rewire(self, x_new, nears_index):
        # Rewire the nodes which are near to new node, in which nodes could be rewired if their cost could be reduced by connecting with new node as their parents
        n = len(self.tree)
        for i in nears_index:
            node = self.tree[i]
            new_cost = x_new.cost + self.cal_dis(x_new, node)
            if i != x_new.parent and new_cost < node.cost and self.check_collision(x_new, node):
                node.parent = n - 1
                node.cost = new_cost
                self.tree[i] = node
                
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
            x_rand = self.random_sample()
            nearest_index = self.nearest(x_rand)
            x_nearest = self.tree[nearest_index]
            x_new = self.steer(nearest_index, x_nearest, x_rand)
            
            if self.verify_node(x_new) and not self.check_collision(x_nearest, x_new):
                nears_index = self.find_nears(x_new)
                x_new = self.choose_best_parent(x_new, nears_index)
                self.tree.append(x_new)
                self.rewire(x_new, nears_index)
            
        best_last_index = self.get_best_index()
        if best_last_index:
            path = self.generate_path(self.tree[best_last_index])
            
        return path

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

    #for i in range(y_range // 5 * 4):
    #    world_state[x_range // 10 * 4 - 1][i] = 1

    #for i in range(y_range // 5 , y_range):
    #    world_state[x_range // 10 * 6 - 1][i] = 1
     
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

    
def show_result(op_path, so_path, world_state, robot_pose, goal_pose, opt, sbt):
    # Plot to show result if we found a path
    OPX = []            # Optimal path
    OPY = []
    SPX = []
    SPY = []
    WX = []             # World
    WY = []
    OX = []             # Obstacle
    OY = []
    SX = robot_pose[0]  # Robot pose
    SY = robot_pose[1]
    GX = goal_pose[0]   # Goal pose
    GY = goal_pose[1]
    OTX = []            # Optimal tree
    OTY = []
    STX = []            # Sub-optimal tree
    STY = []
    
    if op_path:
        for node in op_path:
            OPX.append(node[0])
            OPY.append(node[1])
    
    if so_path:
        for node in so_path:
            SPX.append(node[0])
            SPY.append(node[1])
            
    if opt:
        for node in opt:
            OTX.append(node.x)
            OTY.append(node.y)
    
    if sbt:
        for node in sbt:
            STX.append(node.x)
            STY.append(node.y)

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
    if opt:
        plt.plot(OTX, OTY, ".", label = "Optimal_tree")
    if sbt:
        plt.plot(STX, STY, ".", label = "Sub-optimal_tree")
    
    if op_path:
        plt.plot(OPX, OPY, alpha = 0.9, label = "Optimal_Path", linewidth = 3)
    if so_path:
        plt.plot(SPX, SPY, alpha = 0.9, label = "Sub-optimal_Path", linewidth = 3)
    plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.05), fancybox = True, shadow = True, ncol = 6)
    plt.show()
    plt.close()

def main():
    # Input generation for performance test
    xr = 25
    yr = 25
    world_state = generate_graph(xr, yr)
    # robot_pose, goal_pose = generate_robot_goal(xr, yr, world_state)
    robot_pose, goal_pose = (1, 1),  (xr - 2, yr - 2)
    
    # Implement optimal planner
    search = Search(world_state, robot_pose, goal_pose)
    start = timeit.default_timer()
    optimal_path = search.RRT_star()
    op_tree = search.tree
    stop = timeit.default_timer()
    print("Optimal Planner Time cost: ", stop - start)
    if optimal_path:
        print("Optimal search succeed!")
    else:
        print("No optimal path is found!")
        
    # Implement optimal planner
    search = Search(world_state, robot_pose, goal_pose)
    start = timeit.default_timer()
    sub_path = search.RRT()
    sub_tree = search.tree
    stop = timeit.default_timer()
    print("Sub-Optimal Planner Time cost: ", stop - start)
    if sub_path:
        print("Sub-optimal search succeed!")
    else:
        print("No sub-optimal path is found!")
        
    # Plot result for comparison
    show_result(optimal_path, sub_path, world_state, robot_pose, goal_pose, None, None)
            
if __name__ == '__main__':
    main()