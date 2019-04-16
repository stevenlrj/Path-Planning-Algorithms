# Path-Planning algorithms implementation in Python

(1) A* : graph search algorithm to solve for path planning problems, guaranteed to find optimal solution if it exist.

The path found will be like in the figure below:

![01](A_star_01.png)

If we want to keep our robot stay a specific distance from obstacle for safety consideration, the path found could be like in the figure below:

![02](A_star_02.png)


(2) RRT & RRT* : sampled based algorithm to solve path planning problems, RRT is guaranteed to find sub-optimal solution and RRT* is guaranteed to find optimal solution.

The path found by RRT and RRT* could be like in the figure below:
