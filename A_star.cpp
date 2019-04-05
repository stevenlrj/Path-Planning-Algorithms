#include <iostream>
#include <queue>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct Priority_Queue {
	typedef pair<int, pair<int, int>> element;
	priority_queue<element, vector<element>, greater<element>> elements;

	bool empty() {
		return elements.empty();
	}

	void push(int priority, pair<int, int> node) {
		elements.emplace(priority, node);
	}

	pair<int, int> pop() {
		auto minP = elements.top().second;
		elements.pop();
		return minP;
	}

};

struct pair_hash {
	template <class T1, class T2> 
	size_t operator() (const pair<T1, T2> & p) const {
		size_t h1 = hash<T1>()(p.first);
		size_t h2 = hash<T2>()(p.second);
		return h1 ^ h2;
	}
};

class discrete_planner {
	int x_d;
	int y_d;
  public:
  	discrete_planner () {};

  	vector<pair<int, int>> optimal_planner (vector<vector<int>> & world_state, pair<int, int> & robot_pose, pair<int, int> & goal_pose) {
  		x_d = world_state.size();
  		y_d = world_state[0].size();
  		vector<pair<int, int>> path = {};
  		Priority_Queue frontier;
  		frontier.push(0, robot_pose);

  		unordered_map<pair<int, int>, int, pair_hash> cost;
  		cost[robot_pose] = 0;

  		unordered_map<pair<int, int>, pair<int, int>, pair_hash> parent; 

  		while (!frontier.empty()) {
  			auto cur = frontier.pop();

  			if (cur == goal_pose)
  				path = generate_path(cur, robot_pose, parent);

  			vector<pair<int, int>> neighbors = find_neighbor(cur, world_state);
  			for (auto neighbor: neighbors) {
  				int new_cost = cost[cur] + 1;
  				if (parent.find(neighbor) == parent.end() || new_cost < cost[neighbor]) {
  					cost[neighbor] = new_cost;
  					int priority = new_cost + cal_heuristic(cur, goal_pose);
  					frontier.push(priority, neighbor);
  					parent[neighbor] = cur;
  				}
  			}

  		}
  		return path;
  	}

  	vector<pair<int, int>> generate_path(pair<int, int> goal, pair<int, int> start, unordered_map<pair<int, int>, pair<int, int>, pair_hash> & parent) {
  		vector<pair<int, int>> path;
  		path.push_back(goal);
  		auto node = parent[goal];
  		while (node != start) {
  			path.push_back(node);
  			node = parent[node];
  		}
  		path.push_back(start);
  		reverse(path.begin(), path.end());
  		return path;
  	}

  	bool verify_node(pair<int, int> cur, vector<vector<int>> & world_state) {
  		if (cur.first >= 0 && cur.first < x_d && cur.second >= 0 && cur.second < y_d && !world_state[cur.first][cur.second])
  			return true;
  		else return false;
  	}

  	vector<pair<int, int>> find_neighbor(pair<int, int> cur, vector<vector<int>> & world_state) {
  		vector<pair<int, int>> NS;
  		vector<pair<int, int>> neighbors = {{cur.first+1, cur.second}, {cur.first-1, cur.second}, {cur.first, cur.second+1}, {cur.first, cur.second-1}};
  		for (auto n: neighbors) {
  			if (verify_node(n, world_state)) NS.push_back(n);
  		}
  		return NS;
  	}

  	int cal_heuristic(pair<int, int> cur, pair<int, int> goal) {
  		return abs(cur.first - goal.first) + abs(cur.second - goal.second);
  	}
};

int main () {
	vector<vector<int>> world_state = {{0, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 1, 0, 1, 0}, {0, 0, 0, 1, 0}, {0, 0, 0, 1, 0},};
	pair<int, int> robot_pose = {0, 0};
	pair<int, int> goal_pose = {4, 4}; 
	discrete_planner DP;
	vector<pair<int, int>> path = DP.optimal_planner(world_state, robot_pose, goal_pose);
	return 0;
}

