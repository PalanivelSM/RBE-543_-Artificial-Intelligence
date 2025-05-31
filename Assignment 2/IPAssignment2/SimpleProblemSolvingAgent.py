from search import Problem, UndirectedGraph
from queue import PriorityQueue
from utils import is_in, probability
import sys, random
import numpy as np

# Define the Romania map as an Undirected Graph
romania_map = UndirectedGraph({
    "Arad": {"Zerind": 75, "Sibiu": 140, "Timisoara": 118},
    "Bucharest": {"Urziceni": 85, "Pitesti": 101, "Giurgiu": 90, "Fagaras": 211},
    "Craiova": {"Drobeta": 120, "Rimnicu": 146, "Pitesti": 138},
    "Drobeta": {"Mehadia": 75, "Craiova": 120},
    "Eforie": {"Hirsova": 86},
    "Fagaras": {"Sibiu": 99, "Bucharest": 211},
    "Hirsova": {"Urziceni": 98, "Eforie": 86},
    "Iasi": {"Vaslui": 92, "Neamt": 87},
    "Lugoj": {"Timisoara": 111, "Mehadia": 70},
    "Mehadia": {"Lugoj": 70, "Drobeta": 75},
    "Neamt": {"Iasi": 87},
    "Oradea": {"Zerind": 71, "Sibiu": 151},
    "Pitesti": {"Rimnicu": 97, "Bucharest": 101, "Craiova": 138},
    "Rimnicu": {"Sibiu": 80, "Pitesti": 97, "Craiova": 146},
    "Sibiu": {"Arad": 140, "Oradea": 151, "Fagaras": 99, "Rimnicu": 80},
    "Timisoara": {"Arad": 118, "Lugoj": 111},
    "Urziceni": {"Bucharest": 85, "Hirsova": 98, "Vaslui": 142},
    "Vaslui": {"Iasi": 92, "Urziceni": 142},
    "Zerind": {"Arad": 75, "Oradea": 71}
})
#Romania map Locations
romania_map.locations = {
    "Arad": (91, 492), "Bucharest": (400, 327), "Craiova": (253, 288),
    "Drobeta": (165, 299), "Eforie": (562, 293), "Fagaras": (305, 449),
    "Giurgiu": (375, 270), "Hirsova": (534, 350), "Iasi": (473, 506),
    "Lugoj": (165, 379), "Mehadia": (168, 339), "Neamt": (406, 537),
    "Oradea": (131, 571), "Pitesti": (320, 368), "Rimnicu": (233, 410),
    "Sibiu": (207, 457), "Timisoara": (94, 410), "Urziceni": (456, 350),
    "Vaslui": (509, 444), "Zerind": (108, 531)
}

"""Testing code Area"""


"""Testing code Area"""

#Defining Class Node
class Node:
    def __init__(self, state):
        self.state = state

    def child_node(self, problem, action):
        new_state = problem.result(self.state, action)
        return Node(new_state)

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

# Defining the Romania Problem class with a heuristic function
class Romania_Problem(Problem):
    def __init__(self, initial, goal, Romania_map):
        self.romania_map = Romania_map  # Initializing Romania Map
        self.goal = goal
        super().__init__(initial, goal)  #Initializing from Parent Class Problem

    def actions(self, state):
        """Getting the Neighbouring Cities from the Current cities"""
        return list(self.romania_map.graph_dict.get(state, {}).keys())

    def result(self, state, action):
        """Return the list of all the neighbouring cities"""
        return action

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, current_cost, state1, action, state2):
        """Calculate the path cost based on the distance between cities."""
        cost = self.romania_map.graph_dict.get(state1, {})
        distance = cost.get(state2, float('inf'))
        return current_cost + distance

    def h(self, node):
        """Heuristic function: Use straight-line distance to the goal"""
        locs = self.romania_map.locations
        if node.state in locs and self.goal in locs:
            (x1, y1) = locs[node.state]
            (x2, y2) = locs[self.goal]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5  # Euclidean distance heuristic
        return 0  # Default heuristic if no location data available

    def value(self, state):
        return -self.h(Node(state))

'A* star Search'
def astar_search(problem):
    """ Perform A* search to findd the shortest path from initial state to goal state
    Args:
        problem (Problem): the problem to solve, an instance of Romania_Problem defining the initial state, goal state, and romania map

    Returns:
        tuple:
            the shortest path from initial state to goal state
            total cost of the shortest path from initial state to goal state
    """

    #priority queue to store (priority, state) tuples
    frontier = PriorityQueue()
    frontier.put((0, problem.initial)) #
    #map each state to its parent.
    parent_node = {problem.initial: None}
    # dictionary to store the cost of the shortest path to each state
    path_costs = {problem.initial: 0} #Total Cost of the path

    while not frontier.empty(): # check frontier status
        #Reterieve the state with teh lowest priority- estimated total cost
        current_priority, current_state = frontier.get()

        # Check Current_state is Goal
        if problem.goal_test(current_state):
            path = []
            while current_state is not None:
                path.append(current_state) #Add the Current_state to the path
                current_state = parent_node [current_state]
            path_cost = path_costs [problem.goal] # Get the Total accumulated cost for the path used
            return path[::-1],path_cost

        #Explore neighbours of the current state
        for neighbour in problem.actions(current_state):
            #calculate the new cost to reach the neighbour
            new_cost = problem.path_cost(path_costs [current_state], current_state, None, neighbour) # calculate path_cost
            if neighbour not in path_costs  or new_cost < path_costs[neighbour]:
                path_costs [neighbour] = new_cost
                priority = new_cost + problem.h(Node(neighbour))
                frontier.put((priority, neighbour))
                parent_node[neighbour] = current_state
    return None,float('inf')

'Greedy Best First Search'

def greedy_bf_search(problem):
    """ Perform greedy_bf_search to findd the shortest path from initial state to goal state
        Args:
            problem (Problem): the problem to solve, an instance of Romania_Problem defining the initial state, goal state, and romania map

        Returns:
            tuple:
                the shortest path from initial state to goal state
                total cost of the shortest path from initial state to goal state
        """
    # priority queue to store (priority, state) tuples
    frontier = PriorityQueue() 
    frontier.put((0, problem.initial))
    # map each state to its parent.
    parent_node = {problem.initial: None}
    # dictionary to store the cost of the shortest path to each state
    path_costs = {problem.initial: 0}
    #Keep track of explored states
    explored_states = set()

    while not frontier.empty():  # check frontier status
        #Reterieve the state with teh lowest priority- estimated total cost
        current_priority, current_state = frontier.get()
        # Check Current_state is Goal
        if problem.goal_test(current_state):
            path = []
            while current_state is not None:
                path.append(current_state)
                current_state = parent_node[current_state]
            path_cost = path_costs[problem.goal]
            return path[::-1],path_cost
        #add current states into explored states
        explored_states.add(current_state)

        # Explore neighbours of the current state
        for neighbour in problem.actions(current_state):
            # calculate the new cost to reach the neighbour
            new_cost = problem.path_cost(path_costs[current_state], current_state, None, neighbour)# calculate path_cost
            if neighbour not in explored_states and neighbour not in parent_node:
                path_costs[neighbour] = new_cost
                # priority is used on heuristic value
                priority = problem.h(Node(neighbour))
                frontier.put((priority, neighbour))
                parent_node[neighbour] = current_state
    #Return None and Inf when valid path is found
    return None,float('inf')

'hill_climbing Search'
def hill_climbing(problem):
    """ hill_climbing to findd the shortest path from initial state to goal state
        Args:
            problem (Problem): the problem to solve, an instance of Romania_Problem defining the initial state, goal state, and romania map

        Returns:
            tuple:
                the shortest path from initial state to goal state
                total cost of the shortest path from initial state to goal state
        """
    print('hill Climbing Search')
    #initalise the current state as the initial state
    current_state = problem.initial
    # map each state to its parent.
    parent_node = {current_state: None}
    # dictionary to store the cost of the shortest path to each state
    path_costs = {problem.initial: 0}
    # Set a limit to number of iteration
    iterations = 0
    max_iterations = 80000

    while iterations < max_iterations :
        iterations += 1
        neighbors = problem.actions(current_state)

        # Evaluate neighbors to find the best one
        best_neighbor = None
        best_value = problem.value(current_state)
        # Explore neighbours of the current state
        for neighbour in neighbors:
            neighbor_value = problem.value(neighbour )
            if neighbor_value > best_value:
                best_value = neighbor_value
                best_neighbor = neighbour
        new_cost = problem.path_cost(path_costs[current_state], current_state, None, best_neighbor)
        path_costs[best_neighbor] = new_cost

        # If no better neighbor is found, return the current state
        if best_neighbor is None:
            break

        # Move to the best neighbor
        parent_node[best_neighbor] = current_state
        current_state = best_neighbor

        # Check if the goal is reached
        if problem.goal_test(current_state):
            break

    # Reconstruct the path from initial to goal state
    path = []
    while current_state is not None:
        path.append(current_state)
        current_state = parent_node[current_state]
    path.reverse()

    if problem.goal in path:
        path_cost = path_costs[problem.goal]

    else:
        print("No Valid Path found using Hill Climbing") # If NOPath is found print NoPath Found
        path_cost = 0
        path.append('No Path Found')
    return path,path_cost

def exp_schedule(k=25, lam=0.005, limit=100):
    """One possible schedule function for simulated annealing"""
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)

def simulated_annealing(problem, schedule=exp_schedule()):
    """ hill_climbing to findd the shortest path from initial state to goal state
            Args:
                problem (Problem): the problem to solve, an instance of Romania_Problem defining the following attributes
                 initial - initial state of the problem
                 goal - goal state of the problem
                 value(state) - returns the value of a state
                 path_cost - returns the path cost of a state
                 expand(state) - returns the expanded path from initial state to goal state
                 schedule - Function takes the current step t and return the temperature T .

            Returns:
                tuple:
                    the shortest path from initial state to goal state
                    total cost of the shortest path from initial state to goal state
                    if no valid path is found, return No path found
     """

    # initializing the variables
    states = []
    path = [problem.initial]
    accumulated_cost = {problem.initial: 0}
    current = Node(problem.initial)
    path_cost = 0
    explored = set(path)
    print('simulated annealing Search')

    #search loop
    for t in range(sys.maxsize):
        states.append(current.state)
        T = schedule(t)
        #if temperature T reaches 0, then stop searching
        if T == 0:
            path.append(current.state)
            break
            # return current.state, accumulated_cost.get(problem.goal, path_cost)
        neighbors = current.expand(problem)
        #if no neighbour found, then stop searching too
        if not neighbors:
            path.append(current.state)
            break
            # return current.state, accumulated_cost.get(problem.goal, path_cost)
        next_choice = random.choice(neighbors)
        #calculate the energy difference
        delta_e = problem.value(current.state) - problem.value(next_choice.state)
        #accept the neighbour based on the energy difference and temperature
        if delta_e > 0 or probability(np.exp(-delta_e / T)):
            if next_choice.state not in explored:
                if next_choice.state not in path:
                    path.append(next_choice.state)
                new_cost = problem.path_cost(accumulated_cost[current.state], current.state, None, next_choice.state)
                accumulated_cost[next_choice.state] = new_cost
                explored.add(next_choice.state)
                current = next_choice
            #reterive the path cost if the goal is found
            if problem.goal_test(current.state):
                break
    path_cost = accumulated_cost.get(problem.goal, path_cost)
    # return No path found when there is NO valid path found
    if problem.goal in path:
        path_cost = accumulated_cost.get(problem.goal, path_cost)
    else:
        print("No Valid Path found using simulated annealing")  # If NOPath is found print NoPath Found
        path_cost = 0
        path.append('No Path Found')

    return path,path_cost

# Standardized Problem Solving Agent
class SimpleProblemSolvingAgent:
    """
        A simple problem-solving agent that performs a search using the specified search method.

        This agent is initialized with an initial state, a goal state, and a search method.
        It can be called with a percept to update its state and formulate the goal and problem.
        It will then perform the search to find a sequence of actions leading to the goal.
        """
    def __init__(self, initial_state="None", goal_state = "None", search_method = "None"):

        self.state = initial_state
        self.goal = goal_state
        self.seq = []
        self.search_method = search_method  # Store the search method

    def __call__(self, percept):
        """Update the state based on the percept and formulate the goal and problem.
        Perform the search if the sequence of actions is empty. """
        self.state = self.update_state(self.state, percept)
        if not self.seq:
            goal = self.formulate_goal(self.goal)
            problem = self.formulate_problem(self.state, goal)
            self.seq = self.search(problem)
            if not self.seq:
                return None
        return self.seq.pop(0)
    
    def update_state(self, state, percept):
        """
        Update the state based on the percept.
        Args:
            state (str): The current state of the agent.
            percept (str): The percept (observation or input) that will update the state.
        Returns:
            str: The updated state
        """
        return state  # Placeholder for state updates

    # If no sequence of actions is available, formulate the goal and problem
    def formulate_goal(self, goal):
        """
        Formulate the goal that the agent is trying to achieve.
        Args:
            goal (str): The goal state.
        Returns:
            str: The goal that the agent is trying to reach.
        """
        return goal

    def formulate_problem(self, state, goal):
        """
        Create a problem instance with the given state and goal, and România map.
        Args:
            state (str): The current state of the agent.
            goal (str): The goal state of the agent.

        Returns:
            Romania_Problem: A problem instance that encapsulates the state, goal, and Romania map.
        """
        return Romania_Problem(state, goal, romania_map)

    def search(self, problem):
        """
        Execute the chosen search algorithm
        Args:
            problem : The problem instance that contains the initial state and goal state.
        Returns:
            list: A sequence of actions leading to the goal, or None if no solution is found.
        """
        return self.search_method(problem) if self.search_method else None

