import SimpleProblemSolvingAgent
from SimpleProblemSolvingAgent import *

def main():
    """This function repeatedly ask user for an orign and destination city, and then applies four search algorithm (A*, Greedy best first search
    Hill climbing and simulated annealing) to find teh path between the cites.

    After displaying the results , it asks if the user want to perform another search"""

    print("----------------------------------")
    print("Here are all the possible cities that can be traveled:")
    print(list(romania_map.graph_dict.keys()))
    print("----------------------------------")

    #User Interface
    while True:
        origin = input("Please enter the origin city: ")
        while origin not in romania_map.locations:
            print("Could not find " + origin + ". Please try again.")
            origin = input()

        #Destination City
        destination = input("Please enter the destination city: ")
        while True:
            if destination not in romania_map.locations:
                print("Could not find " + destination + ". Please try again.")
                destination = input()
            elif origin == destination:
                print("The same city can't be both origin and destination. Please try again")
                destination = input()
            else:
                break  # Valid destination found, exit the loop

        #A* Search Algorithm
        astar_searchagent =SimpleProblemSolvingAgent(initial_state= origin, goal_state= destination, search_method= astar_search)
        problem =astar_searchagent.formulate_problem(origin, destination)
        path,path_cost = astar_searchagent.search(problem)

        print("----------------------------------")
        print('A* Search')
        print('Path used: ',"-->".join(path))
        print('Total Cost: ',path_cost)
        print("----------------------------------")

        #BFS Algorithm
        bfs_searchagent = SimpleProblemSolvingAgent(initial_state=origin, goal_state=destination, search_method=greedy_bf_search)
        problem = bfs_searchagent.formulate_problem(origin, destination)
        path,path_cost  = bfs_searchagent.search(problem)

        print('Greedy Best-First Search')
        print('Path used: ',"-->".join(path))
        print('Total Cost: ', path_cost)
        print("----------------------------------")

        #Hill Climbing Algorithm
        hill_climbing_searchagent= SimpleProblemSolvingAgent(initial_state=origin, goal_state=destination, search_method=hill_climbing)
        problem = hill_climbing_searchagent.formulate_problem(origin, destination)
        path, path_cost= hill_climbing_searchagent.search(problem)

        print('Path used: ',"-->".join(path))
        print('Total Cost: ', path_cost)
        print("----------------------------------")

        # simulated annealing Algorithm
        simulated_annealing_searchagent = SimpleProblemSolvingAgent(initial_state=origin, goal_state=destination,search_method=simulated_annealing)
        problem = simulated_annealing_searchagent.formulate_problem(origin, destination)
        path,path_cost= simulated_annealing_searchagent.search(problem)

        print('Path used: ',"-->".join(path))
        print('Total Cost: ', path_cost)
        print("----------------------------------")

        #Did User want to perform another search?
        try_again = input("Would you like to find the best path between the other two cities? (yes/no): ")
        if try_again == "no":
            print("Thank You for Using Our App")
            break
        elif try_again == "yes":
            continue
        else:
            input("Please enter Valid choice (yes/no): ")


if __name__ == "__main__":
    main()

