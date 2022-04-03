
from hmm_inference import *
from robot import *
from utils import normalized
import numpy as np
from matplotlib import pyplot as plt

#define probability
default_direction_probabilities = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}

def filtering2(state,observations,robot):
    initial_Belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forward(initial_Belief, observations, robot)

    manhattan_general =[]
    hitmiss_general = []

    for state, belief in zip(state, beliefs):
        sort = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        estimated_belief = sort[0][0]
        manhattan_dist = manhattan(state,estimated_belief)
        hitmiss_err = hitmiss(state,estimated_belief)
        #add to list of errors
        manhattan_general.append(manhattan_dist)
        hitmiss_general.append(hitmiss_err)
    
    
    return manhattan_general,hitmiss_general

def smoothing2(state,observations,robot):

    initial_Belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forwardbackward(initial_Belief, observations, robot)


    manhattan_general = []
    hitmiss_general = []

    for state, belief in zip(state, beliefs):
        sort = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        estimated_belief = sort[0][0]


        manhattan_dist = manhattan(state, estimated_belief)
        hitmiss_err = hitmiss(state, estimated_belief)
        manhattan_general.append(manhattan_dist)
        hitmiss_general.append(hitmiss_err)

    return manhattan_general, hitmiss_general


def viterbi2(state,observations,robot):


    initial_Belief = normalized({pos: 1 for pos in robot.get_states()})
    cur_m, pred = viterbi(initial_Belief, observations, robot)

    # Lists of errors
    manhattan_general = []
    hitmiss_general = []

    for truth, calc in zip(state, cur_m):
        manhattan_dist = manhattan(truth, calc)
        hitmiss_err = hitmiss(truth, calc)
        manhattan_general.append(manhattan_dist)
        hitmiss_general.append(hitmiss_err)

    
    return manhattan_general,hitmiss_general



def hitmiss(truth,found):
    if truth != found:
        return 1
    else : return 0


def avg_err(errs):
    return sum(errs)/len(errs)

def init_maze(maze,pos):
    #Robot initialization
    robot = Robot(ALL_DIRS, default_direction_probabilities)
    robot.maze = maze
    robot.position = pos
    return robot



if __name__=='__main__':

    #Load the mazes
    m1 = Maze('mazes/rect_3x2_empty.map')
    m2 = Maze('mazes/rect_5x4_empty.map')
    m3 = Maze('mazes/rect_6x10_maze.map')
    m4 = Maze('mazes/rect_6x10_obstacles.map')
    m5 = Maze('mazes/rect_8x8_maze.map')
    maze_maps = [m1, m2, m3, m4, m5]

    m = m3
    pos = (1,1)
    #print(m2)
    iterations = 5
    # Create a list to append the observations and states
    result_manhattan_distance = []
    result_hitmiss_percent = []
    #create a loop for testing
    for n in range(iterations):
                    print("Iteration: ", n)
                    # Initialization
                    robot = init_maze(m,pos)
                    states, obs = robot.simulate(init_state=pos, n_steps=100)


                    filter_manhattan, filter_hitmiss = filtering2(states,obs,robot)

                    smoothing_manhattan, smoothing_hitmiss = smoothing2(states, obs, robot)

                    viterbi_manhattan, viterbi_hitmiss = viterbi2(states, obs, robot)


                    avg_manhattan_error = [avg_err(filter_manhattan), avg_err(smoothing_manhattan),avg_err(viterbi_manhattan)]
                    hitmiss_percent = [avg_err(filter_hitmiss), avg_err(smoothing_hitmiss),avg_err(viterbi_hitmiss)]

                    print("Average manhattan distance between true and estimated pose")
                    print(avg_manhattan_error)

                    print("Wrong estimation percentage")
                    print(hitmiss_percent)
                    result_manhattan_distance.append(avg_manhattan_error)
                    result_hitmiss_percent.append(hitmiss_percent)


    result_hitmiss_percent = np.array(result_hitmiss_percent)
    result_hitmiss_percent = result_hitmiss_percent.dot(100)
    
    result_manhattan_distance = np.array(result_manhattan_distance)

    #Plotting the HitMiss
    plt.figure(0)
    plt.plot(range(iterations),result_hitmiss_percent[:,0])
    plt.plot(range(iterations),result_hitmiss_percent[:,1])
    plt.plot(range(iterations),result_hitmiss_percent[:, 2])

    plt.legend(('Filtering', 'Smoothing', 'Viterbi'))
    plt.title("Algorithms comparisons with miss percentage estimation ")
    plt.xlabel("Number of Iterations")
    plt.ylabel("[%] - Percentage")
    plt.grid()

    #Plotting the Manhattan
    plt.figure(1)
    plt.plot(range(iterations), result_manhattan_distance[:, 0])
    plt.plot(range(iterations), result_manhattan_distance[:, 1])
    plt.plot(range(iterations), result_manhattan_distance[:, 2])

    plt.legend(('Filtering', 'Smoothing', 'Viterbi'))
    plt.title("Algorithms comparisons with Manhattan distance")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Distance (cell size)")
    plt.grid()

    # Final results
    print("--- Conclusion ---")
    print("Overall miss")
    print([avg_err(result_hitmiss_percent[:,0]),avg_err(result_hitmiss_percent[:,1]),avg_err(result_hitmiss_percent[:,2])])

    print("Overall manhattan")
    print([avg_err(result_manhattan_distance[:,0]),avg_err(result_manhattan_distance[:,1]),avg_err(result_manhattan_distance[:,2])])

    plt.show()