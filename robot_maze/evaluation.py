import random

from hmm_inference import *
from robot import *
from test_robot import init_maze
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


def evaluate_function(function, robot, states, obs, verbose=False):
    if verbose:
        print('Running Manhattan evaluation with %s...' % function.__name__)
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    hit_miss = 0
    if function is viterbi:
        ml_states, max_msgs = function(initial_belief, obs, robot)
    else:
        ml_states = function(initial_belief, obs, robot)
    dist = []
    # import manhattan distance
    from scipy.spatial.distance import cityblock
    ctr = 0
    for real, est in zip(states, ml_states):
        if function is viterbi:
            estimate = est
            prob = max_msgs[ctr][estimate]
        else:
            estimate = max(est, key=est.get)
            prob = est[estimate]
        if verbose:
            print('Real pos:', real, '| ML Estimate:', estimate, ' with probability: ', prob)
        if len(dist) > 0:
            dist.append(dist[-1] + cityblock(real, estimate))
        else:
            dist.append(cityblock(real, estimate))
        if real == estimate:
            hit_miss += 1
        ctr += 1

    if verbose:
        print('\nTotal manhattan distance: ')
        print(dist[-1])

    return np.array(dist), 100.0 * hit_miss / len(states)


def evaluate(iterations, n_steps, maze_path, functions=None,
             verbose=False):
    """
    Evaluate hmm algorithms, print hit miss rate and plot manhattan distance results
    :param iterations: Number of iterations each algorithm is tested for
    :param n_steps: Number of steps - length of the sequence
    :param maze_path: Path to the maze
    :param functions: List of functions from hmm_inference like file, default is
            [forward, forwardbackward, viterbi]
    :param verbose:
    """

    if functions is None:
        functions = [forward, forwardbackward, viterbi]

    time = list(range(0, n_steps))
    robot = init_maze(maze_path)
    dist = Counter()
    hitmiss = Counter()

    maze_name = maze_path.split('/')[1]
    maze_name = maze_name.split('.')[0]

    for function in functions:
        dist[function.__name__] = np.zeros(n_steps)
        hitmiss[function.__name__] = 0

    for _ in range(iterations):
        robot.set_random_position()
        states, obs = robot.simulate(n_steps=n_steps)
        for function in functions:
            dist_function, hm = evaluate_function(function, robot, states, obs, verbose)
            dist[function.__name__] += dist_function
            hitmiss[function.__name__] += hm

    plt.figure()
    line_styles = ['b-', 'r--', 'k-.']
    for style, function in zip(line_styles, functions):
        dist[function.__name__] /= iterations
        plt.step(time, dist[function.__name__], style, lw=3, label=function.__name__)

    plt.title(
        'Evaluation for maze %s, n_steps: %d, iterations: %d' % (maze_name, n_steps, iterations),
        fontsize=22)
    plt.xlabel('Time steps', fontsize=16)
    plt.ylabel('Manhattan distance', fontsize=16)
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig('figures/' + INFERENCE_NAME + '/' + maze_name + '.png')

    print('---------Hit miss rate for maze %s' % maze_name)

    for key in hitmiss:
        hitmiss[key] /= iterations
        print('%s: %.2f %%' % (key, hitmiss[key]))


if __name__ == '__main__':
    mazes = [
        'mazes/rect_3x2_empty.map',
        'mazes/rect_5x4_empty.map',
        'mazes/rect_6x10_maze.map',
        'mazes/rect_6x10_obstacles.map',
        'mazes/rect_8x8_maze.map'
    ]

    # use random seed to get same sets of robot positions for different hmm_inferences
    random.seed(0)
    for maze_name in mazes:
        evaluate(iterations=10, n_steps=500, maze_path=maze_name, verbose=False)
