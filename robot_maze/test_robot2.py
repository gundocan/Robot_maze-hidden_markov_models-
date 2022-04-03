"""
Example scripts for Robot in a maze HMM

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from hmm_inference import *
from robot2 import *
from utils import normalized


def init_maze(maze_name='mazes/rect_6x10_obstacles.map'):
    """Create and initialize robot instance for subsequent test"""
    m = Maze(maze_name)
    #    robot = Robot(ALL_DIRS, direction_probabilities)
    robot = Robot()
    robot.maze = m
    robot.pose = (1, 1) + NORTH
    return robot


def test_pt():
    """Try to compute transition probabilities for certain pose"""
    robot = init_maze()
    robot.pose = (2, 10) + NORTH
    print('Robot at', robot.pose)
    for pos in robot.maze.get_free_positions():
        for orient in ORIENTATIONS:
            p = robot.pt(robot.pose, pos + orient)
            if p > 0:
                print('Prob of transition to', pos + orient, 'is', p)


def test_pe():
    """Try to compute the observation probabilities for certain pose"""
    robot = init_maze()
    robot.pose = (1, 5) + NORTH
    print('Robot at', robot.pose)
    for obs in robot.get_observations():
        p = robot.pe(robot.pose, obs)
        if p > 0:
            print('Prob obs', obs, 'is', p)


def test_simulate():
    """Try to generate some data from the robot domain"""
    robot = init_maze()
    print('Generating data...')
    states, observations = robot.simulate(n_steps=5)
    for i, (s, o) in enumerate(zip(states, observations)):
        print('Step:', i + 1, '| State:', s, '| Observation:', o)


def test_filtering():
    """Try to run filtering for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(n_steps=3)
    print('Running filtering...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forward(initial_belief, obs, robot)
    for state, belief in zip(states, beliefs):
        print('Real state:', state)
        print('Sorted beliefs:')
        for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(k, ':', v)


def test_smoothing():
    """Try to run smoothing for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(init_state=(1, 10) + EAST, n_steps=3)
    print('Running smoothing...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forwardbackward(initial_belief, obs, robot)
    for state, belief in zip(states, beliefs):
        print('Real state:', state)
        print('Sorted beliefs:')
        for k, v in sorted(belief.items(), key=lambda x: x[1], reverse=True):
            if v > 0:
                print(k, ':', v)


def test_viterbi():
    """Try to run Viterbi alg. for robot domain"""
    robot = init_maze()
    states, obs = robot.simulate(init_state=(3, 3) + EAST, n_steps=10)
    print('Running Viterbi...')
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    ml_states, max_msgs = viterbi(initial_belief, obs, robot)
    for real, est in zip(states, ml_states):
        print('Real pos:', real, '| ML Estimate:', est)


if __name__ == '__main__':
    print('Uncomment some of the tests in the main section')
    # test_pt()
    # test_pe()
    # test_simulate()
    # test_filtering()
    # test_smoothing()
    # test_viterbi()
