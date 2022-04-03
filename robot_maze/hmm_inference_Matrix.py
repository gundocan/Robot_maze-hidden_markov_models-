"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized
import numpy as np

INFERENCE_NAME = 'hmm_inference_Matrix'

def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    #print(type(sorted(prev_B.items())))
    cur_B = Counter()
    states = hmm.get_states()
    matrix2 = np.array(())
    matrix1 = hmm.get_transition_matrix()
    for i, state in enumerate(states):

        if i == 0: matrix2 = np.append(matrix2, prev_B[state])
        else: matrix2 = np.vstack((matrix2, prev_B[state]))
    final_result = matrix1 @ matrix2
    for i, state in enumerate(states):
        cur_B[state] = final_result[i][0]
    return cur_B

def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B = prior  # This shall be iteratively updated
    Bs = [B]    # This shall be a collection of Bs over time steps
    for n in range(n_steps):
        B = update_belief_by_time_step(B, hmm)
        Bs.append(B)
    return Bs


def update_belief_by_evidence(prev_B, e, hmm):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :return: Counter, current (updated) belief distribution over states
    """
    # Create a new copy of the current belief state
    cur_B = Counter()
    states = hmm.get_states()
    i = 0
    em = hmm.get_observation_matrix()
    matrix1 = np.array(())
    for i, state in enumerate(states):
        if i == 0:
            matrix1 = np.append(matrix1, prev_B[state])
        else:
            matrix1 = np.vstack((matrix1, prev_B[state]))
    observations = hmm.get_observations()
    for i, obs in enumerate(observations):
        if obs == e:
            ind = i
            break
    em = em[:,ind]
    result = em * matrix1

    for i, state in enumerate(states):
        cur_B[state] = result[i]
    #print(cur_B)
    cur_B = normalized(cur_B)

    return cur_B

def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    # Your code here

    cur_B = Counter()
    states = hmm.get_states()
    observations = hmm.get_observations()
    matrix2 = np.array(())
    matrix1 = hmm.get_transition_matrix()
    em = hmm.get_observation_matrix()
    for i, state in enumerate(states):
        if i== 0:
            matrix2 = np.append(matrix2, prev_f[state])
        else:
            matrix2 = np.vstack((matrix2, prev_f[state]))

    res= (matrix1 @ matrix2).T

    ind = 0
    for ind, obs in enumerate(observations):
        if obs == cur_e:
            i = ind
            break
    em = em[:, i].T
    result = (em * res).reshape(len(states),)

    for i, state in enumerate(states):
        cur_B[state] = result[i]
    cur_B = normalized(cur_B)

    return cur_B


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice

    for e in e_seq:
        f = forward1(f, e, hmm)
        fs.append(f)
    return fs


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """

    fs = forward(prior, e_seq, hmm)
    lhood = sum(fs[-1].values())

    return lhood

def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    states = hmm.get_states()
    observations = hmm.get_observations()
    tm = hmm.get_transition_matrix()
    em = hmm.get_observation_matrix()

    ind = 0
    for i, obs in enumerate(observations):
        if obs == next_e:
            ind = i
            break

    em = em[:,ind]
    matrix2 = np.array(())

    for ind, state in enumerate(states):

        if ind == 0:
            matrix2 = np.append(matrix2, next_b[state])
        else:
            matrix2 = np.vstack((matrix2, next_b[state]))

    res = tm @ (em * matrix2.T).reshape(len(states),)

    for i, s in enumerate(states):
        cur_b[s] = res[i]
    return cur_b

def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []  # Smoothed belief distributions
    fs = [priors]
    states = hmm.get_states()
    for e in e_seq:
        fs.append(forward1(fs[-1],e,hmm))

    b = Counter()
    for s in states:
        b[s] = 1

    for i in range(len(fs)-2, -1, -1):
        for s in states:
            fs[i+1][s] *= b[s]
        se.append(normalized(fs[i+1]))
        b = backward1(b, e_seq[i], hmm)
    se.reverse()
    return se

def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()  # Current (updated) max message
    predecessors = {}  # The best of previous states for each current state
    states = hmm.get_states()
    observations = hmm.get_observations()
    tm = hmm.get_transition_matrix()
    em = hmm.get_observation_matrix()

    ind = 0
    for i, obs in enumerate(observations):
        if obs == cur_e:
            ind = i
            break

    em = em[:, ind]

    matrix1 = np.array(())
    for ind, state in enumerate(states):
        if ind == 0:
            matrix1 = np.append(matrix1, prev_m[state])
        else:
            matrix1 = np.vstack((matrix1, prev_m[state]))

    matrix1 = matrix1.reshape(len(states), )

    for i, state in enumerate(states):
        tmp = (tm[i] * matrix1) * em[i]
        it = np.argmax(tmp)
        cur_m[state] = tmp[it]
        predecessors[state] = states[it]
    cur_m = normalized(cur_m)
    return cur_m, predecessors

def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []  # Sequence of max messages

    m = forward1(priors, e_seq[0], hmm)
    ms.append(m)
    ml_seq.append(m.most_common(1)[0][0])
    for i, e in enumerate(e_seq[1:]):
        m, p = viterbi1(ms[-1], e, hmm)
        ms.append(m)
        ml_seq.append(m.most_common(1)[0][0])

    return ml_seq, ms
