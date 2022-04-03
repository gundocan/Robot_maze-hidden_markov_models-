"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized

INFERENCE_NAME = 'hmm_inference'

def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    # Your code here
    for state in hmm.get_states():
        for prev_state in hmm.get_states():
            cur_B[state] += prev_B[prev_state] * hmm.pt(prev_state, state)
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
    Bs = []  # This shall be a collection of Bs over time steps
    # Your code here
    for step in range(n_steps):
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
    cur_B = Counter(prev_B)
    # Your code here
    for state in hmm.get_states():
        cur_B[state] = prev_B[state] * hmm.pe(state, e)
    return cur_B


def forward1(prev_f, cur_e, hmm):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current belief distribution over states
    """
    # Your code here
    cur_f = update_belief_by_time_step(prev_f, hmm)
    cur_f = update_belief_by_evidence(cur_f, cur_e, hmm)

    return cur_f


def forward(init_f, e_seq, hmm):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, i.e., estimates of belief states for all time slices
    """
    f = init_f  # Forward message, updated each iteration
    fs = []  # Sequence of forward messages, one for each time slice
    # Your code here
    for e in e_seq:
        f = forward1(f, e, hmm)
        f = normalized(f)
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
    # Your code here
    f = prior  # Forward message, updated each iteration
    for e in e_seq:
        f = forward1(f, e, hmm)
    return sum(f.values())


def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    # Your code here

    for state in hmm.get_states():
        for next_state in hmm.get_states():
            cur_b[state] += hmm.pe(next_state, next_e) * hmm.pt(state, next_state) * next_b[
                next_state]

    return cur_b


def backward(e_seq, hmm):
    """Propagate the backward message

    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    b = Counter({state: 1 for state in hmm.get_states()})
    bs = []
    for e in reversed(e_seq):
        bs.append(b)
        b = backward1(b, e, hmm)
    return list(reversed(bs))


def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []  # Smoothed belief distributions
    # Your code here
    fs = forward(priors, e_seq, hmm)
    bs = backward(e_seq, hmm)

    for f, b in zip(fs, bs):
        ctr = Counter()
        for state in hmm.get_states():
            ctr[state] = f[state] * b[state]
        se.append(normalized(ctr))

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
    # Your code here

    for state in hmm.get_states():
        best_val = 0
        best_state = None
        for prev_state in hmm.get_sources(state):
            val = hmm.pt(prev_state, state) * prev_m[prev_state]
            if val >= best_val:
                best_val = val
                best_state = prev_state
        cur_m[state] = hmm.pe(state, cur_e) * best_val
        predecessors[state] = best_state

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
    # Your code here

    predecessors = []
    m = forward1(priors, e_seq[0], hmm)
    ms.append(m)
    for e in e_seq[1:]:
        m, cur_pred = viterbi1(m, e, hmm)
        ms.append(m)
        predecessors.append(cur_pred)

    fin_state, fin_prob = max(m.items(), key=lambda x: x[1])
    ml_seq.append(fin_state)

    state = fin_state
    for pred in reversed(predecessors):
        state = pred[state]
        ml_seq.append(state)

    return list(reversed(ml_seq)), ms
