import numpy as np
import math
from MDP import *

def QSolver (mdp, P, Qinit, stoppingCondition) :
    """
    Iterate the Bellman Optimality Equations
    to solve for the Q-function. 

    Parameters
    ----------
    mdp : MDP
        MDP object with rewards, discount factor
        and other relevant information.
    P : np.ndarray
        Estimates of the Transition Probabilities.
    Qinit : np.ndarray
        Initial estimates of Q-function.
    stoppingCondition : lambda
        A function which takes the iteration
        count and difference between
        successive Q-functions and decides
        whether to stop or not.
    """
    iterCnt = 0
    error = math.inf
    Q = Qinit
    while not stoppingCondition(iterCnt, error) :
        Qold = np.copy(Q)
        V = np.max(Q, axis=1)
        Q = mdp.R + gamma * np.sum (P * V, axis=2)
        iterCnt += 1
        error = np.linalg.norm(Q - Qold)
    return Q

def occupancySolver (mdp, )
