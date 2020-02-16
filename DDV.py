import numpy as np
from MDP import *
from BellmanEquation import bellmanEquationSolver
from functools import partial
from itertools import product

class DDV () :
    """
    The DDV algorithm uses an efficient
    sampling strategy to quickly find a
    good policy. 
    
    We give the entire MDP
    object to this algorithm but the 
    algorithm doesn't make use of the 
    MDP's underlying transition 
    probabilities or reward functions.

    Instead, it maintains its own estimates
    which are used by the sampling process.
    """

    def __init__ (self, mdp, epsilon, delta) :
        """
        Constructor.

        Parameters
        ----------
        mdp : MDP
            MDP to solve.
        epsilon : float
            PAC Accuracy parameter.
        delta : float
            PAC Confidence parameter.
        """
        self.mdp = mdp
        self.N = np.zeros(mdp.T.shape)
        self.QUpper = np.ones(mdp.R.shape) * mdp.Vmax
        self.QLower = np.zeros(mdp.R.shape)


    def numberOfSamples (self) :
        """
        The number of times you have to sample
        a particular (state, action) so that the
        uncertainty in its Q-function is small.
        """
        S = self.mdp.S
        A = self.mdp.A
        gamma = self.mdp.gamma

        factor = 1 / (self.epsilon ** 2 * (1 - gamma) ** 4)
        term2 = np.log((S * A) / (self.epsilon * (1 - gamma) ** self.delta))
        return (S + term2) * factor

    def shiftProbabilityMass (self, s, a, delta, findUpper) :
        """
        The algorithm searches for extremal bounds
        for the Q-function. It searches for these 
        extremal bounds within a particular 
        confidence interval of the transition
        probability estimates. 

        findUpper specifies how to shift the
        probability mass. When you want to find 
        an upper bound on the Q-function, you shift
        mass from less valuable states to more 
        valuable states. For the lower bound, you
        do the opposite.

        This algorithm by Strehl and Littman returns
        the transition probabilities. Plugging them
        in the Bellman Equation gives the extremal
        bound on the Q function.

        Parameters
        ----------
        s : int
            The state for which we are finding
            the transition probabilities that 
            maximise the Q function.
        a : int
            The action for which we are finding 
            the transition probabilities.
        delta : float
            Confidence Parameter.
        findUpper : bool 
            True if we have to find the upper
            bound, else False.
        """
        saCount = np.sum(self.N[s][a])

        PHat = self.N / saCount
        Pt = PHat
        
        if findUpper : 
            V = np.max(self.QUpper, axis=1)
        else : 
            V = np.max(self.QLower, axis=1)

        deltaOmega = confidenceRadius(saCount, delta) / 2

        while deltaOmega > 0 : 
            S_ = PHat[s][a] < 1

            donor = np.argmin(V[Pt[s][a] > 0])
            recipient = np.argmax(V[Pt[s][a] < 1 and S_])

            zeta = min(1 - Pt[s][a][donor], Pt[s][a][recipient], deltaOmega)

            if not findUpper :
                donor, recipient = recipient, donor

            Pt[s][a][donor] -= zeta
            Pt[s][a][recipient] += zeta 

            deltaOmega -= zeta

        return Pt[s][a]

    def upperPGoodTuring (self, s, a, delta, M0) :
        """
        The purpose of this function is the
        same as that of upperP. 

        There are some extensions to upperP
        which were proposed by the authors
        which are implemented here.

        Parameters
        ----------
        s : int
            The state for which we are finding
            the transition probabilities that 
            maximise the Q function.
        a : int
            The action for which we are finding 
            the transition probabilities.
        delta : float
            Confidence Parameter.
        M0 : float
            Missing Mass Limit. Don't 
            know what this actually means.
        """
        saCount = np.sum(self.N[s][a])
        PHat = self.N / saCount
        PTilda = PHat

        constant1 = confidenceRadius(saCount, delta / 2) / 2
        # I have no clue what this 
        # second constant is for. 
        # TODO : Find out what this is and 
        # wrap this up in a function.
        constant2 = (1 + 2**0.5) * (np.log(2 / delta) / saCount)**0.5
        deltaOmega = min(constant1, constant2)

        unvisitedSucc = self.N[s][a] == 0

        while deltaOmega > 0 : 
            S_ = PHat[s][a] < 1

            if M0 == 0 :
                S_[unvisitedSucc] = False

            sLower = np.argmin(self.VUpper[PThilda[s][a] > 0])
            sUpper = np.argmax(self.VUpper[PThilda[s][a] < 1 and S_])

            zeta = min(1 - PTilda[s][a][sUpper], PTilda[s][a][sLower], deltaOmega)

            PTilda[s][a][sLower] -= zeta
            PTilda[s][a][sUpper] += zeta

            deltaOmega -= zeta
            
            if unvisitedSucc[sUpper] :
                M0 -= zeta

        return PTilda

    def confidenceRadius (self, saCount, delta) :
        """
        Referred to as omega in the DDV paper.
        Some magic function probably used to
        make the PAC guarantees go through.

        Parameters
        ----------
        saCount : int
            How many times the (state, action) has 
            been sampled by the algorithm.
        delta : float
            A confidence interval parameter.
        """
        top = np.log(2 ** (self.mdp.S) - 2) - np.log(delta))
        return np.sqrt(2 * top / saCount)

    def updateQConfidenceIntervals(self, delta) :
        allStateActions = list(product(range(self.mdp.S), range(self.mdp.A)))

        PUpper = np.stack(map(, allStateActions))
        PLower = np.stack(map(shiftLower, allStateActions))

        PUpper = PUpper.reshape(self.mdp.T.shape)
        PLower = PUpper.reshape(self.mdp.T.shape)

        self.QUpper = bellmanEquationSolver(
            self.mdp, 
            PUpper, 
            self.QUpper, 
            stoppingCondition
        )
         
        self.QLower = bellmanEquationSolver(
            self.mdp, 
            PLower, 
            self.QLower, 
            stoppingCondition
        )

    def ddvLoop (self) :
        """
        Apply the DDV algo.
        TODO : There are a lot of moving 
        parts. Need to sort this.
        """
        m = self.numberOfSamples()
        delta_ = self.delta / (self.mdp.S * self.mdp.A * m)
        
        exploredStates = np.zeros(self.mdp.S, dtype=bool)
        exploredStates[self.mdp.s0] = True

        while True :
            self.updateQConfidenceIntervals()

            VUpper = np.max(self.QUpper[self.mdp.s0])
            VLower = np.max(self.QLower[self.mdp.s0])

            if VUpper - VLower <= self.epsilon :
                break

            ddv = None
            
            s, a = np.unravel_index(ddv.argmax(), ddv.shape)
            s_,r = self.mdp.step(s, a)

            exploredStates[s_] = True
            self.N[s][a][s_] += 1

