# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # TODO check for win, if game state is win return a high number
        if  successorGameState.isWin():
            return 99999999

        # Todo set the necessary data
        ghostList = successorGameState.getGhostPositions()  # ghost position
        foodList = newFood.asList() #food position
        distance = list() #an empty list for the distances to be calculated
        for food in foodList:
            dist = manhattanDistance(food, newPos) #caclulate the distance and append it to the list
            distance.append(dist)

        #     TODO set the next food the one with the shortest distance
        nextFood = min(distance)

        # todo check if the ghost is close to newPos
        for ghost in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            if ghostDistance < 2: #if the ghost is too close
                return -99999999

        #     TODO return the score according to the state of the game + next food
        # the division is responsible to give a higher score to a closer distance
        # so that the agent chooses the bigger number
        finalScore = successorGameState.getScore() + 9 / nextFood
        return finalScore


        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)





class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, state: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # TODO implement minimax dupa algoritmul de la curs
        #  ?? Puteți implementa algoritmul in mod recursiv
        #  acuma se evaluează stări si nu perechi (stare, acțiune).

        # def min-value(state)
        # Initialize V = inf
        # for each successor of state:
        # v = min(v, value(successor))
        # return v
        def minValue(state, typeOfAgent, depth):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # Initialize V = inf
            #     TODO set
            #      -v as infinity
            #      -actions for min agent - pass typeOfAgent as index
            #      -picked action as null
            v = float('inf')
            pickedAction = None
            legalActions = state.getLegalActions(typeOfAgent) # lega; action for ghosts

            # TODO check for legal actions, if no more actions return eval function
            if not legalActions:
                return self.evaluationFunction(state), None

            # for each successor of state:
            for successorAction in legalActions:
                # if the agent is the last one, call the maxValue function for Pacman
                if typeOfAgent == state.getNumAgents() - 1:
                    val, _ = maxValue(state.generateSuccessor(typeOfAgent, successorAction), depth + 1)
                # if the next agent is not the last one - minValue
                else:
                    val, _ = minValue(state.generateSuccessor(typeOfAgent, successorAction), typeOfAgent + 1, depth)

                # if  better action is found
                if v > val:
                    v, pickedAction = val, successorAction

            # return the minimum value and corresponding action
            return v, pickedAction




        # def max-value(state):
        # Initialize v = -inf
        # for each successor of state:
        # v = max(v, value(successor))
        # return v
        def maxValue(state, depth):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # Initialize V = -inf
            #     TODO set
            #      -v as negative infinity
            #      -actions for max agent - pass 0 as index
            #      -picked action as null
            v = -float('inf')
            pickedAction = None
            legalActions = state.getLegalActions(0) #legal actions for PACMAN

            # TODO check for legal actions, if no more actions return eval function
            if not legalActions:
                return self.evaluationFunction(state), None

            # for each successor of the state:
            for successorAction in legalActions:
                val, _ = minValue(state.generateSuccessor(0, successorAction), 1, depth)
                # if a better action is found - update
                if v < val:
                    v, pickedAction = val, successorAction

            # Return the maximum value and the corresponding action
            return v, pickedAction

#TODO start with max
        optimalVal,optimalAction = maxValue(state, 0)
        return optimalAction

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # python pacman.py -p AlphaBetaAgent

        #FAIL: test_cases\q3\6-tied-root.test
#***     Incorrect generated nodes for depth=3
#***         Student generated nodes: A B max min1 min2
#***         Correct generated nodes: A B C max min1 min2
#***     Tree:
#***         max
#***        /   \
#***     min1    min2
#***      |      /  \
#***      A      B   C
#***     10     10   0

        #


#todo trebuie sa îl adaptați pentru situatia in care in joc se găsesc mai multi strigoi (deci trebuie adaptat pseudocodul pentru mai multe nivele de min).
        # def min-value(state, a, b):
        # Initialize v
        # for each successor of state:
        # v=min(v, value(successor, a, b))
        # if v<= a return v
        # b =min(b,v)
        # return v
        def minValue(state, typeOfAgent, depth, alfa , beta):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # Initialize V = inf
            #     TODO set
            #      -v as infinity
            #      -actions for min agent - pass typeOfAgent as index
            #      -picked action as null
            v = float('inf')
            pickedAction = None
            legalActions = state.getLegalActions(typeOfAgent)  # lega; action for ghosts

            # TODO check for legal actions, if no more actions return eval function
            if not legalActions:
                return self.evaluationFunction(state), None

            # for each successor of state:
            for successorAction in legalActions:
                # if the agent is the last one, call the maxValue function for Pacman
                if typeOfAgent == state.getNumAgents() - 1:
                    generatedSucc = state.generateSuccessor(typeOfAgent, legalActions)
                    if depth == self.depth - 1:
                        val, _ = self.evaluationFunction(generatedSucc)
                    else:
                        val, _ = maxValue(generatedSucc, depth+1, alfa, beta )
                # if the next agent is not the last one - minValue
                else:
                    generatedSucc = state.generateSuccessor(typeOfAgent, successorAction)
                    val, _ = minValue(generatedSucc, typeOfAgent + 1, depth, alfa, beta)

                # if  better action is found
                if v > val:
                    v, pickedAction = val, successorAction

                beta = min( beta, v)

                if v < alfa:
                    return v,pickedAction

            # return the minimum value and corresponding action
            return v, pickedAction



        # def max-value(state, a, b):
        # Initialize v
        # for each successor of state:
        # v=max(v, value(successor, a, b))
        # if v>= b return v
        # a =max(a,v)
        # return v

        def maxValue(state, depth, alfa , beta):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            # Initialize V = -inf
            #     TODO set
            #      -v as negative infinity
            #      -actions for max agent - pass 0 as index
            #      -picked action as null
            v = -float('inf')
            val = v
            pickedAction = None
            legalActions = state.getLegalActions(0)  # legal actions for PACMAN

            # TODO check for legal actions, if no more actions return eval function
            if not legalActions:
                return self.evaluationFunction(state), None

            # for each successor of the state:
            for successorAction in legalActions:
                val, _ = minValue(state.generateSuccessor(0, successorAction), 1, depth, alfa , beta)
                # if a better action is found - update
                if v < val:
                    v, pickedAction = val, successorAction

                alfa = max(alfa, v)
                if v > beta:
                    return v, pickedAction

            # Return the maximum value and the corresponding action
            return v, pickedAction

        # TODO start with max and set alfa and beta
        optimalVal, optimalAction = maxValue(gameState, 0, float("-inf"), float("inf"))
        return optimalAction

        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        #TODO foloseste pseudocodul de la curs pentru ajutor

        # def value(state)
        # if the state is a terminal state: return the state's utility
        # if the next agent is MAX: return max-value(state)
        # if the next agent is EXP: return exp-value(state)

        # TODO repair this function maybe :)
        # def value(state, v, p, succ, typeOfAgent, depth):
        #     # TODO check for game state
        #     if state.isWin() or state.isLose() or depth == self.depth:
        #         return self.evaluationFunction(state)
        #
        #     # todo if the next agent is EXP: return exp-value(state)
        #     if typeOfAgent < state.getNumAgents() - 1:
        #         v += p * expValue(succ, typeOfAgent + 1, depth)
        #     # todo if the next agent is MAX: return max-value(state)
        #     else:
        #         v += p * maxValue(succ, depth + 1)

        # det max-value(state):
        # initialize V
        # for each successor of state:
        # v= max(v, value(successor))
        # return v

        def maxValue(state, depth):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # todo initialize V
            v = float("-inf")
            legalActions = state.getLegalActions(0) #legal actions for PACMAN

            # todo for each successor of state:
            for successorAction in legalActions:
                #todo v= max(v, value(successor))
                successor = state.generateSuccessor(0, successorAction)
                v = max(v, expValue(successor, 1, depth))
            #todo return v
            return v


        # del exp-value(state)
        # Initialize v=O
        # for each successor of state:
        # p= probability(successor)
        # v += p * value(successor)
        # return v

        def expValue(state, agentIndex, depth):
            # TODO check for game state
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            #TODO Initialize v=O
            v = 0
            legalActions = state.getLegalActions(agentIndex)

            lenghtOfActions = len(legalActions)
            p = 100 / lenghtOfActions

            #TODO  for each successor of state:
            for successorAction in legalActions:
                successor = state.generateSuccessor(agentIndex, successorAction)
                if agentIndex < state.getNumAgents() - 1:
                    v += p * expValue(successor, agentIndex + 1, depth)
                else:
                    v += p * maxValue(successor, depth + 1)

            return v


        legalActions = gameState.getLegalActions(0)
        #use a lambda as the seccond parameter to the ,max function
        optimalAction = max(legalActions, key=lambda succesorAction: expValue(gameState.generateSuccessor(0, succesorAction), 1, 0))
        return optimalAction

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
