import numpy as np
import copy
import math
import random
from collections import defaultdict

class DecisionNode:
    """
    A decision (action) node. This represents the state that the agent sees (after a random tile has been added).
    """
    def __init__(self, env, parent=None, action_from_parent=None):
        # Make a deep copy so that each node’s environment is independent.
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action_from_parent = action_from_parent
        # Children: mapping from action (0,1,2,3) to a ChanceNode.
        self.children = {}
        self.visits = 0
        self.value = 0.0
        # Untried actions (only legal moves)
        self.untried_actions = [action for action in range(4) if self.env.is_move_legal(action)]
        # Terminal flag (if game is over)
        self.is_terminal = self.env.is_game_over()

    def uct_select_child(self, exploration=math.sqrt(2)):
        """Select a child chance node using the UCT formula."""
        best_value = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visits == 0:
                uct = float('inf')
            else:
                uct = child.value / child.visits + exploration * math.sqrt(math.log(self.visits) / child.visits)
            if uct > best_value:
                best_value = uct
                best_child = child
        return best_child


class ChanceNode:
    """
    A chance node that represents the intermediate state after the player's move (before the random tile is added).
    Its children are decision nodes corresponding to each possible outcome of the tile addition.
    """
    def __init__(self, env, parent, action):
        self.env = copy.deepcopy(env)  # This state results from taking an action (without spawning a tile)
        self.parent = parent
        self.action = action
        # Children: mapping from outcome to decision node.
        # Each outcome is a tuple: (x, y, tile, probability)
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.untried_outcomes = self._get_possible_outcomes()
        # Check terminality: if the board is full, no tile can be added.
        self.is_terminal = self.env.is_game_over() or (len(self.untried_outcomes) == 0)

    def _get_possible_outcomes(self):
        outcomes = []
        board = self.env.board
        empty_cells = list(zip(*np.where(board == 0)))
        if empty_cells:
            prob2 = 0.9 / len(empty_cells)
            prob4 = 0.1 / len(empty_cells)
            for cell in empty_cells:
                outcomes.append((cell[0], cell[1], 2, prob2))
                outcomes.append((cell[0], cell[1], 4, prob4))
        return outcomes

    def sample_outcome(self):
        """
        When the chance node is fully expanded, sample one of its decision-node children
        according to the tile addition probabilities.
        """
        outcomes = []
        weights = []
        for outcome, child in self.children.items():
            outcomes.append(child)
            weights.append(outcome[3])  # outcome[3] holds the probability weight
        # In case all outcomes have zero weight (should not happen), choose uniformly.
        if sum(weights) == 0:
            return random.choice(outcomes)
        chosen = random.choices(outcomes, weights=weights, k=1)[0]
        return chosen


class MCTS:
    """
    Two-phase MCTS for the 2048 game.
    Alternates between decision nodes (agent moves) and chance nodes (random tile additions).
    """
    def __init__(self, env, approximator, iterations=1000, exploration=math.sqrt(2), rollout_depth=100):
        self.root = DecisionNode(env)
        self.approximator = approximator
        self.iterations = iterations
        self.exploration = exploration
        self.rollout_depth = rollout_depth

    def search(self):
        """
        Run MCTS for a given number of iterations and return the best action from the root.
        """
        for _ in range(self.iterations):
            # selection and expansion: traverse the tree to a leaf
            leaf, path = self._tree_policy(self.root)
            # simulation: run a rollout (default: random play)
            reward = self._rollout(leaf)
            # backpropagation: update the nodes along the path with the rollout reward
            self._backpropagate(path, reward)
        # Select the action leading to the child chance node with the highest visit count.
        best_action = None
        best_visits = -1
        for action, chance_node in self.root.children.items():
            if chance_node.visits > best_visits:
                best_visits = chance_node.visits
                best_action = action
        return best_action

    def _tree_policy(self, node):
        """
        Traverse the tree (starting at a decision node) and expand nodes along the way.
        Returns the leaf node and the path (list of nodes) from the root to the leaf.
        """
        path = [node]
        current = node
        while not current.is_terminal:
            # If we are at a decision node:
            if isinstance(current, DecisionNode):
                # Expand an untried action if available.
                if current.untried_actions:
                    action = random.choice(current.untried_actions)
                    current.untried_actions.remove(action)
                    new_env = copy.deepcopy(current.env)
                    # Perform the move WITHOUT spawning a random tile
                    new_env.step(action, spawn_tile=False)
                    # Create a chance node corresponding to the deterministic outcome.
                    chance_child = ChanceNode(new_env, current, action)
                    current.children[action] = chance_child
                    path.append(chance_child)
                    current = chance_child
                    # Now expand the chance node: if it has an untried outcome, expand it.
                    if current.untried_outcomes:
                        outcome = random.choice(current.untried_outcomes)
                        current.untried_outcomes.remove(outcome)
                        new_env2 = copy.deepcopy(current.env)
                        x, y, tile, _ = outcome
                        new_env2.board[x, y] = tile  # simulate the tile addition
                        decision_child = DecisionNode(new_env2, parent=current, action_from_parent=outcome)
                        current.children[outcome] = decision_child
                        path.append(decision_child)
                        current = decision_child
                    else:
                        # In the unlikely event that the chance node has no untried outcomes,
                        # sample one from its children.
                        current = current.sample_outcome()
                        path.append(current)
                else:
                    # If all actions have been tried at this decision node, select one via UCT.
                    chance_child = current.uct_select_child(self.exploration)
                    path.append(chance_child)
                    current = chance_child
            # If we are at a chance node:
            elif isinstance(current, ChanceNode):
                if current.untried_outcomes:
                    outcome = random.choice(current.untried_outcomes)
                    current.untried_outcomes.remove(outcome)
                    new_env2 = copy.deepcopy(current.env)
                    x, y, tile, _ = outcome
                    new_env2.board[x, y] = tile
                    decision_child = DecisionNode(new_env2, parent=current, action_from_parent=outcome)
                    current.children[outcome] = decision_child
                    path.append(decision_child)
                    current = decision_child
                else:
                    # Fully expanded chance node: sample a decision node child according to the outcome probabilities.
                    decision_child = current.sample_outcome()
                    path.append(decision_child)
                    current = decision_child
            # Stop if we reach a terminal decision node.
            if isinstance(current, DecisionNode) and current.is_terminal:
                break
        return current, path

    def _rollout(self, node, depth=0):
        """
        Perform a rollout (simulation) from the given decision node until game over or a rollout depth is reached.
        Uses the environment's built-in randomness (which includes random tile additions).
        Returns the final score as the reward.
        """
        total_reward = 0.0
        discount = 1.0
        
        sim_env = copy.deepcopy(node.env)
        for d in range(depth):
            if sim_env.is_game_over(): break

            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves: break

            action = random.choice(legal_moves)
            _, reward, done, _ = sim_env.step(action)
            total_reward += discount * reward
            discount *= self.gamma
            if done: break

        if not sim_env.is_game_over(): total_reward += discount * self.approximator.value(sim_env.board)

        return total_reward
        # current_env = copy.deepcopy(node.env)
        # depth = 0
        # while not current_env.is_game_over() and depth < self.rollout_depth:
        #     legal_actions = [a for a in range(4) if current_env.is_move_legal(a)]
        #     if not legal_actions:
        #         break
        #     action = random.choice(legal_actions)
        #     current_env.step(action)  # uses default spawn_tile=True
        #     depth += 1
        # return current_env.score

    def _backpropagate(self, path, reward):
        """
        Update the visit count and value of all nodes along the path.
        """
        for node in reversed(path):
            node.visits += 1
            node.value += reward
