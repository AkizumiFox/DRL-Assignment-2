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
        # Make a deep copy so that each node's environment is independent.
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

    def uct_select_child(self, approximator, exploration=math.sqrt(2)):
        """Select a child chance node using the UCT formula."""
        best_value = -float('inf')
        best_child = None
        for action, child in self.children.items():
            if child.visits == 0:
                uct = approximator.value(child.env.board)
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
    def __init__(self, env, parent, action, reward=0):
        self.env = copy.deepcopy(env)  # This state results from taking an action (without spawning a tile)
        self.parent = parent
        self.action = action
        self.reward = reward  # Store the reward obtained from the parent to this node
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
    Modified MCTS for the 2048 game using afterstate value function.
    Alternates between decision nodes (agent moves) and chance nodes (random tile additions).
    """
    def __init__(self, env, approximator, iterations=100, exploration=0, value_norm=20000):
        self.root = DecisionNode(env)
        self.approximator = approximator
        self.iterations = iterations
        self.exploration = exploration
        self.value_norm = value_norm  # Normalization constant for the value function

    def search(self):
        """
        Run MCTS for a given number of iterations and return the best action from the root.
        """
        # First, fully expand the root node to compute all possible afterstates
        if not self.root.children:
            self._expand_decision_node(self.root)
            
        for _ in range(self.iterations):
            # Selection: traverse the tree to a leaf node
            leaf, path, cumulative_reward = self._tree_policy(self.root)
            # Evaluation: use the approximator instead of rollout
            value = self._evaluate_node(leaf, cumulative_reward)
            # Backpropagation: update the nodes along the path with the evaluation
            self._backpropagate(path, value)
            
        # Select the action leading to the child chance node with the highest visit count
        best_action = None
        best_visits = -1
        for action, chance_node in self.root.children.items():
            if chance_node.visits > best_visits:
                best_visits = chance_node.visits
                best_action = action
        return best_action

    def _tree_policy(self, node):
        """
        Traverse the tree (starting at a decision node) and return the leaf node,
        the path from the root to the leaf, and the cumulative reward along that path.
        """
        path = [node]
        current = node
        cumulative_reward = 0
        
        while not current.is_terminal:
            # If we are at a decision node:
            if isinstance(current, DecisionNode):
                # If there are untried actions, expand all of them at once
                if current.untried_actions:
                    self._expand_decision_node(current)
                
                # Select a chance node using UCT
                chance_child = current.uct_select_child(self.approximator, self.exploration)
                cumulative_reward += chance_child.reward  # Add reward from this action
                path.append(chance_child)
                current = chance_child
                
            # If we are at a chance node:
            elif isinstance(current, ChanceNode):
                # If there are untried outcomes, expand all of them at once
                if current.untried_outcomes:
                    self._expand_chance_node(current)
                    
                # Sample a decision node child according to the outcome probabilities
                decision_child = current.sample_outcome()
                path.append(decision_child)
                current = decision_child
                
            # Stop if we reach a leaf node (terminal or newly expanded)
            if (isinstance(current, DecisionNode) and 
                (current.is_terminal or not current.children)) or \
               (isinstance(current, ChanceNode) and not current.children):
                break
                
        return current, path, cumulative_reward

    def _expand_decision_node(self, node):
        """
        Expand a decision node by creating all possible chance node children.
        """
        for action in node.untried_actions:
            new_env = copy.deepcopy(node.env)
            # Perform the move WITHOUT spawning a random tile
            reward = new_env.step(action, spawn_tile=False)[1]  # Get the reward from the action
            # Create a chance node corresponding to the deterministic outcome
            chance_child = ChanceNode(new_env, node, action, reward)
            node.children[action] = chance_child
            # Calculate the value of this afterstate using the approximator
            chance_child.value = self.approximator.value(chance_child.env.board)
        # Remove all untried actions since we've expanded all of them
        node.untried_actions = []

    def _expand_chance_node(self, node):
        """
        Expand a chance node by creating all possible decision node children.
        """
        for outcome in node.untried_outcomes:
            new_env = copy.deepcopy(node.env)
            x, y, tile, _ = outcome
            new_env.board[x, y] = tile  # simulate the tile addition
            decision_child = DecisionNode(new_env, parent=node, action_from_parent=outcome)
            node.children[outcome] = decision_child
        # Remove all untried outcomes since we've expanded all of them
        node.untried_outcomes = []

    def _evaluate_node(self, node, cumulative_reward):
        """
        Evaluate a node using the value function instead of a rollout.
        Returns the normalized total value: (cumulative_reward + node_value) / value_norm
        
        Args:
            node: The leaf node to evaluate
            cumulative_reward: Total reward accumulated along the path so far
            
        Returns:
            float: Normalized value for backpropagation
        """
        if isinstance(node, DecisionNode):
            # For a decision node, we need to expand its afterstates first if not already done
            if not node.children and not node.is_terminal:
                self._expand_decision_node(node)
                
            # If terminal or no legal moves, the value is 0
            if node.is_terminal or not node.children:
                node_value = 0
            else:
                # Value is the maximum of (reward + afterstate value) across all actions
                max_value = float('-inf')
                for action, chance_node in node.children.items():
                    action_value = chance_node.reward + chance_node.value / max(chance_node.visits, 1)
                    max_value = max(max_value, action_value)
                node_value = max_value
                
        elif isinstance(node, ChanceNode):
            # For a chance node, use the weighted average of its children's values
            if not node.children and not node.is_terminal:
                self._expand_chance_node(node)
                
            if node.is_terminal or not node.children:
                # Use the already computed value for this afterstate
                node_value = node.value / max(node.visits, 1)
            else:
                # Calculate weighted average based on outcome probabilities
                total_value = 0.0
                total_weight = 0.0
                for outcome, decision_child in node.children.items():
                    weight = outcome[3]  # Probability weight
                    child_value = decision_child.value / max(decision_child.visits, 1)
                    total_value += weight * child_value
                    total_weight += weight
                node_value = total_value / total_weight if total_weight > 0 else 0
        
        # Return the normalized value: (cumulative_reward + node_value) / value_norm
        return (cumulative_reward + node_value) / self.value_norm

    def _backpropagate(self, path, value):
        """
        Update the statistics of all nodes along the path with the evaluation value.
        Returns the combined reward + value for potential further use.
        
        Args:
            path: List of nodes from root to leaf
            value: Normalized value to backpropagate
            
        Returns:
            float: Total reward + value (un-normalized)
        """
        for node in reversed(path):
            node.visits += 1
            node.value += value
        
        return value * self.value_norm  
