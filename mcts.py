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
        # Cache legal moves to avoid recalculating
        self.untried_actions = [action for action in range(4) if self.env.is_move_legal(action)]
        # Terminal flag (if game is over)
        self.is_terminal = self.env.is_game_over() or not self.untried_actions

    def uct_select_child(self, approximator, exploration=0):
        """Select a child chance node using the UCT formula."""
        best_value = -float('inf')
        best_child = None
        
        # Fast path for no exploration (pure exploitation)
        if exploration <= 0:
            for action, child in self.children.items():
                if child.visits == 0:
                    return child  # First visit any unvisited child
                if child.value > best_value:
                    best_value = child.value
                    best_child = child
            return best_child
            
        # Standard UCT with exploration
        log_visits = math.log(self.visits) if self.visits > 1 else 0
        for action, child in self.children.items():
            if child.visits == 0:
                return child  # First visit any unvisited child
                
            # UCT formula
            exploitation = child.value / child.visits
            exploration_term = exploration * math.sqrt(log_visits / child.visits)
            uct = exploitation + exploration_term
            
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
        # Cache the outcomes
        self.untried_outcomes = self._get_possible_outcomes()
        # Check terminality: if the board is full, no tile can be added.
        self.is_terminal = env.is_game_over() or not self.untried_outcomes

    def _get_possible_outcomes(self):
        """Generate possible outcomes efficiently."""
        outcomes = []
        board = self.env.board
        empty_cells = list(zip(*np.where(board == 0)))
        
        if not empty_cells:
            return []
            
        num_empty = len(empty_cells)
        # Precompute probabilities once
        prob2 = 0.9 / num_empty
        prob4 = 0.1 / num_empty
        
        for cell in empty_cells:
            outcomes.append((cell[0], cell[1], 2, prob2))
            outcomes.append((cell[0], cell[1], 4, prob4))
            
        return outcomes

    def sample_outcome(self):
        """
        When the chance node is fully expanded, sample one of its decision-node children
        according to the tile addition probabilities.
        """
        if not self.children:
            return None
            
        # Optional: Use precomputed probabilities list if calling often
        children_list = list(self.children.values())
        weights = [outcome[3] for outcome in self.children.keys()]
        
        # Faster sampling with numpy
        if len(weights) > 10:  # Arbitrary threshold for numpy optimization
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            idx = np.random.choice(len(children_list), p=weights)
            return children_list[idx]
        # Standard sampling for smaller sets
        elif sum(weights) > 0:
            return random.choices(children_list, weights=weights, k=1)[0]
        else:
            return random.choice(children_list)


class MCTS:
    """
    Modified MCTS for the 2048 game using afterstate value function.
    Alternates between decision nodes (agent moves) and chance nodes (random tile additions).
    """
    def __init__(self, env, approximator, iterations=5000, exploration=0, value_norm=20000):
        self.root = DecisionNode(env)
        self.approximator = approximator
        self.iterations = iterations
        self.exploration = exploration
        self.value_norm = value_norm  # Normalization constant for the value function
        # Cache for board evaluations to avoid redundant calls to approximator
        self._eval_cache = {}

    def search(self):
        """
        Run MCTS for a given number of iterations and return the best action from the root.
        """
        # First, fully expand the root node to compute all possible afterstates
        if not self.root.children:
            self._expand_decision_node(self.root)
            
        # If no legal moves, return None
        if not self.root.children:
            return None
            
        # When exploration is 0 and iterations is small, we can optimize
        if self.exploration == 0:
            # Single-level maximization can be faster with few iterations
            return self._greedy_action_selection()
            
        # Standard MCTS process
        for _ in range(self.iterations):
            # Selection: traverse the tree to a leaf node
            leaf, path, cumulative_reward = self._tree_policy(self.root)
            # Evaluation: use the approximator instead of rollout
            value = self._evaluate_node(leaf, cumulative_reward)
            # Backpropagation: update the nodes along the path with the evaluation
            self._backpropagate(path, value)
            
        # Select the action with highest visit count
        return self._best_action_by_visits()

    def _greedy_action_selection(self):
        """Fast greedy selection based only on immediate rewards and afterstate values."""
        best_action = None
        best_value = -float('inf')
        
        for action, chance_node in self.root.children.items():
            # Direct value estimate without MCTS tree growth
            value = chance_node.reward + chance_node.value
            if value > best_value:
                best_value = value
                best_action = action
                
        return best_action

    def _best_action_by_visits(self):
        """Select action with most visits from root."""
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
        
        # Loop while not at a terminal node
        while not current.is_terminal:
            # Decision node handling
            if isinstance(current, DecisionNode):
                # Expand if there are untried actions
                if current.untried_actions:
                    self._expand_decision_node(current)
                
                # Check if expansion failed (no legal moves)
                if not current.children:
                    current.is_terminal = True
                    break
                
                # Select next node
                chance_child = current.uct_select_child(self.approximator, self.exploration)
                if not chance_child:
                    break
                
                cumulative_reward += chance_child.reward
                path.append(chance_child)
                current = chance_child
                
            # Chance node handling
            elif isinstance(current, ChanceNode):
                # Expand if there are untried outcomes
                if current.untried_outcomes:
                    self._expand_chance_node(current)
                
                # Check if expansion failed
                if not current.children:
                    current.is_terminal = True
                    break
                
                # Sample next node
                decision_child = current.sample_outcome()
                if not decision_child:
                    break
                
                path.append(decision_child)
                current = decision_child
                
            # Terminal check after a complete iteration
            if current.is_terminal:
                break
        
        return current, path, cumulative_reward

    def _expand_decision_node(self, node):
        """
        Expand a decision node by creating all possible chance node children.
        Optimized to reuse the environment.
        """
        env_snapshot = copy.deepcopy(node.env)  # Create one copy to reuse
        
        for action in node.untried_actions:
            # Reset to snapshot before each action
            action_env = copy.deepcopy(env_snapshot)
            # Perform the move WITHOUT spawning a random tile
            reward = action_env.step(action, spawn_tile=False)[1]
            
            # Create a chance node
            chance_child = ChanceNode(action_env, node, action, reward)
            node.children[action] = chance_child
            
            # Calculate and cache value
            board_tuple = tuple(map(tuple, action_env.board))
            if board_tuple in self._eval_cache:
                chance_child.value = self._eval_cache[board_tuple]
            else:
                val = self.approximator.value(action_env.board) 
                self._eval_cache[board_tuple] = val
                chance_child.value = val
                
        # Clear untried actions
        node.untried_actions = []

    def _expand_chance_node(self, node):
        """
        Expand a chance node by creating all possible decision node children.
        Optimized for performance.
        """
        env_snapshot = copy.deepcopy(node.env)  # Create one copy to reuse
        empty_mask = (env_snapshot.board == 0)
        
        # Process outcomes in batches for efficiency
        for outcome in node.untried_outcomes:
            x, y, tile, prob = outcome
            
            # Create a new environment
            outcome_env = copy.deepcopy(env_snapshot)
            outcome_env.board[x, y] = tile
            
            # Create the decision node
            decision_child = DecisionNode(outcome_env, parent=node, action_from_parent=outcome)
            node.children[outcome] = decision_child
            
        # Clear untried outcomes
        node.untried_outcomes = []

    def _evaluate_node(self, node, cumulative_reward):
        """
        Evaluate a node using the value function instead of a rollout.
        Uses caching to avoid redundant calculations.
        """
        # For terminal nodes or empty nodes, return 0 value
        if node.is_terminal or not hasattr(node, 'env'):
            return (cumulative_reward) / self.value_norm
            
        if isinstance(node, DecisionNode):
            # Expand if not already done
            if not node.children and not node.is_terminal:
                self._expand_decision_node(node)
                
            # Calculate max value from children
            if node.children:
                max_value = max(
                    chance_node.reward + chance_node.value 
                    for chance_node in node.children.values()
                )
            else:
                max_value = 0  # No valid moves
                
            node_value = max_value
                
        elif isinstance(node, ChanceNode):
            # Use precalculated value
            node_value = node.value
            
        # Return normalized value
        return (cumulative_reward + node_value) / self.value_norm

    def _backpropagate(self, path, leaf_value):
        """
        Update the statistics of all nodes along the path with their specific rewards and values.
        Optimized for better numerical stability.
        """
        value_update = leaf_value  # Base value for updates
        
        for node in reversed(path):
            node.visits += 1
            
            if isinstance(node, ChanceNode):
                # For chance nodes, include their specific reward
                node_value = node.reward / self.value_norm + value_update
                # Use a moving average update to improve stability
                node.value = node.value + (node_value - node.value) / node.visits
            else:
                # For decision nodes, just update with the leaf value
                node.value = node.value + (value_update - node.value) / node.visits
