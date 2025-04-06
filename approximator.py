import math
import pickle
from collections import defaultdict
from transforms import rot90, rot180, rot270, flip_horizontal

class PatternDefaultDict(defaultdict):
    """A defaultdict implementation that returns a fixed value for missing keys."""
    def __init__(self, init_value):
        super().__init__(float)
        self.init_value = init_value
        
    def __missing__(self, key):
        return self.init_value

class NTupleApproximator:
    def __init__(self, board_size, patterns, v_init=0):
        """
        Initializes the N-Tuple approximator.
        
        Args:
            board_size (int): Size of the 2048 game board (typically 4x4)
            patterns (list): List of coordinate tuples defining the patterns
            v_init (float): Initial value for optimistic initialization
        """
        self.board_size = board_size
        self.patterns = patterns
        
        # Calculate initialization values
        self.init_values = [0] * len(patterns)
        if v_init > 0:
            init_value = v_init / len(patterns)
            for p in range(len(patterns)):
                self.init_values[p] = init_value
        
        # Create weight dictionaries with appropriate default values
        self.weights = [PatternDefaultDict(self.init_values[p]) for p in range(len(patterns))]
        
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)
            
        # Initialize TC learning parameters
        self.tc_enabled = False
        self.tc_error_sum = [defaultdict(float) for _ in patterns]  # E_i
        self.tc_error_abs_sum = [defaultdict(float) for _ in patterns]  # A_i
    
    def optimistic_initialization(self, v_init):
        """
        Apply optimistic initialization to all weights.
        
        Args:
            v_init (float): The initial value to distribute across features
        """
        # Distribute v_init evenly across all patterns
        init_value = v_init / len(self.patterns)
        
        for p in range(len(self.patterns)):
            # Update the init value for this pattern
            self.init_values[p] = init_value
            
            # Create a new dictionary with the updated default value
            self.weights[p] = PatternDefaultDict(init_value)
    
    def generate_symmetries(self, pattern):
        # Generate 8 symmetrical transformations of the given pattern
        syms = [
            pattern,
            rot90(pattern),
            rot180(pattern),
            rot270(pattern),
            flip_horizontal(pattern),
            rot90(flip_horizontal(pattern)),
            rot180(flip_horizontal(pattern)),
            rot270(flip_horizontal(pattern))
        ]
        return syms
    
    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """
        Extract tile values from the board based on the given coordinates
        and convert them into a feature tuple.
        """
        return tuple(self.tile_to_index(board[x, y]) for (x, y) in coords)

    def value(self, board):
        """
        Estimate the board value: sum the evaluations from all patterns
        while considering symmetries.
        """
        total = 0.0
        for p in range(len(self.patterns)):
            pattern_sum = 0.0
            for sym in self.symmetry_patterns[p]:
                index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)
                # Add error handling to deal with unexpected types
                try:
                    weight_value = self.weights[p][index]
                    if isinstance(weight_value, (int, float)):
                        pattern_sum += weight_value
                    else:
                        # If not a number, use 0.0 as default
                        pattern_sum += 0.0
                except Exception as e:
                    # Handle any exception when accessing weights
                    print(f"Warning: Error accessing weight for pattern {p}, index {index}: {e}")
                    # Continue with a default value of 0
                    continue
            total += pattern_sum / 8.0  # Average across 8 symmetries
        return total / len(self.patterns)
    
    def update(self, board, delta, alpha):
        """
        Update the weights for all patterns and their symmetries.
        
        Args:
            board: The game board state
            delta: The TD error
            alpha: Learning rate
            
        Returns:
            float: New value of the board
        """
        if not self.tc_enabled:
            # Standard TD update
            for p in range(len(self.patterns)):
                for sym in self.symmetry_patterns[p]:
                    index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)
                    self.weights[p][index] += alpha * delta
        else:
            # TC update with adaptive learning rates
            for p in range(len(self.patterns)):
                for sym in self.symmetry_patterns[p]:
                    index = tuple(self.tile_to_index(board[r, c]) for (r, c) in sym)
                    
                    # Update TC parameters
                    self.tc_error_sum[p][index] += delta
                    self.tc_error_abs_sum[p][index] += abs(delta)
                    
                    # Calculate coherence
                    if self.tc_error_abs_sum[p][index] != 0:
                        coherence = abs(self.tc_error_sum[p][index]) / self.tc_error_abs_sum[p][index]
                    else:
                        coherence = 1.0
                    
                    # Update weight with adaptive learning rate
                    self.weights[p][index] += alpha * coherence * delta
                    
        return self.value(board)
    
    def enable_tc(self):
        """Enable Temporal Coherence learning"""
        self.tc_enabled = True
        
    def disable_tc(self):
        """Disable Temporal Coherence learning"""
        self.tc_enabled = False
    
    def save(self, filename):
        """Save the weights to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, filename):
        """Load the weights from a file."""
        with open(filename, 'rb') as f:
            self.weights = pickle.load(f) 