import math, random
from checkers import CheckersGame, CheckersBoard
import sys

class Node:
    """ A node class to help with Monte Carlo search. Nodes keep information
    about a game_state, the previous game_state and the move taken, the win
    ratio, and the depth of the game state (i.e. move count).
    """
    def __init__(self, node=None, game_state=None, move=None):
        if isinstance(node, Node):
            self.game_state = node.game_state.generate_successor(move)
            self.parent = node
            self.depth = node.depth + 1
        elif isinstance(game_state, CheckersGame): # only used for generating the root
            self.game_state = game_state
            self.parent = None
            self.depth = 0
        else:
            raise ValueError(f"Cannot create node; no node or game state given.")
            
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

    def __eq__(self, other):
        return self.game_state == other.game_state

class Agent:
    """ Class stub for all Agents (they should implement these methods). """
    def __str__(self):
        return "Unspecified Agent"

    def get_action(self, game_state):
        pass

class RandomAgent(Agent):
    """ An agent whose get_action method randomly choooses a successor state. """
    def __str__(self):
        return "Random Agent"

    def get_action(self, game_state):
        """ Return a random move given game_state. """
        legal_moves = game_state.get_legal_moves()
        return random.choice(legal_moves)

class MonteCarloSearchAgent(Agent):
    """ An agent which runs Monte Carlo Tree Search (MCTS)
    and whose get_action method returns the move which maximizes
    the ratio of wins to visits in the MCTS simulation. """
    def __init__(self, trials=200, debug=False):
        self.trials = trials
        self.debug = debug

    def __str__(self):
        return 'Monte Carlo Search Agent'

    def get_action(self, game_state):
        """ Runs MCTS. After MCTS is run, the node with the highest win to visits ratio is
        selected and returned
        """
        root = self.run_MCTS(game_state)

        best_child = max(root.children, key=lambda n: self.get_win_probability(n))
        return best_child.move

    def run_MCTS(self, game_state):
        """ Generates a MCTS tree for game_state and returns the root node.
        In each trial, a leaf node on the self.root tree is selected and expanded,
        and then a random simulation is run generating a winner.
        The results backpropogated up the tree for each simulation. """
        root = Node(game_state=game_state)
        for i in range(self.trials):
            leaf_node = self.select_leaf_node(root)
            self.expand_leaf_node(leaf_node)
            result = self.simulate(leaf_node)
            self.backpropogate(leaf_node, result)

            if not self.debug:
                sys.stdout.write(f"\r{i} trials complete")
                sys.stdout.flush()
        print()

        return root

    def get_win_probability(self, node):
        """ A node/game_state's win probability is the ratio of wins
        to visits in the MCTS simulations. """
        return node.wins / node.visits

    def select_leaf_node(self, root):
        """ Selects a leaf node (i.e. a node which hasn't had its
        children generated) and returns it.
        """
        current_node = root
        while current_node.children:
            current_node = self.select_child_node_with_UCB1(current_node)
        return current_node

    def select_child_node_with_UCB1(self, node, c=1.4):
        """ Select a successor state node for a game_state node using
        the UCB1 formula. """
        best_child_nodes = []
        best_weight = float('-inf')
        for child_node in node.children:
            if child_node.visits == 0:
                weight = float('inf')
            else:
                weight = (child_node.wins / child_node.visits) + c * math.sqrt(
                    (2 * math.log(node.visits) / child_node.visits)
                )

            if weight > best_weight:
                best_child_nodes = []
                best_weight = weight
            if weight == best_weight:
                best_child_nodes.append(child_node)

        return random.choice(best_child_nodes)

    def select_child_node_randomly(self, node):
        """ Randomly select a successor state node for a game_state node. """
        return random.choice(node.children)

    def expand_leaf_node(self, node):
        """ Expands a leaf node (i.e. generates all of its children
        and adds them to its children list). """
        node.children = [Node(node=node, move=move) for move in node.game_state.get_legal_moves()]

    def simulate(self, node):
        """ Simulates random game play starting at node's game_state.
        Returns the result (who won). """
        game_copy = node.game_state.generate_successor(move=None)
        winner = game_copy.random_play()

        return winner

    def backpropogate(self, node, winner):
        """ Updates win counts for node and all ancestor nodes up until root.
        Winner represents who won. """
        current_node = node
        while current_node:
            current_node.visits += 1
            current_node.wins += int(current_node.game_state.current_player == winner)
            current_node = current_node.parent

class MinimaxSearchAgent:
    def __init__(self, depth=2):
        self.evaluation_function = self.get_player_piece_ratio
        self.depth = depth

    def __str__(self):
        return "MiniMax Agent"

    def get_player_piece_ratio(self, game_state, player=None):
        """ Assumes player is 1 or 2 or unspecified, in which case
        the function will assign player to game_state.current_player.
        """
        if not player:
            player = game_state.current_player
        other_player = game_state.other_player(player)
        player_count = (
            game_state.board.piece_count(player) +
            game_state.board.piece_count(player+2)
        )
        other_player_count = (
            game_state.board.piece_count(other_player) +
            game_state.board.piece_count(other_player + 2)
        )
        if other_player_count == 0:
            return float('inf')
        return player_count / other_player_count

    def get_action(self, game_state):
        action_successor = {action: game_state.generate_successor(action)\
            for action in game_state.get_legal_moves()}

        action_value = {action: self.min_value(successor)\
            for action, successor in action_successor.items()}
        
        return max(action_value, key=lambda x: action_value[x])
    
    def min_value(self, game_state, depth=0):
        """ Recursive min layers in the Minimax algorithm.

        This is the equivalent of every ghost determining the lowest
        value of all the moves they can make. """
        if game_state.is_game_over() or depth == self.depth:
            return self.evaluation_function(game_state)

        v = float('inf')
        for action in game_state.get_legal_moves():
            successor = game_state.generate_successor(action)
            v = min(v, self.max_value(successor, depth=depth))
        return v

    def max_value(self, game_state, depth=0):
        """ The complete max function in the Minimax algorithm.

        This is the equivalent of Pacman deciding the highest value
        of all moves he can make. """
        if game_state.is_game_over() or depth == self.depth:
            return self.evaluation_function(game_state)

        v = float('-inf')
        for action in game_state.get_legal_moves():
            successor = game_state.generate_successor(action)
            v = max(v, self.min_value(successor, depth=depth + 1))
        return v

minimax_agent = MinimaxSearchAgent()
game = CheckersGame()
print(minimax_agent.get_player_piece_ratio(game, 1))