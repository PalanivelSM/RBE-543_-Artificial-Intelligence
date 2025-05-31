from games4e import Game, random
from collections import namedtuple
import numpy as np


# Define a GameState for handling moves, utility, board, and current player
GameState = namedtuple('GameState', 'to_move, utility, board, moves')


class TicTacToe(Game):
    """A Tic-Tac-Toe game class supporting customizable board sizes and win conditions.

    Inherits from game class and allows user for creation of a Tic-Tac-Toe game
    with specified dimensions and a required number of consecutive marks to win.

    Attributes:
        h (int): The height of the board (number of rows).
        v (int): The width of the board (number of columns).
        k (int): The number of consecutive marks required to win.
        initial (GameState): The initial state of the game, including the first player to move,
            initial utility, empty board, and list of possible moves.
    """

    def __init__(self, h=3, v=3, k=3):
        """Initialize a TicTacToe board of size h x v with k in a row to win."""
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Return all legal moves — any unoccupied square."""
        return state.moves

    def result(self, state, move):
        """Return the resulting state from taking a move."""
        if move not in state.moves:
            return state  # Ignore invalid moves
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(
            to_move='O' if state.to_move == 'X' else 'X',
            utility=self.compute_utility(board, move, state.to_move),
            board=board,
            moves=moves
        )


    def utility(self, state, player):
        """Return utility score for the given player."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """Check if the game is over (win or draw)."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        """Display the current state of the board."""
        board = state.board
        for x in range(1, self.h + 1):
            row = ' '.join(board.get((x, y), '.') for y in range(1, self.v + 1))
            print(row)
        print()

    def compute_utility(self, board, move, player):
        """Determine utility based on whether the current move wins the game."""
        if (self.k_in_row(board, move, player, (0, 1)) or
            self.k_in_row(board, move, player, (1, 0)) or
            self.k_in_row(board, move, player, (1, -1)) or
            self.k_in_row(board, move, player, (1, 1))):
            return 1 if player == 'X' else -1
        return 0

    def k_in_row(self, board, move, player, delta_xy):
        """Check if there are k marks in a row (horizontal, vertical, diagonal)."""
        delta_x, delta_y = delta_xy
        x, y = move
        count = 0

        # Count in one direction
        while board.get((x, y)) == player:
            count += 1
            x += delta_x
            y += delta_y

        # Reset and count the other direction
        x, y = move
        while board.get((x, y)) == player:
            count += 1
            x -= delta_x
            y -= delta_y

        count -= 1  # Adjust for counting the move itself twice
        return count >= self.k


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# Define player strategies
def query_player(game, state):
    """Human player inputs a move."""
    print("Current state:")
    game.display(state)
    print("Available moves:", game.actions(state))
    move = None
    if game.actions(state):
        move_string = input("Your move? (e.g., (1, 2)): ")
        try:
            move = eval(move_string)
        except (NameError, SyntaxError):
            print("Invalid move format! Try again.")
            return query_player(game, state)
    return move

def minmax_decision(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""
    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))

def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

def heuristic_alpha_beta_player(game, state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
        This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth + 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if cutoff_test(state, depth):
            return eval_fn(state)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth + 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_cutoff_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = (cutoff_test or (lambda state, depth: depth > d or game.terminal_test(state)))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def ucb(node, C=1.4):
    """Calculate UCB1 value for a node."""
    if node['N'] == 0:
        return np.inf
    return node['U'] / node['N'] + C * np.sqrt(np.log(node['parent']['N']) / node['N'])

def select(n):
    """Select a leaf node in the tree using UCB1."""
    if n['children']:
        return select(max(n['children'], key=ucb))
    else:
        return n

def expand(n, game):
    """Expand a node by adding its children states."""
    if not n['children'] and not game.terminal_test(n['state']):
        n['children'] = [
            {'state': game.result(n['state'], action), 'parent': n, 'children': [], 'N': 0, 'U': 0}
            for action in game.actions(n['state'])
        ]
    return select(n)

def simulate(game, state):
    """Simulate the utility of the current state by playing randomly."""
    player = game.to_move(state)
    while not game.terminal_test(state):
        action = random.choice(list(game.actions(state)))
        state = game.result(state, action)
    v = game.utility(state, player)
    return -v

def backdrop(n, utility):
    """Backdrop the result up the tree, updating utilities and visit counts."""
    while n:
        n['U'] += utility
        n['N'] += 1
        utility = -utility  # Flip utility for the opponent
        n = n['parent']

def mcts_player(game, state, N=1000):
    """Monte Carlo Tree Search player function."""
    root = {'state': state, 'parent': None, 'children': [], 'N': 0, 'U': 0}

    # Run N iterations of MCTS
    for _ in range(N):
        leaf = select(root)
        child = expand(leaf, game)
        result = simulate(game, child['state'])
        backdrop(child, result)

    # Choose the child with the highest visit count
    best_action_index = np.argmax([child['N'] for child in root['children']])
    best_action = game.actions(state)[best_action_index]
    # best_action = max(root['children'], key = lambda p: p['N'])
    return best_action

