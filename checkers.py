import copy
import pygame, sys
import random
import math

class CheckersGame:
    """ Moves are lists of tuples of ((start_x, start_y), (end_x, end_y)).
    For single moves, there is one tuple; for moves with multiple jumps,
    there are multiple tuples.

    Pieces are indicated by number.
        1 = player 1 (bottom)
        2 = player 2 (top)
        3 = player 1 king
        4 = player 2 king
    """
    def __init__(self, board=None, current_player=1, test=False):
        self.board = board if board else CheckersBoard(test=test)
        self.current_player = current_player

    def generate_successor(self, move=None):
        new_game_state = CheckersGame(self.board.copy(), self.current_player)
        if move:
            new_game_state.make_move(move)
        return new_game_state

    def make_move(self, moves):
        """ This method takes a list of moves (can be one move).
        It will execute all of them.
        """
        for move in moves:
            self.board.make_move(move)
        self.current_player = self.other_player()

    def get_legal_moves(self, player=None):
        """ Get and return a list of all legal moves for the given player.      
        """
        if not player:
            player = self.current_player
        jump_moves = []
        single_moves = []

        for y in range(self.board.dim):
            for x in range(self.board.dim):
                piece = self.board[y][x]
                if piece not in [player, player + 2]: # piece is not controlled by the player
                    continue

                jump_moves += self.get_jump_moves(x, y, self.board)
                if not jump_moves: # single moves are only an option if there aren't jump moves
                    single_moves += self.get_single_moves(x, y)

        return jump_moves if jump_moves else single_moves
    
    def get_single_moves(self, x, y):
        """ Get and return a list of all single moves for a piece
        (i.e. not jump moves).
        """
        piece = self.board[y][x]
        single_moves = []
        for px, py in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            if (piece == 1 and py < 0) or (piece == 2 and py > 0):
                continue  # regular pieces can't move towards their own side
            
            nx, ny = x + px, y + py  # position
            if not (nx in range(self.board.dim) and ny in range(self.board.dim)):
                continue  # out of bounds
            if self.board[ny][nx] != 0:
                continue  # new square is occupied

            move = ((x, y), (nx, ny)) # LEGAL SINGLE MOVE
            single_moves.append((move,))
        return single_moves
    
    def get_jump_moves(self, x, y, board):
        """ Get and return a list of all jump moves for a piece.
        Returns a list of move tuples; i.e. [
            ((x, y), (nx, ny)), ((nx, ny), (nnx, nny))
        ]
        """
        piece = board[y][x]
        if piece == 0:
            return []  # not a piece, so no moves

        if piece in [1, 3]:
            player = 1
            enemy_pieces = [2, 4]
        elif piece in [2, 4]:
            player = 2
            enemy_pieces = [1, 3]

        jump_moves = []
        for px, py in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            if (piece == 1 and py < 0) or (piece == 2 and py > 0):
                continue  # regular pieces can't move towards their own side
            
            nx, ny = x + px, y + py  # new position
            if not (nx in range(self.board.dim) and ny in range(self.board.dim)):
                continue  # new square out of bounds
            if board[ny][nx] != 0:
                continue  # new square is occupied already

            jx, jy = x + px // 2, y + py // 2 # jumped piece position
            if board[jy][jx] not in enemy_pieces:
                continue  # did not jump an enemy piece

            move = ((x, y), (nx, ny)) # LEGAL JUMP MOVE
            board_after_move = board.copy()
            board_after_move.make_move(move)

            subsequences = self.get_jump_moves(nx, ny, board_after_move) # recursive

            for seq in subsequences:
                jump_moves.append((move,) + seq)
            if not subsequences: # you can only stop jumping if you can't jump anymore
                jump_moves.append((move,))
            
        return jump_moves
    
    def other_player(self, other_player=None):
        if not other_player:
            other_player = self.current_player
        if other_player == 1:
            return 2
        elif other_player == 2:
            return 1
        else:
            raise ValueError("Player is not 1 or 2.")

    def is_game_over(self):
        # Return True if the game has ended, False otherwise
        return self.get_winner() is not None
    
    def get_winner(self):
        # Determine and return the winner of the game
        if not any(piece in self.board for piece in [1,3]) or not self.get_legal_moves(1):
            return 2
        elif not any(piece in self.board for piece in [2,4]) or not self.get_legal_moves(2):
            return 1
        else:
            return None # game has not ended

    def random_play(self):
        while not self.is_game_over():
            possible_moves = self.get_legal_moves()
            random_move = random.choice(possible_moves)
            self.make_move(random_move)
        winner = self.get_winner()
        return winner

    
class CheckersBoard:
    def __init__(self, board=None, test=False):
        if test:
            self.board = [[0,0,0,1],
                          [0,0,0,0],
                          [0,0,0,0],
                          [0,0,2,0]]

        elif board == None:
            self.board = [[0,1,0,1,0,1,0,1],
                          [1,0,1,0,1,0,1,0],
                          [0,1,0,1,0,1,0,1],
                          [0,0,0,0,0,0,0,0],
                          [0,0,0,0,0,0,0,0],
                          [2,0,2,0,2,0,2,0],
                          [0,2,0,2,0,2,0,2],
                          [2,0,2,0,2,0,2,0]]
        elif isinstance(board, list):
            self.board = board
        self.dim = len(self.board)

    def __getitem__(self, index):
        # Return the row corresponding to 'index', which allows further indexing
        return self.board[index]

    def __contains__(self, item):
        # Return if the board has a certain piece
        return any(item in row for row in self.board)

    def piece_count(self, piece):
        """ Returns the number of piece in the board. """
        count = 0
        for row in self.board:
            for p in row:
                if p == piece:
                    count += 1
        return count


    def copy(self):
        # Create a deep copy of this CheckersBoard
        new_board = CheckersBoard()
        new_board.board = copy.deepcopy(self.board)
        new_board.dim = self.dim
        return new_board

    def make_move(self, move, force=False):
        """ Update the board with a single move. """
        x, y = move[0] # piece
        nx, ny = move[1] # where to move it
        piece = self.board[y][x]

        if piece in [1, 3]:
            player = 1
            enemy_pieces = [2, 4]
        elif piece in [2, 4]:
            player = 2
            enemy_pieces = [1, 3]

        move_is_jump = abs(x - nx) == 2
        if move_is_jump:
            jx, jy = (x + nx) // 2, (y + ny) // 2
            assert self.board[jy][jx] != 0

        assert self.board[ny][nx] == 0
                
        ### START BOARD MODIFICATION
        self.board[y][x] = 0 # pick piece up
        self.board[ny][nx] = piece # put piece down
        if move_is_jump:
            self.board[jy][jx] = 0  # remove jumped piece

        if piece == 1 and ny == self.dim - 1:
            self.board[ny][nx] = 3 # Convert to king if piece reaches end of board.
        
        if piece == 2 and ny == 0:
            self.board[ny][nx] = 4 # Convert to king if piece reaches end of board.
