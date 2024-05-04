from checkers import CheckersGame, CheckersBoard
from search import MonteCarloSearchAgent, RandomAgent, MinimaxSearchAgent

import tkinter as tk
from tkinter import scrolledtext


class Game:
    def __init__(self, agent1, agent2, test=False):
        self.agent1 = agent1
        self.agent2 = agent2
        self.game_state = CheckersGame(test=test)
        self.winner = 0

        self.test = test

    def get_agent(self):
        player = self.game_state.current_player
        return self.player_to_agent(player)

    def player_to_agent(self, player):
        if player == 1:
            return self.agent1
        if player == 2:
            return self.agent2
        else:
            return None


    def run(self):
        i = 1
        while not self.game_state.is_game_over():
            agent = self.get_agent()

            display_info = f"Turn {i}: Player {self.game_state.current_player}, {agent.__str__()}"
            print(display_info)

            move = agent.get_action(self.game_state)
            if self.test:
                print(self.game_state.get_legal_moves())
            self.game_state.make_move(move)
            i += 1

        ### Game is over now
        self.winner = self.game_state.get_winner()
        self.print_winner()

    def run_display(self, display):
        i = 1
        while not self.game_state.is_game_over():
            agent = self.get_agent()

            display_info = f"Turn {i}: Player {self.game_state.current_player}, {agent.__str__()}"
            print(display_info)
            display.insert(tk.END, display_info + "\n") ###

            move = agent.get_action(self.game_state)
            if self.test:
                print(self.game_state.get_legal_moves())
            self.game_state.make_move(move)
            i += 1

        ### Game is over now
        self.winner = self.game_state.get_winner()
        display.insert(tk.END, f"Game Over. Winner: Player {self.winner}\n")

    def print_winner(self):
        winning_agent = self.player_to_agent(self.winner)
        print(f"Winner: Player {self.winner}, {winning_agent.__str__()}")

class SimpleCheckersGUI:
    def __init__(self, game, test=False):
        self.game = game
        self.root = tk.Tk()
        self.root.title("Checkers Game")
        self.canvas = tk.Canvas(self.root, width=400, height=400, borderwidth=0, highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand="true")
        self.tiles = {}
        self.update_board()

        self.test = test

    def update_board(self):
        self.canvas.delete("square")
        for row in range(self.game.game_state.board.dim):
            for col in range(self.game.game_state.board.dim):
                x1 = col * 50
                y1 = row * 50
                x2 = x1 + 50
                y2 = y1 + 50
                color = "white" if (row + col) % 2 == 0 else "black"
                self.tiles[(row, col)] = self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray", fill=color, tags="square")
                piece = self.game.game_state.board[row][col]
                if piece != 0:
                    if piece == 1:  # Assuming 1 for agent1's pieces and 2 for agent2's pieces
                        piece_color = "red"
                    if piece == 2:
                        piece_color = "blue"
                    if piece == 3:
                        piece_color = "pink"
                    if piece == 4:
                        piece_color = "green"

                    self.canvas.create_oval(x1 + 10, y1 + 10, x2 - 10, y2 - 10, outline="gray", fill=piece_color)

        self.root.after(1000, self.next_move)  # Update the board every 1000 ms (1 second)

    def next_move(self):
        if not self.game.game_state.is_game_over():
            agent = self.game.get_agent()
            move = agent.get_action(self.game.game_state)
            if self.test:
                print(self.game.game_state.current_player)
                print(agent.__str__())
                print(self.game.game_state.get_legal_moves())
                print(move)
            self.game.game_state.make_move(move)
            self.update_board()
        else:
            winner = self.game.game_state.get_winner()
            agent = self.game.player_to_agent(winner).__str__()
            print(f"Game Over: Winner is Player {winner}: {agent}")

    def run(self):
        self.root.mainloop()


def main():
    monte_carlo_search_agent = MonteCarloSearchAgent(debug=False)
    random_agent = RandomAgent()
    for i in range(100):
        root = tk.Tk()
        game = Game(random_agent, monte_carlo_search_agent, test=False)
        game.run()
        # app = SimpleCheckersGUI(game, test=True)
        # app.run()

main()