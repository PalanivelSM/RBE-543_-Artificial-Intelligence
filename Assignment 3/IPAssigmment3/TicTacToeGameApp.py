import random
from TicTacToeClass import *
def play_tic_tac_toe():
    """Handles the entire Tic Tac Toe game logic, including player selection and rounds."""

    def choose_player(player_name):
        """
        Prompts the user to select a player type for X and O.

        Parameters:
        player_name (str): The name of the player (e.g., "first Player" or "second Player") to prompt the user.

        Returns:
        Player: The Player corresponding to the chosen player type.
        """

        # Dictionary mapping player choices to their name and function
        player_choices = {
            "1": {
                "name": "Random Player",  # Player description
                "Player": random_player  # Player function
            },
            "2": {
                "name": "MiniMax Player",  # Player description
                "Player": minmax_decision  # Player function
            },
            "3": {
                "name": "Alpha Beta Player",  # Player description
                "Player": alpha_beta_search  # Player function
            },
            "4": {
                "name": "Heuristic Alpha Beta Player",  # Player description
                "Player": heuristic_alpha_beta_player  # Player function
            },
            "5": {
                "name": "MCTS Player",  # Player description
                "Player": mcts_player  # Player function
            },
            "6": {
                "name": "Query Player",  # Player description
                "Player": query_player  # Player function
            }
        }

        # Prompt the user to choose a player
        choice = input(f"Please enter your {player_name}: ")

        # Validate the user input
        while choice not in player_choices:
            choice = input("Could not find your player, please try again: ")

        # Return the function corresponding to the chosen player
        return player_choices[choice]["Player"]

    def face_off(current_game, player_x, player_o, round_number):
        """
        Handles a single round of Tic Tac Toe between Player X and Player O.

        Parameters:
        current_game (object): The current game object containing game state and methods.
        player_x (function): The function representing Player X's strategy.
        player_o (function): The function representing Player O's strategy.
        round_number (int): The current round number in the series.

        Returns:
        str: The outcome of the game in this round, which can be 'X', 'O', or 'Draw'.
        """
        print(f"\nRound {round_number}:")
        print("---------------------")
        print("Available Actions by Player X:", current_game.initial.moves)

        # Initial state of the game
        state = current_game.initial

        # Game loop for a single round
        while not current_game.terminal_test(state):
            current_player_symbol = state.to_move
            current_player = player_x if current_player_symbol == 'X' else player_o
            move = current_player(current_game, state)

            # Validate the move
            if move not in current_game.actions(state):
                print("Invalid move. Try again!")
                continue

            # Update the game state with the move
            state = current_game.result(state, move)
            print(f"\nThe Action taken by Player {current_player_symbol} is: {move}")
            current_game.display(state)

            print(f"Player {current_player_symbol}'s Utility:", current_game.utility(state, current_player_symbol),
                  "\n")

        # Determine the winner or if the game ended in a draw
        if current_game.utility(state, 'X') == 1:
            print(f"Player X won the game in Round {round_number}")
            return 'X'
        elif current_game.utility(state, 'O') == 1:
            print(f"Player O won the game in Round {round_number}")
            return 'O'
        else:
            print(f"Player X and Player O drew the game in Round {round_number}")
            return 'Draw'

    print("\nWelcome to Tic Tac Toe!\n")
    game = TicTacToe()

    print('''Player Selection:
    1. Random Player
    2. MiniMax Player
    3. Alpha Beta Player
    4. Heuristic Alpha Beta Player
    5. MCTS Player
    6. Query Player\n''')

    player_x = choose_player("first Player")
    player_o = choose_player("second Player")

    scores = {'X': 0, 'O': 0, 'Draw': 0}

    for round_num in range(1, 4):
        winner = face_off(game, player_x, player_o, round_num)
        scores[winner] += 1
        print("----------------------------------")
        if scores['X'] == 2:
            print("Player X wins two out of three rounds in the game!")
            print("\n----------------------------------")
            return
        if scores['O'] == 2:
            print("\n Player O wins two out of three rounds in the game!")
            print("\n----------------------------------")
            return

    if scores['X'] == scores['O']:
        print("\n No Player can win two out of three rounds in the game")
    else:
        champions = 'X' if scores['X'] > scores['O'] else 'O'
        print(f"\nPlayer {champions} wins the series!, The Champions")
    print("\n----------------------------------")

def main():
    """
       The main function to start the Tic Tac Toe game and it allows players to play the game multiple times.
    """
    while True:
        play_tic_tac_toe()
        # Did User want to perform another search?
        try_again = input("Would you like to play the game again? (yes/no): ")
        if try_again == "no":
            print("Thank You for Playing our Game!")
            break
        elif try_again == "yes":
            continue
        else:
            input("Please enter Valid choice (yes/no): ")


if __name__ == "__main__":
    main()
