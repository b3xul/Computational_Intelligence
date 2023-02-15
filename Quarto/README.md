# Instructions to import best Player
```python
import Players

game = quarto.Quarto()
game.set_players((Players.RandomPlayer(game), Players.HardPlayer(game)))
winner = game.run()
logging.warning(f"main: Winner: player {winner}")
```

# main
Functions `tournament` and `play_match` were used to make Players compete between each other and obtain these results:

| win % / draw % / lose % | HardPlayer                 | EasyPlayer                 | RandomPlayer               |
| ----------------------- | -------------------------- | -------------------------- | -------------------------- |
| HardPlayer              | 0.00%  / 100.00% /  0.00%  | 90.00% /  10.00%  /  0.00% | 100.00% /  0.00%  /  0.00% |
| EasyPlayer              | 0.00%  /  0.00%  / 100.00% | 40.00%  /  0.00%  / 60.00% | 90.00% /  0.00%  /  10.00% |
| RandomPlayer            | 0.00%  /  0.00%  / 100.00% | 0.00%  /  0.00%  / 100.00% | 40.00%  /  0.00%  / 60.00% |

# QuartoUtilities
Each player creates an instance of this class during initialization (`self.quarto_utilities = QuartoUtilities(game)`).
This class groups useful elements used by the players:
- `self.board` simple `np.ndarray` to represent the current board status. Easy to manipulate.
- `self.winning_lines` is initialized in 0.1ms when the class is instantiated and contains a list with all 536 winning lines, like (0,2,4,8). This let us search the line from the board in a dictionary instead of looking at the binary board to see if the line forms a Quarto or not.
- `def remaining_pieces(self, board: np.ndarray) -> list[int]:` Given a board, returns the list of pieces that still needs to be placed.
- `def get_free_positions(self, board: np.ndarray) -> list[tuple[int, int]]:` Given a board, returns a list of free positions encoded as (x,y) tuples. ex. `[(0,0),(1,2),(3,3)]`
- `def check_lines(self, board: np.ndarray, selected_piece: int, winning: bool = True) -> tuple[int, int] | None:` -> Given a board and a selected piece, returns (if it exists) the (x,y) position where a piece can be placed to obtain (`winning=True`) a Quarto, or to block (`winning=false`) a Quarto.

# EasyPlayer

This is a simple Fixed Rule agent, still able to consistently beat the RandomPlayer. I built this first, since it gave me a sufficiently strong baseline to test my HardPlayer.
- If it can choose a piece which doesn't result in a Quarto, it does, otherwise it chooses a random piece.
- If it can form a Quarto with the selected piece, it does, otherwise place the piece on a random, free position.

# HardPlayer

I chose to build the best MinMax player that I could because I believed that, if symmetries are properly considered, the exploration space can be greatly reduced.
I tried my best to optimize the algorithm so that I could reach a reasonable exploration depth, with the goal of winning consistently against my EasyPlayer.
I decided to include both the decisions that the player must take (where to place the selected piece and which piece to give to the opponent) in a single minmax function, saving the next piece to give to the opponent in a property of the player (`self.next_piece`). In this way when the game asks for the next piece to give to the opponent, after having placed mine, I already have it memorized.

I encoded the evaluation of a move in this structure `MoveEvaluation = namedtuple("MoveEvaluation", ["evaluation", "piece_to_place", "x", "y", "next_piece"])`.
- `"evaluation"`: 1 if placing "piece_to_place" in "x","y" leads to Quarto. 0 if it leads to a Draw. -1 if it leads to a forced Loss.
- `"next_piece"`: None if placing "piece_to_place" in "x","y" leads to Quarto. Otherwise next piece to choose

The algorithm works as follows:
1. If this board (or a symmetrical one) with this selected piece were already analyzed, return that evaluation (see Memoization) (terminal condition)
2. If we can place the selected piece to obtain a Quarto, do it and save that evaluation. (terminal condition)
3. If only one moves remains, but it does not lead to a Quarto, it means it leads to a draw, do it and save that evaluation. (terminal condition)
4a. If there are too many states to explore, use a non-terminal heuristic evaluation function to cutoff the search (see Non-terminal evaluation)
4b. Otherwise we can try exploring all possible positions and all possible remaining pieces (see Alpha-Beta pruning)

## Memoization
I used the HashableBoardPiece class as a wrapper to insert a board and a selected piece as the key of a dictionary.  
This allows the creation of a dictionary like `{(-1, -1, 0, 15), (-1, -1, -1, -1), (-1, -1, -1, 15), (-1, -1, -1, 0)}15: MoveEvaluation(evaluation=0, piece_to_place=15, x=1, y=0, next_piece=1)` so that we can avoid computing the evaluation of the placement of a piece on a certain board that we already seen.

In the hash function I consider 2 boards equal if they have the same pieces in different places. It does not matter in which order those are placed because I sort all lines, remove duplicate and extract a unique string representation for that board. To the board I add the selected piece and those form the dictionary key.

## Non-terminal evaluation
I used an hardcoded value as parameter to decide when to stop the exhaustive search.

```python
branching_factor = len(possible_positions) * (len(possible_positions) - 1)  
if branching_factor ** depth > 150_000:
	# Non-terminal evaluation
```
This definition allows the agent to be able to choose the next move in less than 10 seconds almost every time.
Since it takes into consideration the current state of the board, the maximum depth reachable changes as the board if filled. This means that even if at the start of the game it can't reach depths like 6 or 7 due to the amount of time that would be required, towards the end of the game it is able to do it.

The non-terminal evaluation stops the minmax recursion, deciding heuristically the next move:
1. Instead of trying all positions, if it exists, picks a position that blocks a tris, otherwise pick a random position. The idea is that by blocking a tris, maybe the game will continue for longer, until there are few possible positions remaining. This gives the agent the opportunity to explore all possible states and find the optimal move.
2. If it can choose a piece which doesn't result in a Quarto, it does (in this case the evaluation of the move will be a draw), otherwise it chooses a random piece (since all pieces result in a Quarto, the evaluation of the move will be a draw).

## Alpha-Beta pruning
Terminal states only return MoveEvaluations with 1 or 0.
The previous player takes that value, negated.
If we find a MoveEvaluation = 1, this means that the opponent's best evaluation was -1, so the opponent's best move was a loss.
This means that by doing this move the current player is guaranteed to win, so we can do it and stop the search early.

