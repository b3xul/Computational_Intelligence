# Task3.1: An agent using fixed rules based on *nim-sum* (i.e., an *expert system*)
## Optimal Strategy
If we are in a "cold" state, there are no winning moves, so we just take 1 object from any of the rows. We take only 1 object to give the opponent more opportunities to make the wrong choice.
If we are in an "hot" state, we can compute the optimal move in this way:
1. Compute the modulo k+1 of all rows (`cooked["modulated_rows"] = [n % (state.k + 1) for n in state.rows]`)
2. Compute the nim_sum of the modulated rows  (`cooked["nim_sum"] = nim_sum(cooked["modulated_rows"])`)
3. Find the highest bit set in the nim_sum of the modulated rows  (`cooked["highest_nim_bit"] = setBitNumber(cooked["nim_sum"])`)
4. From any of the modulated rows that have that bit set (`cooked["rows_with_highest_nim_bit_set"] = [i for i, r in enumerate(cooked["modulated_rows"]) if r & cooked["highest_nim_bit"]]`), pick a number of object so that the nim_sum will become = 0. That number is given by `data["modulated_rows"][row] - (data["nim_sum"] ^ data["modulated_rows"][row])`
## Optimize Single Row Strategy
This agent tries to optimize winning every row as if it was the only row in the game. If it is able to take a full row, it does. Otherwise, it tries to leave a row with state.k + 1 elements (or a multiple of it), so that the opponent won't be able to take that row.
## Total Even Odd Strategy
If total remaining objects is even, just remove the max amount of objects from the smaller heap.
If total remaining objects is odd, see which is the largest heap with an amount of objects that repeat in an odd number of heaps (1 1 2 2 3 3 3 5 5 5 6 6 -> the 5 repeats an odd number of times).
We remove from it as many objects as needed to arrive to the previous heap: (removing 2 objects we arrive to 1 1 2 2 3 3 5 5 5 5 6 6, which is a winning position).
# Task3.2: An agent using evolved rules
## Choose_strategy_factory
Genome is represented as the probability of choosing one strategy or another: (`[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]`). I try to maintain this balance (sum of probabilities=1) even after the genetic operators action.
Fitness is computed as the total amount of games won against the other agents.
I didn't use recombination since mutation was enough to reach the optimal solution quickly (`[1.0, 0, 0, 0, 0, 0, 0, 0]` -> meaning that the agent will always choose the optimal strategy).
The mutation zeroes one of the probability and increases of that amount the probability of the best current strategy, converging quickly.
This strategy essentially converges to the optimal strategy between the ones provided. A potential improvement would be to take different parts from different strategies, depending on the game state. This could allow the agent to beat expert systems.
## Make_almost_optimal_strategy
The idea was to hardcode the optimal row and let the correct number of objects to be removed be evolved autonomously. This was not explored much for lack of time. As it is right now the genome represents a single number. In a working solution it would probably require a more complex genome that accounts for the different game situations and evolves the best number based on each situation
# Task3.3: An agent using minmax
## Minmax Strategy
An unoptimized version of minmax that just recursively explored all possible states was very slow: `minmax_rec calls:13 452 859 | Total time taken: 542.213s`

### Memoization
I added a dictionary that mapped positions already explored to the best move found for that position.
For this I customized the hash function for the Nim class so that the dictionary keys where not full positions like (`<1 2 3 5 0 4>, k=2`), but only a stripped version of the position, that excluded zeroes and sorted values. In this way multiple positions could be mapped to the same state. (`<1 2 3 5 0 4>, k=2` and `<1 2 5 0 4 3>, k=2` both correspond to `(1, 2, 3, 4, 5), 2)`.
Since this moves rows, the index of the row cannot be used anymore to select a row in the original state. For this reason I created a new tuple: `MoveEvaluation = namedtuple("MoveEvaluation", ["evaluation", "row_content", "num_objects"])` that contains the evaluation of the ply "remove num_objects from the row that contains row_content items".
This memoization allow us to avoid evaluating multiple times known states, saving time.

### Alpha-beta pruning
A further optimization that I did was a sort of alpha-beta pruning, specialized for Nim.
I only considered possible evaluations as 1, If the move can lead to a win, and -1 if the move leads to a lose.
For this reason, following the minmax strategy, as soon as I find a move that leads to a win, I can stop exploring other possible moves, since I can be sure that that move will lead to a win.

With these 2 improvements, performance greatly increases: `minmax_rec calls: 4 427 583 | Total time taken: 151.419s`

## Minmax Montecarlo Strategy
This is a possible tradeoff to reduce the time needed to explore complex states with high-depth game trees. After a certain depth, stop exploring exhaustively, and just evaluate the state by doing random moves until the end of the game.
- Complete exploration
```
state <[5, 8, 16, 32, 32]>,k=4294967295 | reached level:93 | pruned 2 795 841 branches

minmax_rec calls: 3 100 520 | Total time taken: 67.971s
```
- Montecarlo
```
state <[5, 8, 16, 32, 32]>,k=4294967295 | reached level:26 | pruned 3 591 967 branches
state <[3, 8, 11, 2, 2]>,k=4294967295 | reached level:1 | pruned 23 branches

minmax_rec calls: 1 066 035 | Total time taken: 31.200s
```
The performance greatly increases, but the strategy chooses the wrong move, so its evaluation is not enough representative of the real state of the game.
In Nim where every move can swing the outcome of the game, doing many random moves is probably not a good way to evaluate the state of the game.

## RecursionStats
I kept track of the max depth of the game tree and the amount of branches pruned to understand better how much the solutions were improving and which was a reasonable level for cutting the max-depth.

# Task3.4: An agent using reinforcement learning
I couldn't make this part before the deadline