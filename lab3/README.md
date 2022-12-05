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
## Situations_strategy_factory
Genome is represented as the probability of choosing one strategy or another: (`[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]`). I try to maintain this balance (sum of probabilities=1) even after the genetic operators action.
Fitness is computed as the total amount of games won against the other agents.
I didn't use recombination since mutation was enough to reach the optimal solution quickly (`[1.0, 0, 0, 0, 0, 0, 0, 0]` -> meaning that the agent will always choose the optimal strategy).
The mutation zeroes one of the probability and increases of that amount the probability of the best current strategy, converging quickly.
This strategy essentially converges to the optimal strategy between the ones provided. A potential improvement would be to take different parts from different strategies, depending on the game state. This could allow the agent to beat expert systems.
## Make_almost_optimal_strategy
The idea was to hardcode the optimal row and let the correct number of objects to be removed be evolved autonomously. This was not explored much for lack of time. As it is right now the genome represents a single number. In a working solution it would probably require a more complex genome that accounts for the different game situations and evolves the best number based on each situation