import logging
import random

from lab3.Nim_utilities import Nim, Nimply, cook_status


def optimal_s(state: Nim) -> Nimply:
    """
    Optimal strategy with any k, based on nim sum
    """
    data = cook_status(state)
    if data["nim_sum"] == 0:
        # If we are in a "cold" state, there are no winning moves, so we just take 1 object from any of the rows. We
        # take only 1 object to give the opponent more opportunities to make the wrong choice.
        return Nimply(state.pick_random_row(), 1)

    # elif data["nim_sum"] != 0:

    # 1. Compute the modulo k+1 of all rows
    # 2. Compute the nim_sum of the modulated rows
    # 3. Find the highest bit set in the nim_sum of the modulated rows
    # 4. From any of the modulated rows that have that bit set, pick a number of object so that the nim_sum will
    # become = 0. That number is given by data["modulated_rows"][row] - (data["nim_sum"] ^ data["modulated_rows"][row])
    row = data["rows_with_highest_nim_bit_set"][0]
    num_objects = data["modulated_rows"][row] - (
        data["nim_sum"] ^ data["modulated_rows"][row]
    )
    logging.debug(
        f"Optimal move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
        f"{state.rows[row] - num_objects})"
    )
    return Nimply(row, num_objects)


def optimize_single_row_s(state: Nim) -> Nimply:
    """
    This agent tries to optimize winning every row as if it was the only row in the game. If it is able to take a
    full row, it does. Otherwise, it tries to leave a row with state.k + 1 elements (or a multiple of it), so that the
    opponent won't be able to take that row.
    """
    data = cook_status(state)
    row = num_objects = None
    for r in sorted(state.rows, reverse=True):
        if r == 0:
            break
        if r <= state.k:
            row = state.rows.index(r)
            num_objects = r
            break
        n = r % (state.k + 1)
        if n != 0:
            row = state.rows.index(r)
            num_objects = n
            break
    if row is None:
        row = data["shortest_row"]
        num_objects = 1

    logging.debug(
        f"Chosen move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
        f"{state.rows[row] - num_objects})"
    )
    return Nimply(row, num_objects)


def total_even_odd_s(state: Nim) -> Nimply:
    """
    if total remaining objects is even, just remove the max amount of objects from the smaller heap, otherwise see
    which is the largest heap with an amount of objects that repeat in an odd number of heaps
    (1 1 2 2 3 3 3 5 5 5 6 6 -> the 5 repeats an odd number of times).
    We remove from it as many objects as needed to arrive to the previous heap:
    (removing 2 objects we arrive to 1 1 2 2 3 3 5 5 5 5 6 6, which is a winning position).
    """
    data = cook_status(state)
    if data["total"] % 2 == 0:
        row = data["shortest_row"]
        num_objects = 1
    else:
        # print(data["counter"])
        sorted_rows_by_appearance = sorted(data["counter"], reverse=True)
        for i, r in enumerate(sorted_rows_by_appearance):
            if i == len(sorted_rows_by_appearance) - 1:
                # if we reached the last row without finding an odd amount of heaps with same value, we just remove
                # k or an entire row
                row = state.rows.index(r)
                num_objects = min(state.k, state.rows[row])
                break
            if data["counter"][r] % 2 != 0:
                # the row of len "r" appears an odd number of times -> remove from it as many objects as needed to
                # arrive to the previous heap
                row = state.rows.index(r)
                num_objects = min(state.k, r - sorted_rows_by_appearance[i + 1])
                break

    logging.debug(
        f"Chosen move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
        f"{state.rows[row] - num_objects})"
    )
    return Nimply(row, num_objects)


def pure_random_s(state: Nim) -> Nimply:
    """
    Pure random strategy
    """
    row = state.pick_random_row()
    num_objects = random.randint(1, min(state.rows[row], state.k))
    return Nimply(row, num_objects)


hardcoded_strategies = [
    optimal_s,
    optimize_single_row_s,
    total_even_odd_s,
    pure_random_s,
]
