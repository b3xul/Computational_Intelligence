import logging
from collections import namedtuple, Counter
from functools import reduce
from operator import xor
import random
from typing import List, Tuple

Nimply = namedtuple("Nimply", ["row", "num_objects"])


class Nim:
    def __init__(self, rows: List, k: int = 2**32 - 1) -> None:
        # k is the maximum number of objects that can be removed in one move
        self._rows = rows  # [1, 3, 5, 7, 9]
        self._k = k

    def __bool__(self):
        # False when board is empty, in this way we can put it inside a while
        return sum(self._rows) > 0

    def __str__(self):
        # <1 2 3 5 0 4>, k=2
        return f"<{self._rows}>,k={self._k}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        """
        Returns "useful" version of the state to avoid duplicates: remove zeroes and sort rows by value, so that both
        <1 2 3 5 0 4>, k=2 and <1 2 5 0 4 3>, k=2 are considered equal to (1, 2, 3, 4, 5), 2
        """
        return hash(str(tuple([sorted([x for x in self._rows if x != 0]), self._k])))

    def __eq__(self, other):
        return hash(self) == hash(other)

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        """
        Perform ply on the Nim
        """
        row, num_objects = ply
        # nimming=one move where we remove num_object from row row
        assert self._rows[row] >= num_objects
        assert num_objects <= self._k
        self._rows[row] -= num_objects

    def pick_random_row(self) -> int:
        return random.choice([r for r, c in enumerate(self._rows) if c > 0])


def nim_sum(l: List) -> int:
    return reduce(xor, l)


def setBitNumber(n):
    """
    Find MSB number for given n in O(1). N.B. Only works for numbers up to 32 bit
    """
    # https://www.geeksforgeeks.org/find-significant-set-bit-number/

    # Below steps set bits after MSB (including MSB)

    # Suppose n is 273 (binary is 100010001). It does the following: 100010001 | 010001000 = 110011001
    n |= n >> 1

    # This makes sure 4 bits (From MSB and including MSB) are set. It does the following 110011001 | 001100110 =
    # 111111111
    n |= n >> 2

    n |= n >> 4
    n |= n >> 8
    n |= n >> 16

    # Increment n by 1 so that there is only one set bit which is just before original MSB. n now becomes 1000000000
    n = n + 1

    # Return original MSB after shifting. n now becomes 100000000
    return n >> 1


def cook_status(state: Nim) -> dict:
    """
    Extract numerical values from the current state of the game
    """
    cooked = dict()

    # Pick o sticks from row r
    cooked["possible_moves"] = [
        Nimply(r, o)
        for r, c in enumerate(state.rows)
        for o in range(1, c + 1)
        if o <= state.k
    ]
    # {'possible_moves': [(0, 1), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),...]}

    # cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    # 'active_rows_number': 10

    cooked["shortest_row"] = min(
        (x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1]
    )[0]
    # 'shortest_row': 0, (row index)

    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[
        0
    ]
    # 'longest_row': 9, (row index)

    cooked["modulated_rows"] = [n % (state.k + 1) for n in state.rows]
    # (5, 8, 16, 30, 32), with k=3 becomes [1, 0, 0, 2, 0]

    cooked["nim_sum"] = nim_sum(cooked["modulated_rows"])
    # 'nim_sum': 2

    cooked["highest_nim_bit"] = setBitNumber(cooked["nim_sum"])
    # cooked["nim_sum"]=7 -> cooked["highest_nim_bit"]=4

    cooked["rows_with_highest_nim_bit_set"] = [
        i
        for i, r in enumerate(cooked["modulated_rows"])
        if r & cooked["highest_nim_bit"]
    ]
    # 'modulated_rows': [1, 2, 3, 4, 0] -> 'highest_nim_bit': 4 -> 'rows_with_highest_nim_bit_set': [3]
    # 'highest_nim_bit': 0 ->  'rows_with_highest_nim_bit_set': []

    cooked["total"] = sum(state.rows)
    # (0, 2, 3, 4, 5, 6) -> 'total': 20

    cooked["counter"] = Counter(state.rows)
    # (0, 2, 3, 4, 5, 6) -> counter': Counter({0: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})}
    return cooked


def play_tests(tests: List, competing_strategies: Tuple) -> int:
    """
    Returns amount of win of the starter player over tests, when Nimplies are selected using the 2 competing_strategies
    """
    wins = 0
    for t in tests:
        logging.debug(f"status: Initial board  -> {t}")
        player = 0
        while t:
            ply = competing_strategies[player](t)
            t.nimming(ply)
            logging.debug(f"status: After player {player} -> {t}")
            player = 1 - player
        # player 0 won, since before exiting the while it was the last playing
        winner = 1 - player
        if winner == 0:
            wins += 1
    return wins


# All tests_starter can be won by the starter player
tests_starter = (
    [Nim([i]) for i in range(1, 20)]
    + [
        Nim([1, 1, 1]),
        Nim([1, 1, 2]),
        Nim([1, 2, 2]),
        Nim([2, 2, 2]),
        Nim([3, 3, 3]),
        Nim([1, 2, 4]),
        Nim([1, 2, 3, 4, 5]),
        Nim([1, 2, 3, 4, 5, 5]),
        Nim([1, 2, 3, 5]),
        Nim([5, 8, 16, 32, 32]),
    ]
    + [
        Nim([3], 1),
        Nim([4], 2),
        Nim([5], 1),
        Nim([5], 2),
        Nim([5], 3),
        Nim([6], 3),
        Nim([5, 8, 16, 32, 32], 2),
        Nim([5, 8, 16, 32, 32], 3),
    ]
)
# All tests_Opponent can be won by the opponent player
tests_opponent = [
    Nim([2], 1),
    Nim([3], 2),
    Nim([4], 1),
    Nim([4], 3),
    Nim([5], 4),
    Nim([6], 1),
    Nim([6], 2),
    Nim([1, 2, 3]),
]
# For tests_Unknown I don't know if the starting position is winning for the Starter or the Opponent (an
# improvement would be to add to the Nim class a field to represent this information (just need to
# compute the nim_sum of the starting position in the constructor)
tests_unknown = [Nim(list(range(1, i + 1)), j + 1) for i in range(10) for j in range(i)]
