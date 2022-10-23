import logging
from collections import Counter, defaultdict
import random
from itertools import chain, accumulate
import time
import cProfile as profile
import pstats

from gx_utils import PriorityQueue


def problem(N, seed=None):
    """Creates an instance of the problem"""

    random.seed(seed)
    return [
        list(
            set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2)))
        )
        for n in range(random.randint(N, N * 5))
    ]


def flatten(state):
    """Extracts iterable numbers from a state (tuple of tuples of numbers)"""
    return chain.from_iterable(state)


def goal_test(state, GOAL):
    """Checks if the state contains all the numbers contained in GOAL"""
    return set(flatten(state)) == GOAL


def possible_steps(state, all_lists):
    """Returns the list of all lists with at least one element which is not part of the current solution"""
    current = set(flatten(state))
    return [l for l in all_lists if not set(l) <= current]


def w(state):
    """Computes the weight by dividing the total occurrences of each number in the current state and dividing it by
    the amount of times a number appears only once in the current state. We prefer states with lower weights. Lowest
    weight is 1.0."""
    cnt = Counter()
    cnt.update(tuple(flatten(state)))

    return (sum(cnt.values())) / (sum(cnt[c] == 1 for c in cnt))


def initialize_sol(N, all_lists):
    """Add to the initial state the lists that we already know will be part of the final solution since they are the
    only ones that contain a certain number"""
    initial_state = tuple()

    # Build a dictionary using as key all numbers and as value a tuple of all the tuples that contain that number
    tuples_containing_num = defaultdict(tuple)
    for i, t in enumerate(all_lists):
        for num in t:
            tuples_containing_num[num] = (*tuples_containing_num[num], t)

    for num in range(N):
        if num not in tuples_containing_num:
            # No solutions exist for this N and these lists!
            return None
        if len(tuples_containing_num[num]) == 1:
            initial_state = (*initial_state, *tuples_containing_num[num])

    return initial_state


def solve(N, all_lists):
    GOAL = set(range(N))
    all_lists = tuple(
        set(tuple(_) for _ in all_lists)
    )  # all_lists = ((0, 1), (2, 4), (2,), (0, 4), (1, 2, 4), (1, 2))
    frontier = PriorityQueue()
    nodes = 0

    # We use as state a tuple of tuples
    state = initialize_sol(N, all_lists)
    if state is None:
        return None, None

    state_costs = dict()

    while state is not None and not goal_test(state, GOAL):
        nodes += 1
        next_steps = possible_steps(state, all_lists)
        for s in next_steps:
            next_state = (*state, s)
            next_distinct_numbers = tuple(set(flatten(next_state)))
            next_weight = w(next_state)
            if (next_distinct_numbers not in state_costs) or (
                next_weight < state_costs[next_distinct_numbers]
            ):
                logging.debug(
                    f"Pushing to the frontier {next_state} with priority (weight of the solution)="
                    f"{next_weight}"
                )
                state_costs[next_distinct_numbers] = next_weight
                frontier.push(next_state, p=next_weight)
        state = frontier.pop()
        logging.debug(f"State with best weight:{state}")
    return state, nodes


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    PROFILING = True

    if PROFILING:
        prof = profile.Profile()
        prof.enable()
    for N in [5, 10, 20, 30, 40]:
        available_lists = problem(N, seed=42)
        if PROFILING:
            solving_start = time.process_time()
        solution, nodes = solve(N, available_lists)
        if PROFILING:
            solving_end = time.process_time()
            logging.info(f"SOLUTION time:{(solving_end - solving_start):.3f} seconds")
        if solution:
            weight = sum(len(_) for _ in solution)
            logging.info(
                f"Found solution for N={N} in {nodes} steps: {solution}\nw={weight:,} (bloat="
                f"{(weight - N) / N * 100:.0f}%); iw={w(solution)}"
            )
        else:
            logging.info(f"No solution exists for N={N} with lists={available_lists}")
    if PROFILING:
        prof.disable()
        # print profiling output
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(10)  # top 10 rows
