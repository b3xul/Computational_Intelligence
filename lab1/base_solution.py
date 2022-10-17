import itertools
import random
import logging
from collections import defaultdict

n = 0


def problem(N, seed=None):
    random.seed(seed)
    # returns a list containing from N to 5N (from 5 to 25, from 10 to 50,..) lists Li
    # Each list will contain from N//5 to N//2 distinct numbers between 0 and N-1 (0..4, 0..9)
    return [
        set(
            set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2)))
        )
        for n in range(random.randint(N, N * 5))
    ]


def solution_exists(N, setList):
    for i in range(N):
        if not any(i in sublist for sublist in setList):
            return False
    return True


def solve(N):
    setList = problem(N, seed=42)
    logging.debug(f"N={N} lists={setList}")
    setList = [
        set(item) for item in set(frozenset(item) for item in setList)
    ]  # remove duplicates
    if solution_exists(N, setList):
        solution = setCover(
            setList, target=set(range(N))
        )  # Using * to unpack iterables into a set

        logging.info(
            f"Found solution for N={N} with weight={sumLen(solution)} in {n} steps: {solution}"
        )
    else:
        logging.info("No solution exists")


def sumLen(sol):
    return sum(len(l) for l in sol)


def setCover(setList, target):
    if not setList:
        return None
    bestCover = []
    for i, values in enumerate(setList):
        remaining = target - values
        if remaining == target:  # not adding any number to the solution
            continue
        if not remaining:  # solution found
            return [values]
        subCover = setCover(setList[i + 1 :], remaining)
        if not subCover:
            continue
        global n
        n += 1
        if not bestCover or sumLen(subCover) + len(values) < sumLen(bestCover):
            # print(f"values={values}\nsubCover={subCover}")
            bestCover = [values] + subCover
    return bestCover


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    for N in [5, 10, 20, 22]:
        solve(N)
