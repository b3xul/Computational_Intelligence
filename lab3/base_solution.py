import logging
from collections import namedtuple, Counter
import random
from functools import reduce
from operator import xor
from typing import Callable, List, Tuple
from matplotlib import pyplot as plt

Nimply = namedtuple("Nimply", "row, num_objects")
Individual = namedtuple("Individual", ["genome", "fitness"])

__CALLS__ = dict()


def CallCounter(fn):
    """Annotation @CallCounter"""
    assert (
        fn.__name__ not in __CALLS__
    ), f"Function '{fn.__name__}' already listed in __CALLS__"
    __CALLS__[fn.__name__] = 0

    # logger.debug(f"CallCounter: Counting __CALLS__['{fn.__name__}'] ({fn})")

    def call_count(*args, **kwargs):
        __CALLS__[fn.__name__] += 1
        return fn(*args, **kwargs)

    return call_count


class Nim:
    def __init__(self, rows: List, k: int = 2**32 - 1) -> None:
        # k is the maximum number of objects that can be removed in one move
        self._rows = rows  # [1, 3, 5, 7, 9]
        self._k = k

    def __bool__(self):
        # False when board is empty, in this way we can put it inside a while
        return sum(self._rows) > 0

    def __str__(self):
        # <1 3 5 7 9 11 13 15 17 19 5>
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
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
    Function to find MSB number for given n in O(1). N.B. Only works for numbers up to 32 bit
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
    Function used to extract numerical values from the current state of the game
    """
    cooked = dict()
    cooked["possible_moves"] = [
        (r, o)
        for r, c in enumerate(state.rows)
        for o in range(1, c + 1)
        if o <= state.k
    ]
    # Pick o sticks from row r
    # {'possible_moves': [(0, 1), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),...]}

    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
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


def pure_random_s(state: Nim) -> Nimply:
    """
    Pure random strategy
    """
    row = state.pick_random_row()
    num_objects = random.randint(1, min(state.rows[row], state.k))
    return Nimply(row, num_objects)


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
    logger.debug(
        f"Optimal move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
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

    logger.debug(
        f"Chosen move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
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

    logger.debug(
        f"Chosen move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
        f"{state.rows[row] - num_objects})"
    )
    return Nimply(row, num_objects)


def make_almost_optimal_strategy(genome: float) -> Callable:
    """
    Just used to have more different opponents, I didn't try to evolve this
    """

    def almost_optimal_s(state: Nim) -> Nimply:
        """
        Almost optimal strategy with any k, based on nim sum. number of object taken depends on the genome
        """
        data = cook_status(state)
        if data["nim_sum"] == 0:
            # If we are in a "cold" state, there are no winning moves, so we just take 1 object from any of the rows. We
            # take only 1 object to give the opponent more opportunities to make the wrong choice.
            return Nimply(state.pick_random_row(), 1)

        # elif data["nim_sum"] != 0:
        row = data["rows_with_highest_nim_bit_set"][0]
        # genome = float from 0 to 1
        num_objects = max(1, round(genome * data["modulated_rows"][row]))

        # print(data["modulated_rows"][row])
        # print(num_objects)
        logger.debug(
            f"Optimal move: remove {num_objects} from row {row} ({state.rows[row]}-{num_objects}="
            f"{state.rows[row] - num_objects})"
        )
        return Nimply(row, num_objects)

    return almost_optimal_s


def make_strategy(genome: dict) -> Callable:
    """
    Just used to have more different opponents, I didn't try to evolve this
    """

    def evolvable(state: Nim) -> Nimply:
        data = cook_status(state)
        # genome1: {"p": 0.9999}
        # genome2: {"p": 0.5}
        if random.random() < genome["p"]:
            ply = Nimply(
                data["shortest_row"],
                random.randint(1, min(state.rows[data["shortest_row"]], state.k)),
            )
        else:
            ply = Nimply(
                data["longest_row"],
                random.randint(1, min(state.rows[data["longest_row"]], state.k)),
            )

        return ply

    return evolvable


def choose_strategy_factory(genome: list) -> Callable:
    """
    The genome represents the probability of using the different strategies for a move. After some time,
    the agent will always choose the best strategy between the ones provided.
    """

    def select_strategy(state: Nim) -> Nimply:
        selected_strategy = random.choices(strategies, weights=genome, k=1)[0]
        return selected_strategy(state)

    return select_strategy


# def situations_strategy_factory(genome: dict) -> Callable:
#     # Parametrized rule! Hardcode rules and let it choose the best one. After some time it will always choose the
#     # nim_sum!
#     def evolvable(state: Nim) -> Nimply:
#         data = cook_status(state)
#         row = data["rows_with_highest_nim_bit_set"][0]
#         num_objects = genome[state]
#         return Nimply(row, num_objects)
#
#     return evolvable


def play_tests(tests: List, competing_strategies: Tuple) -> int:
    wins = 0
    for t in tests:
        logger.debug(f"status: Initial board  -> {t}")
        player = 0
        while t:
            ply = competing_strategies[player](t)
            t.nimming(ply)
            logger.debug(f"status: After player {player} -> {t}")
            player = 1 - player
        # player 0 won, since before exiting the while it was the last playing
        winner = 1 - player
        if winner == 0:
            wins += 1
        # logger.info(
        #     f"status: Player {winner} won {tmp.rows},{tmp.k} using {competing_strategies[winner].__name__}!\n\n"
        # )
    return wins


def function_name(f: Callable) -> str:
    """
    Print name of the function and genome used to create it
    """
    return f"{f.__name__} {f.__closure__[0].cell_contents if f.__closure__ else ''}"


@CallCounter
def fitness(strategy: Callable) -> int:
    """
    Fitness is computed as the total amount of games won against the other agents
    """
    matches = [(strategy, s) for s in strategies]
    logger_line = function_name(strategy) + "\t|\t"
    fitness = 0
    for competing_strategies in matches:
        NUM_MATCHES = 2
        correctly_won = 0
        unexpectedly_won = 0
        unknown_won = 0
        for i in range(NUM_MATCHES):
            # All tests_Starter can be won by the starting player
            tests_Starter = (
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
            tests_Opponent = [
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
            tests_Unknown = [
                Nim(list(range(1, i + 1)), j + 1) for i in range(10) for j in range(i)
            ]

            correctly_won += play_tests(tests_Starter, competing_strategies)
            unexpectedly_won += play_tests(tests_Opponent, competing_strategies)
            unknown_won += play_tests(tests_Unknown, competing_strategies)
        logger_line += (
            f"{correctly_won / (len(tests_Starter) * NUM_MATCHES) * 100:.2f}% / "
            f"{unexpectedly_won / (len(tests_Opponent) * NUM_MATCHES) * 100:.2f}% / "
            f"{unknown_won / (len(tests_Unknown) * NUM_MATCHES) * 100:.2f}% |\t"
        )
        fitness += correctly_won + unexpectedly_won + unknown_won
    logger.info(logger_line)
    return fitness


def tournament(population, tournament_size):
    """
    Parent selection: pick tournament size individuals from the population and return the fitter of them
    """

    result = max(random.sample(population, k=tournament_size), key=lambda i: i.fitness)
    return result


# Genetic operators
def cross_over(g1, g2):
    """recombination:
    not yet implemented
    """

    recombined_g = g1
    logger.debug(f"recombined genome={recombined_g}")
    return recombined_g


def mutation(g):
    """ """
    logger.debug(f"old genome:{g}")

    mutated_g = g.copy()

    highest = max(g)
    max_i = g.index(highest)

    candidates = list(filter(lambda prob: 0 < prob < highest, mutated_g))
    if candidates:
        to_reduce = random.choice(candidates)

        mutated_g[max_i] += mutated_g[mutated_g.index(to_reduce)]
        mutated_g[mutated_g.index(to_reduce)] = 0
    else:
        mutated_g = [random.random() for i in range(len(strategies))]
        s = sum(mutated_g)
        mutated_g = [0.5 * i / s for i in mutated_g]
    # while random.random() < MULTIPLE_MUTATIONS_RATE:
    #     logger.debug("multiple mutations")

    logger.debug(f"mutated genome:{mutated_g}")
    return mutated_g


# Graphic performance evaluation
def graphic_eval(fitness_log, g):
    gen_best = [max(f for f in fitness_log if f[0] == x) for x in range(g + 2)]
    plt.figure(figsize=(15, 6))
    plt.scatter([x for x, _ in fitness_log], [y for _, y in fitness_log], marker=".")
    plt.plot([x for x, _ in enumerate(gen_best)], [y for _, y in enumerate(gen_best)])
    plt.savefig(f"evaluation.png")
    plt.close()


def evolve(factory: Callable):
    BEST_FITNESS = fitness(optimal_s)
    # POPULATION_SIZE = 100
    # OFFSPRING_SIZE = 0.8 * POPULATION_SIZE
    # POPULATION_SIZE = len(all_lists)  N=30 -> 10 s
    POPULATION_SIZE = 20
    OFFSPRING_SIZE = POPULATION_SIZE // 2
    TOURNAMENT_SIZE = 2

    population = list()
    initial_probability = 1 / len(strategies)
    genome = [initial_probability] * len(strategies)
    # [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    logger.debug(f"genome:{genome}")
    population.append(Individual(genome, fitness(factory(genome))))

    for s in strategies:
        genome = [initial_probability - initial_probability / 2] * len(strategies)
        genome[strategies.index(s)] = initial_probability + (
            initial_probability / 2 * (len(strategies) - 1)
        )
        # sum([0.5625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]) = 1.0
        logger.debug(f"genome:{genome}")
        population.append(Individual(genome, fitness(factory(genome))))

    while len(population) < POPULATION_SIZE:
        genome = [random.random() for i in range(len(strategies))]
        s = sum(genome)
        genome = [i / s for i in genome]
        # in this way the sum of the probabilities remains 1.0
        logger.debug(f"genome:{genome}")
        population.append(Individual(genome, fitness(factory(genome))))

    logger.debug(f"population:{population}")

    best_solution = max(population, key=lambda i: i.fitness)
    logger.debug(f"best_solution:{best_solution}")

    best_generation = 0

    # Log of (generation number, fitness value)
    fitness_log = [(0, i.fitness) for i in population]
    logger.debug(fitness_log)

    NUM_GENERATIONS = 200

    MUTATION_RATE = 1
    # MULTIPLE_MUTATIONS_RATE = 0.4

    # Evolution
    steady = 0
    for g in range(NUM_GENERATIONS):
        if steady > NUM_GENERATIONS / 3:
            logger.debug("Reached steady state")
            g -= 1
            break
        logger.debug(f"GENERATION {g} population:{population}")
        offspring = list()
        while len(offspring) < OFFSPRING_SIZE:
            # Create OFFSPRING_SIZE individuals, by mutating or recombining parents
            if random.random() < MUTATION_RATE:
                winner = tournament(population, TOURNAMENT_SIZE)
                o = mutation(winner.genome)
            else:
                mother = tournament(population, TOURNAMENT_SIZE)
                father = tournament(population, TOURNAMENT_SIZE)
                while mother == father:
                    father = tournament(population, TOURNAMENT_SIZE)
                logger.debug(f"{mother} and {father} love each other")
                o = cross_over(mother.genome, father.genome)
            f = fitness(factory(o))
            fitness_log.append((g + 1, f))
            next_offspring = Individual(o, f)
            if next_offspring not in offspring and next_offspring not in population:
                logger.debug(f"adding {next_offspring} to population")
                offspring.append(next_offspring)
            logger.debug(f"current offspring:{offspring}")
        logger.debug(f"GENERATION {g} offspring:{offspring}")
        population.extend(offspring)

        # Sort population by fitness and discard the worst individual until only the first POPULATION_SIZE
        # individuals remain
        population = sorted(population, key=lambda i: i.fitness, reverse=True)[
            :POPULATION_SIZE
        ]

        if population[0].fitness > best_solution.fitness:
            steady = 0
            best_solution = population[0]
            best_generation = g
            logger.debug(
                f"New best solution {best_solution} found at generation {g} with weight {best_solution.fitness}"
            )
            if best_solution.fitness >= BEST_FITNESS:
                break
        else:
            steady += 1

    graphic_eval(fitness_log, g)
    # logger.info(f"Best Individual found:{best_solution}")
    return best_solution, best_generation


if __name__ == "__main__":
    logging.basicConfig(format="%(message)s", filename="tournament.tsv", filemode="w")

    logger = logging.getLogger("myLogger")
    logger.setLevel(logging.INFO)

    strategies = [
        optimal_s,
        pure_random_s,
        make_strategy({"p": 0.1}),
        make_strategy({"p": 0.5}),
        make_almost_optimal_strategy(0.1),
        make_almost_optimal_strategy(0.5),
        total_even_odd_s,
        optimize_single_row_s,
    ]
    # strategies = [optimal_s, optimize_single_row_s]
    logger_line = "\t\t\t\t|"
    for s in strategies:
        logger_line += f"\t{function_name(s)}\t|"
    logger.info(logger_line)

    # Full tournament
    # matches = list(product(strategies, repeat=2))
    #     for s in strategies:
    #         evaluate(s)
    # print(list(matches))

    # Check 1 strategy
    factory = choose_strategy_factory
    solution, generation = evolve(factory)

    logger.info(f"| {solution} | {generation} | {__CALLS__} |")
