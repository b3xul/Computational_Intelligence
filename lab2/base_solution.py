import cProfile as profile
import logging
import pstats
import random
import time
from collections import Counter, defaultdict
from collections import namedtuple
from itertools import chain

from matplotlib import pyplot as plt

# Parameters
TOURNAMENT_SIZE = 10

NUM_GENERATIONS = 500

PROFILING = True

Individual = namedtuple("Individual", ["genome", "fitness"])


# Copied from lab1
def problem(N, seed=None):
    """Creates an instance of the problem"""

    random.seed(seed)
    return [
        list(
            set(random.randint(0, N - 1) for n in range(random.randint(N // 5, N // 2)))
        )
        for n in range(random.randint(N, N * 5))
    ]


def initialize_sol(N, all_lists):
    """Add to the initial genome the lists that we already know will be part of the final solution since they are the
    only ones that contain a certain number"""
    initial_genome = tuple()

    # Build a dictionary using as key all numbers and as value a tuple of all the tuples that contain that number
    tuples_containing_num = defaultdict(tuple)
    for i, t in enumerate(all_lists):
        for num in t:
            tuples_containing_num[num] = (*tuples_containing_num[num], t)

    for num in range(N):
        if num not in tuples_containing_num:
            # No solutions exist for this N and these lists!
            return None, None
        if len(tuples_containing_num[num]) == 1:
            initial_genome = (*initial_genome, *tuples_containing_num[num])
        tuples_containing_num[num] = tuple(
            sorted(tuples_containing_num[num], key=lambda x: len(x))
        )
    return initial_genome, tuples_containing_num


def flatten(genome):
    """Extracts iterable numbers from a genome (set of tuples of numbers)"""
    return chain.from_iterable(genome)


def fitness(genome):
    """(amount of numbers covered by the current genome, -total length of the current genome). We prefer individual
    with higher fitness. Max fitness=(N,-N)."""
    cnt = Counter()
    cnt.update(tuple(flatten(genome)))
    return len(set(cnt)), -sum(cnt.values())


def tournament(population, tournament_size=TOURNAMENT_SIZE):
    """Parent selection: pick tournament size individuals from the population and return the fitter of them"""

    result = max(random.sample(population, k=tournament_size), key=lambda i: i.fitness)
    return result


# Genetic operators
def cross_over(g1, g2, GOAL):
    """recombination:
    - take common parts of the 2 genomes
    - remove 0 or more tuples from the longest parent (only if it already covers all numbers)
    - add one of the non-common tuples of the shorter parent to the longest parent"""
    commons = g1.intersection(g2)
    main_parent = max(g1, g2, key=lambda g: len(g)).copy()
    if main_parent == g1:
        other_parent = g2
    else:
        other_parent = g1
    remainders = main_parent.difference(commons)
    # Take 1 or more sample element from the remainders set
    while set(flatten(main_parent)) == GOAL and remainders:
        swap_out = random.sample(remainders, 1)[0]
        logger.debug(f"removing {swap_out} from {main_parent}")
        main_parent.remove(swap_out)
        remainders = main_parent.difference(commons)

    other_remainders = other_parent.difference(commons)
    if other_remainders:
        main_parent.add(random.sample(other_remainders, 1)[0])

    recombined_g = main_parent
    logger.debug(f"recombined genome={recombined_g}")
    return recombined_g


def mutation(g, GOAL, all_lists, MULTIPLE_MUTATIONS_RATE):
    """mutation:
    - remove 0 or more tuples from the genome (only if it already covers all numbers)
    - add 1 or more tuples from the lists not already in the genome
    """
    mutated_g = g.copy()
    logger.debug(f"old genome:{mutated_g}")
    while set(flatten(mutated_g)) == GOAL:
        el = random.sample(mutated_g, 1)[0]  # Take 1 sample element from the tmp set
        logger.debug(f"removing {el}")
        mutated_g.remove(el)
    remainders = set(all_lists).difference(g)
    selected_tuple = random.sample(remainders, 1)[0]
    mutated_g.add(selected_tuple)
    while random.random() < MULTIPLE_MUTATIONS_RATE:
        logger.debug("multiple mutations")
        selected_tuple = random.sample(remainders, 1)[0]
        mutated_g.add(selected_tuple)
    logger.debug(f"mutated genome:{mutated_g}")
    return mutated_g


# Graphic performance evaluation
def graphic_eval(fitness_log, g, N):
    gen_best = [max(f[1] for f in fitness_log if f[0] == x) for x in range(g + 1)]
    plt.figure(figsize=(15, 6))
    plt.scatter([x for x, _ in fitness_log], [y[0] for _, y in fitness_log], marker=".")
    plt.plot([x for x, _ in enumerate(gen_best)], [y for _, y in enumerate(gen_best)])
    plt.savefig(f"evaluation{N}.png")


def solve(N, all_lists):
    logger.info(f"searching solution for N={N}")
    GOAL = set(range(N))
    BEST_FITNESS = (N, -N)

    all_lists = tuple(
        set(tuple(_) for _ in all_lists)
    )  # all_lists = ((0, 1), (2, 4), (2,), (0, 4), (1, 2, 4), (1, 2))
    # We use as state a tuple of tuples
    initial_state, tuples_containing_num = initialize_sol(N, all_lists)
    if initial_state is None:
        return None, None

    population = list()
    POPULATION_SIZE = min([len(all_lists), 100])
    OFFSPRING_SIZE = 0.8 * POPULATION_SIZE
    # Build initial population, composed of POPULATION_SIZE individuals of 1 unique tuples
    for t in all_lists:
        if t not in initial_state:
            genome = (*initial_state, t)
            logger.debug(f"genome:{genome}")
            genome = set(genome)
            # genome is a set of tuple
            logger.debug(f"genome:{genome}")
            population.append(Individual(genome, fitness(genome)))

    best_solution = max(population, key=lambda i: i.fitness)
    best_generation = 0
    logger.debug(f"initial population:{population}\nbest:{best_solution}")
    if best_solution.fitness == BEST_FITNESS:
        logger.debug(f"Found optimal solution in the initial population (Generation 0)")
        return best_solution, best_generation

    # Log of (generation number, fitness value)
    fitness_log = [(0, i.fitness) for i in population]
    logger.debug(fitness_log)

    MUTATION_RATE = 0.5
    MULTIPLE_MUTATIONS_RATE = 0.4
    # Evolution
    steady = 0
    for g in range(NUM_GENERATIONS):
        if steady > NUM_GENERATIONS / 3:
            break
            # MUTATION_RATE = MUTATION_RATE * 1.5
            # MULTIPLE_MUTATIONS_RATE = MULTIPLE_MUTATIONS_RATE + 0.1
            # steady = 0

        logger.debug(f"GENERATION {g} population:{population}")
        offspring = list()
        while len(offspring) < OFFSPRING_SIZE:
            # Create OFFSPRING_SIZE individuals, by mutating or recombining parents
            if random.random() < MUTATION_RATE:
                winner = tournament(population)
                o = mutation(winner.genome, GOAL, all_lists, MULTIPLE_MUTATIONS_RATE)
            else:
                mother = tournament(population)
                father = tournament(population)
                while mother == father:
                    father = tournament(population)
                logger.debug(f"{mother} and {father} love each other")
                o = cross_over(mother.genome, father.genome, GOAL)
            f = fitness(o)
            fitness_log.append((g + 1, f))
            next_offspring = Individual(o, f)
            if next_offspring not in offspring and next_offspring not in population:
                logger.debug(f"adding {next_offspring} to population")
                offspring.append(next_offspring)
            logger.debug(f"current offspring:{offspring}")
            # population.append(Individual(o, f)) would be WRONG!!! We don't want offspring to be selected as parent
            # in the next step of the same generation! They can only be selected as parents in the next generation!
        logger.debug(f"GENERATION {g} offspring:{offspring}")
        # population = population.union(offspring)
        population.extend(offspring)

        # population.extend(offspring)

        # Sort population by fitness and discard the worst individual until only the first POPULATION_SIZE
        # individuals remain
        population = sorted(population, key=lambda i: i.fitness, reverse=True)[
            :POPULATION_SIZE
        ]

        if population[0].fitness > best_solution.fitness:
            steady = 0
            best_solution = population[0]
            best_generation = g
            if best_solution.fitness == BEST_FITNESS:
                break
        else:
            steady += 1

    graphic_eval(fitness_log, g, N)
    logger.info(f"Best Individual found:{best_solution}")
    return best_solution, best_generation


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s")
    logger = logging.getLogger("myLogger")
    logger.setLevel(logging.INFO)

    if PROFILING:
        prof = profile.Profile()
        prof.enable()
    # for N in [5, 10, 20, 30, 40]:
    # for N in [5, 10, 20, 50, 100, 500,1000]:
    for N in [20]:
        available_lists = problem(N, seed=42)
        random.seed()
        if PROFILING:
            solving_start = time.process_time()
        solution, generation = solve(N, available_lists)
        if PROFILING:
            solving_end = time.process_time()
            logger.info(f"SOLUTION time:{(solving_end - solving_start):.3f} seconds")
        if solution:
            weight = sum(len(_) for _ in solution.genome)
            logger.info(
                f"Found solution for N={N} in generation {generation}: {solution}\nweight={weight} (bloat="
                f"{(weight - N) / N * 100:.0f}%) fitness={solution.fitness}"
            )
        else:
            logger.info(f"No solution exists for N={N} with lists={available_lists}")
    if PROFILING:
        prof.disable()
        # print profiling output
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(10)  # top 10 rows
