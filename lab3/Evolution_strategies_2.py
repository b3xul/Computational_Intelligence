import copy
from typing import Callable

from lab3.Hardcoded_strategies_1 import optimal_s, hardcoded_strategies
from lab3.Nim_utilities import *
from lab3.Utilities import function_name, graphic_eval

# from lab3.base_solution import CallCounter

Individual = namedtuple("Individual", ["genome", "fitness"])
logging.basicConfig(format="%(message)s", filename="tournament.tsv", filemode="w")

logger = logging.getLogger("myLogger")
logger.setLevel(logging.INFO)


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
        logging.debug(
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


dumb_evolved_strategies = [
    make_strategy({"p": 0.1}),
    make_strategy({"p": 0.5}),
    make_almost_optimal_strategy(0.1),
    make_almost_optimal_strategy(0.5),
]

strategies_to_choose = hardcoded_strategies + dumb_evolved_strategies


def choose_strategy_factory(genome: list) -> Callable:
    """
    The genome represents the probability of using the different strategies for a move. After some time,
    the agent will always choose the best strategy between the ones provided.
    """

    def select_strategy(state: Nim) -> Nimply:
        selected_strategy = random.choices(strategies_to_choose, weights=genome, k=1)[0]
        return selected_strategy(state)

    return select_strategy


# @CallCounter
def fitness(strategy: Callable) -> int:
    """
    Fitness is computed as the total amount of games won against the other agents
    """
    matches = [(strategy, s) for s in strategies_to_choose]
    logger_line = function_name(strategy) + "\t|\t"
    fitness = 0
    for competing_strategies in matches:
        NUM_MATCHES = 1
        correctly_won = 0
        unexpectedly_won = 0
        unknown_won = 0
        for i in range(NUM_MATCHES):
            # All tests_Starter can be won by the starting player
            tests_Starter = copy.deepcopy(tests_starter)
            tests_Opponent = copy.deepcopy(tests_opponent)
            tests_Unknown = copy.deepcopy(tests_unknown)
            correctly_won += play_tests(tests_Starter, competing_strategies)
            unexpectedly_won += play_tests(tests_Opponent, competing_strategies)
            unknown_won += play_tests(tests_Unknown, competing_strategies)
        logger_line += (
            f"{correctly_won / (len(tests_Starter) * NUM_MATCHES) * 100:.2f}% / "
            f"{unexpectedly_won / (len(tests_Opponent) * NUM_MATCHES) * 100:.2f}% / "
            f"{unknown_won / (len(tests_Unknown) * NUM_MATCHES) * 100:.2f}% |\t"
        )
        fitness += correctly_won + unexpectedly_won + unknown_won
        # logger.info(logger_line)
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

    mutated_g = copy.deepcopy(g)

    highest = max(g)
    max_i = g.index(highest)

    candidates = list(filter(lambda prob: 0 < prob < highest, mutated_g))
    if candidates:
        to_reduce = random.choice(candidates)

        mutated_g[max_i] += mutated_g[mutated_g.index(to_reduce)]
        mutated_g[mutated_g.index(to_reduce)] = 0
    else:
        mutated_g = [random.random() for i in range(len(strategies_to_choose))]
        s = sum(mutated_g)
        mutated_g = [0.5 * i / s for i in mutated_g]
    # while random.random() < MULTIPLE_MUTATIONS_RATE:
    #     logger.debug("multiple mutations")

    logger.debug(f"mutated genome:{mutated_g}")
    return mutated_g


def evolve(factory: Callable):
    BEST_FITNESS = fitness(optimal_s)
    # POPULATION_SIZE = 100
    # OFFSPRING_SIZE = 0.8 * POPULATION_SIZE
    # POPULATION_SIZE = len(all_lists)  N=30 -> 10 s
    POPULATION_SIZE = 20
    OFFSPRING_SIZE = POPULATION_SIZE // 2
    TOURNAMENT_SIZE = 2

    population = list()
    initial_probability = 1 / len(strategies_to_choose)
    genome = [initial_probability] * len(strategies_to_choose)
    # [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    logger.debug(f"genome:{genome}")
    population.append(Individual(genome, fitness(factory(genome))))

    for s in strategies_to_choose:
        genome = [initial_probability - initial_probability / 2] * len(
            strategies_to_choose
        )
        genome[strategies_to_choose.index(s)] = initial_probability + (
            initial_probability / 2 * (len(strategies_to_choose) - 1)
        )
        # sum([0.5625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625]) = 1.0
        logger.debug(f"genome:{genome}")
        population.append(Individual(genome, fitness(factory(genome))))

    while len(population) < POPULATION_SIZE:
        genome = [random.random() for i in range(len(strategies_to_choose))]
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
