import cProfile as profile
import pstats
import time
from itertools import product

from Utilities import function_name
from lab3.Evolution_strategies_2 import (
    dumb_evolved_strategies,
    logger,
    fitness,
)
from lab3.Hardcoded_strategies_1 import *
from lab3.MinMax_3 import *


strategies = hardcoded_strategies + dumb_evolved_strategies + minmax_strategies

if __name__ == "__main__":

    PROFILING = True
    if PROFILING:
        prof = profile.Profile()
        prof.enable()

    if PROFILING:
        solving_start = time.process_time()

    logger_line = "\t\t\t\t|"
    for s in strategies:
        logger_line += f"\t{function_name(s)}\t|"
    logger.info(logger_line)

    # Full tournament
    matches = list(product(strategies, repeat=2))
    for s in strategies:
        fitness(s)

    # # Evolve strategy
    # factory = choose_strategy_factory
    # solution, generation = evolve(factory)
    #
    # logger.info(f"Best individual {solution} found in generation {generation}")

    # # Test minmax
    # fitness(minmax_s)
    # fitness(minmax_montecarlo_s)

    if PROFILING:
        solving_end = time.process_time()
        logging.info(f"Total time taken: {(solving_end - solving_start):.3f}s")
        prof.disable()
        # print profiling output
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(10)  # top 10 rows
