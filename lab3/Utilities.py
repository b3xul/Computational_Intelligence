from typing import Callable
from matplotlib import pyplot as plt


def function_name(f: Callable) -> str:
    """
    Print name of the function and genome used to create it
    """
    return f"{f.__name__} {f.__closure__[0].cell_contents if f.__closure__ else ''}"


# Graphic performance evaluation
def graphic_eval(fitness_log, g):
    gen_best = [max(f for f in fitness_log if f[0] == x) for x in range(g + 2)]
    plt.figure(figsize=(15, 6))
    plt.scatter([x for x, _ in fitness_log], [y for _, y in fitness_log], marker=".")
    plt.plot([x for x, _ in enumerate(gen_best)], [y for _, y in enumerate(gen_best)])
    plt.savefig(f"evaluation.png")
    plt.close()
