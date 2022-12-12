from lab3.Nim_utilities import *

MoveEvaluation = namedtuple(
    "MoveEvaluation", ["evaluation", "row_content", "num_objects"]
)
RecursionStats = namedtuple("RecursionStats", ["level", "pruned"])
seen_states = {}


def montecarlo_rec(state: Nim) -> MoveEvaluation:
    """
    Choose random moves until the game ends, and backpropagate the result. Useful to cut long minmax explorations,
    but not very reliable estimation.
    """
    evaluation = seen_states.get(state, None)
    if evaluation is not None:
        return evaluation

    if not state:
        return MoveEvaluation(-1, None, None)

    possible_moves = cook_status(state)["possible_moves"]
    ply = random.choice(possible_moves)
    state.nimming(ply)
    (val, _, _) = montecarlo_rec(state)
    state.nimming(Nimply(ply.row, -ply.num_objects))
    estimated_evaluation = MoveEvaluation(-val, state.rows[ply.row], ply.num_objects)
    seen_states[Nim([*state.rows], state.k)] = estimated_evaluation

    return estimated_evaluation


# MoveEvaluation = namedtuple("MoveEvaluation", ["evaluation", "row_content", "num_objects"])
# seen_states = { state : MoveEvaluation }
# @CallCounter
def minmax_rec(
    state: Nim,
    maximize: bool,
    level: int,
    pruned: int,
    level_reached: int,
    montecarlo: bool,
) -> (MoveEvaluation, RecursionStats):
    """
    Recursive minmax.
    """
    indent = "\t" * level
    best_evaluation = seen_states.get(state, None)
    if best_evaluation is not None:
        return best_evaluation, RecursionStats(pruned, max(level, level_reached))

    if not state:
        # Current player won
        logging.debug(f"{indent}Player won!")
        return MoveEvaluation(-1, None, None), RecursionStats(
            pruned, max(level, level_reached)
        )

    if montecarlo and level > 25:
        # Hardcoded maximum depth level to visit exhaustively
        return montecarlo_rec(state), RecursionStats(pruned, max(level, level_reached))

    possible_moves = cook_status(state)["possible_moves"]
    logging.debug(f"{indent}possible_moves:{possible_moves}")
    if maximize:
        for ply in possible_moves:
            # Try move
            state.nimming(ply)
            (val, _, _), (pruned, level_reached) = minmax_rec(
                state, False, level + 1, pruned, level_reached, montecarlo
            )
            # Undo move
            state.nimming(Nimply(ply.row, -ply.num_objects))
            if -val == 1:
                # if doing the ply the player won, -val = 1
                # if doing the ply the player (later) lost, -val=-1
                best_evaluation = MoveEvaluation(
                    -val, state.rows[ply.row], ply.num_objects
                )
                pruned += len(possible_moves) - 1 - possible_moves.index(ply)
                break
            else:
                best_evaluation = MoveEvaluation(
                    -val, state.rows[ply.row], ply.num_objects
                )
    else:
        # minimize
        for ply in possible_moves:
            # Try move
            state.nimming(ply)
            (val, _, _), (pruned, level_reached) = minmax_rec(
                state, True, level + 1, pruned, level_reached, montecarlo
            )
            # Undo move
            state.nimming(Nimply(ply.row, -ply.num_objects))
            if -val == 1:
                best_evaluation = MoveEvaluation(
                    -val, state.rows[ply.row], ply.num_objects
                )
                pruned += len(possible_moves) - 1 - possible_moves.index(ply)
                break
            else:
                best_evaluation = MoveEvaluation(
                    -val, state.rows[ply.row], ply.num_objects
                )

    logging.debug(f"{indent}{state} best move:{best_evaluation}")
    seen_states[Nim([*state.rows], state.k)] = best_evaluation
    return best_evaluation, RecursionStats(pruned, max(level_reached, level))


def minmax_s(state: Nim) -> Nimply:
    """
    MinMax strategy to select the best move exhaustively
    """
    (_, row_content, num_objects), (pruned, level) = minmax_rec(
        state=state,
        maximize=True,
        level=0,
        pruned=0,
        level_reached=0,
        montecarlo=False,
    )
    if level > 0:
        logging.info(
            f"state {state} | reached level:{level} | pruned {pruned} branches"
        )
    return Nimply(state.rows.index(row_content), num_objects)


def minmax_montecarlo_s(state: Nim) -> Nimply:
    """
    MinMax strategy to select the best move, applying montecarlo sampling after depth 25
    """
    # MoveEvaluation = namedtuple("MoveEvaluation", ["evaluation", "row_content", "num_objects"])

    (_, row_content, num_objects), pruned, level = minmax_rec(
        state=state,
        maximize=True,
        level=0,
        pruned=0,
        level_reached=0,
        montecarlo=True,
    )
    if level > 0:
        logging.info(
            f"state {state} | reached level:{level} | pruned {pruned} branches"
        )
    return Nimply(state.rows.index(row_content), num_objects)


minmax_strategies = [minmax_s, minmax_montecarlo_s]
