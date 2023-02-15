import logging
import argparse
import time
import cProfile as profile
import pstats

import quarto

import Players


def play_match(
    game: quarto.Quarto,
    competing_players: tuple[type[quarto.Player], type[quarto.Player]],
) -> int:
    """
    Play a match and return its result
    """
    game.set_players((competing_players[0](game), competing_players[1](game)))
    winner = game.run()
    game.reset()
    if winner == 0:
        logging.debug(f"Winner: player {winner}: {competing_players[winner].__name__}")
        return 1
    elif winner == -1:
        logging.debug(f"Draw!")
        return 0
    else:
        logging.debug(f"Winner: player {winner}: {competing_players[winner].__name__}")
        return -1


def tournament(
    player: type[quarto.Player],
    players: list[type[quarto.Player]],
) -> None:
    """
    Perform a tournament between players and log results
    """
    game = quarto.Quarto()

    matches = [(player, s) for s in players]
    logger_line = str(player.__name__).center(29) + "|"
    total_wins = 0
    for competing_players in matches:
        NUM_MATCHES = 1
        current_wins = 0
        current_draws = 0
        current_losses = 0
        for i in range(NUM_MATCHES):
            result = play_match(game, competing_players)
            if result == 0:
                current_draws += 1
            elif result == 1:
                current_wins += 1
            elif result == -1:
                current_losses += 1
        logger_line += (
            f"{current_wins / NUM_MATCHES:^9.2%}/{current_draws / NUM_MATCHES:^9.2%}/"
            f"{current_losses / NUM_MATCHES:^9.2%}|"
        )
        total_wins += current_wins
    logging.info(logger_line)


def main():
    PROFILING = False
    if PROFILING:
        prof = profile.Profile()
        prof.enable()
        solving_start = time.process_time()

    players = [Players.HardPlayer, Players.EasyPlayer, Players.RandomPlayer]

    logger_line = "win % / draw % / lose %".center(29) + "|"
    for player in players:
        logger_line += f"{player.__name__:^29}|"
    logging.info(logger_line)

    for player in players:
        tournament(player, players)

    if PROFILING:
        solving_end = time.process_time()
        logging.info(f"Total time taken: {(solving_end - solving_start) * 1000:.3f}ms")
        prof.disable()
        # print profiling output
        stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
        stats.print_stats(30)  # top 30 rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="count", default=1, help="increase log verbosity"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_const",
        dest="verbose",
        const=2,
        help="log debug messages (same as -vv)",
    )
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()
