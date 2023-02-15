import copy
import logging
import time
import itertools
import random
from collections import namedtuple
from enum import Enum
import numpy as np

import quarto


class QuartoUtilities(object):
    def __init__(self, game: quarto.Quarto) -> None:
        self.game = game
        self.board = game.get_board_status()  # can be modified without interacting with the game.
        self.winning_lines = self.init_winning_lines()  # 0.1 ms
        self.Mode = Enum("Mode", ["ROWS", "COLUMNS", "DIAG1", "DIAG2"])

    def numbers_with_bit_0(self, pos: int) -> list[int]:
        """
        Returns the list of numbers from 0 to 15 with bit in position pos = 0
        """
        return [num for num in range(16) if (num & (1 << pos)) == 0]

    def numbers_with_bit_1(self, pos: int) -> list[int]:
        """
        Returns the list of numbers from 0 to 15 with bit in position pos = 1
        """
        return [num for num in range(16) if (num & (1 << pos)) == (1 << pos)]

    def init_winning_lines(self) -> set[tuple[int]]:
        """
        Before the start of the game, builds a list of all 536 winning lines, like (0,2,4,8).
        This let us search the line from the board in a dictionary instead of looking at the binary board to see if
        the line forms a Quarto or not.
        """
        winning_lines = set()
        for pos in range(4):
            n0 = self.numbers_with_bit_0(pos)
            n1 = self.numbers_with_bit_1(pos)
            # logging.debug(f"{n0}\n{[str(bin(i))[2:].zfill(4) for i in n0]}")
            # logging.debug(f"{n1}\n{[str(bin(i))[2:].zfill(4) for i in n1]}")
            winning_lines.update([c for c in itertools.combinations(n0, 4)])
            winning_lines.update([c for c in itertools.combinations(n1, 4)])

        # logging.debug(winning_lines)
        logging.debug(
            f"{len(winning_lines)} winning positions initialized correctly!"
        )

        return winning_lines

    def remaining_pieces(self, board: np.ndarray) -> list[int]:
        """
        Given a board, returns the list of pieces that still needs to be placed.
        """
        return list(set(range(16)) - set(board.flatten()))

    def get_free_positions(self, board: np.ndarray) -> list[tuple[int, int]]:
        """
        Given a board, returns a list of free positions encoded as (x,y) tuples.
        ex. [(0,0),(1,2),(3,3)]
        """
        positions = np.where(board == -1)
        return list(zip(positions[1], positions[0]))

    def check_lines(
        self, board: np.ndarray, selected_piece: int, winning: bool = True
    ) -> tuple[int, int] | None:
        """
        If winning is True, given a board and the currently selected piece, returns the (x,y) position where the
        piece needs to be placed to obtain a Quarto, or None if no such position exists.
        If winning is False, given a board and the currently selected piece, returns the (x,y) position where the
        piece can be placed to "block" a potential future Quarto, or None if no such position exists.
        """
        for mode in self.Mode:
            match mode:
                case self.Mode.ROWS:
                    lines = board
                case self.Mode.COLUMNS:
                    lines = board.T
                case self.Mode.DIAG1:
                    lines = [np.diag(board, 0)]
                case self.Mode.DIAG2:
                    lines = [np.diag(np.fliplr(board), 0)]

            for lines_index, seq in enumerate(lines):
                # lines_index is the index in the first direction (depends on mode)
                empty_cells_iterator = (i for i, num in enumerate(seq) if num == -1)
                line_index = next(empty_cells_iterator, None)
                # line_index is the index in the second direction (depends on mode)
                if line_index is not None:
                    # at least 1 empty cell remains in the row
                    another = next(empty_cells_iterator, None)
                    if another is None:
                        # exactly 1 empty cell remains in the row: if we have a tris, try to place the selected piece
                        # in the fourth cell

                        # game.get_selected_piece() = 11
                        # row = [ 1  0 10 -1]
                        # candidate = (0, 1, 10, 11) : Sorted tuple of the line, replacing missing piece with the
                        # selected one
                        candidate = tuple(
                            sorted(
                                [selected_piece if i == line_index else num for i, num in enumerate(seq)]
                            )
                        )
                        if (winning and candidate in self.winning_lines) or (not winning):
                            # Winning/Blocking move found
                            # Distinguish cases to report the coordinates of indices in different directions to (x,y)
                            match mode:
                                case self.Mode.ROWS:
                                    # logging.debug(f"Winning position found in the row containing {lines_index},"
                                    #               f"{line_index}")
                                    return line_index, lines_index
                                case self.Mode.COLUMNS:
                                    # logging.debug(f"Winning position found in the column containing {line_index},"
                                    #               f"{lines_index}")
                                    return lines_index, line_index
                                case self.Mode.DIAG1:
                                    # logging.debug(f"Winning position found in the diagonal containing {line_index},"
                                    #               f"{line_index}")
                                    return line_index, line_index
                                case self.Mode.DIAG2:
                                    # logging.debug(f"Winning position found in the antidiagonal containing "
                                    #               f"{self.game.BOARD_SIDE - 1 - line_index},{line_index}")
                                    return self.game.BOARD_SIDE - 1 - line_index, line_index
        return None


class HashableBoardPiece(object):
    """
    Wrapper class used to insert a board and a selected piece as the key of a dictionary.
    This allows the creation of a dictionary like [""]: MoveEvaluation() so that we can avoid computing the
    evaluation of the placement of a piece on a certain board that we already seen.
    """

    def __init__(self, board: np.ndarray, selected_piece: int) -> None:
        self.board = board
        self.selected_piece = selected_piece

    def __repr__(self):
        return str(self.board) + str(self.selected_piece)

    def __hash__(self):
        """
        This hash function maps different boards to the same hash, so that symmetries given by the order of the
        elements on the different lines, or whether the same elements are on a row or on a column does not make any
        difference. Only the exact elements matter.
        """
        board = self.board
        rows = [sorted(board[i, :]) for i in range(board.shape[0])]
        cols = [sorted(board[:, i]) for i in range(board.shape[1])]
        diag1 = [sorted(np.diag(board, 0))]
        diag2 = [sorted(np.diag(np.fliplr(board), 0))]
        all_lines = rows + cols + diag1 + diag2  # list of np.array of lines
        all_tuples = map(tuple, all_lines)
        set_all_tuples = set(all_tuples)
        return hash((str(set_all_tuples), self.selected_piece))

    def __eq__(self, other):
        return hash(self) == hash(other)


MoveEvaluation = namedtuple(
    "MoveEvaluation", ["evaluation", "piece_to_place", "x", "y", "next_piece"]
)


class HardPlayer(quarto.Player):
    """
    Hard player: explore states tree using minmax
    """

    def __init__(self, game: quarto.Quarto) -> None:
        super().__init__(game)
        self.quarto_utilities = QuartoUtilities(game)
        self.next_piece = None
        self.seen_boards = {}
        self.max_depth = 0
        self.total_hits = 0

    def choose_piece(self) -> int:
        """
        If it plays first, choose a random piece. Otherwise use the next_piece contained in the best MoveEvaluation
        found by the last minmax_placing function.
        """
        game = self.get_game()
        selected_piece = None
        board = game.get_board_status()
        remaining_pieces = self.quarto_utilities.remaining_pieces(board)
        if self.next_piece is None:
            for piece in remaining_pieces:
                res = self.quarto_utilities.check_lines(board, piece)
                if res is None:
                    # giving piece to the opponent doesn't result in an immediate loss
                    selected_piece = piece
                    break
            if selected_piece is None:
                selected_piece = random.choice(remaining_pieces)
        else:
            selected_piece = self.next_piece
        return selected_piece

    def place_piece(self) -> tuple[int, int]:
        """
        Choose in which position to place the selected piece and which piece to give to the opponent using minmax
        """
        game = self.get_game()
        self.quarto_utilities.board = game.get_board_status()
        selected_piece = game.get_selected_piece()

        start_time = time.process_time()

        position = self.minmax_placing(self.quarto_utilities.board, game.get_current_player(), selected_piece)

        end_time = time.process_time()
        turn_time = end_time - start_time
        if turn_time > 1:
            logging.debug(
                f"Total time: {turn_time * 1000:.3f}ms ({turn_time:.3f}s)"
            )

        return position

    def minmax_placing(self, board: np.ndarray, starting_player: int, selected_piece: int) -> tuple[int, int]:
        """
        Wrapper for the recursive minmax function
        """
        (val, piece_to_place, x, y, next_piece) = self.minmax_placing_rec(
            board=board,
            starting_player=starting_player,
            current_player=starting_player,
            selected_piece=selected_piece,
            depth=1
        )
        logging.debug(
            f"Best move found by player {starting_player} = place {selected_piece} in {x, y} and give to the opponent"
            f"{next_piece}. This results in an evaluation of {val} (1=win, 0=draw, -1=loss)"
        )
        self.quarto_utilities.board[y, x] = selected_piece
        self.next_piece = next_piece

        logging.debug(f"Total hits: {self.total_hits}")
        logging.debug(f"Total len: {len(self.seen_boards)}")
        return x, y

    def minmax_placing_rec(self, board: np.ndarray, starting_player: int, current_player: int, selected_piece: int,
                           depth: int) -> MoveEvaluation:
        """
        MinMax strategy to find the best placement and the best piece to give to the opponent
        """
        indent = "\t" * depth

        # if depth > self.quarto_utilities.max_depth:
        #     self.quarto_utilities.max_depth = depth
        #     logging.debug(f"Reached depth {depth}")

        # 1. If this board (or a symmetrical one) with this selected piece were already analyzed, return that
        # evaluation (terminal condition)
        best_evaluation = self.seen_boards.get(HashableBoardPiece(board, selected_piece), None)
        if best_evaluation is not None:
            self.total_hits += 1
            logging.debug(f"{indent}Player {current_player} already knows that the board {board}has the best "
                          f"evaluation = {best_evaluation}")
            return best_evaluation

        possible_positions = self.quarto_utilities.get_free_positions(board)
        logging.debug(f"{indent}possible_positions to place {selected_piece}:{possible_positions}")

        # 2. Full board: no pieces can be placed (terminal condition)
        if len(possible_positions) == 0:
            logging.debug(f"{indent}Player {current_player} has no more available moves for piece {selected_piece}, "
                          f"resulting in a draw! EVAL=0")
            return MoveEvaluation(0, None, None, None, None)

        # 3. If we can place the selected piece to obtain a Quarto, do it and save the evaluation. (terminal condition)
        res = self.quarto_utilities.check_lines(board, selected_piece)
        if res is not None:
            x, y = res
            evaluation = MoveEvaluation(1, selected_piece, x, y, None)
            # we must hash the board AFTER placing the piece in x,y
            board[y, x] = selected_piece
            logging.debug(f"{indent}Saving board\n{board} with evaluation={evaluation}")
            self.seen_boards[HashableBoardPiece(copy.deepcopy(board), selected_piece)] = evaluation
            board[y, x] = -1

            logging.debug(f"{indent}If player {current_player} places piece {selected_piece} in ({x},{y}), "
                          f"he wins! EVAL=1")
            return evaluation

        # 4. If only one moves remains, but it does not lead to a Quarto, it means it leads to a draw, do it and save
        # that evaluation. (terminal condition)
        if len(possible_positions) == 1:
            x, y = possible_positions[0]
            evaluation = MoveEvaluation(0, selected_piece, x, y, None)
            logging.debug(f"{indent}Player {current_player} can only place piece {selected_piece} in ({x},{y}), "
                          f"resulting in a draw! EVAL=0")
            board[y, x] = selected_piece
            logging.debug(f"{indent}Saving board\n{board} with evaluation={evaluation}")
            self.seen_boards[HashableBoardPiece(copy.deepcopy(board), selected_piece)] = evaluation
            board[y, x] = -1
            return evaluation

        # 5a. If too many states to explore, use a non-terminal heuristic evaluation function to cutoff the search
        branching_factor = len(possible_positions) * (len(possible_positions) - 1)
        if branching_factor ** depth > 150_000:
            # 6a. Instead of trying all positions, if it exists, pick a position that blocks a tris, otherwise pick a
            # random position
            res = self.quarto_utilities.check_lines(board, selected_piece, winning=False)
            if res is not None:
                # find move that blocks a Quarto
                x, y = res
                logging.debug(f"{indent}If player {current_player} places piece {selected_piece} in ({x},{y}), "
                              f"he stops a quarto!")
            else:
                x, y = random.choice(possible_positions)

            # 7a. If it can choose a piece which doesn't result in a Quarto, it does (evaluation=draw), otherwise it
            # chooses a random piece (evaluation=loss).
            board[y, x] = selected_piece
            remaining_pieces = self.quarto_utilities.remaining_pieces(board)
            next_piece = None
            for piece in remaining_pieces:
                res = self.quarto_utilities.check_lines(board, piece)
                if res is None:
                    # giving piece to the opponent don't result in an immediate loss
                    points = 0
                    next_piece = piece
                    break
            if next_piece is None:
                points = -1
                next_piece = random.choice(remaining_pieces)

            board[y, x] = -1
            best_evaluation = MoveEvaluation(
                points, selected_piece, x, y, next_piece
            )
        else:
            # 5b. We can try exploring all possible positions and all possible remaining pieces
            best_evaluation = MoveEvaluation(-2, selected_piece, None, None, None)
            for (x, y) in possible_positions:
                # Try placing selected piece in all possible positions
                logging.debug(f"{indent}Player {current_player} try to put {selected_piece} in {x, y}")
                board[y, x] = selected_piece

                remaining_pieces = self.quarto_utilities.remaining_pieces(board)
                logging.debug(f"{indent}possible_pieces to give:{remaining_pieces}")
                for next_piece in remaining_pieces:
                    # Try choosing next_piece to give to the opponent
                    logging.debug(f"{indent}Player {current_player} puts {selected_piece} in {x, y} and then try to "
                                  f"give to the opponent {next_piece}")
                    (val, _, _, _, _) = self.minmax_placing_rec(
                        board=board,
                        starting_player=starting_player,
                        current_player=(current_player + 1) % self.get_game().MAX_PLAYERS,
                        selected_piece=next_piece,
                        depth=depth + 1
                    )
                    logging.debug(f"{indent}If player {current_player} places piece {selected_piece} in ({x},{y}), "
                                  f"it results in an EVAL={-val}")
                    if -val > best_evaluation.evaluation:
                        best_evaluation = MoveEvaluation(
                            -val, selected_piece, x, y, next_piece
                        )

                    if -val == 1:
                        # This means that the opponent's best evaluation was -1, so the opponent's best move was a
                        # loss: the current player is guaranteed to win doing this move, so we can stop the search.
                        break
                    # else:
                    # If the current player does this move, the opponent can still obtain a draw or a win,
                    # so we keep trying other positions

                # end of recursion: if best_evaluation.evaluation==1, we found a win, otherwise we can't choose any
                # move that results in a forced win
                logging.debug(f"{indent}best move for player {current_player}, trying to put {selected_piece} in {x, y}"
                              f":{best_evaluation}")

                # Only save evaluation if it comes from an exhaustive search, since we can be sure of it
                logging.debug(f"{indent}Saving board\n{board} with evaluation={best_evaluation}")
                self.seen_boards[HashableBoardPiece(copy.deepcopy(board), selected_piece)] = best_evaluation

                # Undo move
                board[y, x] = -1

                if best_evaluation.evaluation == 1:
                    break

            logging.debug(f"{indent}Saving board\n{board} with evaluation={best_evaluation}")
            self.seen_boards[HashableBoardPiece(copy.deepcopy(board), selected_piece)] = best_evaluation

        logging.debug(f"{indent}best move:{best_evaluation}")
        return best_evaluation


class EasyPlayer(quarto.Player):
    """
    Easy player: if there is a winning move, place the corresponding piece. Avoids giving to the opponent a
    winning piece if possible
    """

    def __init__(self, game: quarto.Quarto) -> None:
        super().__init__(game)
        self.quarto_utilities = QuartoUtilities(game)

    def choose_piece(self) -> int:
        """
        If it can choose a piece which doesn't result in a Quarto, it does, otherwise it chooses a random piece
        """
        game = self.get_game()
        board = game.get_board_status()
        remaining_pieces = self.quarto_utilities.remaining_pieces(board)
        for selected_piece in remaining_pieces:
            res = self.quarto_utilities.check_lines(board, selected_piece)
            if res is None:
                # giving piece to the opponent don't result in an immediate loss
                return selected_piece

        selected_piece = random.choice(remaining_pieces)
        return selected_piece

    def place_piece(self) -> tuple[int, int]:
        """
        If it can form a Quarto with the selected piece, it does, otherwise place the piece on a random, free position
        """
        game = self.get_game()
        board = game.get_board_status()
        selected_piece = game.get_selected_piece()

        res = self.quarto_utilities.check_lines(board, selected_piece)
        if res is not None:
            return res

        position = random.choice(self.quarto_utilities.get_free_positions(board))
        return position


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto) -> None:
        super().__init__(quarto)

    def choose_piece(self) -> int:
        return random.randint(0, 15)

    def place_piece(self) -> tuple[int, int]:
        return random.randint(0, 3), random.randint(0, 3)
