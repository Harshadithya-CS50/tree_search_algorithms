"""
Adversarial Search Algorithms
-----------------------------
This module implements:
1. Minimax Search
2. Alpha-Beta Pruning
3. Heuristic Alpha-Beta Search (Depth Limited)
4. Monte Carlo Tree Search (MCTS)

The implementation is generic and can be adapted to many turn-based,
zero-sum games.

Language: Python 3.10+
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


# ============================================================
# Generic Game Interface
# ============================================================

class GameState(ABC):
    """
    Abstract base class for a game state.

    Any game used with these algorithms should inherit from this class.
    """

    @abstractmethod
    def current_player(self) -> int:
        """Return the current player (1 or -1)."""
        pass

    @abstractmethod
    def legal_moves(self) -> List[Any]:
        """Return a list of legal moves."""
        pass

    @abstractmethod
    def make_move(self, move: Any) -> "GameState":
        """Return a new state after applying a move."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game is over."""
        pass

    @abstractmethod
    def utility(self) -> int:
        """
        Return utility value:
            +1 if player 1 wins
            -1 if player -1 wins
             0 for draw
        """
        pass

    @abstractmethod
    def heuristic(self) -> float:
        """
        Return heuristic evaluation for non-terminal states.
        Positive values favor player 1.
        Negative values favor player -1.
        """
        pass


# ============================================================
# Minimax Search
# ============================================================


def minimax(state: GameState) -> Tuple[float, Optional[Any]]:
    """
    Compute the optimal move using the Minimax algorithm.

    Returns:
        (best_score, best_move)
    """

    def max_value(s: GameState):
        if s.is_terminal():
            return s.utility(), None

        best_score = -math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = min_value(s.make_move(move))
            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move

    def min_value(s: GameState):
        if s.is_terminal():
            return s.utility(), None

        best_score = math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = max_value(s.make_move(move))
            if score < best_score:
                best_score = score
                best_move = move

        return best_score, best_move

    if state.current_player() == 1:
        return max_value(state)
    else:
        return min_value(state)


# ============================================================
# Alpha-Beta Pruning
# ============================================================


def alpha_beta_search(state: GameState) -> Tuple[float, Optional[Any]]:
    """
    Minimax with Alpha-Beta pruning.

    Returns:
        (best_score, best_move)
    """

    def max_value(s: GameState, alpha: float, beta: float):
        if s.is_terminal():
            return s.utility(), None

        best_score = -math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = min_value(s.make_move(move), alpha, beta)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

            if alpha >= beta:
                break

        return best_score, best_move

    def min_value(s: GameState, alpha: float, beta: float):
        if s.is_terminal():
            return s.utility(), None

        best_score = math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = max_value(s.make_move(move), alpha, beta)

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, best_score)

            if alpha >= beta:
                break

        return best_score, best_move

    if state.current_player() == 1:
        return max_value(state, -math.inf, math.inf)
    else:
        return min_value(state, -math.inf, math.inf)


# ============================================================
# Heuristic Alpha-Beta Search
# ============================================================


def heuristic_alpha_beta(
    state: GameState,
    depth_limit: int,
) -> Tuple[float, Optional[Any]]:
    """
    Depth-limited Alpha-Beta search using heuristic evaluation.

    Args:
        state: Current game state
        depth_limit: Maximum search depth

    Returns:
        (best_score, best_move)
    """

    def max_value(s: GameState, alpha: float, beta: float, depth: int):
        if s.is_terminal():
            return s.utility(), None

        # FIX: use <= 0 instead of == 0 to guard against negative depth
        if depth <= 0:
            return s.heuristic(), None

        best_score = -math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = min_value(
                s.make_move(move), alpha, beta, depth - 1
            )

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)

            if alpha >= beta:
                break

        return best_score, best_move

    def min_value(s: GameState, alpha: float, beta: float, depth: int):
        if s.is_terminal():
            return s.utility(), None

        # FIX: use <= 0 instead of == 0 to guard against negative depth
        if depth <= 0:
            return s.heuristic(), None

        best_score = math.inf
        best_move = None

        for move in s.legal_moves():
            score, _ = max_value(
                s.make_move(move), alpha, beta, depth - 1
            )

            if score < best_score:
                best_score = score
                best_move = move

            beta = min(beta, best_score)

            if alpha >= beta:
                break

        return best_score, best_move

    if state.current_player() == 1:
        return max_value(state, -math.inf, math.inf, depth_limit)
    else:
        return min_value(state, -math.inf, math.inf, depth_limit)


# ============================================================
# Monte Carlo Tree Search (MCTS)
# ============================================================


@dataclass
class MCTSNode:
    state: GameState
    parent: Optional["MCTSNode"] = None
    move: Any = None

    wins: float = 0
    visits: int = 0

    # FIX: annotate as Optional so the type matches the None default
    children: Optional[List["MCTSNode"]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.legal_moves())

    def best_child(self, exploration_weight: float = 1.414) -> "MCTSNode":
        """
        Select child using UCB1.

        wins are stored from player 1's perspective (+1 / -1 / 0).
        Multiply exploitation by current_player() so that:
          - player  1 nodes maximise  wins/visits  (want +1)
          - player -1 nodes maximise -wins/visits  (want -1, i.e. minimise)
        Both cases are handled uniformly by Python's max().
        """

        def ucb1(child: "MCTSNode") -> float:
            if child.visits == 0:
                return math.inf

            # FIX: scale exploitation by the choosing player's sign so
            # player -1 correctly prefers moves that lead to -1 outcomes.
            exploitation = (
                child.wins / child.visits
            ) * self.state.current_player()

            exploration = exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )

            return exploitation + exploration

        return max(self.children, key=ucb1)


class MonteCarloTreeSearch:
    def __init__(self, iterations: int = 1000):
        self.iterations = iterations

    def search(self, initial_state: GameState) -> Any:
        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = self._select(root)

            if not node.state.is_terminal():
                node = self._expand(node)

            reward = self._simulate(node.state)
            self._backpropagate(node, reward)

        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        tried_moves = {child.move for child in node.children}

        for move in node.state.legal_moves():
            if move not in tried_moves:
                new_state = node.state.make_move(move)
                child = MCTSNode(new_state, parent=node, move=move)
                node.children.append(child)
                return child

        return node

    def _simulate(self, state: GameState) -> float:
        current = state

        while not current.is_terminal():
            move = random.choice(current.legal_moves())
            current = current.make_move(move)

        return current.utility()

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.wins += reward
            node = node.parent


# ============================================================
# Example Game: Tic-Tac-Toe
# ============================================================


class TicTacToe(GameState):
    """
    Tic-Tac-Toe implementation.

    Player 1  = X
    Player -1 = O
    """

    def __init__(self, board=None, player=1):
        self.board = board or [0] * 9
        self.player = player

    def current_player(self) -> int:
        return self.player

    def legal_moves(self) -> List[int]:
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, move: int) -> "TicTacToe":
        new_board = self.board.copy()
        new_board[move] = self.player
        return TicTacToe(new_board, -self.player)

    def is_terminal(self) -> bool:
        return self.winner() is not None or all(
            cell != 0 for cell in self.board
        )

    def utility(self) -> int:
        winner = self.winner()

        if winner == 1:
            return 1
        elif winner == -1:
            return -1
        return 0

    def heuristic(self) -> float:
        """
        Simple heuristic:
        Count potential winning lines.
        """

        lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        score = 0

        for a, b, c in lines:
            values = [self.board[a], self.board[b], self.board[c]]

            if -1 not in values:
                score += values.count(1)

            if 1 not in values:
                score -= values.count(-1)

        return score

    def winner(self) -> Optional[int]:
        lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        for a, b, c in lines:
            total = self.board[a] + self.board[b] + self.board[c]

            if total == 3:
                return 1
            elif total == -3:
                return -1

        return None

    def display(self) -> None:
        symbols = {
            1: "X",
            -1: "O",
            0: " ",
        }

        for i in range(0, 9, 3):
            print(
                " | ".join(symbols[self.board[j]] for j in range(i, i + 3))
            )
            if i < 6:
                print("---------")


# ============================================================
# Demonstration
# ============================================================

if __name__ == "__main__":
    game = TicTacToe()

    print("Initial Tic-Tac-Toe Board")
    game.display()

    print("\n=== Minimax ===")
    score, move = minimax(game)
    print("Best Move:", move, "Score:", score)

    print("\n=== Alpha-Beta ===")
    score, move = alpha_beta_search(game)
    print("Best Move:", move, "Score:", score)

    print("\n=== Heuristic Alpha-Beta ===")
    score, move = heuristic_alpha_beta(game, depth_limit=3)
    print("Best Move:", move, "Score:", score)

    print("\n=== Monte Carlo Tree Search ===")
    mcts = MonteCarloTreeSearch(iterations=2000)
    move = mcts.search(game)
    print("Best Move:", move)
