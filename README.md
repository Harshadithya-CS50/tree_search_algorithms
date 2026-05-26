# Adversarial Search Algorithms in Python

This project implements four important Artificial Intelligence search algorithms used in turn-based games and decision-making systems.

## Implemented Algorithms

1. Minimax Search
2. Alpha-Beta Pruning
3. Heuristic Alpha-Beta Search
4. Monte Carlo Tree Search (MCTS)

The project also includes:

- A generic game interface (`GameState` ABC)
- A working Tic-Tac-Toe implementation
- Unit test cases
- Demonstration examples

---

## Project Structure

```text
project/
│
├── search_algorithms.py
├── test_search_algorithms.py
└── README.md
```

---

## Requirements

- Python 3.10+

No external libraries are required.

---

## How to Run

### Run the Main Program

```bash
python search_algorithms.py
```

Example Output:

```text
Initial Tic-Tac-Toe Board

  |   |
---------
  |   |
---------
  |   |

=== Minimax ===
Best Move: 0 Score: 0

=== Alpha-Beta ===
Best Move: 0 Score: 0

=== Heuristic Alpha-Beta ===
Best Move: 4 Score: 1

=== Monte Carlo Tree Search ===
Best Move: 4
```

### Run Unit Tests

```bash
python -m unittest test_search_algorithms.py -v
```

---

## Algorithm Explanations

### 1. Minimax Search

Minimax is a recursive adversarial search algorithm for two-player zero-sum games.

- One player (player 1) tries to **maximise** the score.
- The other player (player -1) tries to **minimise** the score.
- Assumes both players play optimally.
- Explores the **entire** game tree, so it is only practical for small games.

**Time Complexity:** `O(b^d)` where `b` = branching factor, `d` = search depth.

---

### 2. Alpha-Beta Pruning

Alpha-Beta pruning is Minimax with a significant optimisation: branches that cannot possibly influence the final decision are pruned and never evaluated.

- `alpha` — the best score the maximising player can already guarantee.
- `beta`  — the best score the minimising player can already guarantee.
- When `alpha >= beta` a cut-off fires and the remaining siblings are skipped.

Produces **identical results** to plain Minimax while evaluating far fewer nodes.

**Best-case Time Complexity:** `O(b^(d/2))` — effectively doubles the searchable depth for the same cost.

---

### 3. Heuristic Alpha-Beta Search

A depth-limited variant of Alpha-Beta pruning designed for games too large to search exhaustively.

When the depth limit is reached the game tree is **not** expanded further; instead a **heuristic evaluation function** estimates the quality of the position without completing the game.

The `depth_limit` parameter trades accuracy for speed:
- Small values → fast but shallow reasoning.
- Large values → slower but stronger play.

---

### 4. Monte Carlo Tree Search (MCTS)

MCTS is a probabilistic search algorithm that builds a partial game tree guided by random simulations ("rollouts"). It is the backbone of modern game-playing AI systems such as AlphaGo.

Each iteration of MCTS runs four phases:

| Phase | Description |
|---|---|
| **Selection** | Descend the tree using UCB1 until a node that is not fully expanded is found. |
| **Expansion** | Add one new child node for an untried move. |
| **Simulation** | Play the game out to completion with random moves. |
| **Backpropagation** | Walk back up the tree updating visit counts and win scores. |

**UCB1 formula** (used in Selection):

```
UCB1 = (wins / visits) * player + C * sqrt(ln(parent_visits) / visits)
```

The `player` multiplier (`+1` or `-1`) ensures each player selects moves in their own favour — without it the algorithm would make actively wrong decisions for player -1.

**Advantages:**
- Works well for very large search spaces.
- Requires no exhaustive tree expansion.
- Does not need a hand-crafted heuristic evaluation function.

---

## Tic-Tac-Toe Example

The project includes a complete Tic-Tac-Toe implementation compatible with all four algorithms.

Board positions are numbered 0–8:

```
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
```

Player 1 = **X**, Player -1 = **O**.

### Features

- Legal move generation
- Terminal state detection
- Utility calculation (`+1` win, `-1` loss, `0` draw)
- Heuristic evaluation (counts open winning lines)

---

## Test Cases

Run with:

```bash
python -m unittest test_search_algorithms.py -v
```

| Test | What it verifies |
|---|---|
| `test_minimax_returns_legal_move` | Minimax returns a move that is on the board |
| `test_alpha_beta_returns_legal_move` | Alpha-Beta returns a legal move |
| `test_heuristic_alpha_beta_returns_legal_move` | Depth-limited search returns a legal move |
| `test_mcts_returns_legal_move` | MCTS returns a legal move |
| `test_terminal_state_utility` | Utility is `+1` when player 1 has won |
| `test_minimax_alpha_beta_agree` | Minimax and Alpha-Beta produce the same score from the opening |
| `test_minimax_alpha_beta_agree_mid_game` | Same consistency check from a non-trivial mid-game position |
| `test_mcts_blocks_immediate_win` | MCTS (as player -1) takes a one-move win |
| `test_heuristic_alpha_beta_negative_depth_guard` | `depth_limit=0` returns heuristic without crashing |

---

## Extending to Other Games

To use these algorithms with a different game, subclass `GameState` and implement the six abstract methods:

```python
class MyGame(GameState):
    def current_player(self) -> int: ...
    def legal_moves(self) -> list: ...
    def make_move(self, move) -> "MyGame": ...
    def is_terminal(self) -> bool: ...
    def utility(self) -> int: ...       # +1 / -1 / 0
    def heuristic(self) -> float: ...   # used by heuristic_alpha_beta only
```

Then pass an instance directly to any of the four search functions.

---
