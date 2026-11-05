# Tic-tac-toe network with hand-written weights

This folder holds a small neural network that plays tic-tac-toe. Nobody trained it. There was no data, no loss function and no gradient descent. Every weight was chosen on purpose to make the network play by a known strategy. It was built to answer a question from the blog post *Sparks of Weight-Space Programming* (Takehi et al., MIT, Nov 2026): can a language model write a working network's raw weights directly?

The result: the network never loses. It was tested against every possible sequence of opponent moves, going first and going second.

## Files

| File | What it is | Needs |
|---|---|---|
| `ttt_weights.json` | The network: every weight and bias, as plain numbers | — |
| `play_ttt.py` | Play against the network in a terminal | Python 3 only |
| `test_net.py` | The tests (exhaustive check plus 10,000-game matches) | Python 3 + numpy |
| `build_weights.py` | The script that wrote the weights, with each unit commented | Python 3 + numpy |

## The network

```
18 inputs  ->  54 ReLU  ->  75 ReLU  ->  115 ReLU  ->  9 outputs
```

- **Parameters:** 14,935 in total, of which 1,485 are nonzero.
- **Inputs (18):** whose turn it is decides the point of view. Inputs 0–8 are 1 where the player to move has a piece. Inputs 9–17 are 1 where the opponent has a piece. Squares are numbered 0–8, left to right and top to bottom.
- **Outputs (9):** one score per square. The move is the square with the highest score.
- **Forward pass:** for each layer, `x = W @ x + b`, then ReLU (negative values become 0) on every layer except the last.

`ttt_weights.json` stores `layers` as a list of `{"W": [...], "b": [...]}`. Each `W` has shape *(outputs × inputs)*, the same layout PyTorch's `nn.Linear` uses.

## How the weights were made

The network carries out the classic tic-tac-toe strategy (Newell & Simon, 1972), checked in this order:

1. Win if you can.
2. Block the opponent's win.
3. Make a fork (two threats at once).
4. Block the opponent's fork.
5. Take the center, then the corner opposite the opponent, then any corner, then an edge.

Each layer computes one stage of this. `build_weights.py` names every hidden unit and lists exactly which inputs it reads and with what weight. Python placed the numbers into matrices, but every value was chosen by hand. Nothing was fitted.

### Layer 1: read the board (54 units)

- **Square state (18 units):** is each square taken, and is it empty.
- **Line detectors (32 units):** four detectors for each of the 8 lines (3 rows, 3 columns, 2 diagonals). Example: the unit "can I win on the top row?" has weight **+1** on each of my three top-row squares, **−4** on each of the opponent's three, and bias **−1**:
  - two of mine and none of theirs: 2 − 1 = **1** (fires)
  - one of mine: 1 − 1 = 0 (silent)
  - any opponent piece: the −4 drives it negative, and ReLU makes it 0 (silent)

  The −4 works as a veto: one opponent piece shuts the unit off no matter what else is on the line. The other three detectors per line count "my pieces on a clean line" and do the same two jobs for the opponent.
- **Opponent corners (4 units):** which corners the opponent holds.

### Layer 2: threats and forks for each square (75 units)

- **Win and block counts:** for each square, add up the "can win" detectors of the lines through it, with −4 if the square is already taken.
- **Forks:** count my one-piece lines through each empty square. "One-piece line" is a combination of layer-1 units (the clean-line count minus twice the can-win detector). A fork means the count is 2 or more. ReLU can't make a yes/no step on its own, so two units do it together: `ReLU(s − 1) − ReLU(s − 2)` equals exactly 1 when s ≥ 2 and 0 otherwise. The same pair of units detects the opponent's fork squares.

### Layer 3: defending against forks (115 units)

This is the hardest part of the strategy. If the opponent has one fork square, you take it. If they have **two or more**, blocking one just loses to the other. The right move then is to make your own two-in-a-row, forcing them to block somewhere harmless. This layer has:

- a **"two or more opponent forks"** detector for each square, which penalizes the naive block;
- **"safe forcing move"** units: playing here makes two-in-a-row, and the square the opponent must answer on is *not* one of their fork squares;
- **"bad forcing move"** units: the forced answer *is* an opponent fork square, so the move backfires.

### Output layer: weigh everything (9 units)

Each square's score is a weighted sum. The gaps between weights set the priority order:

| Feature | Weight |
|---|---|
| Wins the game | +1000 |
| Blocks an opponent win | +300 |
| Creates my fork | +100 |
| Blocks a single opponent fork | +40 |
| Naive block when there are 2+ forks | −60 |
| Safe forcing move / bad forcing move | +30 / −30 |
| Center / corner / edge | +10 / +5 / +2 |
| Corner opposite the opponent | +3 |
| Square already taken | −100000 |

The last row is the illegal-move mask: an empty square gets +100000 − 100000 = 0, and a taken square gets −100000, so the network can never choose it.

## How it was tested

`test_net.py` loads **only** `ttt_weights.json` and does matrix multiplies. It knows nothing about the strategy above.

1. **Exhaustive check.** The network plays its move, then the test branches into *every* possible opponent reply, all the way to the end of every game. This was run with the network as X and as O, against any legal reply and against a perfect opponent (one that only picks minimax-optimal moves).
2. **Random matches.** 10,000 games against a random player in each seat. For comparison, the same matches were run with a perfect player that picks at random among its non-losing moves.

### Results

| Exhaustive check | Wins | Draws | Losses | Illegal moves |
|---|---|---|---|---|
| Network as X vs every possible reply | 72 | 4 | **0** | 0 |
| Network as X vs perfect replies | 0 | 4 | **0** | 0 |
| Network as O vs every possible reply | 346 | 111 | **0** | 0 |
| Network as O vs perfect replies | 0 | 111 | **0** | 0 |

| 10,000 games vs a random player | Win | Draw | Loss |
|---|---|---|---|
| Network, going first | 99.0% | 1.0% | 0.0% |
| Network, going second | 88.3% | 11.7% | 0.0% |
| Perfect player, going first | 96.9% | 3.1% | 0.0% |
| Perfect player, going second | 78.5% | 21.5% | 0.0% |
| Random vs random (X's results) | 58.6% | 13.0% | 28.4% |

The network beats the "perfect" baseline against a random opponent because it plays to win (it goes for forks and immediate wins). The baseline only avoids losing.

### The one bug, and the fix

The first version had **one** losing line, found by the exhaustive test. Playing second, it went: X center, O corner, X opposite corner. At that point X had four possible fork squares. The fork-blocking penalty grew with the number of fork squares, so with four of them it buried the correct reply (a corner that forces X to block harmlessly). The network played an edge instead, which forced X onto a fork square, and lost.

The fix had two parts:

1. The penalty became a clean on/off switch, using the same two-ReLU step trick.
2. The reward for a safe forcing move went from +3 to +30.

After that, the exhaustive test found zero losses. This is like the blog post's experiments that allow a few test runs. It is not a pure one-shot result.

### Honest caveats

- **It compiles a known rulebook.** Tic-tac-toe has a complete written strategy, and this network is that strategy turned into weights. Writing weights for a task with no rulebook, or reading weights someone else trained, is much harder.
- **Code placed the numbers.** That matches how the blog's larger experiments worked. The choices are hand-made: no data, no fitting.

## Run it on your own computer

### 1. Install Python 3

- **Windows:** download from [python.org](https://www.python.org/downloads/). On the first installer screen, check **"Add python.exe to PATH"**.
- **Mac:** download from python.org, or run `brew install python`. Use `python3` instead of `python` in the commands below.
- **Linux:** usually already installed. Use `python3`.

### 2. Put the files in one folder

For example `Documents\tictactoe`. Keep `ttt_weights.json` in the same folder as the scripts.

### 3. Open a terminal in that folder

- **Windows:** open the folder in File Explorer, click the address bar, type `cmd` and press Enter.
- **Mac:** right-click the folder in Finder, then choose *Services → New Terminal at Folder*. Or open Terminal and type `cd ` followed by the folder path.

### 4. Play

```
python play_ttt.py
```

Pick X (you go first) or O (the network goes first). Then type a square number and press Enter:

```
 1 | 2 | 3
 ---------
 4 | 5 | 6
 ---------
 7 | 8 | 9
```

The best you can get is a draw. If Windows says `python` is not recognized, try `py play_ttt.py`, or run the Python installer again with the PATH box checked.

### 5. (Optional) Re-run the tests or rebuild the weights

These two scripts need numpy:

```
pip install numpy
python test_net.py        # runs every check above, takes about 2 seconds
python build_weights.py   # rewrites ttt_weights.json from scratch
```

Run both from inside the folder, because they read and write `ttt_weights.json` in the current folder. To experiment, change a weight in `build_weights.py` (for example, lower the block weight from 300 to 50), rebuild, and re-run the tests to watch it start losing.

### Use the weights in your own code

`play_ttt.py` contains a plain-Python forward pass. The same thing with numpy:

```python
import json
import numpy as np

layers = json.load(open("ttt_weights.json"))["layers"]

def best_move(board, me):
    """board: 9 values, 1 = X, -1 = O, 0 = empty. me: 1 or -1. Returns 0-8."""
    x = np.array([v == me for v in board] + [v == -me for v in board], dtype=float)
    for i, layer in enumerate(layers):
        x = np.array(layer["W"]) @ x + np.array(layer["b"])
        if i < len(layers) - 1:
            x = np.maximum(x, 0)  # ReLU
    return int(np.argmax(x))
```

To load it into PyTorch, copy each layer into an `nn.Linear(in, out)`. `W` goes into `.weight` and `b` into `.bias`, with `nn.ReLU()` between the layers.
