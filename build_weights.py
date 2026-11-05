"""
Hand-writing the weights of a tic-tac-toe network. No training, no data.

Input  (18): mine[0..8], opp[0..8]  -- from the point of view of the side to move
Hidden: three ReLU layers whose units are wired by hand as feature detectors
Output (9): one logit per square; the move is the argmax

Every weight below is set deliberately. The script just writes them into
matrices so they can be saved as raw numbers.
"""
import json
import numpy as np

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
CORNERS, EDGES, CENTER = [0, 2, 6, 8], [1, 3, 5, 7], 4
OPPOSITE = {0: 8, 2: 6, 6: 2, 8: 0}
THRU = {c: [l for l, L in enumerate(LINES) if c in L] for c in range(9)}  # lines through each cell


class Layer:
    """One dense layer, built a unit at a time by naming which inputs it reads."""

    def __init__(self, input_names):
        self.idx = {n: i for i, n in enumerate(input_names)}
        self.rows, self.bias, self.names = [], [], []

    def unit(self, name, w, b=0.0):
        row = np.zeros(len(self.idx))
        for k, v in w.items():
            row[self.idx[k]] += v
        self.rows.append(row)
        self.bias.append(b)
        self.names.append(name)

    def W(self):
        return np.array(self.rows), np.array(self.bias)


def add(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def scale(d, s):
    return {k: v * s for k, v in d.items()}


inputs = [f"m{c}" for c in range(9)] + [f"o{c}" for c in range(9)]

# ---------------- Layer 1: read the board ----------------
L1 = Layer(inputs)
for c in range(9):
    L1.unit(f"occ{c}", {f"m{c}": 1, f"o{c}": 1})                 # square taken
    L1.unit(f"emp{c}", {f"m{c}": -1, f"o{c}": -1}, b=1)         # square empty
for l, L in enumerate(LINES):
    mine = {f"m{c}": 1 for c in L}
    opp = {f"o{c}": 1 for c in L}
    # P = my stones on a line the opponent hasn't touched (0, 1 or 2)
    L1.unit(f"P{l}", add(mine, scale(opp, -4)))
    # W = exactly two of mine, none of theirs  -> I can win on this line
    L1.unit(f"W{l}", add(mine, scale(opp, -4)), b=-1)
    # Q, B = the same two detectors from the opponent's side
    L1.unit(f"Q{l}", add(opp, scale(mine, -4)))
    L1.unit(f"B{l}", add(opp, scale(mine, -4)), b=-1)
for c in CORNERS:
    L1.unit(f"oc{c}", {f"o{c}": 1})                             # opponent holds this corner

# One-stone lines, as linear combinations of layer-1 units:
#   M1 = P - 2W  is 1 iff the line has exactly one of mine and none of theirs
#   O1 = Q - 2B  is the same for the opponent
M1 = {l: {f"P{l}": 1, f"W{l}": -2} for l in range(8)}
O1 = {l: {f"Q{l}": 1, f"B{l}": -2} for l in range(8)}

# ---------------- Layer 2: threats and forks per square ----------------
L2 = Layer(L1.names)
for c in range(9):
    L2.unit(f"emp{c}", {f"emp{c}": 1})
    # number of winning lines this empty square completes
    L2.unit(f"win{c}", add(*[{f"W{l}": 1} for l in THRU[c]], {f"occ{c}": -4}))
    # number of opponent wins this empty square blocks
    L2.unit(f"blk{c}", add(*[{f"B{l}": 1} for l in THRU[c]], {f"occ{c}": -4}))
    # fork detector: s = count of my one-stone lines through an empty square.
    # u - v = ReLU(s-1) - ReLU(s-2) is 1 exactly when s >= 2 (a fork).
    s_me = add(*[M1[l] for l in THRU[c]], {f"occ{c}": -4})
    L2.unit(f"fu{c}", s_me, b=-1)
    L2.unit(f"fv{c}", s_me, b=-2)
    s_op = add(*[O1[l] for l in THRU[c]], {f"occ{c}": -4})
    L2.unit(f"gu{c}", s_op, b=-1)
    L2.unit(f"gv{c}", s_op, b=-2)
for l in range(8):
    L2.unit(f"m1_{l}", M1[l])                                   # pass the one-stone flag up
for c in CORNERS:
    L2.unit(f"oppopp{c}", {f"oc{OPPOSITE[c]}": 1})              # opponent in the opposite corner

oppfork = {c: {f"gu{c}": 1, f"gv{c}": -1} for c in range(9)}   # 1 iff opponent could fork here
F = add(*oppfork.values())                                      # number of opponent fork squares

# ---------------- Layer 3: decide how to defend against forks ----------------
L3 = Layer(L2.names)
K = 9
for c in range(9):
    for n in ("win", "blk", "emp"):
        L3.unit(f"{n}{c}", {f"{n}{c}": 1})
    L3.unit(f"myfork{c}", {f"fu{c}": 1, f"fv{c}": -1})
    L3.unit(f"oppfork{c}", oppfork[c])
    # 1 when this is an opponent fork square AND there are 2+ of them (two units, a - b).
    # Blocking one fork square then just loses to the other, so it gets penalised.
    L3.unit(f"fpa{c}", add(scale(oppfork[c], K), F), b=-K - 1)
    L3.unit(f"fpb{c}", add(scale(oppfork[c], K), F), b=-K - 2)
for c in CORNERS:
    L3.unit(f"oppopp{c}", {f"oppopp{c}": 1})
for l, L in enumerate(LINES):
    for c in L:
        others = add(*[oppfork[e] for e in L if e != c])
        # Playing c makes two-in-a-row on line l and forces the reply on the last square.
        # good: the forced reply is NOT an opponent fork square
        L3.unit(f"good{c}_{l}", add({f"m1_{l}": 1, f"emp{c}": 1}, scale(others, -1)), b=-1)
        # bad: the forced reply hands the opponent their fork
        L3.unit(f"bad{c}_{l}", add({f"m1_{l}": 1, f"emp{c}": 1}, others), b=-2)

# ---------------- Output: one score per square ----------------
OUT = Layer(L3.names)
for c in range(9):
    w = {
        f"win{c}": 1000,       # 1. win
        f"blk{c}": 300,        # 2. block
        f"myfork{c}": 100,     # 3. fork
        f"oppfork{c}": 40,     # 4. block a single fork
        f"fpa{c}": -60,        #    ...but not when there are two
        f"fpb{c}": 60,
        f"emp{c}": 100000,     # illegal-move mask (with the bias below)
    }
    for l in THRU[c]:
        w[f"good{c}_{l}"] = 30     # safe forcing move
        w[f"bad{c}_{l}"] = -30     # forcing move that backfires
    # 5-8. position: center, opposite corner, corner, edge
    w[f"emp{c}"] += 10 if c == CENTER else 5 if c in CORNERS else 2
    if c in CORNERS:
        w[f"oppopp{c}"] = 3
    OUT.unit(f"sq{c}", w, b=-100000)

layers = []
for layer in (L1, L2, L3, OUT):
    Wm, bm = layer.W()
    layers.append({"W": Wm.tolist(), "b": bm.tolist()})

n_params = sum(np.size(l["W"]) + np.size(l["b"]) for l in layers)
shape = [18] + [len(l["b"]) for l in layers]
json.dump({"shape": shape, "activation": "relu (hidden layers), argmax over outputs", "input": "mine[0..8] + opp[0..8], side to move", "layers": layers},
          open("ttt_weights.json", "w"))
print("shape:", " -> ".join(map(str, shape)))
print("parameters:", n_params)
print("nonzero:", sum(int(np.count_nonzero(l["W"])) + int(np.count_nonzero(l["b"])) for l in layers))
