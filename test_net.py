"""Load the raw weights and play. Knows nothing about the rules used to write them."""
import json
import random
from functools import lru_cache
import numpy as np

net = json.load(open("ttt_weights.json"))
Ws = [(np.array(l["W"]), np.array(l["b"])) for l in net["layers"]]

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in LINES:
        if b[a] and b[a] == b[c] == b[d]:
            return b[a]
    return 0


def net_move(b, me):
    x = np.array([1.0 if v == me else 0.0 for v in b] + [1.0 if v == -me else 0.0 for v in b])
    for i, (W, bias) in enumerate(Ws):
        x = W @ x + bias
        if i < len(Ws) - 1:
            x = np.maximum(x, 0)
    return int(np.argmax(x))


@lru_cache(None)
def value(b, turn):  # perfect play: +1 if `turn` wins, 0 draw, -1 loss
    w = winner(b)
    if w:
        return 1 if w == turn else -1
    if 0 not in b:
        return 0
    return max(-value(b[:i] + (turn,) + b[i + 1:], -turn) for i in range(9) if b[i] == 0)


def perfect_moves(b, turn):
    best = value(b, turn)
    return [i for i in range(9) if b[i] == 0 and -value(b[:i] + (turn,) + b[i + 1:], -turn) == best]


def explore(b, turn, me, opp_moves, tally, illegal):
    """Net plays its move; opponent branches over every move `opp_moves` allows."""
    w = winner(b)
    if w or 0 not in b:
        tally["win" if w == me else "loss" if w else "draw"] += 1
        return
    if turn == me:
        m = net_move(b, me)
        if b[m] != 0:
            illegal.append(b)
            return
        explore(b[:m] + (me,) + b[m + 1:], -turn, me, opp_moves, tally, illegal)
    else:
        for m in opp_moves(b, turn):
            explore(b[:m] + (turn,) + b[m + 1:], -turn, me, opp_moves, tally, illegal)


def any_move(b, turn):
    return [i for i in range(9) if b[i] == 0]


def play(px, po, rng):
    b, turn = (0,) * 9, 1
    while not winner(b) and 0 in b:
        m = (px if turn == 1 else po)(b, turn, rng)
        b = b[:m] + (turn,) + b[m + 1:]
        turn = -turn
    return winner(b)


P_net = lambda b, t, r: net_move(b, t)
P_rand = lambda b, t, r: r.choice(any_move(b, t))
P_perfect = lambda b, t, r: r.choice(perfect_moves(b, t))

print("== Every possible opponent move, all game lines ==")
for me, side in ((1, "X (first)"), (-1, "O (second)")):
    for label, opp in (("any legal reply", any_move), ("perfect replies only", perfect_moves)):
        tally, illegal = {"win": 0, "draw": 0, "loss": 0}, []
        explore((0,) * 9, 1, me, opp, tally, illegal)
        print(f"net as {side:10s} vs {label:20s}: {tally}  illegal moves: {len(illegal)}")

print("\n== 10,000 games each vs a random player ==")
N = 10000
for label, px, po, me in (("net (X) vs random", P_net, P_rand, 1), ("random vs net (O)", P_rand, P_net, -1),
                          ("perfect (X) vs random", P_perfect, P_rand, 1), ("random vs perfect (O)", P_rand, P_perfect, -1),
                          ("random (X) vs random", P_rand, P_rand, 1)):
    rng = random.Random(0)
    res = [play(px, po, rng) for _ in range(N)]
    wins, losses = sum(r == me for r in res), sum(r == -me for r in res)
    print(f"{label:24s} win {wins / N:6.1%}  draw {(N - wins - losses) / N:6.1%}  loss {losses / N:6.1%}")
