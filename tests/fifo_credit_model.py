#!/usr/bin/env python3
"""Cycle model of the router/quad ingress FIFO credit rule used in the RTL.
ready(t) = fifo_cnt + in_valid(t) < 2 ; sender output is registered (1-cycle flight).
Asserts: no overflow, no loss, in-order delivery under random traffic and random stalls."""
import random
def run(seed, cycles=20000, p_send=0.9, p_pop=0.5):
    r = random.Random(seed); cnt = 0; fifo = []; flight = None; nxt = 0; got = []
    for _ in range(cycles):
        in_valid = flight is not None
        ready = (cnt + (1 if in_valid else 0)) < 2
        pop = cnt > 0 and r.random() < p_pop
        if in_valid:
            fifo.append(flight); cnt += 1
        if pop:
            got.append(fifo.pop(0)); cnt -= 1
        assert cnt <= 2, "FIFO overflow"
        flight = None
        if ready and r.random() < p_send:
            flight = nxt; nxt += 1
    while fifo or flight is not None:
        if flight is not None: fifo.append(flight); flight = None
        got.append(fifo.pop(0))
    assert got == list(range(nxt)), "lost or reordered flits"
    return nxt
for seed in range(50):
    for ps, pp in ((0.9, 0.5), (1.0, 0.2), (1.0, 1.0), (0.3, 0.9)):
        run(seed, p_send=ps, p_pop=pp)
print("PASS: credit rule never overflows or drops (50 seeds x 4 traffic mixes)")
