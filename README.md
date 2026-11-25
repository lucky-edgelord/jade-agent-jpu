# Trinity-256 — emulator + RTL

**Webpage:** <https://lucky-edgelord.github.io/jade-agent-jpu/> (GitHub Pages, source `/docs`) · [source: `docs/index.html`](docs/index.html)

Heterogeneous many-core RISC-V + neural tensor superchip (256 tiles: RV64 + Coral NPU + NVDLA).

| Path | What |
| --- | --- |
| `emulator/trinity_emu.py` | Single-file, stdlib-only ISA emulator (assembler → decoder → real GEMM / MAC / ACT / AllReduce). |
| `rtl/trinity256_rtl.sv` | Hierarchical SystemVerilog: tile → quad → torus router → chiplet → package. |
| `docs/simulator.html` | [Web simulator](https://lucky-edgelord.github.io/jade-agent-jpu/simulator.html) (UI: `simulator.js`, engine: `trinity_emu.js`). |
| `docs/index.html` | [Project webpage](https://lucky-edgelord.github.io/jade-agent-jpu/) (served by GitHub Pages from `/docs`). |

## Web simulator
Cyberpunk in-browser simulator (editor, breakpoints, step back, registers, memory, machine-code hex): <https://lucky-edgelord.github.io/jade-agent-jpu/simulator.html>

It is a JavaScript port of the emulator: `docs/trinity_emu.js` (works in the browser and Node) + `docs/simulator.html` + `docs/simulator.js`.

## Run
```
python3 emulator/trinity_emu.py            # self-tests + demos
python3 emulator/trinity_emu.py prog.asm   # assemble + run
```
RTL lint (no errors/warnings): `pip install pyslang` then parse `rtl/trinity256_rtl.sv`.

## Emulator fixes
- `LUI` sign-extends (`sext(imm20<<12)`)
- `NOC.REDUCE` honours the shard count in `R[rs2]` (0 = all 4)
- Coral memory split: ITCM 8 KB @ 0x0, DTCM 32 KB @ 0x8000; gap faults
- Semaphores: armed flag + doorbell at zero; `CORAL.ACT` no longer consumes counts
- Bounds checks on `CORAL.MAC` output tile and `NVDLA.GEMM`; illegal instruction halts cleanly

## RTL fixes
- Quad crossbar acked every requesting tile; now acks only the granted one
- Quad SRAM writes clobbered all 4 lanes; now byte-enabled per lane
- Read data returned same cycle as request (SRAM has 1-cycle latency); responses now registered
- `dst_quad == 0xF` as "HBM" collided with real Quad 15; replaced by `gw_hbm` bit, flit is exactly 1024 bits
- Remote NoC accesses carried coordinate bits and bounced back up the uplink; now use the 32-bit offset
- VC2 AllReduce accumulator never cleared; now absorbs `ALLREDUCE_FANIN` contributions, emits the sum, resets
- Gateway flits circled the torus forever and rows 2–3 had no HBM path; fixed, plus UCIe ingress and diagonal routing
- Dual-writer SRAM `always_ff` (illegal) and glitchy `clk & en` gating replaced

## Round 2 RTL fixes
- Router FIFOs (2-entry, round-robin) with credit that counts the in-flight flit, so back-to-back flits are no longer dropped
- Quad NoC ingress FIFO; remote reads stall at the head instead of being lost
- Global NoC / HBM read-response path (`RD_RESP`); HBM response carries the address and the addressed 128-bit lane
- Gateway column token: one row per cycle reaches the HBM/UCIe tap (costs 3/4 of column-3 east bandwidth)
- 39-bit HBM addresses; distinct P1 state; `hbm_self_refresh` now blocks HBM requests

## Known limitations
- One outstanding off-tile read per tile (`up_owner_q`)
- Edge and UCIe injection ignore `edge_in_ready`; UCIe transmit has no backpressure
- Emulator `SEM.ARM` has no destination tile / engine field (the RTL does)
- RTL is lint-clean (slang, 0 diagnostics) but **not simulated**; FIFO credit rule is checked by `tests/fifo_credit_model.py`, a Python model, not the RTL

## Tests
```
python3 emulator/trinity_emu.py
python3 tests/fifo_credit_model.py
node tests/emu_js_test.js      # JS port vs Python results, step-back, breakpoints
```
