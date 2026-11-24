# Trinity-256 — emulator + RTL

**Webpage:** <https://lucky-edgelord.github.io/jade-agent-jpu/> (GitHub Pages, source `/docs`) · [source: `docs/index.html`](docs/index.html)

Heterogeneous many-core RISC-V + neural tensor superchip (256 tiles: RV64 + Coral NPU + NVDLA).

| Path | What |
| --- | --- |
| `emulator/trinity_emu.py` | Single-file, stdlib-only ISA emulator (assembler → decoder → real GEMM / MAC / ACT / AllReduce). |
| `rtl/trinity256_rtl.sv` | Hierarchical SystemVerilog: tile → quad → torus router → chiplet → package. |
| `docs/index.html` | [Project webpage](https://lucky-edgelord.github.io/jade-agent-jpu/) (served by GitHub Pages from `/docs`). |

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

## Known limitations
- Router has no input FIFOs / credit flow control: simultaneous transit flits collide
- No read-response path for global NoC or HBM reads
- `hbm_req_addr` is 34 bits; 288 GB needs 39
- P1 power state is identical to P0 (HBM self-refresh not modelled)
