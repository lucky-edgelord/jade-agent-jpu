#!/usr/bin/env python3
"""
Trinity-256 end-to-end emulator — one tile, for real.

What this is
------------
A single-file, stdlib-only emulator for the Trinity-256 ISA as defined in
`Trinity-256_Architecture___ISA_Reference_Manual_0_xgdb.md` (TRINITY256-PRM-2026-V1).

Unlike the ASM emulator HTML shipped with the spec, this one does NOT
interpret assembler metadata. It:

  1. assembles text -> 32-bit machine-code words (encoding per Manual Ch. 4-5),
  2. loads those words into instruction memory,
  3. fetches each word, DECODES it from its opcode/funct3/funct7 bits, and
  4. executes the decoded instruction against real state:
     32 x 64-bit registers, 2 MB Tile SRAM, 4 MB Quad SRAM, HBM,
     Coral TCM, NVDLA CSB, 64 quad semaphores, power modes, cycle counts.

The Xtrinity engines do real work, not scalar fakes:
  - NVDLA.GEMM really multiplies an INT8 MxK matrix (Tile SRAM) by an INT8
    KxN matrix (HBM weights) into INT32 MxN accumulators (Tile SRAM).
  - CORAL.MAC really multiplies 16x16 INT8 tiles into a 16x16 INT32 tile.
  - CORAL.ACT really transforms the INT32 tensor in Tile SRAM (ReLU /
    SwiGLU / Softmax / RMSNorm).
  - NOC.REDUCE really sums the same SRAM word across 4 simulated tiles.

Spec gaps this emulator had to fill (documented, not hidden)
-----------------------------------------------------------
The manual leaves several things unspecified. The emulator defines
"Emulator ABI v1" for them, and says so in the trace when they matter:

  G1. NVDLA.GEMM has no K dimension and no pointers. Manual: M=R[rs1],
      N=R[rs2]. ABI v1: K is taken from x15 (the "K register", set by the
      program before GEMM); A (INT8, MxK) lives at Tile SRAM offset 0x1000;
      B (INT8, KxN, weights) lives at HBM offset 0; C (INT32, MxN) is
      written at Tile SRAM offset 0. rd = total MACs (M*N*K).
      If x15 == 0 the GEMM is degenerate (0 MACs) and a warning is logged.
  G2. CORAL.ACT has no tensor length and its rs1 is ambiguous (pointer?
      value?). ABI v1: R[rs1] is a Tile SRAM *address*; the op runs over
      the element count of the most recent GEMM (M*N), in place, INT32.
      imm: 0=ReLU, 1=SwiGLU, 2=Softmax, 3=RMSNorm.
  G3. CORAL.MAC has no result destination. ABI v1: A tile (16x16 INT8) at
      Tile SRAM address R[rs1], B tile at R[rs2]; the 16x16 INT32 result
      is written to Tile SRAM at R[rs1] + 0x1000; rd = 256 (MAC count).
  G4. SwiGLU on integers needs a scale. ABI v1: values are treated as
      milli-units, i.e. SwiGLU(x) = round(x * sigmoid(x / 1000)).
      Softmax/RMSNorm outputs are INT32 milli-units (x1000) as well.
  G5. NOC.REDUCE has no shard values. ABI v1: the emulator simulates 4
      tiles; REDUCE reads the 32-bit word at Tile SRAM address R[rs1]
      from each tile and sums them; rd = the sum. The host preloads
      tiles 1-3 via `load_tile_sram(tile, addr, words)`.

Running it
----------
    python3 trinity_emu.py            # runs the built-in self-tests + demos
    python3 trinity_emu.py prog.asm   # assemble + run a program file

The manual's Chapter 6 sample is run literally in the self-test
(`demo_manual_sample`) — it assembles and halts, but computes nothing,
because of G1/G2. The corrected program (`demo_real_gemm`) is the one
that actually multiplies matrices, and its result is checked against a
plain-Python reference matmul.
"""

import math
import struct
import sys

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

# ---------------------------------------------------------------------------
# Address map (Manual Ch. 3)
# ---------------------------------------------------------------------------
CORAL_BASE = 0x0000_0000
NVDLA_CSB_BASE = 0x0010_0000
TILE_SRAM_BASE = 0x0020_0000
TILE_SRAM_SIZE = 2 * 1024 * 1024
QUAD_SRAM_BASE = 0x0100_0000
QUAD_SRAM_SIZE = 4 * 1024 * 1024
SEM_BASE = 0x0140_0000
REMOTE_BASE = 0x0001_0000_0000
HBM_BASE = 0x8000_0000_0000
ITCM_SIZE = 0x2000          # 8 KB   @ 0x0000
DTCM_BASE = 0x8000
DTCM_SIZE = 0x8000          # 32 KB  @ 0x8000
HBM_SIZE = 4 * 1024 * 1024  # modelled window (spec: up to 288 GB)

# Emulator ABI v1 layout (see module docstring, G1)
GEMM_A_OFF = 0x1000   # Tile SRAM offset of A (INT8, MxK)
GEMM_C_OFF = 0x0000   # Tile SRAM offset of C (INT32, MxN)
GEMM_B_OFF = 0x0000   # HBM offset of B / weights (INT8, KxN)


def sext(val, bits):
    """Sign-extend `val` (treated as `bits`-bit) to a Python int."""
    if val & (1 << (bits - 1)):
        return val - (1 << bits)
    return val


def to_s32(val):
    return sext(val & MASK32, 32)


def to_s64(val):
    return sext(val & MASK64, 64)


def to_u64(val):
    return val & MASK64


# ---------------------------------------------------------------------------
# Assembler: text -> 32-bit words. Encoding per Manual Ch. 4.1 / 5.1.
# ---------------------------------------------------------------------------
REGS = {f"x{i}": i for i in range(32)}
REGS.update({"zero": 0, "ra": 1, "sp": 2, "gp": 3, "tp": 4,
             "t0": 5, "t1": 6, "t2": 7, "s0": 8, "fp": 8, "s1": 9,
             "a0": 10, "a1": 11, "a2": 12, "a3": 13, "a4": 14, "a5": 15,
             "a6": 16, "a7": 17, "s2": 18, "s3": 19, "s4": 20, "s5": 21,
             "s6": 22, "s7": 23, "s8": 24, "s9": 25, "s10": 26, "s11": 27,
             "t3": 28, "t4": 29, "t5": 30, "t6": 31})


def parse_reg(tok):
    tok = tok.strip().lower()
    if tok not in REGS:
        raise ValueError(f"unknown register '{tok}'")
    return REGS[tok]


def parse_imm(tok):
    tok = tok.strip()
    neg = tok.startswith("-")
    if neg:
        tok = tok[1:]
    if tok.lower().startswith("0x"):
        val = int(tok, 16)
    else:
        val = int(tok, 10)
    return -val if neg else val


def enc_r(opcode, rd, funct3, rs1, rs2, funct7):
    return ((funct7 & 0x7F) << 25 | (rs2 & 0x1F) << 20 | (rs1 & 0x1F) << 15
            | (funct3 & 0x7) << 12 | (rd & 0x1F) << 7 | (opcode & 0x7F))


def _range_check(val, lo, hi, what):
    # FIX (2026-11-24): immediates used to be silently masked (& 0xFFF),
    # so `ADDI x11, x10, 0x2000` quietly assembled to `ADDI x11, x10, 0`.
    # Out-of-range immediates are now a hard assembly error.
    if not (lo <= val <= hi):
        raise ValueError(f"{what} {val} out of range [{lo}, {hi}]")


def enc_i(opcode, rd, funct3, rs1, imm):
    _range_check(imm, -2048, 2047, "I-type immediate")
    return ((imm & 0xFFF) << 20 | (rs1 & 0x1F) << 15 | (funct3 & 0x7) << 12
            | (rd & 0x1F) << 7 | (opcode & 0x7F))


def enc_s(opcode, funct3, rs1, rs2, imm):
    _range_check(imm, -2048, 2047, "S-type immediate")
    imm &= 0xFFF
    return ((imm >> 5) << 25 | (rs2 & 0x1F) << 20 | (rs1 & 0x1F) << 15
            | (funct3 & 0x7) << 12 | (imm & 0x1F) << 7 | (opcode & 0x7F))


def enc_b(opcode, funct3, rs1, rs2, offset):
    _range_check(offset, -4096, 4094, "B-type offset")
    if offset % 2:
        raise ValueError(f"B-type offset {offset} is not 2-byte aligned")
    off = offset & 0x1FFF
    return (((off >> 12) & 1) << 31 | ((off >> 5) & 0x3F) << 25
            | (rs2 & 0x1F) << 20 | (rs1 & 0x1F) << 15 | (funct3 & 0x7) << 12
            | ((off >> 1) & 0xF) << 8 | ((off >> 11) & 1) << 7
            | (opcode & 0x7F))


def enc_u(opcode, rd, imm20):
    _range_check(imm20, 0, 0xFFFFF, "U-type immediate")
    return ((imm20 & 0xFFFFF) << 12 | (rd & 0x1F) << 7 | (opcode & 0x7F))


def enc_j(opcode, rd, offset):
    _range_check(offset, -1048576, 1048574, "J-type offset")
    if offset % 2:
        raise ValueError(f"J-type offset {offset} is not 2-byte aligned")
    off = offset & 0x1FFFFF
    return (((off >> 20) & 1) << 31 | ((off >> 1) & 0x3FF) << 21
            | ((off >> 11) & 1) << 20 | ((off >> 12) & 0xFF) << 12
            | (rd & 0x1F) << 7 | (opcode & 0x7F))


def assemble(src):
    """Assemble Trinity assembly text. Returns (words, listing).

    listing: list of (addr, word, text) for the trace / hex dump.
    Supports labels (`loop:`), `;` / `#` comments, and every mnemonic in
    Manual Ch. 4.2 and 5.1.
    """
    lines = []
    labels = {}
    addr = 0
    for raw in src.splitlines():
        line = raw.split(";")[0].split("#")[0].strip()
        if not line:
            continue
        while ":" in line:
            lbl, rest = line.split(":", 1)
            labels[lbl.strip().upper()] = addr
            line = rest.strip()
            if not line:
                break
        if line:
            lines.append((addr, line))
            addr += 4

    words, listing = [], []
    for addr, line in lines:
        toks = line.replace(",", " ").split()
        op = toks[0].upper()

        def target(tok):
            t = tok.upper()
            if t in labels:
                return labels[t] - addr
            return parse_imm(tok)

        if op in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
            rd, rs1, rs2 = parse_reg(toks[1]), parse_reg(toks[2]), parse_reg(toks[3])
            f3 = {"AND": 7, "OR": 6, "XOR": 4}.get(op, 0)
            f7 = 0x20 if op == "SUB" else 0x01 if op == "MUL" else 0x00
            word = enc_r(0x33, rd, f3, rs1, rs2, f7)
        elif op == "ADDI":
            word = enc_i(0x13, parse_reg(toks[1]), 0, parse_reg(toks[2]),
                         parse_imm(toks[3]))
        elif op == "LUI":
            word = enc_u(0x37, parse_reg(toks[1]), parse_imm(toks[2]))
        elif op == "LW":
            import re as _re
            m = _re.match(r"^(-?\d+|0x[0-9a-fA-F]+)\(([^)]+)\)$", toks[2])
            if not m:
                raise ValueError(f"bad LW operand '{toks[2]}'")
            word = enc_i(0x03, parse_reg(toks[1]), 2, parse_reg(m.group(2)),
                         parse_imm(m.group(1)))
        elif op == "SW":
            import re as _re
            m = _re.match(r"^(-?\d+|0x[0-9a-fA-F]+)\(([^)]+)\)$", toks[2])
            if not m:
                raise ValueError(f"bad SW operand '{toks[2]}'")
            word = enc_s(0x23, 2, parse_reg(m.group(2)), parse_reg(toks[1]),
                         parse_imm(m.group(1)))
        elif op in ("BEQ", "BNE"):
            word = enc_b(0x63, 0 if op == "BEQ" else 1,
                         parse_reg(toks[1]), parse_reg(toks[2]), target(toks[3]))
        elif op == "JAL":
            word = enc_j(0x6F, parse_reg(toks[1]), target(toks[2]))
        elif op == "ECALL":
            word = 0x00000073
        elif op == "CORAL.MAC":
            word = enc_r(0x0B, parse_reg(toks[1]), 1,
                         parse_reg(toks[2]), parse_reg(toks[3]), 0x01)
        elif op == "CORAL.ACT":
            word = enc_i(0x0B, parse_reg(toks[1]), 2,
                         parse_reg(toks[2]), parse_imm(toks[3]))
        elif op == "NVDLA.GEMM":
            word = enc_r(0x2B, parse_reg(toks[1]), 1,
                         parse_reg(toks[2]), parse_reg(toks[3]), 0x02)
        elif op == "SEM.ARM":
            word = enc_i(0x5B, 0, 1, parse_reg(toks[1]), parse_imm(toks[2]))
        elif op == "PWR.MODE":
            word = enc_i(0x5B, 0, 2, 0, parse_imm(toks[1]))
        elif op == "NOC.REDUCE":
            word = enc_r(0x5B, parse_reg(toks[1]), 3,
                         parse_reg(toks[2]), parse_reg(toks[3]), 0x04)
        elif op == "HALT":
            word = 0x0000007F
        else:
            raise ValueError(f"unknown mnemonic '{op}' in: {line}")
        words.append(word)
        listing.append((addr, word, line))
    return words, listing


# ---------------------------------------------------------------------------
# Decoder: 32-bit word -> instruction dict. This is what actually runs.
# ---------------------------------------------------------------------------
def decode(word):
    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F
    imm_i = sext((word >> 20) & 0xFFF, 12)
    imm_s = sext((((word >> 25) & 0x7F) << 5) | ((word >> 7) & 0x1F), 12)
    imm_b = sext((((word >> 31) & 1) << 12) | (((word >> 7) & 1) << 11)
                 | (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1), 13)
    imm_u = sext(word & 0xFFFFF000, 32)
    imm_j = sext((((word >> 31) & 1) << 20) | (((word >> 12) & 0xFF) << 12)
                 | (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1), 21)

    if opcode == 0x33:  # R-type base
        op = {(0, 0x00): "ADD", (0, 0x20): "SUB", (0, 0x01): "MUL",
              (7, 0x00): "AND", (6, 0x00): "OR", (4, 0x00): "XOR"}.get((funct3, funct7))
        if op is None:
            raise ValueError(f"undecoded R-type word 0x{word:08x}")
        return {"op": op, "rd": rd, "rs1": rs1, "rs2": rs2}
    if opcode == 0x13:
        return {"op": "ADDI", "rd": rd, "rs1": rs1, "imm": imm_i}
    if opcode == 0x37:
        return {"op": "LUI", "rd": rd, "imm": imm_u}
    if opcode == 0x03:
        return {"op": "LW", "rd": rd, "rs1": rs1, "imm": imm_i}
    if opcode == 0x23:
        return {"op": "SW", "rs1": rs1, "rs2": rs2, "imm": imm_s}
    if opcode == 0x63:
        op = {0: "BEQ", 1: "BNE"}.get(funct3)
        if op is None:
            raise ValueError(f"undecoded branch word 0x{word:08x}")
        return {"op": op, "rs1": rs1, "rs2": rs2, "imm": imm_b}
    if opcode == 0x6F:
        return {"op": "JAL", "rd": rd, "imm": imm_j}
    if opcode == 0x73:
        return {"op": "ECALL"}
    if opcode == 0x0B:
        if funct3 == 1 and funct7 == 0x01:
            return {"op": "CORAL.MAC", "rd": rd, "rs1": rs1, "rs2": rs2}
        if funct3 == 2:
            return {"op": "CORAL.ACT", "rd": rd, "rs1": rs1, "imm": imm_i}
        raise ValueError(f"undecoded custom-0 word 0x{word:08x}")
    if opcode == 0x2B:
        if funct3 == 1 and funct7 == 0x02:
            return {"op": "NVDLA.GEMM", "rd": rd, "rs1": rs1, "rs2": rs2}
        raise ValueError(f"undecoded custom-1 word 0x{word:08x}")
    if opcode == 0x5B:
        if funct3 == 1:
            return {"op": "SEM.ARM", "rs1": rs1, "imm": imm_i}
        if funct3 == 2:
            return {"op": "PWR.MODE", "imm": imm_i}
        if funct3 == 3 and funct7 == 0x04:
            return {"op": "NOC.REDUCE", "rd": rd, "rs1": rs1, "rs2": rs2}
        raise ValueError(f"undecoded custom-2 word 0x{word:08x}")
    if opcode == 0x7F:
        return {"op": "HALT"}
    raise ValueError(f"unknown opcode 0x{opcode:02x} in word 0x{word:08x}")


# ---------------------------------------------------------------------------
# CPU / tile
# ---------------------------------------------------------------------------
class TrinityTile:
    def __init__(self, trace=False):
        self.regs = [0] * 32
        self.pc = 0
        self.program = []          # list of 32-bit words
        self.listing = []          # (addr, word, text)
        self.sram = bytearray(TILE_SRAM_SIZE)
        self.quad_sram = bytearray(QUAD_SRAM_SIZE)
        self.coral_tcm = bytearray(0x10000)   # 8K ITCM + 32K DTCM + CSRs
        self.hbm = bytearray(HBM_SIZE)
        self.csb = {}              # NVDLA CSB registers (addr -> value)
        # 4 tiles, for NOC.REDUCE: tile 0 is this CPU; 1-3 are shard tiles.
        self.shard_srams = [self.sram, bytearray(TILE_SRAM_SIZE),
                            bytearray(TILE_SRAM_SIZE), bytearray(TILE_SRAM_SIZE)]
        self.semaphores = [0] * 64
        self.sem_armed = [False] * 64
        self.doorbells = [0] * 64   # pending doorbells (count reached 0)
        self.power_mode = 3        # P3 sentinel at reset, per spec
        self.cycles = 0
        self.halted = False
        self.trace = trace
        self.log = []
        self.coral_macs = 0
        self.nvdla_macs = 0
        self.last_gemm = None     # (M, N, K)
        self.warnings = []

    # -- memory ------------------------------------------------------------
    def _mem(self, addr, size):
        if CORAL_BASE <= addr < CORAL_BASE + ITCM_SIZE or \
                DTCM_BASE <= addr < DTCM_BASE + DTCM_SIZE:
            if addr + size > (CORAL_BASE + ITCM_SIZE if addr < DTCM_BASE
                              else DTCM_BASE + DTCM_SIZE):
                raise ValueError(f"access straddles TCM boundary: 0x{addr:08x}")
            return self.coral_tcm, addr - CORAL_BASE
        if NVDLA_CSB_BASE <= addr < NVDLA_CSB_BASE + 0x100000:
            return None, addr  # CSB: handled by caller
        if TILE_SRAM_BASE <= addr < TILE_SRAM_BASE + TILE_SRAM_SIZE:
            return self.sram, addr - TILE_SRAM_BASE
        if QUAD_SRAM_BASE <= addr < QUAD_SRAM_BASE + QUAD_SRAM_SIZE:
            return self.quad_sram, addr - QUAD_SRAM_BASE
        if HBM_BASE <= addr < HBM_BASE + HBM_SIZE:
            return self.hbm, addr - HBM_BASE
        raise ValueError(f"memory access outside modelled map: 0x{addr:012x}")

    def read32(self, addr):
        if NVDLA_CSB_BASE <= addr < NVDLA_CSB_BASE + 0x100000:
            return self.csb.get(addr, 0)
        if SEM_BASE <= addr < SEM_BASE + 0x10000:
            return self.semaphores[(addr - SEM_BASE) // 4 % 64]
        buf, off = self._mem(addr, 4)
        return struct.unpack_from("<I", buf, off)[0]

    def write32(self, addr, val):
        if NVDLA_CSB_BASE <= addr < NVDLA_CSB_BASE + 0x100000:
            self.csb[addr] = val & MASK32
            return
        if SEM_BASE <= addr < SEM_BASE + 0x10000:
            self.semaphores[(addr - SEM_BASE) // 4 % 64] = val & MASK32
            return
        buf, off = self._mem(addr, 4)
        struct.pack_into("<I", buf, off, val & MASK32)

    def read_sram_i8(self, off):
        return sext(self.sram[off], 8)

    def read_sram_i32(self, off):
        return to_s32(struct.unpack_from("<i", self.sram, off)[0])

    def write_sram_i32(self, off, val):
        struct.pack_into("<i", self.sram, off, to_s32(val))

    # -- host loading (what a loader / DBB DMA would do) -------------------
    def load_program(self, src):
        self.program, self.listing = assemble(src)
        self.pc = 0
        self.halted = False

    def load_matrix_i8_sram(self, off, rows):
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.sram[off + r * len(row) + c] = v & 0xFF

    def load_matrix_i8_hbm(self, off, rows):
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                self.hbm[off + r * len(row) + c] = v & 0xFF

    def load_tile_sram(self, tile, addr, words):
        """Preload a 32-bit word list into shard tile `tile`'s SRAM."""
        base = addr - TILE_SRAM_BASE
        for i, w in enumerate(words):
            struct.pack_into("<I", self.shard_srams[tile], base + 4 * i, w & MASK32)

    # -- registers ---------------------------------------------------------
    def r(self, i):
        return self.regs[i]

    def w(self, i, val):
        if i != 0:
            self.regs[i] = to_u64(val)

    # -- execution ---------------------------------------------------------
    def step(self):
        if self.halted:
            return False
        idx = self.pc // 4
        if idx < 0 or idx >= len(self.program):
            self.halted = True
            self.log.append(f"PC 0x{self.pc:04x} outside program — halted")
            return False
        word = self.program[idx]
        try:
            ins = decode(word)
        except ValueError as e:
            self.halted = True
            self.log.append(f"PC 0x{self.pc:04x}: illegal instruction — {e}")
            return False
        op = ins["op"]
        pc0 = self.pc
        next_pc = self.pc + 4
        cost = 1
        note = ""

        if op == "ADD":
            self.w(ins["rd"], self.r(ins["rs1"]) + self.r(ins["rs2"]))
        elif op == "SUB":
            self.w(ins["rd"], self.r(ins["rs1"]) - self.r(ins["rs2"]))
        elif op == "MUL":
            self.w(ins["rd"], self.r(ins["rs1"]) * self.r(ins["rs2"]))
            cost = 3
        elif op == "AND":
            self.w(ins["rd"], self.r(ins["rs1"]) & self.r(ins["rs2"]))
        elif op == "OR":
            self.w(ins["rd"], self.r(ins["rs1"]) | self.r(ins["rs2"]))
        elif op == "XOR":
            self.w(ins["rd"], self.r(ins["rs1"]) ^ self.r(ins["rs2"]))
        elif op == "ADDI":
            self.w(ins["rd"], self.r(ins["rs1"]) + ins["imm"])
        elif op == "LUI":
            self.w(ins["rd"], ins["imm"])
        elif op == "LW":
            addr = (self.r(ins["rs1"]) + ins["imm"]) & MASK64
            self.w(ins["rd"], to_s32(self.read32(addr)))
            cost = 2
        elif op == "SW":
            addr = (self.r(ins["rs1"]) + ins["imm"]) & MASK64
            self.write32(addr, self.r(ins["rs2"]))
            cost = 2
        elif op == "BEQ":
            if self.r(ins["rs1"]) == self.r(ins["rs2"]):
                next_pc = pc0 + ins["imm"]
        elif op == "BNE":
            if self.r(ins["rs1"]) != self.r(ins["rs2"]):
                next_pc = pc0 + ins["imm"]
        elif op == "JAL":
            self.w(ins["rd"], pc0 + 4)
            next_pc = pc0 + ins["imm"]
        elif op == "ECALL":
            a7 = self.r(17)
            if a7 == 1:
                note = f"ECALL print a0={to_s64(self.r(10))}"
                self.log.append(f"  [ECALL] a0 = {to_s64(self.r(10))}")
            else:
                note = f"ECALL a7={a7} (unimplemented syscall — halt)"
                self.halted = True
        elif op == "CORAL.MAC":
            cost, note = self._coral_mac(ins)
        elif op == "CORAL.ACT":
            cost, note = self._coral_act(ins)
        elif op == "NVDLA.GEMM":
            cost, note = self._nvdla_gemm(ins)
        elif op == "SEM.ARM":
            sem = self.r(ins["rs1"]) & 63
            self.semaphores[sem] = ins["imm"] & 0xFFF
            self.sem_armed[sem] = self.semaphores[sem] > 0
            note = f"sem#{sem} <- {self.semaphores[sem]}"
        elif op == "PWR.MODE":
            self.power_mode = ins["imm"] & 0x3
            note = f"power -> P{self.power_mode}"
        elif op == "NOC.REDUCE":
            cost, note = self._noc_reduce(ins)
        elif op == "HALT":
            self.halted = True
            note = "halt"
        else:  # pragma: no cover
            raise AssertionError(op)

        self.regs[0] = 0
        self.pc = next_pc
        self.cycles += cost
        if self.trace:
            txt = self.listing[idx][2] if idx < len(self.listing) else "?"
            self.log.append(f"pc=0x{pc0:04x} word=0x{word:08x} {txt:<38} "
                            f"cycles+{cost} {note}")
        return not self.halted

    def run(self, max_steps=100000):
        steps = 0
        while not self.halted and steps < max_steps:
            if not self.step():
                break
            steps += 1
        if steps >= max_steps:
            self.warnings.append("step limit reached — possible infinite loop")
        return steps

    # -- Xtrinity engines (real work) --------------------------------------
    def _sem_complete(self, sem):
        """Engine-completion auto-decrement; fires a doorbell at zero."""
        if self.sem_armed[sem] and self.semaphores[sem] > 0:
            self.semaphores[sem] -= 1
            if self.semaphores[sem] == 0:
                self.sem_armed[sem] = False
                self.doorbells[sem] += 1

    def _nvdla_gemm(self, ins):
        m = self.r(ins["rs1"])
        n = self.r(ins["rs2"])
        k = self.r(15)  # ABI v1: K register is x15 (see G1)
        if k == 0:
            msg = ("G1 warning: NVDLA.GEMM with K (x15) == 0 — the manual "
                   "gives no K dimension, so this GEMM is degenerate (0 MACs)")
            self.warnings.append(msg)
            self.log.append(f"  [WARN] {msg}")
            self.last_gemm = (m, n, 0)
            self.w(ins["rd"], 0)
            return 2, "GEMM degenerate, 0 MACs"
        macs = m * n * k
        a_end = GEMM_A_OFF + m * k
        c_end = GEMM_C_OFF + 4 * m * n
        if a_end > TILE_SRAM_SIZE or c_end > TILE_SRAM_SIZE or k * n > HBM_SIZE:
            raise ValueError(f"NVDLA.GEMM {m}x{k}x{n} exceeds modelled memory")
        if c_end > GEMM_A_OFF:
            msg = (f"G1 warning: C ({c_end - GEMM_C_OFF} B) overlaps A at "
                   f"0x{GEMM_A_OFF:04x}; A is overwritten during the GEMM")
            self.warnings.append(msg)
            self.log.append(f"  [WARN] {msg}")
        for i in range(m):
            for j in range(n):
                acc = 0
                for kk in range(k):
                    a = self.read_sram_i8(GEMM_A_OFF + i * k + kk)
                    b = sext(self.hbm[GEMM_B_OFF + kk * n + j], 8)
                    acc += a * b
                self.write_sram_i32(GEMM_C_OFF + 4 * (i * n + j), acc)
        self.nvdla_macs += macs
        self.last_gemm = (m, n, k)
        self.w(ins["rd"], macs)
        # doorbell: a completed GEMM decrements semaphore 0 if armed
        self._sem_complete(0)
        return max(2, (macs + 511) // 512), f"GEMM {m}x{k} * {k}x{n} = {macs} MACs -> SRAM[0x{GEMM_C_OFF:04x}]"

    def _coral_act(self, ins):
        addr = self.r(ins["rs1"])
        kind = ins["imm"]
        if self.last_gemm is None:
            count = 1
        else:
            count = self.last_gemm[0] * self.last_gemm[1]
        off = addr - TILE_SRAM_BASE
        if not (0 <= off < TILE_SRAM_SIZE - 4 * count):
            msg = (f"G2 warning: CORAL.ACT address 0x{addr:08x} is outside "
                   f"Tile SRAM for {count} elements — no-op")
            self.warnings.append(msg)
            self.log.append(f"  [WARN] {msg}")
            self.w(ins["rd"], 0)
            return 2, "ACT no-op (bad address)"
        vals = [self.read_sram_i32(off + 4 * i) for i in range(count)]
        if self.sem_armed[0] and self.semaphores[0] > 0:
            msg = "sem#0 still counting: CORAL.ACT issued before producer done"
            self.warnings.append(msg)
            self.log.append(f"  [WARN] {msg}")
        elif self.doorbells[0] > 0:
            self.doorbells[0] -= 1
        names = {0: "ReLU", 1: "SwiGLU", 2: "Softmax", 3: "RMSNorm"}
        name = names.get(kind, f"ACT#{kind}")
        if kind == 0:
            out = [max(0, v) for v in vals]
        elif kind == 1:
            out = [round(v * (1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, v / 1000.0))))))
                   for v in vals]
        elif kind == 2:
            mx = max(vals) if vals else 0
            exps = [math.exp((v - mx) / 1000.0) for v in vals]
            s = sum(exps) or 1.0
            out = [round(e / s * 1000) for e in exps]
        elif kind == 3:
            rms = math.sqrt(sum(v * v for v in vals) / max(1, len(vals))) or 1.0
            out = [round(v / rms * 1000) for v in vals]
        else:
            out = vals
        for i, v in enumerate(out):
            self.write_sram_i32(off + 4 * i, v)
        self.coral_macs += 128
        self.w(ins["rd"], count)
        return 2, f"ACT {name} on {count} elems @ 0x{addr:08x}"

    def _coral_mac(self, ins):
        a_addr = self.r(ins["rs1"])
        b_addr = self.r(ins["rs2"])
        a_off, b_off = a_addr - TILE_SRAM_BASE, b_addr - TILE_SRAM_BASE
        if not (0 <= a_off and a_off + 0x1000 + 1024 <= TILE_SRAM_SIZE
                and 0 <= b_off and b_off + 256 <= TILE_SRAM_SIZE):
            self.warnings.append("CORAL.MAC address outside Tile SRAM — no-op")
            self.w(ins["rd"], 0)
            return 1, "MAC no-op (bad address)"
        for i in range(16):
            for j in range(16):
                acc = 0
                for kk in range(16):
                    acc += (self.read_sram_i8(a_off + i * 16 + kk)
                            * self.read_sram_i8(b_off + kk * 16 + j))
                self.write_sram_i32(a_off + 0x1000 + 4 * (i * 16 + j), acc)
        self.coral_macs += 256
        self.w(ins["rd"], 256)
        return 1, "MAC 16x16 tile -> SRAM[rs1+0x1000], 256 MACs"

    def _noc_reduce(self, ins):
        addr = self.r(ins["rs1"])
        off = addr - TILE_SRAM_BASE
        shards = self.r(ins["rs2"]) or 4
        if shards > 4:
            self.warnings.append(f"NOC.REDUCE shards={shards} > 4 modelled; clamped")
            shards = 4
        if not (0 <= off <= TILE_SRAM_SIZE - 4):
            self.warnings.append("NOC.REDUCE address outside Tile SRAM — no-op")
            self.w(ins["rd"], 0)
            return 2, "REDUCE no-op (bad address)"
        total = 0
        for t in range(shards):
            total += to_s32(struct.unpack_from("<I", self.shard_srams[t], off)[0])
        self.w(ins["rd"], total)
        return 8, f"AllReduce sum over {shards} tiles = {total}"


# ---------------------------------------------------------------------------
# Reference models (for the tests — plain Python, independent of the CPU)
# ---------------------------------------------------------------------------
def ref_gemm(a, b):
    m, k, n = len(a), len(a[0]), len(b[0])
    return [[sum(a[i][kk] * b[kk][j] for kk in range(k)) for j in range(n)]
            for i in range(m)]


# ---------------------------------------------------------------------------
# Demos / self-tests
# ---------------------------------------------------------------------------
MANUAL_SAMPLE = """\
; Manual Ch. 6.1, verbatim: Transformer MLP projection + SwiGLU
PWR.MODE 0
LUI  x10, 0x200
ADDI x11, x0, 64
ADDI x12, x0, 128
SEM.ARM x0, 1
NVDLA.GEMM x13, x11, x12
CORAL.ACT  x14, x13, 1
SW   x14, 0(x10)
HALT
"""

REAL_GEMM = """\
; Corrected program: 4x8 * 8x4 INT8 GEMM, then SwiGLU on C.
; ABI v1: x15=K, A at SRAM 0x1000, B at HBM 0, C at SRAM 0.
PWR.MODE 0
LUI  x10, 0x200          ; x10 = Tile SRAM base 0x00200000
ADDI x11, x0, 4          ; M = 4
ADDI x12, x0, 4          ; N = 4
ADDI x15, x0, 8          ; K = 8  (manual gives no K; ABI v1 reads x15)
SEM.ARM x0, 1
NVDLA.GEMM x13, x11, x12 ; x13 <- MACs; C -> SRAM[0]
CORAL.ACT  x14, x10, 1   ; SwiGLU in place on C @ SRAM base; x14 <- elems
SW   x14, 0x40(x10)      ; store element count at SRAM[0x40]
HALT
"""

REDUCE_PROG = """\
; Loop test (sum 1..10) + CORAL.MAC tile + NOC.REDUCE over 4 tiles.
ADDI x5, x0, 0           ; sum = 0
ADDI x6, x0, 1           ; i = 1
ADDI x7, x0, 11          ; limit = 11
loop:
ADD  x5, x5, x6
ADDI x6, x6, 1
BNE  x6, x7, loop
LUI  x10, 0x200
; NOTE: ADDI immediates are 12-bit, so 0x2000/0x2100/0x3000 do NOT fit
; in one ADDI (they would be masked to 0x000/0x100/0x000). Use LUI.
LUI  x11, 0x202          ; A tile @ 0x00202000 (SRAM+0x2000)
LUI  x12, 0x202
ADDI x12, x12, 0x100     ; B tile @ 0x00202100 (SRAM+0x2100)
CORAL.MAC x13, x11, x12  ; 16x16 tile -> SRAM+0x3000
LUI  x16, 0x203          ; shard scalar @ 0x00203000 (SRAM+0x3000)
NOC.REDUCE x17, x16, x0  ; sum the word at that offset over 4 tiles
HALT
"""


def demo_manual_sample():
    print("== Demo 1: Manual Ch. 6.1 sample, run literally ==")
    cpu = TrinityTile(trace=True)
    cpu.load_program(MANUAL_SAMPLE)
    print("   assembled words:")
    for addr, word, text in cpu.listing:
        print(f"     0x{addr:04x}: 0x{word:08x}  {text}")
    cpu.run()
    for line in cpu.log:
        print("   " + line)
    print(f"   halted={cpu.halted} cycles={cpu.cycles} "
          f"x13(MACs)={cpu.r(13)} x14={cpu.r(14)} sem0={cpu.semaphores[0]}")
    print("   warnings:", cpu.warnings or "none")
    print("   -> It assembles, decodes and halts. But x13 = 0 MACs: the manual")
    print("      never gives K, so no real GEMM could run. That is spec gap G1.")
    return cpu


def demo_real_gemm():
    print("\n== Demo 2: Corrected program — a real 4x8 * 8x4 GEMM + SwiGLU ==")
    a = [[1, 2, 0, 1, 3, 0, 2, 1],
         [0, 1, 2, 0, 1, 1, 0, 2],
         [2, 0, 1, 1, 0, 2, 1, 0],
         [1, 1, 1, 0, 2, 0, 1, 1]]
    b = [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [1, 0, 1, 2],
         [0, 2, 1, 0], [1, 1, 0, 1], [0, 0, 2, 1], [2, 1, 0, 0]]
    expected = ref_gemm(a, b)
    cpu = TrinityTile(trace=True)
    cpu.load_matrix_i8_sram(GEMM_A_OFF, a)
    cpu.load_matrix_i8_hbm(GEMM_B_OFF, b)
    cpu.load_program(REAL_GEMM)
    cpu.run()
    for line in cpu.log:
        print("   " + line)
    got = [[cpu.read_sram_i32(4 * (i * 4 + j)) for j in range(4)] for i in range(4)]
    # SwiGLU was applied in place, so compare against activated reference:
    exp_act = [round(v * (1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, v / 1000.0))))))
               for row in expected for v in row]
    got_flat = [v for row in got for v in row]
    print(f"   halted={cpu.halted} cycles={cpu.cycles} "
          f"x13(MACs)={cpu.r(13)} x14(elems)={cpu.r(14)} sem0={cpu.semaphores[0]}")
    print(f"   C (after SwiGLU) = {got_flat}")
    print(f"   expected         = {exp_act}")
    assert cpu.r(13) == 4 * 4 * 8, f"MAC count {cpu.r(13)} != 128"
    assert got_flat == exp_act, "GEMM+SwiGLU result mismatch vs reference"
    assert cpu.read_sram_i32(0x40) == 16, "element-count store failed"
    print("   PASS: GEMM + SwiGLU match the independent Python reference.")
    return cpu


def demo_reduce_and_mac():
    print("\n== Demo 3: Loop + CORAL.MAC tile + NOC.REDUCE ==")
    cpu = TrinityTile(trace=True)
    # A tile: identity-ish (1s on diagonal), B tile: all 2s, at SRAM offsets
    a = [[1 if i == j else 0 for j in range(16)] for i in range(16)]
    b = [[2 for _ in range(16)] for _ in range(16)]
    cpu.load_matrix_i8_sram(0x2000, a)
    cpu.load_matrix_i8_sram(0x2100, b)
    # shard tiles 1..3 hold the same word at SRAM+0x3000 as tile 0 will
    # compute there (C[0][0] = 2). Preload them with 10, 20, 30; tile 0
    # computes 2, so the reduce should give 2+10+20+30 = 62.
    for tile, val in ((1, 10), (2, 20), (3, 30)):
        cpu.load_tile_sram(tile, TILE_SRAM_BASE + 0x3000, [val])
    cpu.load_program(REDUCE_PROG)
    cpu.run()
    for line in cpu.log:
        print("   " + line)
    print(f"   loop sum 1..10: x5={cpu.r(5)} (expect 55)")
    print(f"   CORAL.MAC: x13={cpu.r(13)} (expect 256); "
          f"C[0][0]={cpu.read_sram_i32(0x3000)} (expect 2)")
    print(f"   NOC.REDUCE: x17={cpu.r(17)} (expect 2+10+20+30 = 62)")
    assert cpu.r(5) == 55
    assert cpu.r(13) == 256 and cpu.read_sram_i32(0x3000) == 2
    assert cpu.r(17) == 62
    print("   PASS: loop, tile MAC and 4-tile AllReduce all check out.")
    return cpu


def test_encodings():
    """Manual Appendix A: the six worked examples must re-encode exactly."""
    cases = [
        ("ADDI x11, x0, 64", 0x04000593),
        ("LUI x10, 0x200", 0x00200537),
        ("NVDLA.GEMM x13, x11, x12", 0x04C596AB),
        ("CORAL.ACT x14, x13, 1", 0x0016A70B),
        ("NOC.REDUCE x15, x14, x11", 0x08B737DB),
        ("HALT", 0x0000007F),
    ]
    print("== Encoding test: Manual Appendix A ==")
    for src, want in cases:
        (got,), _ = assemble(src)
        status = "OK " if got == want else "FAIL"
        print(f"   {status} {src:<28} -> 0x{got:08x} (manual: 0x{want:08x})")
        assert got == want, f"{src}: got 0x{got:08x}, want 0x{want:08x}"
    # decode round-trip: the words must decode back to the same instruction
    for src, _ in cases:
        (word,), _ = assemble(src)
        ins = decode(word)
        assert ins["op"] == src.split()[0], f"decode mismatch for {src}: {ins}"
    print("   PASS: all six Appendix A words encode and decode correctly.")
    # x0 hardwired, and a bad opcode must fail loudly, not silently run
    cpu = TrinityTile()
    cpu.load_program("ADDI x0, x0, 99\nHALT\n")
    cpu.run()
    assert cpu.r(0) == 0
    try:
        decode(0x00000001)  # opcode 0x01: not a Trinity opcode
    except ValueError:
        pass
    else:
        raise AssertionError("bad opcode did not raise")
    print("   PASS: x0 stays 0; unknown opcodes raise instead of executing.")
    # FIX: out-of-range immediates must fail loudly, not mask silently.
    try:
        assemble("ADDI x11, x10, 0x2000")
    except ValueError as e:
        print(f"   PASS: ADDI 0x2000 rejected as it should be: {e}")
    else:
        raise AssertionError("ADDI x11, x10, 0x2000 did not raise")


def test_fixes():
    print("\n== Regression tests for fixes ==")
    # LUI sign-extends (manual: sext(imm20 << 12))
    cpu = TrinityTile(); cpu.load_program("LUI x5, 0x80000\nHALT\n"); cpu.run()
    assert to_s64(cpu.r(5)) == -(1 << 31), hex(cpu.r(5))
    cpu = TrinityTile(); cpu.load_program("LUI x5, 0x200\nHALT\n"); cpu.run()
    assert cpu.r(5) == 0x200000
    print("   PASS: LUI sign extension")
    # NOC.REDUCE honours shard count
    cpu = TrinityTile()
    for t, v in ((1, 10), (2, 20), (3, 30)):
        cpu.load_tile_sram(t, TILE_SRAM_BASE + 0x100, [v])
    cpu.write32(TILE_SRAM_BASE + 0x100, 1)
    cpu.load_program("LUI x10, 0x200\nADDI x10, x10, 0x100\nADDI x11, x0, 2\n"
                     "NOC.REDUCE x12, x10, x11\nHALT\n")
    cpu.run(); assert cpu.r(12) == 11, cpu.r(12)
    print("   PASS: NOC.REDUCE shard count (2 shards -> 1+10)")
    # TCM gap is unmapped; DTCM works
    cpu = TrinityTile(); cpu.write32(0x8000, 7); assert cpu.read32(0x8000) == 7
    try:
        cpu.read32(0x4000)
    except ValueError:
        pass
    else:
        raise AssertionError("TCM gap readable")
    print("   PASS: ITCM/DTCM split, gap faults")
    # semaphore doorbell lifecycle; ACT does not consume counts
    cpu = TrinityTile(); cpu.last_gemm = (1, 1, 1)
    cpu.sem_armed[0] = True; cpu.semaphores[0] = 2
    cpu._sem_complete(0); assert cpu.semaphores[0] == 1 and cpu.doorbells[0] == 0
    cpu._sem_complete(0); assert cpu.semaphores[0] == 0 and cpu.doorbells[0] == 1
    print("   PASS: semaphore fires doorbell at zero")
    # CORAL.MAC near end of SRAM must no-op, not crash
    cpu = TrinityTile()
    cpu.load_program("LUI x11, 0x3FF\nCORAL.MAC x13, x11, x11\nHALT\n"); cpu.run()
    assert cpu.r(13) == 0 and cpu.warnings
    print("   PASS: CORAL.MAC output-tile bounds")


def main():
    if len(sys.argv) > 1:
        src = open(sys.argv[1]).read()
        cpu = TrinityTile(trace=True)
        cpu.load_program(src)
        for addr, word, text in cpu.listing:
            print(f"0x{addr:04x}: 0x{word:08x}  {text}")
        cpu.run()
        for line in cpu.log:
            print(line)
        print(f"halted={cpu.halted} cycles={cpu.cycles} "
              f"regs x10={cpu.r(10)} x13={cpu.r(13)} x14={cpu.r(14)}")
        print("warnings:", cpu.warnings or "none")
        return 0
    test_encodings()
    test_fixes()
    demo_manual_sample()
    demo_real_gemm()
    demo_reduce_and_mac()
    print("\nALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
