/*!
 * Trinity-256 emulator — JavaScript port of emulator/trinity_emu.py
 * Works in the browser (window.Trinity) and Node (require). No dependencies.
 *
 * Adds over the Python version: step-back (undo journal), per-instruction run
 * counters, a disassembler, breakpoints in run(), and `.sram/.hbm/.shard`
 * preload directives so programs can carry their own data.
 *
 * Emulator ABI v1 (spec gaps, same as the Python emulator):
 *   G1 GEMM: K = x15, A (INT8 MxK) @ Tile SRAM 0x1000, B (INT8 KxN) @ HBM 0, C (INT32 MxN) @ Tile SRAM 0
 *   G2 ACT : R[rs1] = Tile SRAM address; runs over M*N elements of the last GEMM, in place
 *   G3 MAC : A tile @ R[rs1], B tile @ R[rs2], 16x16 INT32 result @ R[rs1]+0x1000
 *   G4 SwiGLU/Softmax/RMSNorm treat INT32 as milli-units
 *   G5 NOC.REDUCE: sums the word at R[rs1] over R[rs2] tiles (0 = 4)
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.Trinity = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const C = {
    ITCM_SIZE: 0x2000, DTCM_BASE: 0x8000, DTCM_SIZE: 0x8000,
    NVDLA_CSB_BASE: 0x100000, TILE_SRAM_BASE: 0x200000, TILE_SRAM_SIZE: 2 * 1024 * 1024,
    QUAD_SRAM_BASE: 0x1000000, QUAD_SRAM_SIZE: 4 * 1024 * 1024, SEM_BASE: 0x1400000,
    HBM_BASE: 0x800000000000, HBM_SIZE: 4 * 1024 * 1024,
    GEMM_A_OFF: 0x1000, GEMM_C_OFF: 0, GEMM_B_OFF: 0,
    MAX_MACS: 20000000
  };

  const sext = (v, bits) => { const s = 32 - bits; return (v << s) >> s; };
  const h = (n, w) => (n >>> 0).toString(16).padStart(w, '0');

  function pyRound(x) { // Python's round-half-to-even, for parity with the reference emulator
    const f = Math.floor(x), d = x - f;
    if (d < 0.5) return f;
    if (d > 0.5) return f + 1;
    return f % 2 === 0 ? f : f + 1;
  }

  // ---------------------------------------------------------------- assembler
  const REG_NAMES = ['zero','ra','sp','gp','tp','t0','t1','t2','s0','s1','a0','a1','a2','a3','a4','a5','a6','a7',
    's2','s3','s4','s5','s6','s7','s8','s9','s10','s11','t3','t4','t5','t6'];
  const REGS = {};
  for (let i = 0; i < 32; i++) { REGS['x' + i] = i; REGS[REG_NAMES[i]] = i; }
  REGS.fp = 8;

  function parseReg(tok) {
    const t = (tok || '').trim().toLowerCase();
    if (!(t in REGS)) throw new Error(`unknown register '${tok}'`);
    return REGS[t];
  }
  function parseImm(tok) {
    tok = (tok || '').trim();
    let neg = false;
    if (tok[0] === '-') { neg = true; tok = tok.slice(1); } else if (tok[0] === '+') tok = tok.slice(1);
    let v;
    if (/^0x[0-9a-f]+$/i.test(tok)) v = parseInt(tok, 16);
    else if (/^\d+$/.test(tok)) v = parseInt(tok, 10);
    else throw new Error(`bad immediate '${tok}'`);
    return neg ? -v : v;
  }
  function rangeCheck(v, lo, hi, what) {
    if (!(v >= lo && v <= hi)) throw new Error(`${what} ${v} out of range [${lo}, ${hi}]`);
  }
  const encR = (op, rd, f3, rs1, rs2, f7) =>
    (((f7 & 0x7f) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | ((rd & 31) << 7) | (op & 0x7f)) >>> 0;
  function encI(op, rd, f3, rs1, imm) {
    rangeCheck(imm, -2048, 2047, 'I-type immediate');
    return (((imm & 0xfff) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | ((rd & 31) << 7) | (op & 0x7f)) >>> 0;
  }
  function encS(op, f3, rs1, rs2, imm) {
    rangeCheck(imm, -2048, 2047, 'S-type immediate');
    imm &= 0xfff;
    return (((imm >> 5) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) | ((f3 & 7) << 12) | ((imm & 31) << 7) | (op & 0x7f)) >>> 0;
  }
  function encB(op, f3, rs1, rs2, offset) {
    rangeCheck(offset, -4096, 4094, 'B-type offset');
    if (offset % 2) throw new Error(`B-type offset ${offset} is not 2-byte aligned`);
    const off = offset & 0x1fff;
    return ((((off >> 12) & 1) << 31) | (((off >> 5) & 0x3f) << 25) | ((rs2 & 31) << 20) | ((rs1 & 31) << 15) |
      ((f3 & 7) << 12) | (((off >> 1) & 0xf) << 8) | (((off >> 11) & 1) << 7) | (op & 0x7f)) >>> 0;
  }
  function encU(op, rd, imm20) {
    rangeCheck(imm20, 0, 0xfffff, 'U-type immediate');
    return (((imm20 & 0xfffff) << 12) | ((rd & 31) << 7) | (op & 0x7f)) >>> 0;
  }
  function encJ(op, rd, offset) {
    rangeCheck(offset, -1048576, 1048574, 'J-type offset');
    if (offset % 2) throw new Error(`J-type offset ${offset} is not 2-byte aligned`);
    const off = offset & 0x1fffff;
    return ((((off >> 20) & 1) << 31) | (((off >> 1) & 0x3ff) << 21) | (((off >> 11) & 1) << 20) |
      (((off >> 12) & 0xff) << 12) | ((rd & 31) << 7) | (op & 0x7f)) >>> 0;
  }

  const MEM_RE = /^(-?\d+|-?0x[0-9a-fA-F]+)\(([^)]+)\)$/;

  function assemble(src) {
    const labels = {}, lines = [], preload = [];
    let addr = 0;
    src.split(/\r?\n/).forEach((raw, ix) => {
      let line = raw.split(';')[0].split('#')[0].trim();
      if (!line) return;
      const ln = ix + 1;
      if (line[0] === '.') {
        const t = line.replace(/,/g, ' ').trim().split(/\s+/), d = t[0].toLowerCase();
        try {
          if (d === '.sram' || d === '.hbm') preload.push({ kind: d.slice(1), off: parseImm(t[1]), vals: t.slice(2).map(parseImm), line: ln });
          else if (d === '.shard') preload.push({ kind: 'shard', tile: parseImm(t[1]), off: parseImm(t[2]), vals: t.slice(3).map(parseImm), line: ln });
          else throw new Error(`unknown directive '${t[0]}'`);
        } catch (e) { throw new Error(`line ${ln}: ${e.message}`); }
        return;
      }
      while (line.includes(':')) {
        const k = line.indexOf(':'), lbl = line.slice(0, k).trim().toUpperCase();
        if (lbl in labels) throw new Error(`line ${ln}: duplicate label '${lbl}'`);
        labels[lbl] = addr;
        line = line.slice(k + 1).trim();
        if (!line) break;
      }
      if (line) { lines.push({ addr, line, ln }); addr += 4; }
    });

    const words = [], listing = [];
    for (const { addr: a, line, ln } of lines) {
      try {
        const toks = line.replace(/,/g, ' ').trim().split(/\s+/);
        const op = toks[0].toUpperCase();
        const need = (n) => { if (toks.length !== n + 1) throw new Error(`${op} requires ${n} operand${n === 1 ? '' : 's'}`); };
        const target = (tok) => { const t = tok.toUpperCase(); return t in labels ? labels[t] - a : parseImm(tok); };
        let word;
        if (['ADD', 'SUB', 'MUL', 'AND', 'OR', 'XOR'].includes(op)) {
          need(3);
          const f3 = { AND: 7, OR: 6, XOR: 4 }[op] || 0;
          const f7 = op === 'SUB' ? 0x20 : op === 'MUL' ? 0x01 : 0;
          word = encR(0x33, parseReg(toks[1]), f3, parseReg(toks[2]), parseReg(toks[3]), f7);
        } else if (op === 'ADDI') { need(3); word = encI(0x13, parseReg(toks[1]), 0, parseReg(toks[2]), parseImm(toks[3])); }
        else if (op === 'LUI') { need(2); word = encU(0x37, parseReg(toks[1]), parseImm(toks[2])); }
        else if (op === 'LW' || op === 'SW') {
          need(2);
          const m = MEM_RE.exec(toks[2]);
          if (!m) throw new Error(`bad ${op} operand '${toks[2]}'`);
          word = op === 'LW'
            ? encI(0x03, parseReg(toks[1]), 2, parseReg(m[2]), parseImm(m[1]))
            : encS(0x23, 2, parseReg(m[2]), parseReg(toks[1]), parseImm(m[1]));
        } else if (op === 'BEQ' || op === 'BNE') {
          need(3); word = encB(0x63, op === 'BEQ' ? 0 : 1, parseReg(toks[1]), parseReg(toks[2]), target(toks[3]));
        } else if (op === 'JAL') { need(2); word = encJ(0x6f, parseReg(toks[1]), target(toks[2])); }
        else if (op === 'ECALL') word = 0x73;
        else if (op === 'CORAL.MAC') { need(3); word = encR(0x0b, parseReg(toks[1]), 1, parseReg(toks[2]), parseReg(toks[3]), 1); }
        else if (op === 'CORAL.ACT') { need(3); word = encI(0x0b, parseReg(toks[1]), 2, parseReg(toks[2]), parseImm(toks[3])); }
        else if (op === 'NVDLA.GEMM') { need(3); word = encR(0x2b, parseReg(toks[1]), 1, parseReg(toks[2]), parseReg(toks[3]), 2); }
        else if (op === 'SEM.ARM') { need(2); word = encI(0x5b, 0, 1, parseReg(toks[1]), parseImm(toks[2])); }
        else if (op === 'PWR.MODE') { need(1); word = encI(0x5b, 0, 2, 0, parseImm(toks[1])); }
        else if (op === 'NOC.REDUCE') { need(3); word = encR(0x5b, parseReg(toks[1]), 3, parseReg(toks[2]), parseReg(toks[3]), 4); }
        else if (op === 'HALT') word = 0x7f;
        else throw new Error(`unknown mnemonic '${op}'`);
        words.push(word >>> 0);
        listing.push({ addr: a, word: word >>> 0, text: line, line: ln });
      } catch (e) { throw new Error(`line ${ln}: ${e.message}`); }
    }
    return { words, listing, preload };
  }

  // ------------------------------------------------------------------ decoder
  function decode(word) {
    word = word >>> 0;
    const opcode = word & 0x7f, rd = (word >> 7) & 31, f3 = (word >> 12) & 7,
      rs1 = (word >> 15) & 31, rs2 = (word >> 20) & 31, f7 = (word >> 25) & 0x7f;
    const immI = sext((word >>> 20) & 0xfff, 12);
    const immS = sext((((word >>> 25) & 0x7f) << 5) | ((word >> 7) & 31), 12);
    const immB = sext((((word >>> 31) & 1) << 12) | (((word >> 7) & 1) << 11) | (((word >>> 25) & 0x3f) << 5) | (((word >> 8) & 0xf) << 1), 13);
    const immU = (word & 0xfffff000) | 0;
    const immJ = sext((((word >>> 31) & 1) << 20) | (((word >> 12) & 0xff) << 12) | (((word >> 20) & 1) << 11) | (((word >>> 21) & 0x3ff) << 1), 21);
    const bad = (k) => new Error(`undecoded ${k} word 0x${h(word, 8)}`);
    switch (opcode) {
      case 0x33: {
        const k = f3 + ':' + f7;
        const op = { '0:0': 'ADD', '0:32': 'SUB', '0:1': 'MUL', '7:0': 'AND', '6:0': 'OR', '4:0': 'XOR' }[k];
        if (!op) throw bad('R-type');
        return { op, rd, rs1, rs2 };
      }
      case 0x13: return { op: 'ADDI', rd, rs1, imm: immI };
      case 0x37: return { op: 'LUI', rd, imm: immU };
      case 0x03: return { op: 'LW', rd, rs1, imm: immI };
      case 0x23: return { op: 'SW', rs1, rs2, imm: immS };
      case 0x63: { const op = { 0: 'BEQ', 1: 'BNE' }[f3]; if (!op) throw bad('branch'); return { op, rs1, rs2, imm: immB }; }
      case 0x6f: return { op: 'JAL', rd, imm: immJ };
      case 0x73: return { op: 'ECALL' };
      case 0x0b:
        if (f3 === 1 && f7 === 1) return { op: 'CORAL.MAC', rd, rs1, rs2 };
        if (f3 === 2) return { op: 'CORAL.ACT', rd, rs1, imm: immI };
        throw bad('custom-0');
      case 0x2b:
        if (f3 === 1 && f7 === 2) return { op: 'NVDLA.GEMM', rd, rs1, rs2 };
        throw bad('custom-1');
      case 0x5b:
        if (f3 === 1) return { op: 'SEM.ARM', rs1, imm: immI };
        if (f3 === 2) return { op: 'PWR.MODE', imm: immI };
        if (f3 === 3 && f7 === 4) return { op: 'NOC.REDUCE', rd, rs1, rs2 };
        throw bad('custom-2');
      case 0x7f: return { op: 'HALT' };
      default: throw new Error(`unknown opcode 0x${h(opcode, 2)} in word 0x${h(word, 8)}`);
    }
  }

  function disasm(word) {
    let i;
    try { i = decode(word); } catch (e) { return `.word 0x${h(word, 8)}`; }
    const x = (n) => 'x' + n;
    switch (i.op) {
      case 'ADD': case 'SUB': case 'MUL': case 'AND': case 'OR': case 'XOR':
      case 'CORAL.MAC': case 'NVDLA.GEMM': case 'NOC.REDUCE':
        return `${i.op} ${x(i.rd)}, ${x(i.rs1)}, ${x(i.rs2)}`;
      case 'ADDI': case 'CORAL.ACT': return `${i.op} ${x(i.rd)}, ${x(i.rs1)}, ${i.imm}`;
      case 'LUI': return `LUI ${x(i.rd)}, 0x${h(i.imm >>> 12, 1)}`;
      case 'LW': return `LW ${x(i.rd)}, ${i.imm}(${x(i.rs1)})`;
      case 'SW': return `SW ${x(i.rs2)}, ${i.imm}(${x(i.rs1)})`;
      case 'BEQ': case 'BNE': return `${i.op} ${x(i.rs1)}, ${x(i.rs2)}, ${i.imm >= 0 ? '+' : ''}${i.imm}`;
      case 'JAL': return `JAL ${x(i.rd)}, ${i.imm >= 0 ? '+' : ''}${i.imm}`;
      case 'SEM.ARM': return `SEM.ARM ${x(i.rs1)}, ${i.imm}`;
      case 'PWR.MODE': return `PWR.MODE ${i.imm}`;
      default: return i.op;
    }
  }

  // --------------------------------------------------------------------- tile
  class TrinityTile {
    constructor() {
      this.sram = new Uint8Array(C.TILE_SRAM_SIZE);
      this.quad = new Uint8Array(C.QUAD_SRAM_SIZE);
      this.tcm = new Uint8Array(0x10000);
      this.hbm = new Uint8Array(C.HBM_SIZE);
      this.shards = [this.sram, new Uint8Array(C.TILE_SRAM_SIZE), new Uint8Array(C.TILE_SRAM_SIZE), new Uint8Array(C.TILE_SRAM_SIZE)];
      this.program = []; this.listing = []; this.preload = [];
      this.reset();
    }

    reset() {
      this.regs = new Array(32).fill(0n);
      this.pc = 0; this.cycles = 0; this.halted = false;
      this.csb = new Map();
      this.sem = new Array(64).fill(0); this.armed = new Array(64).fill(false); this.doorbells = new Array(64).fill(0);
      this.power = 3; this.coralMacs = 0; this.nvdlaMacs = 0; this.lastGemm = null;
      this.log = []; this.trace = []; this.warnings = [];
      this.runs = new Int32Array(this.program.length);
      this.hist = []; this.J = null;
      for (const b of [this.sram, this.quad, this.tcm, this.hbm, this.shards[1], this.shards[2], this.shards[3]]) b.fill(0);
      this._applyPreload();
    }

    loadProgram(src) {
      const a = assemble(src);
      this.program = a.words; this.listing = a.listing; this.preload = a.preload;
      this.reset();
    }

    _applyPreload() {
      for (const p of this.preload) {
        let buf, size;
        if (p.kind === 'sram') { buf = this.sram; size = C.TILE_SRAM_SIZE; }
        else if (p.kind === 'hbm') { buf = this.hbm; size = C.HBM_SIZE; }
        else {
          if (!(p.tile >= 0 && p.tile < 4)) throw new Error(`line ${p.line}: invalid shard tile ${p.tile}`);
          buf = this.shards[p.tile]; size = C.TILE_SRAM_SIZE;
        }
        const width = p.kind === 'shard' ? 4 : 1;
        if (p.off < 0 || p.off + p.vals.length * width > size) throw new Error(`line ${p.line}: .${p.kind} preload out of range`);
        p.vals.forEach((v, i) => {
          const o = p.off + i * width;
          for (let k = 0; k < width; k++) buf[o + k] = (v >> (8 * k)) & 255;
        });
      }
    }

    // ---- registers / memory
    r(i) { return this.regs[i]; }
    w(i, v) { if (i !== 0) this.regs[i] = BigInt.asUintN(64, v); }
    _warn(msg) { this.warnings.push(msg); this.log.push(`  [WARN] ${msg}`); }

    _mem(addr, size) {
      const { ITCM_SIZE, DTCM_BASE, DTCM_SIZE } = C;
      if ((addr >= 0 && addr < ITCM_SIZE) || (addr >= DTCM_BASE && addr < DTCM_BASE + DTCM_SIZE)) {
        const end = addr < DTCM_BASE ? ITCM_SIZE : DTCM_BASE + DTCM_SIZE;
        if (addr + size > end) throw new Error(`access straddles TCM boundary: 0x${h(addr, 8)}`);
        return [this.tcm, addr];
      }
      if (addr >= C.TILE_SRAM_BASE && addr < C.TILE_SRAM_BASE + C.TILE_SRAM_SIZE) {
        if (addr + size > C.TILE_SRAM_BASE + C.TILE_SRAM_SIZE) throw new Error(`access straddles Tile SRAM end: 0x${h(addr, 8)}`);
        return [this.sram, addr - C.TILE_SRAM_BASE];
      }
      if (addr >= C.QUAD_SRAM_BASE && addr < C.QUAD_SRAM_BASE + C.QUAD_SRAM_SIZE) {
        if (addr + size > C.QUAD_SRAM_BASE + C.QUAD_SRAM_SIZE) throw new Error(`access straddles Quad SRAM end: 0x${h(addr, 8)}`);
        return [this.quad, addr - C.QUAD_SRAM_BASE];
      }
      if (addr >= C.HBM_BASE && addr < C.HBM_BASE + C.HBM_SIZE) {
        if (addr + size > C.HBM_BASE + C.HBM_SIZE) throw new Error('access straddles HBM window end');
        return [this.hbm, addr - C.HBM_BASE];
      }
      throw new Error(`memory access outside modelled map: 0x${addr.toString(16)}`);
    }

    read32(addr) {
      if (addr >= C.NVDLA_CSB_BASE && addr < C.NVDLA_CSB_BASE + 0x100000) return this.csb.get(addr) || 0;
      if (addr >= C.SEM_BASE && addr < C.SEM_BASE + 0x10000) return this.sem[((addr - C.SEM_BASE) >> 2) % 64] >>> 0;
      const [b, o] = this._mem(addr, 4);
      return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0;
    }

    write32(addr, val) {
      val = val >>> 0;
      if (addr >= C.NVDLA_CSB_BASE && addr < C.NVDLA_CSB_BASE + 0x100000) { this.csb.set(addr, val); return; }
      if (addr >= C.SEM_BASE && addr < C.SEM_BASE + 0x10000) {
        const i = ((addr - C.SEM_BASE) >> 2) % 64;
        this.sem[i] = val; this.armed[i] = val > 0; return;
      }
      const [b, o] = this._mem(addr, 4);
      this._w32(b, o, val);
    }

    _w32(buf, off, v) { // journaled little-endian 32-bit store
      if (this.J) this.J.push({ buf, off, old: [buf[off], buf[off + 1], buf[off + 2], buf[off + 3]] });
      buf[off] = v & 255; buf[off + 1] = (v >>> 8) & 255; buf[off + 2] = (v >>> 16) & 255; buf[off + 3] = (v >>> 24) & 255;
    }
    _ri8(off) { return (this.sram[off] << 24) >> 24; }
    _ri32(off) { const b = this.sram; return b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24); }

    // ---- history
    _snap(idx) {
      return {
        idx, counted: false, regs: this.regs.slice(), pc: this.pc, cycles: this.cycles, halted: this.halted,
        sem: this.sem.slice(), armed: this.armed.slice(), doorbells: this.doorbells.slice(), power: this.power,
        cm: this.coralMacs, nm: this.nvdlaMacs, lastGemm: this.lastGemm ? this.lastGemm.slice() : null,
        logLen: this.log.length, traceLen: this.trace.length, warnLen: this.warnings.length,
        csb: new Map(this.csb), mem: []
      };
    }

    stepBack() {
      const s = this.hist.pop();
      if (!s) return false;
      for (let i = s.mem.length - 1; i >= 0; i--) {
        const e = s.mem[i];
        for (let k = 0; k < 4; k++) e.buf[e.off + k] = e.old[k];
      }
      this.regs = s.regs; this.pc = s.pc; this.cycles = s.cycles; this.halted = s.halted;
      this.sem = s.sem; this.armed = s.armed; this.doorbells = s.doorbells; this.power = s.power;
      this.coralMacs = s.cm; this.nvdlaMacs = s.nm; this.lastGemm = s.lastGemm; this.csb = s.csb;
      this.log.length = s.logLen; this.trace.length = s.traceLen; this.warnings.length = s.warnLen;
      if (s.counted) this.runs[s.idx]--;
      return true;
    }

    // ---- execution
    step() {
      if (this.halted) return false;
      if (this.pc % 4 !== 0) { this.halted = true; this.log.push(`PC 0x${h(this.pc, 4)} misaligned — halted`); return false; }
      const idx = this.pc / 4;
      if (idx < 0 || idx >= this.program.length) {
        this.halted = true; this.log.push(`PC 0x${h(this.pc, 4)} outside program — halted`); return false;
      }
      const snap = this._snap(idx);
      this.hist.push(snap);
      if (this.hist.length > 25000) this.hist.splice(0, 5000);
      this.J = snap.mem;
      const word = this.program[idx], pc0 = this.pc;
      let next = pc0 + 4, cost = 1, note = '';
      try {
        let ins;
        try { ins = decode(word); }
        catch (e) { this.halted = true; this.log.push(`PC 0x${h(pc0, 4)}: illegal instruction — ${e.message}`); return false; }
        this.runs[idx]++; snap.counted = true;
        const R = (i) => this.regs[i];
        switch (ins.op) {
          case 'ADD': this.w(ins.rd, R(ins.rs1) + R(ins.rs2)); break;
          case 'SUB': this.w(ins.rd, R(ins.rs1) - R(ins.rs2)); break;
          case 'MUL': this.w(ins.rd, R(ins.rs1) * R(ins.rs2)); cost = 3; break;
          case 'AND': this.w(ins.rd, R(ins.rs1) & R(ins.rs2)); break;
          case 'OR': this.w(ins.rd, R(ins.rs1) | R(ins.rs2)); break;
          case 'XOR': this.w(ins.rd, R(ins.rs1) ^ R(ins.rs2)); break;
          case 'ADDI': this.w(ins.rd, R(ins.rs1) + BigInt(ins.imm)); break;
          case 'LUI': this.w(ins.rd, BigInt(ins.imm)); break;
          case 'LW': {
            const a = Number(BigInt.asUintN(64, R(ins.rs1) + BigInt(ins.imm)));
            this.w(ins.rd, BigInt(this.read32(a) | 0)); cost = 2; break;
          }
          case 'SW': {
            const a = Number(BigInt.asUintN(64, R(ins.rs1) + BigInt(ins.imm)));
            this.write32(a, Number(R(ins.rs2) & 0xffffffffn)); cost = 2; break;
          }
          case 'BEQ': if (R(ins.rs1) === R(ins.rs2)) next = pc0 + ins.imm; break;
          case 'BNE': if (R(ins.rs1) !== R(ins.rs2)) next = pc0 + ins.imm; break;
          case 'JAL': this.w(ins.rd, BigInt(pc0 + 4)); next = pc0 + ins.imm; break;
          case 'ECALL': {
            const a7 = R(17);
            if (a7 === 1n) { const v = BigInt.asIntN(64, R(10)); note = `ECALL print a0=${v}`; this.log.push(`  [ECALL] a0 = ${v}`); }
            else { note = `ECALL a7=${a7} (unimplemented syscall — halt)`; this.halted = true; }
            break;
          }
          case 'CORAL.MAC': [cost, note] = this._coralMac(ins); break;
          case 'CORAL.ACT': [cost, note] = this._coralAct(ins); break;
          case 'NVDLA.GEMM': [cost, note] = this._nvdlaGemm(ins); break;
          case 'SEM.ARM': {
            const s = Number(R(ins.rs1) & 63n);
            this.sem[s] = ins.imm & 0xfff; this.armed[s] = this.sem[s] > 0; note = `sem#${s} <- ${this.sem[s]}`; break;
          }
          case 'PWR.MODE': this.power = ins.imm & 3; note = `power -> P${this.power}`; break;
          case 'NOC.REDUCE': [cost, note] = this._nocReduce(ins); break;
          case 'HALT': this.halted = true; note = 'halt'; break;
          default: throw new Error('unhandled ' + ins.op);
        }
        this.regs[0] = 0n; this.pc = next; this.cycles += cost;
        this.trace.push(`pc=0x${h(pc0, 4)} word=0x${h(word, 8)} ${this.listing[idx].text.padEnd(34)} cycles+${cost} ${note}`);
        return !this.halted;
      } catch (e) {
        this.halted = true; this.log.push(`PC 0x${h(pc0, 4)}: fault — ${e.message}`); return false;
      } finally { this.J = null; }
    }

    /** Run until halt, step limit or (just before) an instruction index in `stop`. */
    run(max = 100000, stop = null) {
      let steps = 0;
      while (!this.halted && steps < max) {
        if (!this.step()) { steps++; break; }
        steps++;
        if (stop && stop.has(this.pc / 4)) return { steps, reason: 'breakpoint' };
      }
      if (!this.halted && steps >= max) { this._warn('step limit reached — possible infinite loop'); return { steps, reason: 'limit' }; }
      return { steps, reason: 'halt' };
    }

    _semComplete(s) {
      if (this.armed[s] && this.sem[s] > 0) {
        this.sem[s]--;
        if (this.sem[s] === 0) { this.armed[s] = false; this.doorbells[s]++; }
      }
    }

    _nvdlaGemm(ins) {
      const m = Number(this.regs[ins.rs1]), n = Number(this.regs[ins.rs2]), k = Number(this.regs[15]);
      if (m === 0 || n === 0 || k === 0) {
        this._warn('G1: NVDLA.GEMM with zero dimension (M, N, or K/x15 == 0) — degenerate GEMM (0 MACs). Set K in x15.');
        this.lastGemm = [m, n, k]; this.w(ins.rd, 0n);
        return [2, 'GEMM degenerate, 0 MACs'];
      }
      const macs = m * n * k;
      const aEnd = C.GEMM_A_OFF + m * k, cEnd = C.GEMM_C_OFF + 4 * m * n;
      if (aEnd > C.TILE_SRAM_SIZE || cEnd > C.TILE_SRAM_SIZE || k * n > C.HBM_SIZE) throw new Error(`NVDLA.GEMM ${m}x${k}x${n} exceeds modelled memory`);
      if (macs > C.MAX_MACS) throw new Error(`NVDLA.GEMM ${macs} MACs exceeds the web emulator budget (${C.MAX_MACS})`);
      if (cEnd > C.GEMM_A_OFF) this._warn(`G1: C (${cEnd - C.GEMM_C_OFF} B) overlaps A at 0x${h(C.GEMM_A_OFF, 4)}; A is overwritten during the GEMM`);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let kk = 0; kk < k; kk++) acc += this._ri8(C.GEMM_A_OFF + i * k + kk) * ((this.hbm[C.GEMM_B_OFF + kk * n + j] << 24) >> 24);
        this._w32(this.sram, C.GEMM_C_OFF + 4 * (i * n + j), acc | 0);
      }
      this.nvdlaMacs += macs; this.lastGemm = [m, n, k]; this.w(ins.rd, BigInt(macs));
      this._semComplete(0);
      return [Math.max(2, Math.floor((macs + 511) / 512)), `GEMM ${m}x${k} * ${k}x${n} = ${macs} MACs -> SRAM[0x${h(C.GEMM_C_OFF, 4)}]`];
    }

    _coralAct(ins) {
      const addr = Number(this.regs[ins.rs1]), kind = ins.imm;
      const count = this.lastGemm ? this.lastGemm[0] * this.lastGemm[1] : 1;
      const off = addr - C.TILE_SRAM_BASE;
      if (count <= 0 || off < 0 || off + 4 * count > C.TILE_SRAM_SIZE || off % 4 !== 0) {
        this._warn(`G2: CORAL.ACT address 0x${addr.toString(16)} is invalid/unaligned for ${count} elements — no-op. x${ins.rs1} must hold a Tile SRAM address.`);
        this.w(ins.rd, 0n); return [2, 'ACT no-op (bad address)'];
      }
      const vals = []; for (let i = 0; i < count; i++) vals.push(this._ri32(off + 4 * i));
      if (this.armed[0] && this.sem[0] > 0) this._warn('sem#0 still counting: CORAL.ACT issued before producer done');
      else if (this.doorbells[0] > 0) this.doorbells[0]--;
      const names = ['ReLU', 'SwiGLU', 'Softmax', 'RMSNorm'];
      let out;
      if (kind === 0) out = vals.map((v) => Math.max(0, v));
      else if (kind === 1) out = vals.map((v) => pyRound(v * (1 / (1 + Math.exp(-Math.max(-10, Math.min(10, v / 1000)))))));
      else if (kind === 2) {
        const mx = vals.length ? Math.max(...vals) : 0;
        const ex = vals.map((v) => Math.exp(Math.max(-20, Math.min(0, (v - mx) / 1000))));
        const s = ex.reduce((a, b) => a + b, 0) || 1;
        out = ex.map((e) => pyRound(e / s * 1000));
      } else if (kind === 3) {
        const rms = Math.sqrt(vals.reduce((a, v) => a + v * v, 0) / Math.max(1, vals.length)) || 1;
        out = vals.map((v) => pyRound(v / rms * 1000));
      } else out = vals;
      out.forEach((v, i) => this._w32(this.sram, off + 4 * i, v | 0));
      this.coralMacs += 128; this.w(ins.rd, BigInt(count));
      return [2, `ACT ${names[kind] || 'ACT#' + kind} on ${count} elems @ 0x${h(addr, 8)}`];
    }

    _coralMac(ins) {
      const aOff = Number(this.regs[ins.rs1]) - C.TILE_SRAM_BASE, bOff = Number(this.regs[ins.rs2]) - C.TILE_SRAM_BASE;
      if (!(aOff >= 0 && aOff + 0x1000 + 1024 <= C.TILE_SRAM_SIZE && bOff >= 0 && bOff + 256 <= C.TILE_SRAM_SIZE)) {
        this._warn('G3: CORAL.MAC address outside Tile SRAM — no-op'); this.w(ins.rd, 0n); return [1, 'MAC no-op (bad address)'];
      }
      for (let i = 0; i < 16; i++) for (let j = 0; j < 16; j++) {
        let acc = 0;
        for (let k = 0; k < 16; k++) acc += this._ri8(aOff + i * 16 + k) * this._ri8(bOff + k * 16 + j);
        this._w32(this.sram, aOff + 0x1000 + 4 * (i * 16 + j), acc | 0);
      }
      this.coralMacs += 256; this.w(ins.rd, 256n);
      return [1, 'MAC 16x16 tile -> SRAM[rs1+0x1000], 256 MACs'];
    }

    _nocReduce(ins) {
      const off = Number(this.regs[ins.rs1]) - C.TILE_SRAM_BASE;
      let shards = Number(this.regs[ins.rs2]) || 4;
      if (shards > 4) { this._warn(`NOC.REDUCE shards=${shards} > 4 modelled; clamped`); shards = 4; }
      if (off < 0 || off > C.TILE_SRAM_SIZE - 4 || off % 4 !== 0) {
        this._warn('NOC.REDUCE address outside/unaligned in Tile SRAM — no-op'); this.w(ins.rd, 0n); return [2, 'REDUCE no-op (bad address)'];
      }
      let total = 0;
      for (let t = 0; t < shards; t++) { const b = this.shards[t]; total += b[off] | (b[off + 1] << 8) | (b[off + 2] << 16) | (b[off + 3] << 24); }
      this.w(ins.rd, BigInt(total));
      return [8, `AllReduce sum over ${shards} tiles = ${total}`];
    }
  }

  // ----------------------------------------------------------------- examples
  // one directive per matrix row (directives are single-line)
  const rowsDir = (kind, base, rows) => rows.map((r, i) => `.${kind} 0x${(base + i * r.length).toString(16)} ${r.join(', ')}`).join('\n');
  const GEMM_A = [[1, 2, 0, 1, 3, 0, 2, 1], [0, 1, 2, 0, 1, 1, 0, 2], [2, 0, 1, 1, 0, 2, 1, 0], [1, 1, 1, 0, 2, 0, 1, 1]];
  const GEMM_B = [[1, 0, 2, 1], [0, 1, 1, 0], [2, 1, 0, 1], [1, 0, 1, 2], [0, 2, 1, 0], [1, 1, 0, 1], [0, 0, 2, 1], [2, 1, 0, 0]];
  const gemmProg = (act, title) => `; ${title}
; ABI v1: A (INT8 4x8) @ Tile SRAM 0x1000, B (INT8 8x4) @ HBM 0, C (INT32 4x4) @ Tile SRAM 0
${rowsDir('sram', 0x1000, GEMM_A)}
${rowsDir('hbm', 0, GEMM_B)}

PWR.MODE 0
LUI  x10, 0x200          ; x10 = Tile SRAM base 0x00200000
ADDI x11, x0, 4          ; M = 4
ADDI x12, x0, 4          ; N = 4
ADDI x15, x0, 8          ; K = 8  (manual has no K; ABI v1 reads x15)
SEM.ARM x0, 1            ; arm semaphore 0
NVDLA.GEMM x13, x11, x12 ; x13 <- MACs; C -> SRAM[0]
CORAL.ACT  x14, x10, ${act}   ; activation in place on C; x14 <- element count
SW   x14, 0x40(x10)      ; store element count at SRAM[0x40]
HALT
`;
  const EXAMPLES = {
    gemm: { title: 'GEMM 4x8·8x4 + SwiGLU', src: gemmProg(1, 'Real INT8 GEMM on the NVDLA model, then SwiGLU on the Coral model') },
    softmax: { title: 'GEMM + Softmax', src: gemmProg(2, 'Same GEMM, followed by Softmax (INT32 milli-units)') },
    reduce: {
      title: 'Loop + CORAL.MAC + NOC.REDUCE',
      src: `; Sum 1..10, a 16x16 INT8 tile multiply, then an AllReduce over 4 tiles
.sram 0x2000 ${Array.from({ length: 256 }, (_, i) => (Math.floor(i / 16) === i % 16 ? 1 : 0)).join(',')}
.sram 0x2100 ${Array(256).fill(2).join(',')}
.shard 1 0x3000 10
.shard 2 0x3000 20
.shard 3 0x3000 30

ADDI x5, x0, 0           ; sum = 0
ADDI x6, x0, 1           ; i = 1
ADDI x7, x0, 11          ; limit
loop:
ADD  x5, x5, x6
ADDI x6, x6, 1
BNE  x6, x7, loop
LUI  x11, 0x202          ; A tile @ SRAM+0x2000 (identity)
LUI  x12, 0x202
ADDI x12, x12, 0x100     ; B tile @ SRAM+0x2100 (all 2s)
CORAL.MAC x13, x11, x12  ; C = A*B -> SRAM+0x3000, C[0][0] = 2
LUI  x16, 0x203          ; word @ SRAM+0x3000 on every tile
NOC.REDUCE x17, x16, x0  ; 2 + 10 + 20 + 30 = 62
HALT
`
    },
    hello: {
      title: 'ECALL print loop',
      src: `; ECALL with a7 = 1 prints a0 to the console
ADDI a7, x0, 1
ADDI a0, x0, 0
ADDI t0, x0, 1
ADDI t1, x0, 6
loop:
ADD  a0, a0, t0          ; running triangular sum
ECALL                    ; prints 1, 3, 6, 10, 15
ADDI t0, t0, 1
BNE  t0, t1, loop
HALT
`
    },
    manual: {
      title: "Manual Ch.6.1 sample (verbatim — doesn't compute)",
      src: `; The spec's own example. It assembles and halts, but no K is given (gap G1)
; and x13 is not a valid ACT address (gap G2), so it warns and does nothing.
PWR.MODE 0
LUI  x10, 0x200
ADDI x11, x0, 64
ADDI x12, x0, 128
SEM.ARM x0, 1
NVDLA.GEMM x13, x11, x12
CORAL.ACT  x14, x13, 1
SW   x14, 0(x10)
HALT
`
    }
  };

  return { TrinityTile, assemble, decode, disasm, pyRound, REGS, REG_NAMES, C, EXAMPLES, GEMM_A, GEMM_B };
});
