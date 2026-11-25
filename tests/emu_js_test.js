#!/usr/bin/env node
// Checks the JS port against the Python emulator's known results. Run: node tests/emu_js_test.js
const assert = require('assert');
const path = require('path');
const T = require(path.join(__dirname, '..', 'docs', 'trinity_emu.js'));
const { TrinityTile, assemble, decode, disasm, EXAMPLES, pyRound, GEMM_A, GEMM_B } = T;
const ok = (m) => console.log('  PASS: ' + m);
const load = (src) => { const c = new TrinityTile(); c.loadProgram(src); return c; };

// 1. Manual Appendix A encodings + decode round trip
for (const [src, want] of [['ADDI x11, x0, 64', 0x04000593], ['LUI x10, 0x200', 0x00200537], ['NVDLA.GEMM x13, x11, x12', 0x04C596AB],
  ['CORAL.ACT x14, x13, 1', 0x0016A70B], ['NOC.REDUCE x15, x14, x11', 0x08B737DB], ['HALT', 0x7F]]) {
  const { words } = assemble(src);
  assert.strictEqual(words[0], want >>> 0, src);
  assert.strictEqual(decode(words[0]).op, src.split(' ')[0]);
}
ok('Appendix A encodings + decode');

// 2. Immediate range is an error, not a silent mask
assert.throws(() => assemble('ADDI x11, x10, 0x2000'), /out of range/);
ok('out-of-range immediate rejected');

// 3. LUI sign extension, x0 hardwired
let c = load('LUI x5, 0x80000\nADDI x0, x0, 99\nHALT'); c.run();
assert.strictEqual(BigInt.asIntN(64, c.regs[5]), -(1n << 31n)); assert.strictEqual(c.regs[0], 0n);
ok('LUI sign extension, x0 hardwired');

// 4. GEMM + SwiGLU equals the Python emulator's output and an independent reference
c = load(EXAMPLES.gemm.src); c.run();
const got = Array.from({ length: 16 }, (_, i) => c._ri32(4 * i));
const PY = [2, 5, 6, 3, 5, 4, 1, 2, 4, 2, 4, 4, 3, 4, 4, 2]; // output of trinity_emu.py demo 2
assert.deepStrictEqual(got, PY);
const ref = GEMM_A.flatMap((row) => GEMM_B[0].map((_, j) => row.reduce((s, a, k) => s + a * GEMM_B[k][j], 0)));
assert.deepStrictEqual(ref.map((v) => pyRound(v / (1 + Math.exp(-v / 1000)))), PY);
assert.strictEqual(c.regs[13], 128n); assert.strictEqual(c.regs[14], 16n); assert.strictEqual(c._ri32(0x40), 16);
assert.strictEqual(c.cycles, 13); assert.strictEqual(c.sem[0], 0);
ok('GEMM + SwiGLU matches Python (cycles=13, 128 MACs)');

// 5. Loop + CORAL.MAC + NOC.REDUCE
c = load(EXAMPLES.reduce.src); c.run();
assert.strictEqual(c.regs[5], 55n); assert.strictEqual(c.regs[13], 256n); assert.strictEqual(c._ri32(0x3000), 2); assert.strictEqual(c.regs[17], 62n);
ok('loop=55, MAC=256, AllReduce=62');

// 6. ECALL print
c = load(EXAMPLES.hello.src); c.run();
assert.deepStrictEqual(c.log.filter((l) => l.includes('[ECALL]')).map((l) => l.split('= ')[1]), ['1', '3', '6', '10', '15']);
ok('ECALL prints 1,3,6,10,15');

// 7. Manual sample warns exactly as the Python emulator does and halts
c = load(EXAMPLES.manual.src); c.run();
assert.ok(c.halted && c.regs[13] === 0n && c.warnings.length === 2 && c.warnings[0].startsWith('G1') && c.warnings[1].startsWith('G2'));
ok('manual sample: G1 + G2 warnings, halts');

// 8. Step back restores registers, memory, cycles, run counters and logs exactly
for (const key of ['gemm', 'reduce', 'hello']) {
  c = load(EXAMPLES[key].src);
  const sram0 = Buffer.from(c.sram), hbm0 = Buffer.from(c.hbm), sh0 = Buffer.from(c.shards[1]);
  const n = c.run().steps;
  let back = 0; while (c.stepBack()) back++;
  assert.strictEqual(back, n, key);
  assert.ok(c.regs.every((r) => r === 0n) && c.pc === 0 && c.cycles === 0 && !c.halted && c.log.length === 0 && c.trace.length === 0);
  assert.ok(Buffer.from(c.sram).equals(sram0) && Buffer.from(c.hbm).equals(hbm0) && Buffer.from(c.shards[1]).equals(sh0));
  assert.ok(c.runs.every((v) => v === 0));
}
ok('step back restores full state for gemm / reduce / hello');

// 9. Step back one instruction from the middle, then re-run to the same final state
c = load(EXAMPLES.gemm.src); c.run(); const fin = JSON.stringify([c.regs.map(String), c.cycles, Buffer.from(c.sram).toString('base64')]);
c.stepBack(); c.stepBack(); c.run();
assert.strictEqual(JSON.stringify([c.regs.map(String), c.cycles, Buffer.from(c.sram).toString('base64')]), fin);
ok('back 2 then forward reproduces final state');

// 10. Breakpoints
c = load(EXAMPLES.reduce.src);
const r = c.run(100000, new Set([6])); // instruction 6 = first one after the loop
assert.strictEqual(r.reason, 'breakpoint'); assert.strictEqual(c.pc, 24); assert.strictEqual(c.regs[5], 55n);
assert.strictEqual(c.run().reason, 'halt');
ok('breakpoint stops before the instruction, run resumes');

// 11. Bounds / faults halt cleanly, never throw out of step()
c = load('LUI x11, 0x3FF\nCORAL.MAC x13, x11, x11\nHALT'); c.run(); assert.ok(c.regs[13] === 0n && c.warnings.length === 1);
c = load('LUI x5, 0x40000\nLW x6, 0(x5)\nHALT'); c.run(); assert.ok(c.halted && c.log.some((l) => l.includes('fault')));
c = load('ADDI x1, x0, 1\nBEQ x0, x0, -4\nHALT'); const lim = c.run(50); assert.strictEqual(lim.reason, 'limit');
ok('MAC bounds, memory fault and step limit handled');

// 12. TCM split + semaphore doorbell lifecycle + NOC shard count
c = new TrinityTile(); c.write32(0x8000, 7); assert.strictEqual(c.read32(0x8000), 7); assert.throws(() => c.read32(0x4000));
c.armed[0] = true; c.sem[0] = 2; c._semComplete(0); assert.ok(c.sem[0] === 1 && c.doorbells[0] === 0); c._semComplete(0); assert.ok(c.sem[0] === 0 && c.doorbells[0] === 1);
c = load('LUI x10, 0x200\nADDI x10, x10, 0x100\nADDI x11, x0, 2\nNOC.REDUCE x12, x10, x11\nHALT\n.shard 1 0x100 10\n.shard 2 0x100 20'); c.write32(0x200100, 1); c.run();
assert.strictEqual(c.regs[12], 11n);
ok('TCM split, doorbell, NOC shard count');

// 13. Disassembler round trip over every example
for (const e of Object.values(EXAMPLES)) {
  const a = assemble(e.src);
  const re = assemble(a.words.map((w) => disasm(w)).join('\n'));
  assert.deepStrictEqual(re.words, a.words, e.title);
}
ok('disassembly re-assembles to identical words');
console.log('\nALL JS EMULATOR CHECKS PASSED.');
