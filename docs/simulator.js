(function () {
  'use strict';
  const T = window.Trinity, $ = (id) => document.getElementById(id);
  const ed = $('ed'), gut = $('gut');
  const cpu = new T.TrinityTile();
  const bps = new Set();          // breakpoint source lines
  let dirty = true, uiMsgs = [], prevRegs = null;
  const hx = (n, w) => n.toString(16).padStart(w, '0');
  const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const store = (k, v) => { try { v == null ? localStorage.getItem(k) : localStorage.setItem(k, v); } catch (e) {} };
  const load = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  // ---------------------------------------------------------------- tabs
  function selectTab(group, name) {
    document.querySelectorAll(`.tabs[data-group="${group}"] button`).forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    document.querySelectorAll(`[data-pane^="${group}:"]`).forEach((p) => p.classList.toggle('on', p.dataset.pane === `${group}:${name}`));
    if (group === 'R') render();
  }
  document.querySelectorAll('.tabs').forEach((t) => t.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) selectTab(t.dataset.group, b.dataset.tab);
  }));
  document.querySelector('[data-open="ref"]').addEventListener('click', () => selectTab('L', 'ref'));

  // ---------------------------------------------------------------- actions
  const msg = (t) => { uiMsgs.push(t); };
  function assembleNow() {
    try {
      cpu.loadProgram(ed.value); dirty = false; prevRegs = null; uiMsgs = [];
      msg(`[OK] assembled ${cpu.program.length} instruction${cpu.program.length === 1 ? '' : 's'}` + (cpu.preload.length ? `, ${cpu.preload.length} preload block${cpu.preload.length === 1 ? '' : 's'}` : ''));
      render(); return true;
    } catch (e) { msg('[ERR] ' + e.message); dirty = true; render(); selectTab('B', 'con'); return false; }
  }
  const ready = () => (dirty || !cpu.program.length ? assembleNow() : true);
  const stopSet = () => new Set(cpu.listing.filter((l) => bps.has(l.line)).map((l) => l.addr / 4));

  function doRun() {
    if (!ready()) return;
    if (cpu.halted) { msg('[INFO] program halted — press Reset to run again'); return render(); }
    const r = cpu.run(100000, stopSet());
    if (r.reason === 'breakpoint') msg(`[BREAK] at 0x${hx(cpu.pc, 4)} after ${r.steps} steps`);
    else if (r.reason === 'halt') msg(`[HALT] ${r.steps} steps, ${cpu.cycles} cycles`);
    render();
  }
  function doStep() {
    if (!ready()) return;
    if (cpu.halted) { msg('[INFO] program halted — press Back to rewind or Reset'); return render(); }
    cpu.step();
    if (cpu.halted && !cpu.log.some((l) => /fault|illegal|outside|misaligned/.test(l))) msg(`[HALT] ${cpu.cycles} cycles`);
    render();
  }
  function doBack() {
    if (!ready()) return;
    uiMsgs = uiMsgs.filter((m) => !/^\[(HALT|BREAK|INFO)\]/.test(m));
    if (!cpu.stepBack()) msg('[INFO] nothing to undo');
    render();
  }
  function doReset() {
    if (dirty) return assembleNow();
    try { cpu.reset(); prevRegs = null; uiMsgs = []; msg('[OK] reset'); } catch (e) { msg('[ERR] ' + e.message); }
    render();
  }
  $('btnAsm').onclick = assembleNow; $('btnRun').onclick = doRun; $('btnStep').onclick = doStep;
  $('btnBack').onclick = doBack; $('btnReset').onclick = doReset;

  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); doRun(); }
    else if (e.key === 'F10') { e.preventDefault(); e.shiftKey ? doBack() : doStep(); }
  });
  ed.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') { e.preventDefault(); const s = ed.selectionStart; ed.setRangeText('  ', s, ed.selectionEnd, 'end'); onEdit(); }
  });

  // ---------------------------------------------------------------- editor
  function onEdit() { dirty = true; store('trinity_src', ed.value); drawGutter(); }
  ed.addEventListener('input', onEdit);
  ed.addEventListener('scroll', () => { gut.scrollTop = ed.scrollTop; });
  gut.addEventListener('click', (e) => {
    const d = e.target.closest('div[data-n]'); if (!d) return;
    const n = +d.dataset.n; bps.has(n) ? bps.delete(n) : bps.add(n); render();
  });
  function curLine() { const l = cpu.listing[cpu.pc / 4]; return l && !dirty ? l.line : -1; }
  function drawGutter() {
    const n = ed.value.split('\n').length, cl = curLine();
    let s = '';
    for (let i = 1; i <= n; i++) s += `<div data-n="${i}" class="${bps.has(i) ? 'bp ' : ''}${i === cl ? 'cur' : ''}">${i}</div>`;
    gut.innerHTML = s; gut.scrollTop = ed.scrollTop;
  }

  // ---------------------------------------------------------------- examples / files
  const sel = $('exSel');
  sel.innerHTML = '<option value="">Start from an example…</option>' + Object.entries(T.EXAMPLES).map(([k, v]) => `<option value="${k}">${esc(v.title)}</option>`).join('');
  sel.onchange = () => {
    if (!sel.value) return;
    ed.value = T.EXAMPLES[sel.value].src; bps.clear(); store('trinity_src', ed.value); sel.value = ''; assembleNow(); drawGutter();
  };
  function download(name, data, type) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([data], { type: type || 'text/plain' }));
    a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  $('btnSave').onclick = () => download('program.asm', ed.value);
  $('btnOpen').onclick = () => $('file').click();
  $('file').onchange = (e) => {
    const f = e.target.files[0]; if (!f) return;
    const r = new FileReader(); r.onload = () => { ed.value = r.result; bps.clear(); onEdit(); assembleNow(); }; r.readAsText(f); e.target.value = '';
  };
  const hexText = () => cpu.program.map((w) => hx(w, 8)).join('\n') + '\n';
  $('btnHex').onclick = () => { if (ready()) download('program.hex', hexText()); };
  $('btnBin').onclick = () => {
    if (!ready()) return;
    const b = new Uint8Array(cpu.program.length * 4);
    cpu.program.forEach((w, i) => { b[4 * i] = w & 255; b[4 * i + 1] = (w >>> 8) & 255; b[4 * i + 2] = (w >>> 16) & 255; b[4 * i + 3] = (w >>> 24) & 255; });
    download('program.bin', b, 'application/octet-stream');
  };
  $('btnCopyHex').onclick = async () => {
    if (!ready()) return;
    const t = hexText(), btn = $('btnCopyHex');
    try { await navigator.clipboard.writeText(t); }
    catch (e) { const ta = document.createElement('textarea'); ta.value = t; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); } catch (_) {} ta.remove(); }
    btn.textContent = 'Copied ✓'; setTimeout(() => (btn.textContent = 'Copy as hex'), 1200);
  };

  // ---------------------------------------------------------------- renderers
  function renderRegs() {
    let s = '';
    for (let i = 0; i < 32; i++) {
      const v = cpu.regs[i], chg = prevRegs && prevRegs[i] !== v;
      s += `<div class="reg${chg ? ' chg' : ''}"><b>x${i}</b><i>${T.REG_NAMES[i]}</i><code>0x${hx(v, 16)}<em>${BigInt.asIntN(64, v)}</em></code></div>`;
    }
    $('regs').innerHTML = s; prevRegs = cpu.regs.slice();
  }
  const MEM = { sram: [() => cpu.sram, 0x200000], hbm: [() => cpu.hbm, 0x800000000000], quad: [() => cpu.quad, 0x1000000], tcm: [() => cpu.tcm, 0], s1: [() => cpu.shards[1], 0x200000], s2: [() => cpu.shards[2], 0x200000], s3: [() => cpu.shards[3], 0x200000] };
  function renderMem() {
    const [get, base] = MEM[$('memReg').value], buf = get();
    let off = parseInt($('memOff').value || '0', 16); if (!(off >= 0)) off = 0;
    off = Math.min(off & ~15, buf.length - 256); if (off < 0) off = 0;
    const last = cpu.hist[cpu.hist.length - 1], chg = new Set();
    if (last) for (const e of last.mem) if (e.buf === buf) for (let k = 0; k < 4; k++) chg.add(e.off + k);
    const word = $('memMode').value === 'w';
    let s = '';
    for (let r = 0; r < 16; r++) {
      const o = off + r * 16, a = base + o;
      s += `<div><span class="a">${a > 0xffffffff ? hx(a, 12) : hx(a, 8)}</span>  `;
      if (word) {
        for (let c = 0; c < 4; c++) {
          const p = o + c * 4, v = buf[p] | (buf[p + 1] << 8) | (buf[p + 2] << 16) | (buf[p + 3] << 24);
          const cl = chg.has(p) || chg.has(p + 1) || chg.has(p + 2) || chg.has(p + 3) ? 'chg' : v === 0 ? 'z' : '';
          s += `<span class="${cl}">${String(v).padStart(11)}</span> `;
        }
      } else {
        let asc = '';
        for (let c = 0; c < 16; c++) {
          const v = buf[o + c], cl = chg.has(o + c) ? 'chg' : v === 0 ? 'z' : '';
          s += `<span class="${cl}">${hx(v, 2)}</span> `; asc += v >= 32 && v < 127 ? esc(String.fromCharCode(v)) : '.';
        }
        s += ` <span class="asc">${asc}</span>`;
      }
      s += '</div>';
    }
    $('mem').innerHTML = s;
  }
  ['memReg', 'memMode'].forEach((id) => ($(id).onchange = renderMem)); $('memOff').oninput = renderMem;
  const stepMem = (d) => { $('memOff').value = hx(Math.max(0, (parseInt($('memOff').value || '0', 16) || 0) + d), 1); renderMem(); };
  $('memPrev').onclick = () => stepMem(-256); $('memNext').onclick = () => stepMem(256);

  function renderCode() {
    let s = '';
    cpu.listing.forEach((l, i) => {
      const cur = !dirty && cpu.pc === l.addr, bp = bps.has(l.line);
      s += `<tr class="row${cur ? ' cur' : ''}${bp ? ' bp' : ''}" data-line="${l.line}"><td>${bp ? '● ' : ''}0x${hx(l.addr, 4)}</td><td>${hx(l.word, 8)}</td><td>${esc(T.disasm(l.word))}</td><td>${l.line}</td><td>${cpu.runs[i] || 0}</td></tr>`;
    });
    $('codeBody').innerHTML = s || '<tr><td colspan="5">Assemble a program to see machine code.</td></tr>';
  }
  $('codeBody').addEventListener('click', (e) => {
    const r = e.target.closest('tr[data-line]'); if (!r) return;
    const n = +r.dataset.line; bps.has(n) ? bps.delete(n) : bps.add(n); render();
  });

  const cls = (l) => /\[ERR\]|fault|illegal|misaligned|outside program/.test(l) ? 'err' : /\[WARN\]/.test(l) ? 'warn' : /\[ECALL\]/.test(l) ? 'out' : /\[(OK|HALT)\]/.test(l) ? 'ok' : 'inf';
  function renderCon() {
    const lines = uiMsgs.concat(cpu.log).slice(-400);
    const c = $('con'); c.innerHTML = lines.map((l) => `<div class="${cls(l)}">${esc(l)}</div>`).join('') || '<div class="inf">Console output appears here.</div>';
    c.scrollTop = c.scrollHeight;
    const t = $('trace'); t.textContent = cpu.trace.slice(-300).join('\n') || 'Trace of executed instructions appears here.'; t.scrollTop = t.scrollHeight;
  }
  function renderSem() {
    let s = '<div class="a">#   count  armed  doorbells</div>';
    for (let i = 0; i < 8; i++) s += `<div>${String(i).padEnd(3)} ${String(cpu.sem[i]).padEnd(6)} ${String(cpu.armed[i]).padEnd(6)} ${cpu.doorbells[i]}</div>`;
    $('semView').innerHTML = s + '<div class="asc">semaphores 8–63 not shown unless used</div>';
  }
  function renderStatus() {
    const st = $('sState'); st.textContent = dirty ? 'EDITED' : cpu.halted ? 'HALTED' : 'READY'; st.className = 'st' + (cpu.halted ? ' h' : '');
    $('sPc').textContent = '0x' + hx(cpu.pc, 4); $('sCyc').textContent = cpu.cycles; $('sPwr').textContent = 'P' + cpu.power;
    $('sSem').textContent = cpu.sem[0]; $('sNv').textContent = cpu.nvdlaMacs; $('sCo').textContent = cpu.coralMacs; $('sHist').textContent = cpu.hist.length;
  }
  function render() { renderStatus(); renderRegs(); renderMem(); renderSem(); renderCode(); renderCon(); drawGutter(); }

  // ---------------------------------------------------------------- boot
  ed.value = load('trinity_src') || T.EXAMPLES.gemm.src;
  assembleNow();
})();
