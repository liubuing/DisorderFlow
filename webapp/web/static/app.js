/* DisorderFlow — 前端逻辑（无框架） */

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  const WS = {
    design: {
      label: "设计",
      modes: [["single", "单结构"], ["batch", "批量"], ["target", "靶点约束"], ["disease", "疾病靶点"]],
    },
    validate: {
      label: "验证",
      modes: [["af2", "结构预测"], ["confidence", "置信度"], ["iterative", "迭代精修"], ["loop", "闭环优化"]],
    },
    targets: { label: "靶点", modes: [] },
    runs: { label: "记录", modes: [] },
  };

  const WS_LABEL = { design: "设计", validate: "验证", targets: "靶点", runs: "记录" };

  const OP = {
    design: { single: "design_single", batch: "design_batch", target: "design_target", disease: "design_disease" },
    validate: { af2: "validate_af2", confidence: "validate_confidence", iterative: "validate_iterative", loop: "validate_loop" },
  };

  const state = {
    ws: "design",
    mode: { design: "single", validate: "af2" },
    boot: null,
    disease: null,          // 已加载的疾病靶点 {upload_id, chain, is_idp}
    job: null,
    es: null,
    result: null,
    runs: [],
    selectedRun: null,
  };

  /* ── 基础工具 ─────────────────────────────────────────────────────────── */

  async function api(path, options = {}) {
    const headers = { "X-DF": "1", ...(options.headers || {}) };
    if (options.json !== undefined) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
    }
    delete options.json;
    const res = await fetch(path, { ...options, headers });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
      throw new Error(detail);
    }
    return res.headers.get("content-type")?.includes("json") ? res.json() : res;
  }

  let toastTimer = null;
  function toast(message, kind = "err") {
    const el = $("#toast");
    el.textContent = message;
    el.className = "toast" + (kind === "ok" ? " ok" : "");
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
  }

  const pad = (n) => String(n).padStart(2, "0");
  function clock(seconds) {
    const s = Math.max(0, Math.round(seconds || 0));
    return `${pad(Math.floor(s / 60))}:${pad(s % 60)}`;
  }
  function stamp(iso) {
    if (!iso) return "—";
    return iso.replace("T", " ").slice(5, 16);
  }

  /* ── 导航 ─────────────────────────────────────────────────────────────── */

  function setWorkspace(name, push = true) {
    state.ws = name;
    $$(".ws-nav button").forEach((b) => b.classList.toggle("on", b.dataset.ws === name));
    $$(".ws").forEach((s) => { s.hidden = s.dataset.ws !== name; });
    renderRail();
    updateSummary();
    if (push) history.replaceState(null, "", `#${name}/${state.mode[name] || ""}`);
    const panel = activePanel();
    if (panel) showPanel(panel);
  }

  function setMode(name, push = true) {
    const ws = state.ws;
    if (!WS[ws].modes.length) return;
    state.mode[ws] = name;
    $$("[data-panel]", $(`#${ws}-inputs`)).forEach((p) => { p.hidden = p.dataset.panel !== name; });
    renderRail();
    updateSummary();
    if (push) history.replaceState(null, "", `#${ws}/${name}`);
  }

  function activePanel() {
    const inputs = $(`#${state.ws}-inputs`);
    if (!inputs) return null;
    const modes = WS[state.ws].modes;
    if (!modes.length) return null;
    const current = state.mode[state.ws];
    return $(`[data-panel="${current}"]`, inputs);
  }

  function showPanel(panel) {
    if (panel) syncToolVisibility(panel);
  }

  function currentRegion() {
    const panel = activePanel();
    if (!panel) return "";
    const field = $("[data-field='epitope']", panel) || $("[data-field='region']", panel);
    return field ? field.value : "";
  }

  function renderRail() {
    const wrap = $("#rail-modes");
    wrap.innerHTML = "";
    const modes = WS[state.ws].modes;
    if (!modes.length) {
      wrap.hidden = true;
    } else {
      wrap.hidden = false;
      wrap.appendChild(label("模式"));
      modes.forEach(([key, text], index) => {
        const b = document.createElement("button");
        b.className = "rail-item" + (state.mode[state.ws] === key ? " on" : "");
        b.innerHTML = `<span>${text}</span><span class="k">${index + 1}</span>`;
        b.onclick = () => setMode(key);
        wrap.appendChild(b);
      });
    }
    renderRailRuns();
  }

  function label(text) {
    const d = document.createElement("div");
    d.className = "rail-label";
    d.textContent = text;
    return d;
  }

  function renderRailRuns() {
    const box = $("#rail-runs");
    box.innerHTML = "";
    if (!state.runs.length) {
      box.innerHTML = '<div class="rail-empty">暂无记录</div>';
      return;
    }
    state.runs.slice(0, 8).forEach((run) => {
      const row = document.createElement("div");
      row.className = "rail-run";
      const kind = run.status === "completed" ? "ok" : run.status === "failed" ? "err" : "warn";
      row.innerHTML = `<i class="st ${kind}"></i><span>${stamp(run.started_at)}</span>
                       <span style="margin-left:auto">${run.mode}</span>`;
      row.title = `${run.run_id} · ${run.status}`;
      row.onclick = () => { setWorkspace("runs"); selectRun(run.run_id); };
      box.appendChild(row);
    });
  }

  /* ── 参数收集 ─────────────────────────────────────────────────────────── */

  function collect(panel) {
    const params = {};
    $$("[data-field]", panel).forEach((el) => {
      let value = el.value;
      if (el.type === "range") value = Number(value);
      params[el.dataset.field] = value;
    });
    $$("[data-seg]", panel).forEach((seg) => {
      const on = $(".on", seg);
      params[seg.dataset.seg] = on ? on.dataset.value : "";
    });
    $$("[data-switch]", panel).forEach((sw) => {
      params[sw.dataset.switch] = sw.classList.contains("on");
    });
    $$("[data-drop]", panel).forEach((drop) => {
      const key = drop.dataset.drop;
      if (key === "files") {
        params.file_ids = JSON.parse(drop.dataset.ids || "[]");
      } else if (key === "pdb") {
        params.pdb_id = drop.dataset.uploadId || "";
      } else if (key === "target") {
        params.target_id = state.disease && drop.dataset.fromDisease
          ? state.disease.upload_id
          : drop.dataset.uploadId || "";
      } else if (key === "antibody") {
        params.antibody_id = drop.dataset.uploadId || "";
      }
    });
    if (state.ws === "design" && state.mode.design === "disease" && state.disease) {
      params.target_id = state.disease.upload_id;
      params.chain = state.disease.chain;
    }
    return params;
  }

  function syncToolVisibility(panel) {
    const seg = $("[data-seg='tool']", panel);
    if (!seg) return;
    const tool = $(".on", seg)?.dataset.value || "";
    $$("[data-for]", panel).forEach((row) => {
      const allowed = row.dataset.for.split("|");
      row.hidden = !allowed.includes(tool);
    });
  }

  /* ── 上传 ─────────────────────────────────────────────────────────────── */

  async function uploadFiles(drop, files) {
    const list = [...files];
    if (!list.length) return;
    const multiple = drop.dataset.multiple === "1";
    const ids = multiple ? JSON.parse(drop.dataset.ids || "[]") : [];
    const names = multiple ? JSON.parse(drop.dataset.names || "[]") : [];

    drop.classList.add("over");
    try {
      for (const file of multiple ? list : list.slice(0, 1)) {
        const res = await api(`/api/upload?name=${encodeURIComponent(file.name)}`, {
          method: "POST",
          body: file,
        });
        if (multiple) {
          ids.push(res.id);
          names.push(res.name);
        } else {
          ids.length = 0; names.length = 0;
          ids.push(res.id); names.push(res.name);
        }
        const meta = drop.parentElement.querySelector("[data-meta]");
        if (meta && res.chains?.length) {
          meta.textContent = res.chains
            .map((c) => `${c.id} · ${c.length}aa`).join("   ");
        }
        if (drop.dataset.drop === "target") fillChainSelect(drop.closest(".panel"), res.chains);
      }
      drop.dataset.ids = JSON.stringify(ids);
      drop.dataset.names = JSON.stringify(names);
      if (!multiple) drop.dataset.uploadId = ids[0] || "";
      drop.classList.add("loaded");
      renderReady();
      if (window.DFBio && !multiple
          && String(names[0] || "").toLowerCase().endsWith(".pdb")) {
        window.DFBio.loadPreview(api, drop, currentRegion());
      }
      $(".dt", drop).textContent = multiple
        ? `${names.length} 个文件 · ${names[0]}${names.length > 1 ? " …" : ""}`
        : names[0];
    } catch (err) {
      toast(`上传失败：${err.message}`);
    } finally {
      drop.classList.remove("over");
    }
  }

  function fillChainSelect(panel, chains) {
    const select = $("select[data-field='chain']", panel);
    if (!select || !chains?.length) return;
    select.innerHTML = chains
      .map((c) => `<option value="${c.id}">${c.id}</option>`).join("");
  }

  function wireDrops(root = document) {
    $$("[data-drop]", root).forEach((drop) => {
      const input = $("input[type=file]", drop);
      drop.addEventListener("click", () => input.click());
      input.addEventListener("change", () => uploadFiles(drop, input.files));
      drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
      drop.addEventListener("dragleave", () => drop.classList.remove("over"));
      drop.addEventListener("drop", (e) => {
        e.preventDefault();
        drop.classList.remove("over");
        uploadFiles(drop, e.dataTransfer.files);
      });
    });
  }

  /* ── 控件行为 ─────────────────────────────────────────────────────────── */

  function wireControls(root = document) {
    $$("[data-seg]", root).forEach((seg) => {
      seg.addEventListener("click", (e) => {
        const btn = e.target.closest("button[data-value]");
        if (!btn) return;
        $$("button", seg).forEach((b) => b.classList.toggle("on", b === btn));
        syncToolVisibility(seg.closest(".panel"));
        updateSummary();
        renderReady();
      });
    });

    $$("[data-switch]", root).forEach((sw) => {
      sw.addEventListener("click", () => {
        const on = sw.classList.toggle("on");
        sw.setAttribute("aria-pressed", String(on));
      });
    });

    $$("input.text, textarea.text, select.text", root).forEach((el) => {
      el.addEventListener("input", renderReady);
      el.addEventListener("change", renderReady);
    });

    $$(".range input[type=range]", root).forEach((input) => {
      const out = input.parentElement.querySelector("output");
      const sync = () => {
        if (!out) return;
        const step = Number(input.step || 1);
        out.textContent = step < 0.1 ? Number(input.value).toFixed(3)
          : step < 1 ? Number(input.value).toFixed(2)
          : input.value;
      };
      input.addEventListener("input", sync);
      sync();
    });
  }

  /* ── 结果渲染 ─────────────────────────────────────────────────────────── */

  function resultRoot() {
    return state.ws === "validate" ? $("#validate-results") : $("#design-results");
  }

  function readiness(panel) {
    const op = currentOp();
    const drop = (key) => $(`[data-drop='${key}']`, panel);
    const field = (name) => $(`[data-field='${name}']`, panel);
    const seg = $("[data-seg='tool']", panel);

    if (op === "design_batch") {
      const ids = JSON.parse(drop("files")?.dataset.ids || "[]");
      return [
        ["结构文件", ids.length ? `${ids.length} 个` : "未选择", ids.length > 0],
        ["工具", seg ? $(".on", seg).textContent.trim() : "—", true],
      ];
    }
    if (op === "validate_af2") {
      const text = (field("fasta")?.value || "").trim();
      return [
        ["FASTA", text ? `${text.split(/\s+/).length} 行` : "未填写", text.length > 0],
        ["模型数", field("models")?.value || "1", true],
        ["回收次数", field("recycles")?.value || "1", true],
      ];
    }
    if (op === "design_disease") {
      return [
        ["靶点", state.disease ? `链 ${state.disease.chain}` : "未加载", Boolean(state.disease)],
        ["抗体序列", (field("sequence")?.value || "").trim() ? "已就绪" : "未选择", Boolean((field("sequence")?.value || "").trim())],
        ["区域", field("region")?.value || "默认 CDR", true],
      ];
    }

    const pdbId = drop("pdb")?.dataset.uploadId;
    const items = [["结构文件", pdbId ? (JSON.parse(drop("pdb").dataset.names || "[]")[0] || "已上传") : "未选择", Boolean(pdbId)]];
    if (op === "design_target") {
      const targetId = drop("target")?.dataset.uploadId;
      const antibodyId = drop("antibody")?.dataset.uploadId;
      return [
        ["靶点结构", targetId ? "已上传" : "未选择", Boolean(targetId)],
        ["表位区域", field("epitope")?.value || "未分析", Boolean((field("epitope")?.value || "").trim())],
        ["抗体结构", antibodyId ? "已上传" : "未选择", Boolean(antibodyId)],
      ];
    }
    if (field("region")) items.push(["设计区域", field("region").value, Boolean(field("region").value)]);
    if (seg) items.push(["工具", $(".on", seg).textContent.trim(), true]);
    return items;
  }

  function renderReady() {
    const root = resultRoot();
    const box = $("[data-role=ready]", root);
    if (!box) return;
    const panel = activePanel();
    if (!panel) { box.innerHTML = ""; return; }
    const items = readiness(panel);
    box.innerHTML = '<div class="ready-title">运行前检查</div>' +
      items.map(([label, value, ok]) => `
        <div class="ready-row ${ok ? "ok" : ""}">
          <i></i><span>${escapeHtml(label)}</span><b title="${escapeHtml(value)}">${escapeHtml(value)}</b>
        </div>`).join("") +
      '<div class="ready-foot">Ctrl / ⌘ + Enter 运行</div>';
  }

  function renderResult(result) {
    const root = resultRoot();
    const summaryEl = $("[data-role=summary]", root);
    const tableWrap = $("[data-role=table-wrap]", root);
    const table = $("[data-role=table]", root);
    const empty = $("[data-role=empty]", root);
    const log = $("[data-role=log]", root);
    const logBody = $("[data-role=log-body]", root);
    const arts = $("[data-role=artifacts]", root);
    const copy = $("[data-role=copy]", root);

    summaryEl.textContent = result.summary || "完成";

    const hasRows = Boolean(result.columns?.length && result.rows?.length);
    root.classList.toggle("has-data", hasRows);
    if (window.DFBio) {
      window.DFBio.renderStats(root, result);
      window.DFBio.renderProfile(root, result);
    }
    tableWrap.hidden = !hasRows;
    empty.hidden = Boolean(hasRows);
    if (!hasRows && result.summary) {
      $("p", empty).textContent = result.summary;
    }

    if (hasRows) {
      renderTable(table, result.columns, result.rows);
      copy.hidden = false;
      copy.onclick = () => copyTSV(result);
    } else {
      copy.hidden = true;
    }

    log.hidden = !result.log;
    logBody.textContent = result.log || "";

    arts.innerHTML = "";
    (result.artifacts || []).forEach((item) => {
      const a = document.createElement("a");
      a.href = item.url;
      a.textContent = `↓ ${item.name}`;
      a.title = `${(item.size / 1024).toFixed(1)} KB`;
      arts.appendChild(a);
    });
  }

  const HIDDEN_COLUMNS = /^full sequence$/i;

  function visibleColumns(columns) {
    return columns.map((c, i) => ({ name: c, index: i }))
      .filter((c) => !HIDDEN_COLUMNS.test(c.name.trim()));
  }

  function renderTable(table, columns, rows) {
    const cols = visibleColumns(columns);
    const shown = cols.map((c) => c.index);
    const qualityIndex = cols.findIndex((c) => /quality/i.test(c.name));
    const head = `<thead><tr>${cols.map((c) => {
      const numeric = /^(Rank|PPL|Entropy|pLDDT|ipTM|PAE|Composite|Position|pTM|InterfacePAE)$/i.test(c.name);
      return `<th class="${numeric ? "n" : ""}">${escapeHtml(c.name)}</th>`;
    }).join("")}</tr></thead>`;

    const body = rows.map((row) => `<tr>${shown.map((index, position) => {
      const cell = row[index];
      if (position === qualityIndex) return `<td>${quality(cell)}</td>`;
      const raw = String(cell ?? "");
      const text = escapeHtml(raw);
      const numeric = /^(Rank|PPL|Entropy|pLDDT|ipTM|PAE|Composite|Position|pTM|InterfacePAE)$/i
        .test(cols[position].name);
      const inner = window.DFBio && /sequence|residue/i.test(cols[position].name)
        && /^[A-Z]+$/.test(raw)
        ? window.DFBio.residueHtml(raw) : text;
      return `<td class="${numeric ? "n" : ""}" title="${text}"><span class="cell">${inner}</span></td>`;
    }).join("")}</tr>`).join("");

    table.innerHTML = head + `<tbody>${body}</tbody>`;
  }

  function quality(value) {
    const key = String(value || "").toUpperCase();
    const cls = key === "HIGH" ? "high" : key === "MEDIUM" ? "med" : key === "LOW" ? "low" : "";
    return `<span class="q ${cls}"><i></i>${escapeHtml(key || "—")}</span>`;
  }

  function copyTSV(result) {
    const text = [result.columns.join("\t"),
      ...result.rows.map((r) => r.join("\t"))].join("\n");
    navigator.clipboard.writeText(text)
      .then(() => toast("已复制 TSV", "ok"))
      .catch(() => toast("复制失败"));
  }

  function escapeHtml(value) {
    return value.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /* ── 运行 ─────────────────────────────────────────────────────────────── */

  function currentOp() {
    const ws = state.ws;
    if (!OP[ws]) return null;
    return OP[ws][state.mode[ws]];
  }

  async function run() {
    if (state.job) return;
    const op = currentOp();
    if (!op) return;
    const panel = activePanel();
    if (!panel) return;

    const params = collect(panel);
    if (op === "validate_af2") {
      const text = $("[data-field=fasta]", panel).value.trim();
      if (!text) return toast("请先粘贴 FASTA 序列");
    } else if (op === "design_batch") {
      if (!params.file_ids?.length) return toast("请先选择 PDB 文件");
    } else if (op === "design_disease") {
      if (!state.disease) return toast("请先加载靶点");
      if (!(params.sequence || "").trim()) return toast("请先选择支架或输入序列");
    } else if (op === "design_target") {
      if (!params.target_id) return toast("请先上传靶点结构");
      if (!params.antibody_id) return toast("请先上传抗体结构");
      if (!(params.epitope || "").trim()) return toast("请先分析并确认表位");
    } else if (["design_single", "validate_confidence", "validate_iterative", "validate_loop"].includes(op)) {
      if (!params.pdb_id) return toast("请先上传 PDB");
    }

    setRunning(true, "提交任务");
    try {
      const { job_id } = await api("/api/run", { method: "POST", json: { op, params } });
      watch(job_id);
    } catch (err) {
      setRunning(false);
      toast(err.message);
    }
  }

  function setRunning(active, stage = "") {
    const bar = $("#actionbar");
    bar.classList.toggle("running", active);
    $("#ab-run").disabled = active;
    $("#ab-cancel").hidden = !active;
    $("#ab-fill").style.width = active ? "2%" : "0";
    $("#ab-stage").textContent = stage;
    if (!active) { $("#ab-clock").textContent = ""; }
  }

  function watch(jobId) {
    state.job = { id: jobId };
    if (state.es) state.es.close();
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    state.es = es;

    es.onmessage = (event) => {
      const snap = JSON.parse(event.data);
      state.job = snap;
      $("#ab-fill").style.width = `${snap.pct}%`;
      $("#ab-stage").textContent = snap.stage || "";
      $("#ab-clock").textContent = clock(snap.elapsed);

      if (snap.status === "done") {
        es.close(); state.es = null; state.job = null;
        setRunning(false);
        state.result = snap.result;
        renderResult(snap.result);
        refreshRuns();
        toast(snap.result.summary || "完成", "ok");
      } else if (snap.status === "error") {
        es.close(); state.es = null; state.job = null;
        setRunning(false);
        toast(snap.error || "运行失败");
        refreshRuns();
      } else if (snap.status === "cancelled") {
        es.close(); state.es = null; state.job = null;
        setRunning(false);
        toast("已取消", "ok");
      }
    };
    es.onerror = () => {
      es.close(); state.es = null; state.job = null;
      setRunning(false);
      toast("与服务器的连接中断");
    };
  }

  async function cancel() {
    if (!state.job) return;
    try { await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }); }
    catch (err) { toast(err.message); }
  }

  function updateSummary() {
    const ws = WS[state.ws];
    document.body.classList.toggle("no-bar", !ws.modes.length);
    if (!ws.modes.length) {
      $("#ab-op").textContent = ws.label;
      $("#ab-summary").textContent = "";
      return;
    }
    const mode = state.mode[state.ws];
    const text = (ws.modes.find(([k]) => k === mode) || [null, mode])[1];
    $("#ab-op").textContent = `${ws.label} · ${text}`;
    const panel = activePanel();
    const seg = panel && $("[data-seg='tool']", panel);
    const parts = [];
    if (seg) parts.push($(".on", seg)?.textContent.trim());
    const samples = panel && $("[data-field='samples']", panel);
    if (samples) parts.push(`${samples.value} 条`);
    $("#ab-summary").textContent = parts.join(" · ");
    $("#ab-run").disabled = Boolean(state.job);
    renderReady();
  }

  /* ── 靶点知识库 ───────────────────────────────────────────────────────── */

  function renderKnowledge() {
    const { diseases, scaffolds } = state.boot.knowledge;

    const tbody = $("#disease-table tbody");
    tbody.innerHTML = diseases.map((d) => `
      <tr data-key="${d.key}">
        <td>${escapeHtml(d.cn)}</td>
        <td>${escapeHtml(d.en)}</td>
        <td class="n">${d.conformations.length}</td>
        <td>${d.idp ? '<span class="q low"><i></i>无序</span>' : "折叠"}</td>
      </tr>`).join("");
    $$("#disease-table tbody tr").forEach((row) => {
      row.onclick = () => selectDisease(row.dataset.key);
    });

    const select = $("#scaffold-select");
    select.innerHTML = scaffolds.map((s) =>
      `<option value="${s.key}">${escapeHtml(s.name)} (${escapeHtml(s.source)})</option>`).join("");
    select.onchange = () => showScaffold(select.value);
    if (scaffolds.length) showScaffold(scaffolds[0].key);

    // 设计/疾病靶点面板的下拉
    const dselect = $("[data-field='disease']");
    if (dselect) {
      dselect.innerHTML = diseases.map((d) =>
        `<option value="${d.key}">${escapeHtml(d.cn)} (${escapeHtml(d.en)})</option>`).join("");
      dselect.onchange = () => fillConformations(dselect.value, null);
    }
    fillConformations(diseases[0]?.key, null);
    if (diseases.length) selectDisease(diseases[0].key);
  }

  function fillConformations(diseaseKey, chosen) {
    const disease = state.boot.knowledge.diseases.find((d) => d.key === diseaseKey);
    const select = $("[data-field='conformation']");
    if (!disease || !select) return;
    select.innerHTML = disease.conformations.map((c) =>
      `<option value="${c.key}">${escapeHtml(c.type)} (${escapeHtml(c.pdb)})</option>`).join("");
    if (chosen) select.value = chosen;
  }

  function selectDisease(key) {
    const disease = state.boot.knowledge.diseases.find((d) => d.key === key);
    if (!disease) return;
    $$("#disease-table tbody tr").forEach((r) =>
      r.classList.toggle("on", r.dataset.key === key));
    $("#target-summary").textContent = disease.summary;
    $("#conf-table tbody").innerHTML = disease.conformations.map((c) => `
      <tr><td>${escapeHtml(c.key)}</td><td>${escapeHtml(c.pdb)}</td>
      <td>${escapeHtml(c.type)}</td><td>${escapeHtml(c.resolution || "—")}</td></tr>`).join("");
    state.pickedDisease = key;
  }

  function showScaffold(key) {
    const scaffold = state.boot.knowledge.scaffolds.find((s) => s.key === key);
    if (!scaffold) return;
    $("#scaffold-detail").textContent =
      `${scaffold.name} · ${scaffold.length} aa\nCDR: ${JSON.stringify(scaffold.cdr)}\n\n` +
      `${scaffold.notes}\n\n${scaffold.sequence}`;
    const seq = $("[data-panel='disease'] [data-field='sequence']");
    if (seq && !seq.value.trim()) seq.value = scaffold.sequence;
  }

  /* ── 表位 ─────────────────────────────────────────────────────────────── */

  async function runEpitope(kind) {
    const panel = kind === "target"
      ? $("[data-panel='target']")
      : $("[data-panel='disease']");
    const drop = kind === "target" ? $("[data-drop='target']", panel) : null;
    const uploadId = kind === "target"
      ? drop?.dataset.uploadId
      : state.disease?.upload_id;
    if (!uploadId) return toast(kind === "target" ? "请先上传靶点结构" : "请先加载靶点");

    const topN = Number($("[data-field='top_n']", panel)?.value || 20);
    try {
      const res = await api("/api/epitope", {
        method: "POST",
        json: { upload_id: uploadId, chain: state.disease?.chain || "A", top_n: topN },
      });
      if (kind === "target") {
        $("[data-field='epitope']", panel).value = res.region || "";
        renderResult({
          status: "ok",
          summary: `表位打分 · Top ${topN}`,
          columns: res.columns,
          rows: res.rows,
          log: res.table,
          artifacts: [],
        });
      } else {
        $("[data-field='region']", panel).value = res.region || "";
        renderEpitopeTable(res);
        toast(`表位 Top ${topN} 已计算`, "ok");
      }
    } catch (err) {
      toast(`表位分析失败：${err.message}`);
    }
  }

  function renderEpitopeTable(res) {
    const box = $("#epi-result");
    const table = $("table", box);
    table.innerHTML =
      `<thead><tr>${res.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>` +
      `<tbody>${res.rows.map((r) => `<tr>${r.map((c) =>
        `<td class="${typeof c === "number" ? "n" : ""}">${escapeHtml(String(c))}</td>`).join("")}</tr>`).join("")}</tbody>`;
    box.hidden = false;
    $("#epi-region").value = res.region || "";
  }

  async function loadDisease() {
    const disease = $("[data-field='disease']").value;
    const conformation = $("[data-field='conformation']").value;
    try {
      const res = await api("/api/disease/load", {
        method: "POST", json: { disease, conformation },
      });
      state.disease = { upload_id: res.upload_id, chain: res.chain, is_idp: res.is_idp };
      const status = $("#disease-state");
      status.className = "note" + (res.is_idp ? " is-idp" : "");
      status.textContent = res.is_idp
        ? `已加载 · 链 ${res.chain} · IDP 靶点，置信度不可靠，建议 AF2 验证`
        : `已加载 · 链 ${res.chain} · ${res.info.split("\n")[0]}`;
      toast("靶点已加载", "ok");
    } catch (err) {
      toast(`加载失败：${err.message}`);
    }
  }

  /* ── 记录 ─────────────────────────────────────────────────────────────── */

  async function refreshRuns() {
    try {
      const res = await api("/api/runs");
      state.runs = res.runs;
      renderRailRuns();
      renderRunsTable();
    } catch (_) { /* 静默 */ }
  }

  function renderRunsTable() {
    const tbody = $("#runs-table tbody");
    if (!tbody) return;
    if (!state.runs.length) {
      tbody.innerHTML = '<tr class="blank"><td colspan="6">还没有运行记录 · 去「设计」跑一次</td></tr>';
      return;
    }
    tbody.innerHTML = state.runs.map((run) => {
      const kind = run.status === "completed" ? "ok" : run.status === "failed" ? "err" : "warn";
      return `<tr data-run="${run.run_id}">
        <td>${stamp(run.started_at)}</td>
        <td>${escapeHtml(WS_LABEL[run.workspace] || run.workspace)}</td>
        <td>${escapeHtml(run.mode)}</td>
        <td><span class="q ${kind === "ok" ? "high" : kind === "err" ? "low" : "med"}"><i></i>${escapeHtml(run.status)}</span></td>
        <td class="n">${run.duration_s != null ? run.duration_s + "s" : "—"}</td>
        <td class="n">${run.artifacts}</td>
      </tr>`;
    }).join("");
    $$("#runs-table tbody tr").forEach((row) => {
      row.onclick = () => selectRun(row.dataset.run);
    });
  }

  async function selectRun(runId) {
    state.selectedRun = runId;
    $$("#runs-table tbody tr").forEach((r) =>
      r.classList.toggle("on", r.dataset.run === runId));
    try {
      const res = await api(`/api/runs/${runId}`);
      $("#manifest-body").textContent = JSON.stringify(res.manifest, null, 2);
      $("#artifact-list").innerHTML = res.artifacts.map((a) => `
        <a href="${a.url}"><span>${escapeHtml(a.name)}</span>
        <span class="sz">${(a.size / 1024).toFixed(1)} KB</span></a>`).join("")
        || '<div class="rail-empty">没有产物</div>';
    } catch (err) {
      toast(err.message);
    }
  }

  /* ── 设置 ─────────────────────────────────────────────────────────────── */

  async function openSettings() {
    const dlg = $("#settings");
    dlg.showModal();
    try {
      const cfg = await api("/api/config");
      $("#config-text").value = cfg.text;
      $("#config-path").textContent = cfg.path;
    } catch (err) { toast(err.message); }
    renderEnv();
  }

  function renderEnv() {
    const s = state.boot?.status || {};
    const rows = [
      ["设备", s.device || "—"],
      ["CUDA", s.cuda ? "可用" : "不可用"],
      ["PyTorch", s.torch || "—"],
      ["BFN", s.bfn_ready ? "已加载" : "未加载"],
      ["检查点", s.checkpoint || "—"],
    ];
    $("#env-grid").innerHTML = rows.map(([k, v]) =>
      `<dt>${k}</dt><dd>${escapeHtml(String(v))}</dd>`).join("");
  }

  async function saveConfig() {
    try {
      const res = await api("/api/config", {
        method: "PUT", json: { text: $("#config-text").value },
      });
      $("#config-msg").textContent = `已保存${res.backup ? "，备份 " + res.backup : ""}`;
      toast("配置已保存", "ok");
    } catch (err) {
      $("#config-msg").textContent = err.message;
      toast(err.message);
    }
  }

  /* ── 事件绑定 ─────────────────────────────────────────────────────────── */

  function wireEvents() {
    $$(".ws-nav button").forEach((b) => {
      b.onclick = () => setWorkspace(b.dataset.ws);
    });

    $("#ab-run").onclick = run;
    $("#ab-cancel").onclick = cancel;
    $("#settings-btn").onclick = openSettings;
    $("#settings-close").onclick = () => $("#settings").close();
    $("#config-save").onclick = saveConfig;
    $("#runs-refresh").onclick = refreshRuns;
    $("#use-target").onclick = () => {
      if (!state.pickedDisease) return;
      setWorkspace("design");
      setMode("disease");
      const select = $("[data-field='disease']");
      select.value = state.pickedDisease;
      fillConformations(state.pickedDisease, null);
    };
    $("#epi-run").onclick = async () => {
      const drop = $("[data-drop='epitope']");
      if (!drop?.dataset.uploadId) return toast("请先上传 PDB");
      const chain = $("[data-field='epi_chain']").value || "A";
      const topN = Number($("[data-field='epi_topn']").value || 20);
      try {
        const res = await api("/api/epitope", {
          method: "POST", json: { upload_id: drop.dataset.uploadId, chain, top_n: topN },
        });
        renderEpitopeTable(res);
      } catch (err) { toast(`分析失败：${err.message}`); }
    };

    $$("[data-load-model]").forEach((btn) => {
      btn.onclick = async () => {
        $("#model-msg").textContent = "加载中…";
        try {
          await api(`/api/models/${btn.dataset.loadModel}/load`, { method: "POST" });
          $("#model-msg").textContent = "已加载";
          const boot = await api("/api/bootstrap");
          state.boot.status = boot.status;
          renderEnv();
          renderStatusChip();
        } catch (err) {
          $("#model-msg").textContent = err.message;
        }
      };
    });

    $$("[data-action]").forEach((btn) => {
      btn.onclick = async () => {
        const action = btn.dataset.action;
        if (action === "template") await makeTemplate(btn);
        else if (action === "epitope-target") await runEpitope("target");
        else if (action === "epitope-disease") await runEpitope("disease");
        else if (action === "load-disease") await loadDisease();
      };
    });

    document.addEventListener("keydown", (e) => {
      const typing = /input|textarea|select/i.test(document.activeElement?.tagName || "");
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        run();
      } else if (!typing && /^[1-4]$/.test(e.key) && WS[state.ws].modes.length) {
        const target = WS[state.ws].modes[Number(e.key) - 1];
        if (target) setMode(target[0]);
      } else if (e.key === "Escape" && !$("#settings").open) {
        if (state.job) cancel();
      }
    });

    window.addEventListener("hashchange", () => {
      const [ws, mode] = location.hash.replace("#", "").split("/");
      if (WS[ws]) { setWorkspace(ws, false); if (mode) setMode(mode, false); }
    });
  }

  async function makeTemplate(btn) {
    const panel = btn.closest(".panel");
    const text = $("[data-field='template']", panel).value.trim();
    if (!text) return toast("请粘贴 FASTA 序列");
    btn.disabled = true;
    try {
      const res = await api("/api/template", { method: "POST", json: { fasta: text } });
      const drop = $("[data-drop='pdb']", panel);
      drop.dataset.uploadId = res.id;
      drop.dataset.ids = JSON.stringify([res.id]);
      drop.dataset.names = JSON.stringify([res.name]);
      drop.classList.add("loaded");
      if (window.DFBio) window.DFBio.loadPreview(api, drop, currentRegion());
      $(".dt", drop).textContent = res.name;
      const meta = drop.parentElement.querySelector("[data-meta]");
      if (meta && res.chains?.length) {
        meta.textContent = res.chains.map((c) => `${c.id} · ${c.length}aa`).join("   ");
      }
      toast("模板已生成", "ok");
    } catch (err) {
      toast(err.message);
    } finally {
      btn.disabled = false;
    }
  }

  function renderStatusChip() {
    const s = state.boot.status;
    const chip = $("#chip-device");
    $(".dot", chip).className = "dot " + (s.bfn_ready ? "ok" : "warn");
    $("span", chip).textContent = `${s.device.toUpperCase()}${s.bfn_ready ? " · 就绪" : ""}`;
  }

  /* ── 启动 ─────────────────────────────────────────────────────────────── */

  async function boot() {
    wireControls();
    wireDrops();
    wireEvents();

    try {
      state.boot = await api("/api/bootstrap");
    } catch (err) {
      toast(`无法连接服务：${err.message}`);
      return;
    }
    state.runs = state.boot.runs || [];
    renderStatusChip();
    renderKnowledge();
    renderRunsTable();

    const [hashWs, hashMode] = location.hash.replace("#", "").split("/");
    const ws = WS[hashWs] ? hashWs : "design";
    if (hashMode && WS[ws].modes.some(([k]) => k === hashMode)) state.mode[ws] = hashMode;
    setWorkspace(ws, false);
    setMode(state.mode[ws], false);
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
