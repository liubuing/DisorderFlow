/* DisorderFlow — 生物学表达层
 *
 * 三件事：
 *   1. 残基按理化性质着色；
 *   2. 结果统计条（pLDDT / ipTM / PPL，带 AF2 置信度分档配色）；
 *   3. 主链 CA 投影预览，可高亮设计区域。
 *
 * 不持有应用状态：需要的数据都由 app.js 传入。
 */

(() => {
  "use strict";

  const AA_GROUP = {};
  const GROUPS = {
    hydrophobic: "AVILMFWY",
    polar: "STNQ",
    positive: "KRH",
    negative: "DE",
    special: "GPC",
  };
  Object.entries(GROUPS).forEach(([group, letters]) => {
    letters.split("").forEach((c) => { AA_GROUP[c] = group; });
  });

  const CHAIN_COLORS = ["#4d8dff", "#b285ff", "#3fb950", "#d29922", "#f85149", "#2dd4bf"];

  /* ── 序列 ─────────────────────────────────────────────────────────── */

  function residueHtml(sequence) {
    return [...sequence].map((c) =>
      `<i class="res ${AA_GROUP[c] || ""}">${c}</i>`).join("");
  }

  /* ── 统计条 ───────────────────────────────────────────────────────── */

  function renderStats(root, result) {
    const box = root.querySelector("[data-role=stats]");
    if (!box) return;

    const columns = result.columns || [];
    const rows = result.rows || [];
    const at = (name) => columns.findIndex(
      (c) => c.toLowerCase() === name.toLowerCase());
    const mean = (index) => {
      const values = rows.map((r) => parseFloat(r[index]))
        .filter((v) => !Number.isNaN(v));
      return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
    };

    const chips = [];
    const plddt = at("pLDDT");
    if (plddt >= 0) {
      const value = mean(plddt);
      if (value !== null) {
        const normalised = value > 1 ? value / 100 : value;   // 兼容 0–1 与 0–100
        const band = normalised >= 0.8 ? "high" : normalised >= 0.6 ? "med" : "low";
        chips.push([band, value.toFixed(3), "pLDDT"]);
      }
    }
    const iptm = at("ipTM");
    if (iptm >= 0) {
      const value = mean(iptm);
      if (value !== null) {
        const band = value >= 0.7 ? "high" : value >= 0.4 ? "med" : "low";
        chips.push([band, value.toFixed(3), "ipTM"]);
      }
    }
    const ppl = at("PPL");
    if (ppl >= 0) {
      const value = mean(ppl);
      if (value !== null) chips.push(["", value.toFixed(2), "PPL 均值"]);
    }
    const recovery = at("recovery");
    if (recovery >= 0) {
      const value = mean(recovery);
      if (value !== null) chips.push(["", `${(value * 100).toFixed(1)}%`, "恢复率"]);
    }
    if (rows.length) chips.push(["", String(rows.length), "候选"]);

    if (!chips.length) { box.hidden = true; return; }
    box.hidden = false;
    box.innerHTML = chips.map(([band, value, label]) =>
      `<span class="stat ${band}"><b>${escapeHtml(value)}</b>` +
      `<span>${escapeHtml(label)}</span></span>`).join("");
  }

  /* ── 逐残基色带（pLDDT） ──────────────────────────────────────────── */

  function renderProfile(root, result) {
    const box = root.querySelector("[data-role=profile]");
    if (!box) return;
    const columns = result.columns || [];
    const rows = result.rows || [];
    const at = (pattern) => columns.findIndex((c) => pattern.test(c));

    const iPos = at(/^position$/i);
    const iVal = at(/^plddt$/i);
    if (iPos < 0 || iVal < 0) { box.hidden = true; return; }

    const points = rows
      .map((r) => [Number(r[iPos]), Number(r[iVal])])
      .filter(([p, v]) => Number.isFinite(p) && Number.isFinite(v));
    if (!points.length) { box.hidden = true; return; }

    const first = points[0][0];
    const last = points[points.length - 1][0];
    const span = Math.max(last - first + 1, 1);
    const W = 720, H = 46, TOP = 6;
    const width = W / span;
    const band = (value) => {
      const v = value > 1 ? value / 100 : value;
      return v >= 0.8 ? "#3fb950" : v >= 0.6 ? "#d29922" : "#f85149";
    };

    const bars = points.map(([pos, value]) => {
      const v = value > 1 ? value / 100 : value;
      const x = ((pos - first) / span) * W;
      const height = Math.max(1.5, v * (H - TOP - 6));
      return `<rect x="${x.toFixed(1)}" y="${(H - 6 - height).toFixed(1)}"` +
             ` width="${Math.max(width - 0.6, 0.8).toFixed(1)}" height="${height.toFixed(1)}"` +
             ` fill="${band(value)}" fill-opacity=".82"/>`;
    }).join("");

    box.hidden = false;
    box.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"
           aria-label="逐残基 pLDDT">${bars}
        <line x1="0" y1="${H - 5}" x2="${W}" y2="${H - 5}"
              stroke="rgba(255,255,255,.12)" stroke-width="1"/>
      </svg>
      <div class="pf-foot"><span>残基 ${first}–${last}</span>
        <span>逐残基 pLDDT · 绿 ≥0.8 · 黄 ≥0.6 · 红 &lt;0.6</span></div>`;
  }

  /* ── 区域解析 ─────────────────────────────────────────────────────── */

  function parseRegion(spec) {
    const map = {};
    String(spec || "").replace(/\s+/g, " ").trim().split(" ").forEach((token) => {
      const m = token.match(/^([A-Za-z0-9]+):([\d,\-]+)$/);
      if (!m) return;
      const set = new Set();
      m[2].split(",").forEach((segment) => {
        const [start, end] = segment.split("-").map(Number);
        if (Number.isFinite(start) && Number.isFinite(end)) {
          for (let i = start; i <= end; i += 1) set.add(i);
        } else if (Number.isFinite(start)) {
          set.add(start);
        }
      });
      map[m[1]] = set;
    });
    return map;
  }

  /* ── 骨架投影 ─────────────────────────────────────────────────────── */

  function drawPreview(box, data, regionSpec) {
    const chains = (data.chains || []).filter((c) => c.atoms && c.atoms.length > 1);
    if (!chains.length) { box.remove(); return; }

    const all = chains.flatMap((c) => c.atoms);

    // 选方差最大的两个坐标轴做投影：避免退化视角，也不需要真的做 PCA
    const axes = [0, 1, 2].map((axis) => {
      const values = all.map((a) => a[axis + 1]);
      const avg = values.reduce((x, y) => x + y, 0) / values.length;
      const variance = values.reduce((acc, v) => acc + (v - avg) ** 2, 0) / values.length;
      return { axis, variance };
    }).sort((a, b) => b.variance - a.variance);
    const ax = axes[0].axis;
    const ay = axes[1].axis;

    const xs = all.map((a) => a[ax + 1]);
    const ys = all.map((a) => a[ay + 1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const W = 300, H = 150, PAD = 12;
    const scale = Math.min((W - 2 * PAD) / Math.max(maxX - minX, 1e-6),
                           (H - 2 * PAD) / Math.max(maxY - minY, 1e-6));
    const offsetX = (W - (maxX - minX) * scale) / 2 - minX * scale;
    const offsetY = (H - (maxY - minY) * scale) / 2 - minY * scale;
    const project = (atom) => [
      (atom[ax + 1] * scale + offsetX).toFixed(1),
      (H - (atom[ay + 1] * scale + offsetY)).toFixed(1),
    ];

    const region = parseRegion(regionSpec);
    let body = "";

    chains.forEach((chain, index) => {
      const color = CHAIN_COLORS[index % CHAIN_COLORS.length];
      const trail = chain.atoms.map((atom, i) =>
        `${i ? "L" : "M"}${project(atom).join(" ")}`).join(" ");
      body += `<path d="${trail}" fill="none" stroke="${color}" stroke-opacity=".38"` +
              ` stroke-width="1.3" stroke-linejoin="round" stroke-linecap="round"/>`;

      const highlighted = region[chain.id];
      if (!highlighted || !highlighted.size) return;
      for (let i = 1; i < chain.atoms.length; i += 1) {
        const previous = chain.atoms[i - 1];
        const current = chain.atoms[i];
        if (!highlighted.has(current[0]) || !highlighted.has(previous[0])) continue;
        body += `<path d="M${project(previous).join(" ")} L${project(current).join(" ")}"` +
                ` stroke="${color}" stroke-width="2.8" stroke-linecap="round" fill="none"/>`;
      }
    });

    box.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet"
           role="img" aria-label="主链 CA 投影">${body}</svg>
      <div class="pv-legend">
        ${chains.map((c, i) => `<span><i style="background:${CHAIN_COLORS[i % CHAIN_COLORS.length]}"></i>${escapeHtml(c.id)} · ${c.count} aa</span>`).join("")}
        ${regionSpec ? "<em>亮段 = 设计区域</em>" : ""}
      </div>`;
  }

  async function loadPreview(api, drop, regionSpec) {
    const uploadId = drop.dataset.uploadId;
    if (!uploadId) return;
    const panel = drop.closest(".panel");
    const key = drop.dataset.drop || "pdb";
    let box = panel ? panel.querySelector(`.preview[data-for="${key}"]`) : null;
    if (!box) {
      box = document.createElement("div");
      box.className = "preview";
      box.dataset.for = key;
      const row = drop.closest(".row");
      if (row && row.parentElement) row.after(box);
      else drop.parentElement.appendChild(box);
    }
    box.innerHTML = '<div class="pv-empty">读取骨架…</div>';
    try {
      const data = await api("/api/structure", {
        method: "POST", json: { upload_id: uploadId },
      });
      drawPreview(box, data, regionSpec);
    } catch (_) {
      box.remove();
    }
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  window.DFBio = { residueHtml, renderStats, renderProfile, loadPreview, drawPreview, parseRegion };
})();
