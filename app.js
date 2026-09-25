/* ═════════════════════════════════════════════════════════════════════════════
   ig-autobot — SPA app.js
   Views: Dashboard | Gallery | Agents | Privacy
   Data: dashboard_data.json, gallery.json, agents.json, privacy.html
   ═════════════════════════════════════════════════════════════════════════════ */

(function () {
  "use strict";

  const API = {
    dashboard: "dashboard_data.json",
    gallery:   "gallery.json",
    agents:    "agents.json",
  };

  // ── state ──────────────────────────────────────────────────────────────────
  let DASHBOARD = null;
  let GALLERY   = null;
  let AGENTS    = null;
  let FILTER    = "all";
  let SEARCH    = "";

  // ── helpers ────────────────────────────────────────────────────────────────

  function $(id) { return document.getElementById(id); }

  function showView(name) {
    document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    const el = $(`view-${name}`);
    if (el) el.classList.add("active");
    const btn = document.querySelector(`.tab[data-view="${name}"]`);
    if (btn) btn.classList.add("active");
  }

  function fmtDate(ts) {
    if (!ts || ts === "—") return "—";
    try {
      const s = String(ts);
      if (s.match(/^\d{4}-\d{2}-\d{2}/)) return s.slice(0, 10);
      if (s.match(/^\d{4}-\d{2}-\d{2}T/)) return s.slice(0, 16).replace("T", " ");
      return s;
    } catch { return String(ts); }
  }

  function platformIcon(p) {
    const map = { instagram:"📷", linkedin:"💼", pinterest:"📌", youtube:"🎬", threads:"🧵", bluesky:"🦋", facebook:"📘" };
    return map[p] || "📌";
  }

  function capitalize(s) { return String(s).charAt(0).toUpperCase() + String(s).slice(1); }

  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }

  // ── Tab navigation ─────────────────────────────────────────────────────────

  document.getElementById("tabs").addEventListener("click", e => {
    const btn = e.target.closest(".tab");
    if (!btn) return;
    showView(btn.dataset.view);
  });

  // ── Load data ──────────────────────────────────────────────────────────────

  async function loadAll() {
    try {
      const [dRes, gRes, aRes] = await Promise.all([
        fetch(API.dashboard),
        fetch(API.gallery),
        fetch(API.agents),
      ]);
      if (!dRes.ok || !gRes.ok || !aRes.ok) throw new Error("Data fetch failed");
      DASHBOARD = await dRes.json();
      GALLERY   = await gRes.json();
      AGENTS    = await aRes.json();
    } catch (err) {
      console.error("Failed to load data:", err);
      $("last-updated").textContent = "⚠ Data unavailable";
      return;
    }

    $("last-updated").textContent = new Date().toLocaleString();
    $("footer-generated").textContent = DASHBOARD.generated_at || "—";

    renderDashboard();
    renderGallery();
    renderAgentList();
    loadPrivacy();
  }

  // ── Dashboard ──────────────────────────────────────────────────────────────

  function renderDashboard() {
    const m = DASHBOARD.main;
    const w = DASHBOARD.wilma;

    // Main stats
    $("main-total").textContent   = m.total_posted.toLocaleString();
    $("main-queue").textContent   = m.queue_count;
    $("main-active-pid").textContent = m.active_bundle?.post_id ?? "—";

    // Main platform bars
    renderPlatformBars("main-platform-bars", m.posts_by_platform, false);

    // Main active bundle
    const ab = m.active_bundle;
    if (!ab || !ab.post_id) {
      $("main-active-bundle").innerHTML = '<p class="muted">No active bundle</p>';
    } else {
      const posted = ab.platforms_posted || [];
      const pending = ab.platforms_prepared || [];
      const parts = [`<strong>Post #${ab.post_id}</strong>`];
      if (ab.image) parts.push(`<span class="badge posted">🖼 Image</span>`);
      if (ab.reel)  parts.push(`<span class="badge posted">🎬 Reel</span>`);
      if (ab.carousel) parts.push(`<span class="badge posted">📸 Carousel</span>`);
      posted.forEach(p => parts.push(`<span class="badge posted">✓ ${p}</span>`));
      const unposted = pending.filter(p => !posted.includes(p));
      unposted.forEach(p => parts.push(`<span class="badge pending">⟳ ${p}</span>`));
      $("main-active-bundle").innerHTML = parts.join(" ");
    }

    // Main queue
    renderQueue("main-queue-list", m.queue);

    // Wilma stats
    $("wilma-total").textContent = w.total_posted.toLocaleString();
    $("wilma-day").textContent   = w.current_day === null ? "—" : String(w.current_day);
    $("wilma-queue").textContent = w.queue_count;

    renderPlatformBars("wilma-platform-bars", w.posts_by_platform, true);

    // Wilma topic
    const topic = w.last_topic || "—";
    $("wilma-topic").textContent = topic;

    // Wilma queue
    renderQueue("wilma-queue-list", w.queue);

    // Pillars
    const pillars = m.pillar_counts || {};
    const sorted = Object.entries(pillars).sort((a,b) => b[1] - a[1]);
    const max = sorted.length ? sorted[0][1] : 1;
    $("pillar-list").innerHTML = sorted.map(([name, count]) => `
      <div class="pillar-item">
        <div class="p-name">${capitalize(name)}</div>
        <div class="p-count" style="--w:${clamp(count/max*100, 20, 100)}%">${count}</div>
      </div>
    `).join("");

    // Recent posts
    renderRecent();
  }

  function renderPlatformBars(containerId, byPlatform, isWilma) {
    const el = $(containerId);
    if (!byPlatform || Object.keys(byPlatform).length === 0) {
      el.innerHTML = '<p class="muted">No data</p>';
      return;
    }
    const total = Object.values(byPlatform).reduce((a,b) => a+b, 0) || 1;
    const maxCount = Math.max(...Object.values(byPlatform), 1);
    el.innerHTML = Object.entries(byPlatform)
      .sort((a,b) => b[1] - a[1])
      .map(([plat, count]) => `
        <div class="platform-bar">
          <span class="bar-label">${platformIcon(plat)} ${capitalize(plat)}</span>
          <div class="bar-track">
            <div class="bar-fill${isWilma ? ' wilma' : ''}" style="width:${(count/maxCount*100)}%"></div>
          </div>
          <span class="bar-val">${count}</span>
        </div>
      `).join("");
  }

  function renderQueue(containerId, items) {
    const el = $(containerId);
    if (!items || items.length === 0) {
      el.innerHTML = '<p class="muted">Empty</p>';
      return;
    }
    el.innerHTML = items.map(item => `
      <div class="queue-item">
        <span class="q-badge">#${item.post_id ?? '?'}</span>
        <span class="q-plat">${item.platforms?.join(', ') ?? '—'}</span>
        <span class="q-plat" style="margin-left:auto">${item.format ?? '—'}</span>
      </div>
    `).join("");
  }

  function renderRecent() {
    const m = DASHBOARD.main;
    const w = DASHBOARD.wilma;
    const all = [
      ...m.recent.map(r => ({...r, identity: "main"})),
      ...w.recent.map(r => ({...r, identity: "wilma"})),
    ];
    all.sort((a,b) => b.post_id - a.post_id);
    const top = all.slice(0, 12);
    $("recent-list").innerHTML = top.map(r => `
      <div class="recent-item">
        <span class="r-icon">${r.identity === "wilma" ? "🟢" : "🔵"}</span>
        <span class="r-plat">${platformIcon(r.platform)} ${capitalize(r.platform)}</span>
        <span class="r-id">#${r.post_id}</span>
        <span class="r-tag">${r.identity === "wilma" ? "Wilma" : "Main"}</span>
      </div>
    `).join("");
  }

  // ── Gallery ────────────────────────────────────────────────────────────────

  function renderGallery() {
    GALLERY = GALLERY || [];
    applyFilterAndSearch();
  }

  function applyFilterAndSearch() {
    const grid = $("gallery-grid");
    const empty = $("gallery-empty");
    const count = $("gallery-count");

    let items = GALLERY || [];
    if (FILTER !== "all") items = items.filter(e => e.type === FILTER);
    if (SEARCH.trim()) {
      const q = SEARCH.trim().toLowerCase();
      items = items.filter(e =>
        e.title.toLowerCase().includes(q) ||
        e.caption.toLowerCase().includes(q) ||
        e.pillar.toLowerCase().includes(q)
      );
    }

    count.textContent = `${items.length} post${items.length !== 1 ? 's' : ''}`;

    if (items.length === 0) {
      grid.innerHTML = "";
      empty.style.display = "block";
      return;
    }
    empty.style.display = "none";

    grid.innerHTML = items.map(item => {
      const mediaSrc = item.image || "";
      const isVideo = mediaSrc.endsWith(".mp4");
      return `
        <div class="gallery-card" data-id="${item.id}"
             onclick="window._openLightbox(${item.id})"
             onkeydown="if(event.key==='Enter')window._openLightbox(${item.id})"
             tabindex="0" role="button" aria-label="Open post #${item.id}">
          <div class="card-type-badge">${item.type || '?'}</div>
          ${isVideo
            ? `<video class="card-media" src="${mediaSrc}" loading="lazy" muted preload="metadata"></video>`
            : `<img class="card-media" src="${mediaSrc}" loading="lazy" alt="${item.title || 'Post'}">`
          }
          <div class="card-overlay">
            <div class="card-title">${item.title || 'Post #' + item.id}</div>
            <div class="card-meta">
              <span>${platformIcon(item.platform)} ${capitalize(item.platform)}</span>
              <span>${capitalize(item.pillar)}</span>
              <span>#${item.id}</span>
            </div>
          </div>
        </div>
      `;
    }).join("");
  }

  // Expose to inline onclick
  window._openLightbox = function (id) {
    const item = (GALLERY || []).find(e => e.id === id);
    if (!item) return;
    const img = $("lightbox-img");
    const metaTitle = $("lightbox-title");
    const metaCaption = $("lightbox-caption");
    const metaDetails = $("lightbox-details");
    const lb = $("lightbox");

    if (item.image.endsWith(".mp4")) {
      img.outerHTML = `<video id="lightbox-img" src="${item.image}" controls autoplay style="max-width:90vw;max-height:70vh;border-radius:8px;"></video>`;
    } else {
      img.src = item.image;
      img.outerHTML = `<img id="lightbox-img" src="${item.image}" alt="${item.title || ''}" style="max-width:90vw;max-height:70vh;border-radius:8px;">`;
    }

    metaTitle.textContent       = item.title || `Post #${item.id}`;
    metaCaption.textContent     = item.caption || "—";
    metaDetails.textContent     = `${platformIcon(item.platform)} ${capitalize(item.platform)} · ${capitalize(item.pillar)} · ${fmtDate(item.date)}`;

    lb.classList.add("show");
  };

  function closeLightbox() {
    const lb = $("lightbox");
    lb.classList.remove("show");
    // Restore img element
    const existing = document.getElementById("lightbox-img");
    if (existing) {
      existing.outerHTML = '<img id="lightbox-img" src="" alt="">';
    }
  }

  $("lightbox-close").addEventListener("click", closeLightbox);
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeLightbox();
  });
  $("lightbox").addEventListener("click", e => {
    if (e.target === $("lightbox")) closeLightbox();
  });

  // Gallery filter buttons
  document.getElementById("filter-group").addEventListener("click", e => {
    const btn = e.target.closest(".filter-btn");
    if (!btn) return;
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    FILTER = btn.dataset.filter;
    applyFilterAndSearch();
  });

  // Gallery search
  let searchTimer = null;
  $("gallery-search").addEventListener("input", e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      SEARCH = e.target.value;
      applyFilterAndSearch();
    }, 200);
  });

  // ── Agents ────────────────────────────────────────────────────────────────

  function renderAgentList() {
    const list = $("agent-list");
    list.innerHTML = (AGENTS || []).map((a, i) => `
      <div class="agent-item ${a.status === 'active' ? '' : 'archived'}"
           data-idx="${i}"
           onclick="window._selectAgent(${i})"
           onkeydown="if(event.key==='Enter')window._selectAgent(${i})"
           tabindex="0" role="button" aria-label="Select ${a.name}">
        <div class="a-name">${a.name}</div>
        <div class="a-schedule">${a.schedule}</div>
        <span class="a-status ${a.status}">${a.status}</span>
      </div>
    `).join("");
    // Select first active by default
    const firstActive = (AGENTS || []).find(a => a.status === "active");
    if (firstActive) {
      window._selectAgent((AGENTS || []).indexOf(firstActive));
    }
  }

  window._selectAgent = function (idx) {
    const a = (AGENTS || [])[idx];
    if (!a) return;
    const list = $("agent-list");
    list.querySelectorAll(".agent-item").forEach(el => el.classList.remove("active"));
    const el = list.querySelector(`.agent-item[data-idx="${idx}"]`);
    if (el) el.classList.add("active");

    const detail = $("agent-detail");
    const kindLabels = {
      publish:   "Publish",
      carousel:  "Carousel",
      quote:     "Quote",
      generate:  "Generate",
    };
    const kindLabel = kindLabels[a.kind] || capitalize(a.kind);

    detail.innerHTML = `
      <h2>${a.name}</h2>
      <span class="d-kind">${kindLabel}</span>
      <div class="d-row"><span class="d-label">File</span><span class="d-value"><code>${a.file}</code></span></div>
      <div class="d-row"><span class="d-label">Status</span><span class="d-value"><span class="a-status ${a.status}">${a.status}</span></span></div>
      <div class="d-row"><span class="d-label">Trigger</span><span class="d-value">${a.trigger}</span></div>
      <div class="d-row"><span class="d-label">Platform</span><span class="d-value">${a.platform}</span></div>
      <div class="d-row"><span class="d-label">Schedule</span><span class="d-value">${a.schedule}</span></div>
      <div class="d-description">
        ${describeAgent(a)}
      </div>
      ${a.status === "active" && a.kind !== "generate" ? `
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--card-border)">
        <button class="run-btn" onclick="window._triggerAgent('${a.file}')"
                disabled="${!window._ghToken}"
                title="${window._ghToken ? 'Click to trigger via GitHub Actions' : 'Set GH_TOKEN in app.js to enable'}">
          ${window._ghToken ? "▶ Run now" : "▶ Run now (disabled)"}
        </button>
        <p class="muted" style="font-size:0.72rem;margin-top:4px">
          ${window._ghToken ? "Triggers a manual dispatch of this workflow via the GitHub API." : "Add a GitHub PAT with repo scope to app.js to enable manual triggers."}
        </p>
      </div>
      ` : ""}
    `;
  };

  function describeAgent(a) {
    const descs = {
      "master_publish.yml":        "Main publish workflow. Posts the active bundle to all configured platforms (Instagram, LinkedIn, Pinterest, YouTube, Threads, Bluesky). Runs daily at 17:00 UTC.",
      "master_carousel.yml":       "Generates and posts Instagram carousel slides for the main identity. Runs Mon/Wed/Fri at 13:30 UTC. Uses dark palette. Falls back to static image if no slides are queued.",
      "master_linkedin_carousel.yml":"LinkedIn carousel for the main identity. Posts multi-image carousel to LinkedIn. Runs Wed at 16:00 UTC.",
      "master_quote.yml":          "Posts quotes from the book to Instagram. Runs 5 times per day (02, 06, 10, 18, 22 UTC). Strips chapter attribution — only the quote text + book footer appears.",
      "wilma_publish.yml":         "Wilma identity publish. Posts Wilma's daily content to LinkedIn and Bluesky. Runs daily at 16:30 UTC.",
      "wilma_carousel.yml":        "Wilma LinkedIn carousel. Posts multi-image carousel to Wilma's LinkedIn. Runs Fri+Sun at 15:00 UTC. Uses cream palette.",
      "master_wilma_gen.yml":      "Generates the next day's Wilma content (image + caption). Runs daily at 11:30 UTC via wilma_bot.py --mode generate_all.",
      "master_content_gen.yml":    "Generates content bundles for the main identity (image, caption, topic selection). Runs Mon/Tue/Thu/Sat at 02:00 UTC.",
    };
    return descs[a.file] || `Workflow: ${a.name}. Schedule: ${a.schedule}. Platform: ${a.platform}.`;
  }

  // ── Manual trigger (GitHub API) ────────────────────────────────────────────

  // Set this to a GitHub PAT with repo scope to enable manual triggers
  window._ghToken = null; // e.g. "ghp_xxxxxxxxxxxxxxxxxxxx"

  window._triggerAgent = async function (file) {
    if (!window._ghToken) return;
    const ownerRepo = "iyeque/ig-autobot";
    const fullName = `${ownerRepo}/${file}`;
    try {
      const resp = await fetch(`https://api.github.com/repos/${ownerRepo}/actions/workflows/${encodeURIComponent(file)}/dispatches`, {
        method: "POST",
        headers: {
          "Authorization": `token ${window._ghToken}`,
          "Accept": "application/vnd.github.v3+json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ref: "master" }),
      });
      if (resp.ok) {
        alert(`✔ Dispatched ${file} — check GitHub Actions for the run.`);
      } else {
        const body = await resp.text();
        alert(`✘ Failed to dispatch ${file}\n${resp.status}: ${body.slice(0, 200)}`);
      }
    } catch (err) {
      alert(`✘ Network error: ${err.message}`);
    }
  };

  // ── Privacy ────────────────────────────────────────────────────────────────

  async function loadPrivacy() {
    try {
      const resp = await fetch("privacy.html");
      if (!resp.ok) throw new Error("privacy.html not found");
      const html = await resp.text();
      $("privacy-container").innerHTML = html;
      // Re-bind any inner links to SPA navigation
      $("privacy-container").querySelectorAll('a[href="index.html"]').forEach(a => {
        a.onclick = () => { showView("dashboard"); };
      });
    } catch {
      $("privacy-container").innerHTML = '<p class="muted">Privacy policy not available. See <a href="PRIVACY.md" target="_blank">PRIVACY.md</a>.</p>';
    }
  }

  // ── Boot ───────────────────────────────────────────────────────────────────

  document.addEventListener("DOMContentLoaded", () => {
    showView("dashboard");
    loadAll();
  });

})();
