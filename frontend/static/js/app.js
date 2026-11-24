/**
 * VEHICLE TRACE — Frontend Application Logic
 */
let currentCaseId = null;
let currentResults = null;
let currentAllVehicles = [];
let pollTimer = null;
let selectedTrackId = null;

document.addEventListener("DOMContentLoaded", () => {
  setupTabs();
  setupUpload();
  setupModal();
  setupMatching();
  setupFilterToggle();
  loadCases();
});

// Tab Navigation
function setupTabs() {
  document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-tab");
      if (targetId && document.getElementById(targetId)) {
        window.switchTab(targetId);
      }
    });
  });
}

// Global UI Navigation Helpers for DriveX Interface
window.triggerUploadClick = function () {
  const inp = document.getElementById("media-upload-input");
  if (inp) inp.click();
};

window.setSearchMode = function (mode) {
  document.querySelectorAll(".search-tab-pill").forEach(p => p.classList.remove("active"));
  if (window.event && window.event.currentTarget) {
    window.event.currentTarget.classList.add("active");
  }
  if (mode === "registry") {
    window.switchTab("matching-tab");
  } else {
    window.switchTab("workspace-tab");
  }
};

window.scrollToSection = function (id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: "smooth" });
  }
};

window.switchTab = function (tabId) {
  document.querySelectorAll(".nav-tab").forEach(t => {
    if (t.getAttribute("data-tab") === tabId) {
      t.classList.add("active");
    } else {
      t.classList.remove("active");
    }
  });
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));
  const targetPane = document.getElementById(tabId);
  if (targetPane) {
    targetPane.classList.add("active");
    targetPane.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (tabId === "cases-tab") {
    loadCases();
  }
};

window.quickFilterRegistry = function (category) {
  window.switchTab("matching-tab");
};

// Media Upload & Ingestion
function setupUpload() {
  const fileInput = document.getElementById("media-upload-input");
  if (fileInput) {
    fileInput.addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const lbl = document.getElementById("selected-file-label");
      if (lbl) {
        lbl.innerText = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
        lbl.style.color = "#0f172a";
        lbl.style.fontWeight = "600";
      }

      window.switchTab("workspace-tab");
      await ingestMedia(file);
    });
  }

  const exportBtn = document.getElementById("btn-export-report");
  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (currentCaseId) {
        window.open(`/api/reports/${currentCaseId}/html`, "_blank");
      }
    });
  }
}

window.openVehicleReport = function(tid) {
  if (!currentCaseId) {
    alert("Please load or ingest a case first to generate individual vehicle reports.");
    return;
  }
  window.open(`/api/reports/${currentCaseId}/tracks/${tid}/html`, "_blank");
};

async function ingestMedia(file) {
  const isVideo = file.type.startsWith("video/") || file.name.match(/\.(mp4|avi|mov|mkv)$/i);
  const formData = new FormData();
  formData.append("file", file);
  formData.append("title", `Investigation ${file.name}`);

  // Display local preview
  const imgDisp = document.getElementById("image-display");
  const vidDisp = document.getElementById("video-display");
  const dropPrompt = document.getElementById("dropzone-prompt");
  dropPrompt.style.display = "none";

  if (isVideo) {
    imgDisp.style.display = "none";
    vidDisp.style.display = "block";
    vidDisp.src = URL.createObjectURL(file);
  } else {
    vidDisp.style.display = "none";
    imgDisp.style.display = "block";
    imgDisp.src = URL.createObjectURL(file);
  }

  // Clear previous outputs
  document.getElementById("vehicles-container").innerHTML = "<div class='empty-state'>Analyzing evidence...</div>";
  document.getElementById("frame-thumbnails").innerHTML = "<div class='empty-state'>Capturing frames...</div>";

  try {
    if (isVideo) {
      // Async Video Processing
      document.getElementById("job-progress-container").style.display = "block";
      updateProgress(10, "Uploading CCTV video...");

      const resp = await fetch("/api/analysis/video", { method: "POST", body: formData });
      const data = await resp.json();
      currentCaseId = data.job_id;
      startPollingVideo(currentCaseId);
    } else {
      // Sync Image Processing
      const resp = await fetch("/api/analysis/image", { method: "POST", body: formData });
      const data = await resp.json();
      currentCaseId = data.case_id;
      currentResults = data.results;
      renderImageResults(data.results);
      document.getElementById("btn-export-report").disabled = false;
    }
  } catch (err) {
    alert("Analysis failed: " + err);
  }
}

// Video Polling Loop
function startPollingVideo(jobId) {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const resp = await fetch(`/api/analysis/jobs/${jobId}`);
      const statusData = await resp.json();
      updateProgress(statusData.progress, `Processing video (${statusData.progress.toFixed(0)}%)...`);

      if (statusData.status === "completed") {
        clearInterval(pollTimer);
        updateProgress(100, "Analysis complete.");
        setTimeout(() => {
          document.getElementById("job-progress-container").style.display = "none";
        }, 1200);

        // Fetch results and frame evidence
        const resResp = await fetch(`/api/analysis/jobs/${jobId}/results`);
        const resData = await resResp.json();
        currentResults = resData;
        renderVideoResults(resData);
        loadFrameEvidence(jobId);
        document.getElementById("btn-export-report").disabled = false;
      } else if (statusData.status === "failed") {
        clearInterval(pollTimer);
        updateProgress(0, "Processing failed: " + statusData.error_message);
      }
    } catch (err) {
      console.error("Poll error:", err);
    }
  }, 1000);
}

function updateProgress(pct, msg) {
  document.getElementById("job-progress-fill").style.width = `${pct}%`;
  document.getElementById("job-percent-text").innerText = `${pct.toFixed(0)}%`;
  document.getElementById("job-status-text").innerText = msg;
}

const INDIAN_STATE_CODES = new Set([
  "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ", "HP", "HR", "JH", "JK",
  "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK",
  "TN", "TR", "TS", "TG", "UK", "UA", "UP", "WB"
]);

function isIndianPlateFormat(text) {
  if (!text) return false;
  const t = text.toUpperCase().replace(/[^A-Z0-9?]/g, "");
  if (t.length < 5 || t.length > 11) return false;

  // Must have at least 3 known characters (cannot be all ?)
  const known = t.replace(/[^A-Z0-9]/g, "");
  if (known.length < 3) return false;

  // Bharat Series: YY BH NNNN XX
  const bhMatch = t.match(/^([\d?]{2})(BH|B\?|\?H)([\d?]{4})([A-Z?]{1,2})$/);
  if (bhMatch) return true;

  // Standard Series: State(2) + RTO(1-2) + Series(0-3) + Number(1-4)
  const stdMatch = t.match(/^([A-Z?]{2})([\d?]{1,2})([A-Z?]{0,3})([\d?]{1,4})$/);
  if (!stdMatch) return false;

  const st = stdMatch[1];
  if (!st.includes("?")) {
    if (!INDIAN_STATE_CODES.has(st)) return false;
  } else {
    // If one char is ?, verify at least one state code matches
    let canMatch = false;
    for (const sc of INDIAN_STATE_CODES) {
      if (sc[0] === st[0] || sc[1] === st[1]) {
        canMatch = true;
        break;
      }
    }
    if (!canMatch) return false;
  }

  return true;
}

function formatPlateHtml(rawText) {
  if (!rawText) return "NOT_READABLE";
  return rawText.split("").map(ch => {
    if (ch === "?") {
      return `<span class="occluded-digit" title="Digit occluded / not clearly visible">?</span>`;
    }
    return ch;
  }).join("");
}

// Plate Visibility & Indian Format Classifier
function isPlateCompletelyVisible(v) {
  if (!v) return false;
  const plate = v.plate || {};
  const text = (plate.text || "").trim();
  const pCropSrc = v.crops?.plate_crop || v.evidence?.plate_crop_path || "";

  // 1. Must have an actual cropped plate image
  if (!pCropSrc || pCropSrc === "None" || pCropSrc.includes("No Plate") || pCropSrc === "null") {
    return false;
  }

  // 2. Plate text must be non-empty and not generic unreadable placeholder
  if (!text || text === "NOT_READABLE" || text === "UNKNOWN" || text === "None" || text === "null") {
    return false;
  }

  // 3. MUST strictly match official Indian vehicle registration syntax (Standard or BH series)
  if (!isIndianPlateFormat(text)) {
    return false;
  }

  // 4. Plate status must not be unreadable or unknown
  const status = (plate.status || "").toLowerCase();
  if (status === "unreadable" || status === "unknown") {
    return false;
  }

  return true;
}

// Render Results with Visibility Filtering
function renderImageResults(results) {
  currentAllVehicles = results.vehicles || [];
  renderFilteredVehicles();
}

function renderVideoResults(results) {
  currentAllVehicles = results.vehicles || [];
  renderFilteredVehicles();
}

function renderFilteredVehicles() {
  const container = document.getElementById("vehicles-container");
  container.innerHTML = "";

  const filterCheckbox = document.getElementById("filter-visible-plate-only");
  const filterActive = filterCheckbox ? filterCheckbox.checked : true;

  const vehiclesToDisplay = filterActive
    ? currentAllVehicles.filter(isPlateCompletelyVisible)
    : currentAllVehicles;

  const countBadge = document.getElementById("vehicle-count-badge");
  if (countBadge) {
    if (filterActive && currentAllVehicles.length !== vehiclesToDisplay.length) {
      countBadge.innerText = `${vehiclesToDisplay.length} with Complete Plate (${currentAllVehicles.length - vehiclesToDisplay.length} hidden)`;
    } else {
      countBadge.innerText = `${vehiclesToDisplay.length} Identified`;
    }
  }

  if (!vehiclesToDisplay.length) {
    if (currentAllVehicles.length > 0 && filterActive) {
      container.innerHTML = `
        <div class="empty-state">
          <div style="font-size:28px;margin-bottom:8px;">🔍</div>
          <h3 style="margin-bottom:6px;color:#e2e8f0;">No Vehicles with Completely Visible Plates</h3>
          <p style="font-size:13px;line-height:1.5;color:#94a3b8;margin-bottom:12px;">
            ${currentAllVehicles.length} vehicle track(s) were detected, but their license plates were not completely visible, obscured by angle, or missing.
          </p>
          <button class="btn-secondary" style="font-size:12px;padding:6px 14px;" onclick="window.togglePlateFilter(false)">
            Show All ${currentAllVehicles.length} Tracks (Including Occluded)
          </button>
        </div>
      `;
    } else {
      container.innerHTML = "<div class='empty-state'>No vehicle tracks found.</div>";
    }
    return;
  }

  vehiclesToDisplay.forEach(v => {
    container.appendChild(createVehicleCard(v));
  });

  // Select first displayed track for candidate matching
  if (vehiclesToDisplay.length > 0) {
    selectTrackForMatching(vehiclesToDisplay[0]);
  }
}

window.togglePlateFilter = function (checked) {
  const cb = document.getElementById("filter-visible-plate-only");
  if (cb) {
    cb.checked = checked;
    renderFilteredVehicles();
  }
};

function setupFilterToggle() {
  const cb = document.getElementById("filter-visible-plate-only");
  if (cb) {
    cb.addEventListener("change", () => {
      renderFilteredVehicles();
    });
  }
}

const COLOR_SWATCH_MAP = {
  white: { hex: "#FFFFFF", border: "#94A3B8", label: "White" },
  black: { hex: "#0F172A", border: "#334155", label: "Black" },
  silver_grey: { hex: "#94A3B8", border: "#64748B", label: "Silver / Grey" },
  red: { hex: "#DC2626", border: "#B91C1C", label: "Red" },
  blue: { hex: "#2563EB", border: "#1D4ED8", label: "Blue" },
  green: { hex: "#16A34A", border: "#15803D", label: "Green" },
  yellow: { hex: "#EAB308", border: "#CA8A04", label: "Yellow / Gold" },
  brown: { hex: "#78350F", border: "#451A03", label: "Brown" },
  orange: { hex: "#EA580C", border: "#C2410C", label: "Orange" },
  other: { hex: "#6B7280", border: "#4B5563", label: "Other" },
  unknown: { hex: "#E2E8F0", border: "#CBD5E1", label: "Unknown" }
};

function getColorSwatch(name) {
  const k = (name || "").toLowerCase().replace(/[\s-]/g, "_");
  return COLOR_SWATCH_MAP[k] || { hex: "#94A3B8", border: "#64748B", label: name || "Unknown" };
}

function createVehicleCard(v) {
  const card = document.createElement("div");
  card.className = "vehicle-card";

  const tid = v.track_id;
  const plate = v.plate || {};
  const veh = v.vehicle || {};
  const colour = veh.colour || {};
  const make = veh.make || {};
  const model = veh.model || {};
  const vtype = veh.type || veh.vehicle_type || {};

  const vCropSrc = v.crops?.vehicle_crop || v.evidence?.vehicle_crop_path || "";
  const pCropSrc = v.crops?.plate_crop || v.evidence?.plate_crop_path || "";

  const fusionStatus = v.evidence?.fusion_status || colour.status || "predicted";

  const hasOccluded = (plate.text || "").includes("?");
  const isIndian = isIndianPlateFormat(plate.text);
  const statusBadge = hasOccluded
    ? `<span class="plate-status-badge badge-occluded">PARTIAL OCCLUSION</span>`
    : (isIndian
      ? `<span class="plate-status-badge badge-indian">VERIFIED INDIAN REG</span>`
      : `<span class="plate-status-badge">${plate.status?.toUpperCase() || "UNKNOWN"}</span>`);

  const colInfo = getColorSwatch(colour.value);
  const paintMetrics = colour.paint_metrics;

  // Only find/display closely matching registered vehicles if the extracted number has missing digits ('?')
  const closeMatches = hasOccluded ? (v.close_matches || []) : [];
  let matchesHtml = "";

  if (hasOccluded) {
    if (closeMatches.length > 0) {
      matchesHtml = `
        <div class="registry-matches-panel">
          <div class="registry-matches-hdr">
            <span class="hdr-title">📋 Closely Matching Registered Vehicles</span>
            <span class="match-count-pill">${closeMatches.length} Found in Registry</span>
          </div>
          <div class="registry-matches-list">
            ${closeMatches.map(m => `
              <div class="registry-match-card" title="Click to open Candidate Matching for this record" onclick="triggerMatchingForTrack('${tid}')">
                <div class="match-row-top">
                  <span class="match-plate">${m.registration_number}</span>
                  <div class="match-score-pill ${m.match_percentage >= 80 ? 'score-high' : 'score-med'}">
                    <span>${m.match_percentage}% MATCH</span>
                  </div>
                </div>
                <div class="match-attrs-grid">
                  <div class="attr-chip"><span class="chip-k">Make:</span> <span class="chip-v">${m.manufacturer}</span></div>
                  <div class="attr-chip"><span class="chip-k">Model:</span> <span class="chip-v">${m.vehicle_model}</span></div>
                  <div class="attr-chip"><span class="chip-k">Colour:</span> <span class="chip-v">${m.body_colour}</span></div>
                  <div class="attr-chip"><span class="chip-k">Category:</span> <span class="chip-v">${m.category_type}</span></div>
                  <div class="attr-chip"><span class="chip-k">State RTO:</span> <span class="chip-v">${m.state}</span></div>
                </div>
                ${m.attributes_matched && m.attributes_matched.length > 0 ? `
                  <div class="match-tags-row">
                    ${m.attributes_matched.map(t => `<span class="tag-matched">${t}</span>`).join("")}
                  </div>
                ` : ''}
              </div>
            `).join("")}
          </div>
        </div>
      `;
    } else {
      matchesHtml = `
        <div class="registry-matches-panel">
          <div class="registry-matches-hdr">
            <span class="hdr-title">📋 Closely Matching Registered Vehicles</span>
            <span class="match-count-pill empty">0 Found</span>
          </div>
          <div class="no-matches-msg">
            No authorized registry vehicles closely match pattern <b>${plate.text}</b>.
          </div>
        </div>
      `;
    }
  }

  card.innerHTML = `
    <div class="card-top">
      <div class="track-identity">
        <span class="track-dot"></span>
        <span class="track-title">Vehicle Track #${tid}</span>
      </div>
      <span class="badge badge-${fusionStatus}">${fusionStatus.toUpperCase()} EVIDENCE</span>
    </div>

    <div class="crops-grid">
      <div class="crop-card">
        ${vCropSrc ? `<img src="${vCropSrc}" alt="Vehicle Body">` : `<div style="height:100px;line-height:100px;font-size:11px;color:#64748b;">No Body Crop</div>`}
        <span class="crop-lbl">Vehicle Body Crop</span>
      </div>
      <div class="crop-card">
        ${pCropSrc ? `<img src="${pCropSrc}" alt="Plate Crop">` : `<div style="height:100px;line-height:100px;font-size:11px;color:#64748b;">No Plate Crop</div>`}
        <span class="crop-lbl">License Plate OCR Crop</span>
      </div>
    </div>

    <!-- Official High-Security Indian License Plate Box -->
    <div class="plate-display-box">
      <div class="hsrp-plate-wrap">
        <div class="hsrp-blue-band">
          <span class="hsrp-chakra">☸</span>
          <span class="hsrp-ind">IND</span>
        </div>
        <span class="hsrp-plate-text">${formatPlateHtml(plate.text)}</span>
      </div>
      ${statusBadge}
    </div>

    <!-- High-Readability Attribute Matrix -->
    <div class="attr-matrix-grid">
      <div class="attr-cell">
        <span class="cell-label">Body Colour</span>
        <div class="cell-val-wrap">
          <span class="color-swatch-dot" style="background:${colInfo.hex}; border-color:${colInfo.border};"></span>
          <span class="cell-primary-val">${colInfo.label}</span>
          <span class="confidence-pill ${colour.score >= 0.7 ? 'conf-high' : 'conf-mid'}">${Math.round((colour.score || 0) * 100)}%</span>
        </div>
      </div>

      <div class="attr-cell">
        <span class="cell-label">Manufacturer</span>
        <div class="cell-val-wrap">
          <span class="cell-primary-val">${make.value && make.value.toLowerCase() !== 'unknown' ? make.value : 'Uncertain'}</span>
          <span class="confidence-pill ${make.score >= 0.7 ? 'conf-high' : 'conf-mid'}">${Math.round((make.score || 0) * 100)}%</span>
        </div>
      </div>

      <div class="attr-cell">
        <span class="cell-label">Vehicle Model</span>
        <div class="cell-val-wrap">
          <span class="cell-primary-val">${model.value && model.value.toLowerCase() !== 'unknown' ? model.value : 'Uncertain'}</span>
          <span class="confidence-pill ${model.score >= 0.7 ? 'conf-high' : 'conf-mid'}">${Math.round((model.score || 0) * 100)}%</span>
        </div>
      </div>

      <div class="attr-cell">
        <span class="cell-label">Category / Class</span>
        <div class="cell-val-wrap">
          <span class="cell-primary-val">${(vtype.value || 'Car').toUpperCase()}</span>
          <span class="category-pill">${(vtype.value || 'Car').toUpperCase()}</span>
        </div>
      </div>
    </div>

    ${paintMetrics ? `
      <div class="paint-metrics-row">
        <span class="metric-title">CIE L*a*b* Paint Panel Analysis:</span>
        <span class="metric-chip">L: ${paintMetrics.L}</span>
        <span class="metric-chip">a: ${paintMetrics.a}</span>
        <span class="metric-chip">b: ${paintMetrics.b}</span>
        <span class="metric-chip">Chroma: ${paintMetrics.chroma}</span>
      </div>
    ` : ''}

    ${matchesHtml}

    <div class="card-actions">
      <button class="btn-card btn-review" onclick="openReviewModal('${tid}', '${plate.text || ""}', '${colour.value || ""}', '${make.value || ""}', '${model.value || ""}', '${vtype.value || ""}')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>
        Manual Review
      </button>
      <button class="btn-card btn-match" onclick="triggerMatchingForTrack('${tid}')">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        Match Registry
      </button>
      <button class="btn-card btn-report" onclick="openVehicleReport('${tid}')" title="Generate Individual Forensic Investigation Report for this Vehicle">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>
        Vehicle Report
      </button>
    </div>
  `;

  return card;
}

async function loadFrameEvidence(jobId) {
  try {
    const resp = await fetch(`/api/analysis/jobs/${jobId}/frames`);
    const data = await resp.json();
    const box = document.getElementById("frame-thumbnails");
    box.innerHTML = "";

    const frames = data.frames || [];
    if (!frames.length) {
      box.innerHTML = "<div class='empty-state'>No frame evidence.</div>";
      return;
    }

    frames.slice(0, 16).forEach(f => {
      const thumb = document.createElement("div");
      thumb.className = "frame-thumb";
      thumb.innerHTML = `
        <img src="${f.vehicle_crop || f.plate_crop}" alt="Frame">
        <span class="lbl">#${f.track_id} ${(f.timestamp_ms / 1000).toFixed(1)}s</span>
      `;
      box.appendChild(thumb);
    });
  } catch (err) {
    console.error("Frame evidence load error:", err);
  }
}

// Manual Review & Audit Modal
function setupModal() {
  const modal = document.getElementById("review-modal");
  document.getElementById("modal-close-btn").addEventListener("click", () => modal.style.display = "none");
  document.getElementById("modal-cancel-btn").addEventListener("click", () => modal.style.display = "none");

  const form = document.getElementById("correction-form");
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const tid = document.getElementById("modal-track-id").value;
    const attr = document.getElementById("modal-attr-select").value;
    const newVal = document.getElementById("modal-new-val").value;
    const invId = document.getElementById("modal-inv-id").value;
    const reason = document.getElementById("modal-reason").value;

    try {
      const resp = await fetch(`/api/cases/${currentCaseId}/confirm`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          track_id: tid,
          attribute_name: attr,
          corrected_value: newVal,
          investigator_id: invId,
          reason: reason
        })
      });
      const data = await resp.json();
      modal.style.display = "none";
      alert(`Correction saved successfully for ${tid}!`);
    } catch (err) {
      alert("Failed to save correction: " + err);
    }
  });
}

window.openReviewModal = function (tid, plate, colour, make, model, vtype) {
  const modal = document.getElementById("review-modal");
  document.getElementById("modal-track-id").value = tid;

  const attrSelect = document.getElementById("modal-attr-select");
  const origInput = document.getElementById("modal-orig-val");

  function updateOrig() {
    const a = attrSelect.value;
    if (a === "plate") origInput.value = plate;
    else if (a === "colour") origInput.value = colour;
    else if (a === "make") origInput.value = make;
    else if (a === "model") origInput.value = model;
    else if (a === "vehicle_type") origInput.value = vtype;
  }

  attrSelect.onchange = updateOrig;
  updateOrig();
  modal.style.display = "flex";
};

// Candidate Matching
function setupMatching() {
  document.getElementById("btn-run-match").addEventListener("click", () => {
    if (selectedTrackId && currentCaseId) {
      runMatching(currentCaseId, selectedTrackId);
    } else {
      alert("Please select a track from the Analysis Workspace first.");
    }
  });
}

function selectTrackForMatching(v) {
  selectedTrackId = v.track_id;
  const col = getColorSwatch(v.vehicle?.colour?.value);
  const trackEl = document.getElementById("crit-track");
  const plateEl = document.getElementById("crit-plate");
  const colEl = document.getElementById("crit-colour");
  const mmEl = document.getElementById("crit-make-model");
  const typeEl = document.getElementById("crit-type");

  if (trackEl) trackEl.innerHTML = `<span class="crit-badge-track">#${v.track_id}</span>`;
  if (plateEl) {
    plateEl.innerHTML = `
      <span class="crit-badge-plate">
        <span class="crit-ind">IND</span>
        <span>${formatPlateHtml(v.plate?.text || "Unknown")}</span>
      </span>
    `;
  }
  if (colEl) {
    colEl.innerHTML = `
      <span class="crit-badge-attr">
        <span class="color-swatch-dot" style="background:${col.hex}; border-color:${col.border};"></span>
        ${col.label}
      </span>
    `;
  }
  if (mmEl) {
    mmEl.innerHTML = `
      <span class="crit-badge-attr">${v.vehicle?.make?.value || "Unknown"} / ${v.vehicle?.model?.value || "Unknown"}</span>
    `;
  }
  if (typeEl) {
    typeEl.innerHTML = `
      <span class="crit-badge-attr">${v.vehicle?.type?.value || v.vehicle?.vehicle_type?.value || "Car"}</span>
    `;
  }
}

window.triggerMatchingForTrack = function (tid) {
  selectedTrackId = tid;
  window.switchTab("matching-tab");
  runMatching(currentCaseId, tid);
};

async function runMatching(caseId, trackId) {
  const tbody = document.getElementById("candidates-table-body");
  tbody.innerHTML = `<tr><td colspan="7" class="empty-row">Querying registry database and ranking candidates...</td></tr>`;

  try {
    const resp = await fetch(`/api/matching/cases/${caseId}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ track_id: trackId, use_investigator_corrections: true })
    });
    const data = await resp.json();
    const trackResult = data.tracks_matched[trackId];
    if (!trackResult) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No match results for track ${trackId}.</td></tr>`;
      return;
    }

    const ranking = trackResult.ranking_result;
    const candidates = ranking.candidates || [];

    if (ranking.status === "NO_CONFIDENT_MATCH" || !candidates.length) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="empty-row" style="color:#f87171;">
            <b>NO_CONFIDENT_MATCH:</b> Visual evidence was below the acceptance threshold (${ranking.acceptance_threshold}). No registered vehicle matched confidently.
          </td>
        </tr>
      `;
      return;
    }

    tbody.innerHTML = "";
    candidates.forEach(c => {
      const reg = c.registry_details;
      const comp = c.compatibility;
      const row = document.createElement("tr");

      const plateBadgeHtml = `
        <div class="table-plate-badge">
          <span class="table-plate-ind">IND</span>
          <span class="table-plate-txt">${c.registration_number}</span>
        </div>
      `;
      const scorePct = (c.ranking_score * 100).toFixed(1);
      const isTopScore = c.ranking_score >= 0.8;

      row.innerHTML = `
        <td>${plateBadgeHtml}</td>
        <td>
          <div class="reg-veh-title">${reg.make} ${reg.model}</div>
          <div class="reg-veh-sub">
            <span class="sub-chip">${reg.colour.toUpperCase()}</span>
            <span class="sub-chip">${reg.vehicle_type || 'Vehicle'}</span>
            <span class="sub-rto">${reg.rto}, ${reg.state}</span>
          </div>
          <div class="reg-owner">Owner: ${reg.owner_masked}</div>
        </td>
        <td>
          <span class="score-pill ${comp.plate_score >= 0.8 ? 'pill-high' : 'pill-mid'}">${(comp.plate_score * 100).toFixed(0)}%</span>
          <div class="score-note">${comp.plate_notes}</div>
        </td>
        <td>
          <span class="score-pill ${comp.make_score >= 0.7 ? 'pill-high' : 'pill-low'}">${(comp.make_score * 100).toFixed(0)}%</span>
        </td>
        <td>
          <span class="score-pill ${comp.colour_score >= 0.7 ? 'pill-high' : 'pill-low'}">${(comp.colour_score * 100).toFixed(0)}%</span>
        </td>
        <td>
          <div class="total-rank-wrap">
            <span class="rank-pct ${isTopScore ? 'rank-top' : 'rank-mid'}">${scorePct}%</span>
            <div class="rank-bar-bg"><div class="rank-bar-fill" style="width: ${scorePct}%;"></div></div>
          </div>
        </td>
        <td><span class="status-match-pill ${isTopScore ? 'status-top' : 'status-cand'}">${isTopScore ? 'TOP MATCH' : 'CANDIDATE'}</span></td>
      `;
      tbody.appendChild(row);
    });

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-row" style="color:#f87171;">Matching failed: ${err}</td></tr>`;
  }
}

// Cases List
async function loadCases() {
  const tbody = document.getElementById("cases-table-body");
  try {
    const resp = await fetch("/api/cases");
    const cases = await resp.json();
    tbody.innerHTML = "";

    if (!cases.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="empty-row">No investigation cases recorded.</td></tr>`;
      return;
    }

    cases.forEach(c => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><b>${c.id}</b></td>
        <td>${c.title}</td>
        <td><span class="badge badge-unknown">${c.source_type.toUpperCase()}</span></td>
        <td><span class="badge badge-${c.status === 'completed' ? 'predicted' : 'low_confidence'}">${c.status.toUpperCase()}</span></td>
        <td>${c.progress}%</td>
        <td>${c.created_at ? new Date(c.created_at).toLocaleString() : '-'}</td>
        <td><a href="/api/reports/${c.id}/html" target="_blank" class="btn-card btn-review" style="text-decoration:none; display:inline-block;">Report</a></td>
      `;
      tbody.appendChild(row);
    });
  } catch (err) {
    console.error("Failed to load cases:", err);
  }
}
