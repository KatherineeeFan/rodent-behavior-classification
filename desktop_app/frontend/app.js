const CLASS_HEX = {
  body_grooming: "#d97742",
  face_grooming: "#c9528a",
  freezing: "#3b7dc4",
  exploratory: "#4f9d4f",
};

let state = {
  videoPath: null,
  segments: [],
  labels: {},
  editIndex: null,
};

const el = (id) => document.getElementById(id);

function setStatus(text) {
  el("status").textContent = text;
}

function renderLegend(classes, labels) {
  el("legend").innerHTML = classes.map(c => `
    <span><span class="dot" style="background:${CLASS_HEX[c]}"></span>${labels[c] || c}</span>
  `).join("");
}

function totalDuration() {
  const v = el("video-player");
  if (v.duration && !isNaN(v.duration)) return v.duration;
  return state.segments.length ? state.segments[state.segments.length - 1].end : 1;
}

function renderTimeline() {
  const container = el("timeline");
  container.innerHTML = "";
  const dur = totalDuration();
  state.segments.forEach((seg, i) => {
    const width = ((seg.end - seg.start) / dur) * 100;
    const div = document.createElement("div");
    div.className = `seg seg-${seg.label}`;
    div.style.width = width + "%";
    div.title = `${seg.start}s - ${seg.end}s : ${state.labels[seg.label] || seg.label} (${Math.round(seg.confidence * 100)}%)`;
    div.onclick = () => openEditModal(i);
    container.appendChild(div);
  });
}

function statusSuffix(seg) {
  if (seg.corrected) return " · Corrected";
  if (seg.reviewed) return " · Confirmed";
  return " · Not reviewed";
}

function renderSegmentList() {
  const list = el("segment-list");
  list.innerHTML = state.segments.map((seg, i) => `
    <div class="segment-row">
      <span>${seg.start}s - ${seg.end}s</span>
      <span class="tag" style="background:${CLASS_HEX[seg.label]}">${state.labels[seg.label] || seg.label}${statusSuffix(seg)}</span>
      <span class="conf">Confidence ${Math.round(seg.confidence * 100)}%</span>
      <button data-jump="${i}">Jump</button>
      <button data-confirm="${i}">Confirm Correct</button>
      <button data-edit="${i}">Correct</button>
    </div>
  `).join("");

  list.querySelectorAll("[data-jump]").forEach(btn => {
    btn.onclick = () => jumpTo(state.segments[parseInt(btn.dataset.jump)].start);
  });
  list.querySelectorAll("[data-edit]").forEach(btn => {
    btn.onclick = () => openEditModal(parseInt(btn.dataset.edit));
  });
  list.querySelectorAll("[data-confirm]").forEach(btn => {
    btn.onclick = () => confirmCorrect(parseInt(btn.dataset.confirm));
  });
}

async function confirmCorrect(index) {
  const seg = state.segments[index];
  seg.reviewed = true;
  renderSegmentList();
  await autosave();
  setStatus(`Confirmed segment ${seg.start}s-${seg.end}s as correct`);
}

function jumpTo(t) {
  el("video-player").currentTime = t;
  el("video-player").play();
}

function openEditModal(index) {
  state.editIndex = index;
  const seg = state.segments[index];
  el("edit-range").textContent = `Time range: ${seg.start}s - ${seg.end}s. Current label: ${state.labels[seg.label] || seg.label}`;
  const select = el("edit-select");
  select.innerHTML = Object.keys(CLASS_HEX).map(c =>
    `<option value="${c}" ${c === seg.label ? "selected" : ""}>${state.labels[c] || c}</option>`
  ).join("");
  el("edit-modal").classList.remove("hidden");
}

el("edit-cancel").onclick = () => el("edit-modal").classList.add("hidden");

el("edit-confirm").onclick = async () => {
  const newLabel = el("edit-select").value;
  const seg = state.segments[state.editIndex];
  seg.label = newLabel;
  seg.corrected = newLabel !== seg.orig_label;
  seg.reviewed = true;
  el("edit-modal").classList.add("hidden");
  renderTimeline();
  renderSegmentList();
  await autosave();
};

async function autosave() {
  if (!state.videoPath) return;
  await window.pywebview.api.save_annotation(state.videoPath, state.segments);
  setStatus("Correction auto-saved");
}

el("btn-import").onclick = async () => {
  const path = await window.pywebview.api.select_video();
  if (!path) return;

  state.videoPath = path;

  const urlResult = await window.pywebview.api.get_video_url(path);
  if (urlResult.error) {
    setStatus("Failed to load video: " + urlResult.error);
    return;
  }
  el("video-player").src = urlResult.url;

  setStatus("Running model inference, please wait (longer videos take longer)...");
  el("progress-wrap").style.display = "block";
  el("progress-bar").style.width = "20%";

  const result = await window.pywebview.api.run_inference(path);
  el("progress-wrap").style.display = "none";

  if (result.error) {
    setStatus("Error: " + result.error);
    return;
  }

  state.segments = result.segments.map(seg => ({
    ...seg,
    orig_label: seg.label,
    reviewed: false,
    corrected: false,
  }));
  state.labels = result.labels;
  renderLegend(result.classes, result.labels);
  renderTimeline();
  renderSegmentList();
  setStatus(`Recognition complete: ${result.segments.length} behavior segments found. Click a colored block or the "Correct" button in the list to change a label.`);
};

el("btn-settings").onclick = async () => {
  const cfg = await window.pywebview.api.set_shared_folder();
  if (cfg.shared_export_folder) {
    setStatus("Shared folder set to: " + cfg.shared_export_folder);
  }
};

el("btn-export").onclick = async () => {
  if (!state.videoPath) {
    setStatus("Import a video first before exporting.");
    return;
  }
  setStatus("Checking for annotation changes and exporting this video, please wait...");
  const result = await window.pywebview.api.export_clips(state.videoPath);
  if (result.error) {
    setStatus("Export failed: " + result.error);
    return;
  }
  const action = result.updated ? "Re-cut" : "No changes since last export, reused existing clips for";
  setStatus(`${action} ${result.video_name}: ${result.clip_count} clip(s) in ${result.manifest_name}.`);
};

el("video-player").addEventListener("loadedmetadata", renderTimeline);
