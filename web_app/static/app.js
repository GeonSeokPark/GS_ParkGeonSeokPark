"use strict";

const byId = (id) => document.getElementById(id);
const photo = byId("photo");
const dropzone = byId("dropzone");
const start = byId("start");
const statusLabel = byId("status");
const progress = byId("progress");
const stageItems = [...document.querySelectorAll("#stages li")];
const labels = {
  align: "FFHQ 기준으로 얼굴을 정렬하는 중입니다.",
  reconstruct: "혼합 잠재벡터에서 사진을 재구성하는 중입니다.",
  refine: "사진에 맞춰 노이즈 맵을 보정하는 중입니다.",
  sweep: "워터마크 조합을 적용하고 점수를 비교하는 중입니다.",
};
const storageKey = "deep-protect-active-job";
const requestedJob = new URLSearchParams(location.search).get("job");
let job = /^[0-9a-f]{24}$/.test(requestedJob || "") ? requestedJob : sessionStorage.getItem(storageKey);
let state = null;
let busy = false;
let timer;
let schema = null;
let settingsFor = null;
const controls = new Map();
const presetButtons = {tuned: byId("preset-tuned"), paper: byId("preset-paper")};

function canStart() {
  return state?.status === "ready" || state?.status === "done" ||
    (state?.status === "failed" && state.aligned_available);
}

function message(text, error = false) {
  if (statusLabel.textContent !== text) statusLabel.textContent = text;
  statusLabel.classList.toggle("error", error);
}

function setBusy(value) {
  busy = value;
  photo.disabled = value;
  dropzone.setAttribute("aria-disabled", String(value));
  start.disabled = value || !schema || !canStart();
  byId("settings-fields").disabled = value || !schema;
  for (const button of Object.values(presetButtons)) button.disabled = value || !schema;
}

function jobUrl(path = "") {
  return "/api/jobs/" + job + path;
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "로컬 앱 요청이 실패했습니다.");
  return data;
}

function clearView() {
  clearTimeout(timer);
  state = null;
  byId("alignment").hidden = true;
  byId("result").hidden = true;
  byId("empty").hidden = false;
  byId("error-log").hidden = true;
  byId("progress-detail").hidden = true;
  byId("elapsed").hidden = true;
  stageItems.forEach((item) => item.classList.remove("active", "done", "skipped"));
}

async function upload(file) {
  if (!file || busy) return;
  if (file.size > 25 * 1024 * 1024 || file.size === 0) {
    message("25 MB 이하의 JPG 또는 PNG를 선택해 주세요.", true);
    return;
  }
  if (!/\.(png|jpe?g)$/i.test(file.name)) {
    message("JPG 또는 PNG 사진을 선택해 주세요.", true);
    return;
  }
  clearView();
  setBusy(true);
  byId("file-name").textContent = file.name;
  message("선택한 사진을 정렬 작업에 전달하는 중입니다.");
  try {
    const result = await api("/api/jobs", {
      method: "POST", body: file, headers: {"Content-Type": "application/octet-stream"},
    });
    job = result.id;
    sessionStorage.setItem(storageKey, job);
    history.replaceState(null, "", "/?job=" + job);
    await poll();
  } catch (error) {
    setBusy(false);
    message(error.message, true);
  }
}

function renderState(data) {
  state = data;
  if (schema && settingsFor !== data.id) {
    settingsFor = data.id;
    if (data.settings) applySettings(data.settings);
  }
  const running = data.status === "running" || data.status === "aligning";
  setBusy(running);
  const stage = data.stage;
  const current = stageItems.findIndex((item) => item.dataset.stage === stage);
  stageItems.forEach((item, index) => {
    const skipped = item.dataset.stage === "refine" && data.settings?.noise_steps === 0;
    item.classList.toggle("skipped", skipped);
    item.classList.toggle("active", index === current && running);
    item.classList.toggle("done", !skipped && (data.status === "done" || index < current || (index === 0 && data.status === "ready")));
  });
  byId("refine-label").textContent = data.settings?.noise_steps === 0 ? "노이즈 보정 생략" : "노이즈 보정";
  if (data.aligned_available || ["ready", "running", "done"].includes(data.status)) {
    const image = byId("aligned");
    if (image.getAttribute("src") !== jobUrl("/aligned.png")) image.src = jobUrl("/aligned.png");
    byId("alignment").hidden = false;
  }
  byId("progress-detail").hidden = data.status !== "running";
  byId("elapsed").hidden = !data.started;
  if (data.started) {
    const end = data.status === "running" ? Date.now() / 1000 : data.updated;
    const seconds = Math.max(0, Math.round(end - data.started));
    byId("elapsed").textContent = Math.floor(seconds / 60) + "분 " + (seconds % 60) + "초";
  }
  if (running) {
    message(labels[stage]);
    if (data.total) {
      progress.max = data.total;
      progress.value = Math.min(data.current, data.total);
      byId("progress-label").textContent = data.current + " / " + data.total;
    }
  } else if (data.status === "ready") {
    message("정렬된 얼굴을 확인한 다음 변환을 시작하세요.");
  } else if (data.status === "failed") {
    message(data.error, true);
    const log = byId("error-log");
    log.href = jobUrl("/run.log");
    log.hidden = false;
  } else if (data.status === "done") {
    message("변환 완료. 설정을 바꿔 같은 사진으로 다시 실행할 수 있습니다.");
  }
  start.firstChild.textContent = data.status === "done" || (data.status === "failed" && data.aligned_available)
    ? "같은 사진으로 다시 실행 " : "보호 이미지 만들기 ";
}

function poll() {
  const activeJob = job;
  return api(jobUrl()).then(async (data) => {
    if (job !== activeJob) return;
    renderState(data);
    if (data.status === "done") {
      const result = await api(jobUrl("/metrics.json"));
      if (job === activeJob) renderResult(result);
    } else if (["aligning", "running"].includes(data.status)) {
      timer = setTimeout(poll, 1400);
    }
  }).catch((error) => {
    if (job !== activeJob) return;
    message(error.message, true);
    if (busy) {
      timer = setTimeout(poll, 4000);
    }
  });
}

const score = (value, places = 3) => typeof value === "number" ? value.toFixed(places) : "∞";
const el = (tag, text, className) => {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
};

function readSettings() {
  return Object.fromEntries([...controls].map(([key, {spec, input}]) => [
    key, spec.kind === "range" ? Number(input.value) : spec.kind === "toggle" ? input.checked : input.value,
  ]));
}

function syncSettings() {
  const values = readSettings();
  for (const [key, {spec, input, output}] of controls) {
    if (!output) continue;
    const value = values[key];
    output.textContent = spec.levels ? spec.levels[value] : [value, spec.unit].filter((v) => v !== "").join(" ");
    input.setAttribute("aria-valuetext", output.textContent);
  }
  let active = null;
  for (const [name, button] of Object.entries(presetButtons)) {
    const matches = Object.keys(values).every((key) => values[key] === schema.presets[name][key]);
    button.setAttribute("aria-pressed", String(matches));
    if (matches) active = name;
  }
  byId("preset-name").textContent = active ? presetButtons[active].textContent : "사용자 설정";
}

function applySettings(values) {
  for (const [key, {spec, input}] of controls) {
    const value = values[key] ?? schema.presets.tuned[key];
    if (spec.kind === "toggle") input.checked = value;
    else input.value = String(value);
  }
  syncSettings();
}

function settingControl(key, spec) {
  const item = el("div", undefined, "setting");
  const header = el("div", undefined, "setting-header");
  const id = "setting-" + key;
  const helpId = id + "-help";
  const label = el("label", spec.label);
  label.htmlFor = id;
  const help = el("button", "?", "setting-help");
  help.type = "button";
  help.setAttribute("aria-label", spec.label + " 설명");
  help.setAttribute("aria-describedby", helpId);
  help.setAttribute("aria-expanded", "false");
  const tooltip = el("span", spec.help, "setting-tooltip");
  tooltip.id = helpId;
  tooltip.setAttribute("role", "tooltip");
  header.append(label, help);
  item.append(header, tooltip);

  const input = document.createElement(spec.kind === "select" ? "select" : "input");
  input.id = id;
  input.setAttribute("aria-describedby", helpId);
  let output;
  if (spec.kind === "range") {
    input.type = "range";
    Object.assign(input, {min: spec.min, max: spec.max, step: spec.step});
    output = el("output", undefined, "setting-value");
    output.setAttribute("for", id);
    item.append(output);
  } else if (spec.kind === "toggle") {
    input.type = "checkbox";
  } else {
    for (const [value, title] of Object.entries(spec.options)) {
      const option = el("option", title);
      option.value = value;
      input.append(option);
    }
  }
  help.addEventListener("click", () => {
    const open = help.getAttribute("aria-expanded") !== "true";
    document.querySelectorAll(".setting-help[aria-expanded='true']").forEach((button) => {
      button.setAttribute("aria-expanded", "false");
    });
    help.setAttribute("aria-expanded", String(open));
  });
  help.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      help.setAttribute("aria-expanded", "false");
      input.focus();
    }
  });
  input.addEventListener("input", syncSettings);
  controls.set(key, {spec, input, output});
  item.append(input);
  return item;
}

async function loadSettings() {
  const note = byId("settings-message");
  try {
    schema = await api("/api/settings");
    const parent = byId("settings-fields");
    for (const [group, title] of Object.entries(schema.groups)) {
      const grid = el("div", undefined, "setting-grid");
      for (const [key, spec] of Object.entries(schema.fields)) {
        if (spec.group === group) grid.append(settingControl(key, spec));
      }
      if (group === "main") {
        parent.append(grid);
      } else {
        const section = el("details");
        section.append(el("summary", title), grid);
        parent.append(section);
      }
    }
    applySettings(state?.settings || schema.presets.tuned);
    settingsFor = state?.id;
    note.textContent = "새 실행부터 적용됩니다. 각 ?에 커서를 올리거나 눌러 설명을 확인하세요.";
  } catch (_) {
    schema = null;
    note.textContent = "설정 API에 연결할 수 없습니다. 실행 창에서 서버를 다시 시작하고 이 페이지를 새로 고쳐 주세요.";
  }
  setBusy(busy);
}

function renderResult(result) {
  const candidates = [...result.trials].sort((a, b) => result.selected_id
    ? b.ssim - a.ssim
    : a.jpeg85_cosine - b.jpeg85_cosine || b.ssim - a.ssim);
  const selection = byId("selection-info");
  const metrics = byId("metrics");
  const rows = byId("candidate-rows");
  const buttons = new Map();
  byId("compare-source").src = jobUrl("/aligned.png");
  byId("download-json").href = jobUrl("/metrics.json?download=1");
  byId("candidate-summary").textContent = "워터마크 후보 " + result.trials.length + "개";
  byId("cutoff-note").textContent = "CosSim은 같은 로컬 ArcFace 모델의 점수입니다. JPEG 85의 " +
    score(result.jpeg85_cutoff, 2) + "은 이번 실행의 후보 선택 기준입니다.";
  const previous = byId("previous-result");
  previous.hidden = !state?.parent;
  if (state?.parent) previous.href = "/?job=" + state.parent;
  const first = candidates[0];
  if (!result.selected_id) {
    message(result.trials.length + "개 비교 완료. 자동 선택 기준을 통과한 후보가 없어 가장 낮은 CosSim 후보를 표시합니다.");
  }

  function select(row) {
    buttons.forEach((button, id) => button.setAttribute("aria-pressed", String(id === row.id)));
    const passed = row.jpeg85_cosine <= result.jpeg85_cutoff;
    selection.classList.toggle("warning", !passed);
    const note = row.id === result.selected_id ? "자동 선택" : !result.selected_id && row.id === first.id
      ? "최저 JPEG 85 CosSim 후보" : "후보 미리보기";
    selection.textContent = note + " · " + row.prompt + " · ε " + score(row.epsilon) + " · " +
      row.steps + "회 · CosSim " + score(row.jpeg85_cosine) +
      (passed ? " · 기준 충족" : " · 기준 " + score(result.jpeg85_cutoff, 2) + " 초과");
    const image = jobUrl(row.id === result.selected_id ? "/protected.png" : "/candidates/" + row.id + ".png");
    byId("compare-protected").src = image;
    byId("download-png").href = image + "?download=1";
    byId("download-png").download = row.id === result.selected_id ? "protected.png" : "protected-" + row.id + ".png";
    byId("download-label").textContent = row.id === result.selected_id ? "보호 PNG 다운로드" : "현재 후보 PNG 다운로드";
    const delta = byId("watermark-delta");
    delta.hidden = row.watermark_mae === undefined;
    if (!delta.hidden) delta.textContent = "워터마크만 적용한 픽셀 차이 · 평균 " +
      score(row.watermark_mae, 2) + " / 255 · 최대 " + row.watermark_max + " / 255";
    metrics.replaceChildren();
    for (const [name, value] of [
      ["MAE ↓", score(row.mae, 2)], ["SSIM ↑", score(row.ssim)],
      ["PSNR ↑", score(row.psnr_db, 2) + " dB"], ["LPIPS ↓", score(row.lpips)],
    ]) {
      const cell = el("div", undefined, "metric");
      cell.append(el("span", name, "metric-name"), el("strong", value, "metric-value"));
      metrics.append(cell);
    }
    const identity = byId("identity");
    const heading = el("p", "ArcFace CosSim ↓");
    const table = document.createElement("table");
    table.setAttribute("aria-label", "워터마크 전후 신원 유사도");
    const header = table.createTHead().insertRow();
    for (const name of ["비교 조건", "워터마크 전", "워터마크 후"]) {
      const th = el("th", name);
      th.scope = "col";
      header.append(th);
    }
    const body = table.createTBody();
    for (const [label, key] of [["PNG", "cosine"], ["JPEG 95", "jpeg95_cosine"], ["JPEG 85", "jpeg85_cosine"]]) {
      const tr = body.insertRow();
      for (const value of [label, score(result.pre_watermark[key]), score(row[key])]) tr.append(el("td", value));
    }
    identity.replaceChildren(heading, table);
  }

  rows.replaceChildren();
  byId("alternatives").open = !result.selected_id;
  for (const row of candidates) {
    const tr = rows.insertRow();
    for (const value of [row.prompt, score(row.epsilon), String(row.steps), score(row.ssim), score(row.jpeg85_cosine)]) {
      tr.append(el("td", value));
    }
    const button = el("button", row.id === result.selected_id ? "선택 결과" : "보기", "candidate");
    button.type = "button";
    button.setAttribute("aria-label", row.prompt + ", ε " + score(row.epsilon) + ", " + row.steps + "회 미리보기");
    button.addEventListener("click", () => select(row));
    buttons.set(row.id, button);
    tr.insertCell().append(button);
  }
  select(candidates.find((row) => row.id === result.selected_id) || first);
  byId("result").hidden = false;
  byId("empty").hidden = true;
}

photo.addEventListener("change", () => {
  const file = photo.files[0];
  photo.value = "";
  upload(file);
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    if (!busy) photo.click();
  }
});
document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => event.preventDefault());
dropzone.addEventListener("dragover", () => { if (!busy) dropzone.classList.add("dragging"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragging"));
dropzone.addEventListener("drop", (event) => {
  dropzone.classList.remove("dragging");
  upload(event.dataTransfer.files[0]);
});
start.addEventListener("click", async () => {
  if (busy || !job || !schema || !canStart()) return;
  const settings = readSettings();
  setBusy(true);
  try {
    const result = await api(jobUrl("/start"), {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(settings),
    });
    if (result.id !== job) {
      job = result.id;
      sessionStorage.setItem(storageKey, job);
      history.replaceState(null, "", "/?job=" + job);
      clearView();
    }
    await poll();
  } catch (error) {
    setBusy(false);
    message(error.message, true);
  }
});
byId("split").addEventListener("input", (event) => {
  byId("comparator").style.setProperty("--split", event.target.value + "%");
});

for (const [name, button] of Object.entries(presetButtons)) {
  button.addEventListener("click", () => { if (!busy && schema) applySettings(schema.presets[name]); });
}
document.addEventListener("click", (event) => {
  if (!event.target.closest(".setting-help")) {
    document.querySelectorAll(".setting-help[aria-expanded='true']").forEach((button) => {
      button.setAttribute("aria-expanded", "false");
    });
  }
});

setBusy(false);
loadSettings();
if (job) poll();
