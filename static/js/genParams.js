// static/js/genParams.js
//
// Image / Video generation-parameters popup (ComfyUI-backed). See
// data/studio/PLAN-IMAGE-VIDEO-TABS.md for the design — this implements plan
// steps 6-7: the popup UI (§7 parameter schema), workflow selection (§4),
// global-defaults-overridden-per-session persistence (§6.5), the in-bubble
// progress bar (§5), and the /api/comfy/* send path (§6.3).
//
// Copies the `Z-Image ^` model-picker pattern wholesale (open/close/roll-up
// animation via the shared .model-picker-menu class — see modelPicker.js).

import Storage from './storage.js';
import uiModule from './ui.js';
import sessionModule from './sessions.js';
import chatRenderer from './chatRenderer.js?v=20260722emailfastindex1';
import { formatElapsed } from './research/jobs.js?v=20260630researchthumb';
import { bindMenuDismiss } from './escMenuStack.js';

const { esc, showToast, scrollHistory } = uiModule;

let API_BASE = '';
let _inited = false;

// ── Baseline parameter schemas (plan §7) — the floor every effective value
// falls back to. Global defaults (Storage.loadGenDefaults()) and per-session
// overrides (Storage.loadGenSessions()) are both PARTIAL objects layered on
// top of these, so adding a new field later never orphans an old saved chat.
const IMAGE_BASELINE = Object.freeze({
  workflow: 'Custom',
  mode: 'txt2img',            // 'txt2img' | 'img2img'
  input_image: null,          // {id, name, previewUrl}
  loras: null,                // null = "use the registry's LIVE defaults"; else {key: weight}
  size_preset: '1216x672',
  width: 1216,
  height: 672,
  denoise: 0.6,
  batch: 1,
  negative_prompt: 'text, letters, lettering, logo, watermark, signature',
  seed: 0,
  randomize_seed: true,
  steps: 30,
  cfg: 4.5,
  sampler: 'res_multistep',
  scheduler: 'simple',
  shift: 3,
  unet: '',
  clip: '',
  vae: '',
});
const VIDEO_BASELINE = Object.freeze({
  workflow: 'Custom',
  input_image: null,
  frames: 81,
  fps: 16,
  size_preset: '720x720',
  width: 720,
  height: 720,
  negative_prompt: 'text, letters, lettering, logo, watermark, signature',
  seed: 0,
  randomize_seed: true,
  steps: 4,
  cfg: 1,
  sampler: 'euler',
  scheduler: 'simple',
  lora_high_weight: 1.0,
  lora_low_weight: 1.0,
});
const SIZE_PRESETS_IMAGE = ['1216x672', '1152x864', '1024x1024', 'custom'];
const SIZE_PRESETS_VIDEO = ['720x720', 'custom'];

function _baseline(kind) { return kind === 'video' ? VIDEO_BASELINE : IMAGE_BASELINE; }

// ── Persistence ──
// Editing scope: which layer field edits in the currently-open popup write
// to. Defaults to 'session' (tweak this chat only); the small scope switch
// in the popup header lets the user explicitly promote to 'default' (the
// baseline every NEW chat starts from).
let _editScope = 'session';
let _currentSessionId = null;
let _currentKind = 'image';
let _currentTab = 'simple';

function _loadDefaults() { return Storage.loadGenDefaults() || {}; }
function _saveDefaults(d) { Storage.saveGenDefaults(d); }
function _loadSessions() { return Storage.loadGenSessions() || {}; }
function _saveSessions(s) { Storage.saveGenSessions(s); }

function _effective(kind, sessionId) {
  const defaults = _loadDefaults();
  const sessions = _loadSessions();
  const sessionOverride = (sessionId && sessions[sessionId] && sessions[sessionId][kind]) || {};
  const globalOverride = defaults[kind] || {};
  return { ..._baseline(kind), ...globalOverride, ...sessionOverride };
}

function _heartbeatEnabled() {
  const d = _loadDefaults();
  return !!d._heartbeat;
}
function _setHeartbeat(on) {
  const d = _loadDefaults();
  d._heartbeat = !!on;
  _saveDefaults(d);
}

// Writes ONE field of the given kind's params into whichever layer
// `_editScope` currently points at, then re-renders.
function _setField(kind, key, value) {
  if (_editScope === 'default') {
    const d = _loadDefaults();
    d[kind] = { ...(d[kind] || {}), [key]: value };
    _saveDefaults(d);
  } else {
    const sessions = _loadSessions();
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    if (!sid) { // No session yet (pre-first-message) — fall back to defaults.
      const d = _loadDefaults();
      d[kind] = { ...(d[kind] || {}), [key]: value };
      _saveDefaults(d);
      return;
    }
    sessions[sid] = { ...(sessions[sid] || {}) };
    sessions[sid][kind] = { ...(sessions[sid][kind] || {}), [key]: value };
    _saveSessions(sessions);
  }
}

function _setFields(kind, patch) {
  Object.entries(patch).forEach(([k, v]) => _setField(kind, k, v));
}

// ── Options / workflows (fetched from the backend the other agent is
// building — cached briefly, tolerant of it not existing yet). ──
let _optionsCache = null;
let _optionsFetchedAt = 0;
let _workflowsCache = null;
let _workflowsFetchedAt = 0;
const _CACHE_MS = 60000;

async function _fetchOptions(force) {
  if (!force && _optionsCache && Date.now() - _optionsFetchedAt < _CACHE_MS) return _optionsCache;
  try {
    const res = await fetch(`${API_BASE}/api/comfy/options`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _optionsCache = await res.json();
  } catch (e) {
    console.warn('[genParams] /api/comfy/options unavailable:', e.message || e);
    _optionsCache = { loras: [], loras_all: [], unets: [], vaes: [], clips: [], samplers: [], schedulers: [] };
  }
  _optionsFetchedAt = Date.now();
  return _optionsCache;
}

async function _fetchWorkflows(force) {
  if (!force && _workflowsCache && Date.now() - _workflowsFetchedAt < _CACHE_MS) return _workflowsCache;
  try {
    const res = await fetch(`${API_BASE}/api/comfy/workflows`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _workflowsCache = await res.json();
  } catch (e) {
    console.warn('[genParams] /api/comfy/workflows unavailable:', e.message || e);
    _workflowsCache = { workflows: [] };
  }
  _workflowsFetchedAt = Date.now();
  return _workflowsCache;
}

async function _applyWorkflow(kind, name) {
  _setField(kind, 'workflow', name);
  if (!name || name === 'Custom') { _renderPanel(); return; }
  try {
    const res = await fetch(`${API_BASE}/api/comfy/workflows/${encodeURIComponent(name)}/params`, { credentials: 'same-origin' });
    const data = await res.json();
    if (!data || data.ok === false) {
      showToast('Could not read that workflow: ' + (data && data.error || 'unknown error'));
      _renderPanel();
      return;
    }
    // Seed every recognized field — but every field stays editable afterward
    // (plan §4a). Only apply keys that exist on our schema so a stray
    // introspected field can't corrupt the form.
    const schema = _baseline(kind);
    const patch = {};
    Object.keys(data.params || {}).forEach(k => {
      if (Object.prototype.hasOwnProperty.call(schema, k)) patch[k] = data.params[k];
    });
    _setFields(kind, patch);
    if (Array.isArray(data.unsupported) && data.unsupported.length) {
      showToast('Workflow has ' + data.unsupported.length + ' node(s) this panel doesn\'t expose yet — left as saved.');
    }
  } catch (e) {
    showToast('Failed to load workflow params: ' + (e.message || 'network error'));
  }
  _renderPanel();
}

// ── Input-image upload (start frame for video, source for img2img) ──
// Local preview is a blob: URL (same trick fileHandler.js uses) so the
// thumbnail shows instantly; the real upload runs in the background.
//
// Uploads go to POST /api/comfy/upload, NOT the general-purpose /api/upload.
// This matters: ComfyUI's LoadImage node resolves filenames against ComfyUI's
// OWN input/ directory, so the file has to land there. /api/comfy/upload
// forwards the bytes to ComfyUI's native /upload/image and returns the name
// ComfyUI stored it under — verified live 2026-08-03, response shape
// {filename, subfolder, type} (ComfyUI itself returns the key as "name"; the
// Odysseus route translates it). That returned filename is what LoadImage
// needs, so it becomes params.input_image directly.
//
// The other accepted form is a data/generated_images/ filename from the
// gallery picker, which the backend's _resolve_input_image() detects via
// GENERATED_IMAGE_RE and forwards the same way. Both paths converge.
async function _uploadInputImage(kind, file) {
  if (!file) return;
  const previewUrl = URL.createObjectURL(file);
  _setField(kind, 'input_image', { comfy_filename: null, name: file.name, previewUrl, uploading: true });
  _renderPanel();
  try {
    const fd = new FormData();
    fd.append('file', file, file.name || 'input.png');
    const res = await fetch(`${API_BASE}/api/comfy/upload`, { method: 'POST', body: fd, credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    const name = data.filename || data.name;
    if (!name) throw new Error('no filename in response');
    // Non-default subfolders must be PREFIXED onto the name, not sent
    // separately — ComfyUI's folder_paths.annotated_filepath() convention.
    const ref = data.subfolder ? `${data.subfolder}/${name}` : name;
    _setField(kind, 'input_image', { comfy_filename: ref, name: file.name, previewUrl, uploading: false });
  } catch (e) {
    showToast('Image upload failed: ' + (e.message || 'network error'));
    _setField(kind, 'input_image', null);
  }
  _renderPanel();
}

function _clearInputImage(kind) {
  const sid = _currentSessionId || sessionModule.getCurrentSessionId();
  const cur = _effective(kind, sid).input_image;
  if (cur && cur.previewUrl) { try { URL.revokeObjectURL(cur.previewUrl); } catch (_) {} }
  _setField(kind, 'input_image', null);
  _renderPanel();
}

// ── DOM wiring (module load time — index.html's elements already exist by
// the time this module evaluates, since app.js module scripts defer until
// the DOM is parsed) ──
let _wrap, _btn, _menu, _tabsEl, _bodyEl;

function _wireDom() {
  _wrap = document.getElementById('gen-params-wrap');
  _btn = document.getElementById('gen-params-btn');
  _menu = document.getElementById('gen-params-menu');
  _tabsEl = document.getElementById('gen-params-tabs');
  _bodyEl = document.getElementById('gen-params-body');
  if (!_wrap || !_btn || !_menu || !_bodyEl) return;

  _btn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (_menu.classList.contains('hidden') || _menu.classList.contains('closing')) {
      _open();
    } else {
      _close();
    }
  });
  _btn.addEventListener('pointerdown', (e) => e.stopPropagation());

  if (_tabsEl) {
    _tabsEl.addEventListener('click', (e) => {
      const tabBtn = e.target.closest('[data-gen-tab]');
      if (!tabBtn) return;
      _currentTab = tabBtn.dataset.genTab === 'advanced' ? 'advanced' : 'simple';
      _tabsEl.querySelectorAll('.gen-params-tab').forEach(b => b.classList.toggle('active', b === tabBtn));
      _renderPanel();
    });
  }

  document.addEventListener('click', (e) => {
    if (_menu.classList.contains('hidden')) return;
    if (_wrap.contains(e.target)) return;
    // The gallery picker (DEFECT 2) is deliberately appended to document.body
    // rather than nested inside _wrap (so it can render as a full-viewport
    // modal) — without this check, clicking a thumbnail inside it would
    // bubble up and be misread as "clicked outside the params popup",
    // closing the popup underneath it too.
    if (_galleryModal && _galleryModal.contains(e.target)) return;
    _close();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !_menu.classList.contains('hidden')) _close();
  });

  // Delegated listeners for the dynamically-rendered body — avoids re-wiring
  // on every re-render (cookbookServe.js's data-field convention, mirrored).
  _bodyEl.addEventListener('click', _onBodyClick);
  _bodyEl.addEventListener('change', _onBodyChange);
  _bodyEl.addEventListener('input', _onBodyInput);
}

function _open() {
  _menu.classList.remove('closing', 'hidden');
  _btn.classList.add('active');
  Promise.all([_fetchOptions(), _fetchWorkflows()]).then(() => _renderPanel());
  _renderPanel();
}

function _close() {
  if (_menu.classList.contains('hidden')) return;
  _menu.classList.add('closing');
  _btn.classList.remove('active');
  const onDone = () => {
    _menu.removeEventListener('animationend', onDone);
    _menu.classList.remove('closing');
    _menu.classList.add('hidden');
  };
  _menu.addEventListener('animationend', onDone, { once: true });
  setTimeout(() => { if (!_menu.classList.contains('hidden')) onDone(); }, 200);
}

// ── Public API ──

export function init(apiBase) {
  API_BASE = apiBase || '';
  if (_inited) return;
  _inited = true;
  _wireDom();
  _currentSessionId = sessionModule.getCurrentSessionId ? sessionModule.getCurrentSessionId() : null;
  // app.js's initModeToggle() runs (and calls setMode(currentMode), which
  // calls onModeChange) earlier in startOdysseusApp() than this init() —
  // onModeChange no-ops before _wireDom() above has run. Re-derive the
  // persisted mode and re-sync once now, so a page load that starts in
  // Image/Video mode still shows the params trigger button.
  try {
    const mode = (Storage.loadToggleState() || {}).mode;
    if (mode) onModeChange(mode);
  } catch (_) {}
  // DEFECT 3: a page load (not just a same-tab session switch) can also land
  // on a session with a job still running server-side (e.g. this tab was
  // reloaded, or a job was started in another tab) — reconnect it the same
  // way onSessionSwitch() does.
  try { _reattachSession(_currentSessionId); } catch (_) {}
}

export function onModeChange(mode) {
  if (!_wrap) return; // DOM not wired (index.html markup missing) — no-op
  const isGen = mode === 'image' || mode === 'video';
  _wrap.style.display = isGen ? '' : 'none';
  if (isGen) {
    if (_currentKind !== mode) {
      _currentKind = mode;
      _currentTab = 'simple';
      if (_tabsEl) {
        _tabsEl.querySelectorAll('.gen-params-tab').forEach(b => b.classList.toggle('active', b.dataset.genTab === 'simple'));
      }
      if (!_menu.classList.contains('hidden')) _renderPanel();
    }
    _fetchOptions().catch(() => {});
    _fetchWorkflows().catch(() => {});
  } else {
    _close();
  }
}

export function onSessionSwitch(sessionId) {
  _currentSessionId = sessionId;
  if (_menu && !_menu.classList.contains('hidden')) _renderPanel();
  // DEFECT 3: reconnect the just-selected session to its in-flight job (if
  // any) — see _reattachSession()'s docstring for why this doesn't need to
  // wait for the async chat-history rebuild that follows this same hook.
  try { _reattachSession(sessionId); } catch (_) {}
}

export function isBusy(sessionId) {
  const sid = sessionId || _currentSessionId;
  return _activeJobs.has(sid);
}

// ── Panel rendering ──

function _sizeOptionsHtml(kind, current) {
  const presets = kind === 'video' ? SIZE_PRESETS_VIDEO : SIZE_PRESETS_IMAGE;
  return presets.map(p => `<option value="${p}"${p === current ? ' selected' : ''}>${p === 'custom' ? 'Custom' : p.replace('x', ' × ')}</option>`).join('');
}

function _selectHtml(list, current, allowEmpty) {
  let html = allowEmpty ? `<option value=""${!current ? ' selected' : ''}>(workflow default)</option>` : '';
  (list || []).forEach(v => { html += `<option value="${esc(v)}"${v === current ? ' selected' : ''}>${esc(v)}</option>`; });
  return html;
}

function _badgeClass(status) {
  const s = String(status || '').toLowerCase();
  return 'gen-params-badge-' + (s || 'keep');
}

// DEFECT 1: a curated registry row is only offered as a selectable default
// once the backend has confirmed (via /api/comfy/options's live /object_info
// resolution — see src/comfy_graphs.py's resolve_lora_entries()) that
// ComfyUI currently recognizes SOME spelling of its comfy_name.
// `available` is only ever `false` when the backend actually checked and
// found no match; treat a missing key (options not loaded yet, or the
// degraded-fallback response) as available so nothing is wrongly greyed out
// before the real answer has arrived.
function _isLoraAvailable(entry) {
  return !entry || entry.available !== false;
}

function _liveLoraDefaults(relevant) {
  return Object.fromEntries(
    relevant.filter(l => l.status === 'LIVE' && _isLoraAvailable(l)).map(l => [l.key, l.default_weight ?? 1.0])
  );
}

function _loraRowsHtml(kind, effective) {
  const opts = _optionsCache || { loras: [] };
  const relevant = (opts.loras || []); // registry already scopes style|character; show all, both slots
  const selected = effective.loras === null ? _liveLoraDefaults(relevant) : effective.loras;
  if (!relevant.length) {
    return `<div class="gen-params-hint">No LoRA registry loaded yet (needs /api/comfy/options from the backend).</div>`;
  }
  // Mode banner. With no STYLE LoRA checked the backend skips the style
  // trigger entirely, so the render is plain base-model output. Say so
  // explicitly -- otherwise "why does this still look like anime?" (when a
  // style LoRA is silently on) and "why is my prompt being edited?" (when the
  // trigger was applied unconditionally) are both invisible from the UI.
  const anyStyle = relevant.some(l =>
    (l.kind || '').toLowerCase() === 'style' &&
    !_isLoraAvailable(l) === false &&
    Object.prototype.hasOwnProperty.call(selected, l.key));
  const banner = anyStyle
    ? `<div class="gen-params-hint">Studio mode &mdash; the style trigger is added to your prompt automatically.</div>`
    : `<div class="gen-params-hint"><strong>General mode</strong> &mdash; no style LoRA selected, so your prompt is sent to the plain base model unchanged. This is the equivalent of Z-Image-General.</div>`;
  return banner + relevant.map(l => {
    const unavailable = !_isLoraAvailable(l);
    const checked = !unavailable && Object.prototype.hasOwnProperty.call(selected, l.key);
    const weight = checked ? selected[l.key] : (l.default_weight ?? 1.0);
    const unavailableTitle = 'Not available on the ComfyUI server yet — it needs a restart to pick up new LoRA files.';
    return `<div class="gen-params-lora-row${unavailable ? ' gen-params-lora-unavailable' : ''}" data-lora-key="${esc(l.key)}"${unavailable ? ` title="${esc(unavailableTitle)}"` : ''}>
      <input type="checkbox" data-lora-toggle="${esc(l.key)}" ${checked ? 'checked' : ''} ${unavailable ? 'disabled' : ''}>
      <span class="gen-params-lora-name" title="${esc(unavailable ? unavailableTitle : (l.note || ''))}">${esc(l.name || l.key)}</span>
      ${l.status ? `<span class="gen-params-badge ${_badgeClass(l.status)}">${esc(l.status)}</span>` : ''}
      ${unavailable ? `<span class="gen-params-badge gen-params-badge-unavailable" title="${esc(unavailableTitle)}">unavailable</span>` : ''}
      <input type="range" min="0" max="1.5" step="0.05" value="${Number(weight)}" data-lora-weight="${esc(l.key)}" ${checked ? '' : 'disabled'}>
      <span class="gen-params-lora-weight">${Number(weight).toFixed(2)}</span>
    </div>`;
  }).join('');
}

function _inputImageHtml(kind, effective, label) {
  const img = effective.input_image;
  return `
    <div class="gen-params-imgrow">
      <button type="button" class="gen-params-btn" data-action="upload-input-image">${img ? 'Replace' : 'Upload'} ${esc(label)}</button>
      <button type="button" class="gen-params-btn" data-action="pick-gallery-image">Pick from Gallery</button>
      <span class="gen-params-hint">${img ? esc(img.name) + (img.uploading ? ' (uploading…)' : '') : 'None selected'}</span>
      ${img ? `<button type="button" class="gen-params-btn" data-action="clear-input-image">Clear</button>` : ''}
    </div>
    ${img && img.previewUrl ? `<img class="gen-params-thumb" src="${img.previewUrl}" alt="">` : ''}
    <input type="file" accept="image/*" data-role="gen-params-file-input" style="display:none">
  `;
}

// ── DEFECT 2: "Pick from Gallery" — the backend already accepts a gallery
// filename directly (routes/comfy_routes.py's _resolve_input_image() detects
// GENERATED_IMAGE_RE and forwards those bytes to ComfyUI transparently, plan
// §14.1); this wires the missing frontend picker onto that existing bridge.
// Reuses GET /api/gallery/library (static/js/gallery.js's own _fetchLibrary)
// rather than inventing a second listing endpoint.
let _galleryModal = null;   // the overlay element, while a picker is open
let _galleryModalClose = null;

function _isVideoFilename(name) {
  return /\.(mp4|mov|webm|mkv|m4v)$/i.test(name || '');
}

async function _openGalleryPicker(kind) {
  if (_galleryModalClose) { _galleryModalClose(); }

  const overlay = document.createElement('div');
  overlay.className = 'gen-params-gallery-overlay';
  overlay.innerHTML = `
    <div class="gen-params-gallery-modal">
      <div class="gen-params-gallery-modal-header">
        <span>Pick from Gallery</span>
        <button type="button" class="gen-params-gallery-close" aria-label="Close">&times;</button>
      </div>
      <div class="gen-params-gallery-modal-body">
        <div class="gen-params-hint">Loading…</div>
      </div>
    </div>`;
  document.body.appendChild(overlay);
  _galleryModal = overlay;

  const panel = overlay.querySelector('.gen-params-gallery-modal');
  const close = bindMenuDismiss(panel, () => {
    overlay.remove();
    if (_galleryModal === overlay) _galleryModal = null;
    _galleryModalClose = null;
  });
  _galleryModalClose = close;
  overlay.querySelector('.gen-params-gallery-close').addEventListener('click', close);

  const bodyEl = overlay.querySelector('.gen-params-gallery-modal-body');
  try {
    const res = await fetch(`${API_BASE}/api/gallery/library?sort=recent&limit=48`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    // A video can't be an img2img source or a Wan start frame — only offer
    // stills, for both the Image and Video tab pickers.
    const items = (data.items || []).filter(it => it && it.url && !_isVideoFilename(it.filename || it.url));
    if (!items.length) {
      bodyEl.innerHTML = `<div class="gen-params-hint">No images in the Gallery yet.</div>`;
      return;
    }
    bodyEl.innerHTML = `<div class="gen-params-gallery-grid"></div>`;
    const grid = bodyEl.querySelector('.gen-params-gallery-grid');
    items.forEach(it => {
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'gen-params-gallery-item';
      cell.title = it.prompt || it.filename || '';
      cell.innerHTML = `<img src="${esc(it.url)}" alt="" loading="lazy">`;
      cell.addEventListener('click', () => {
        _setField(kind, 'input_image', {
          id: null, gallery_filename: it.filename, name: it.filename, previewUrl: it.url, uploading: false,
        });
        close();
        _renderPanel();
      });
      grid.appendChild(cell);
    });
  } catch (e) {
    bodyEl.innerHTML = `<div class="gen-params-hint">Could not load the Gallery: ${esc(e.message || 'network error')}</div>`;
  }
}

function _renderImagePanel(effective, advanced) {
  const opts = _optionsCache || {};
  const wf = (_workflowsCache && _workflowsCache.workflows || []).filter(w => w.kind === 'image' || w.kind === 'unknown');
  let html = '';
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Workflow</span>
    <select class="settings-select" data-field="workflow">
      <option value="Custom"${effective.workflow === 'Custom' ? ' selected' : ''}>Custom (defaults)</option>
      ${wf.map(w => `<option value="${esc(w.name)}"${effective.workflow === w.name ? ' selected' : ''}>${esc(w.name)}</option>`).join('')}
    </select></div>`;

  if (!advanced) {
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Mode</span>
      <div class="gen-params-tabs" style="margin:0;border:0;padding:0;flex:1;">
        <button type="button" class="gen-params-tab${effective.mode !== 'img2img' ? ' active' : ''}" data-action="set-mode" data-mode-value="txt2img">txt2img</button>
        <button type="button" class="gen-params-tab${effective.mode === 'img2img' ? ' active' : ''}" data-action="set-mode" data-mode-value="img2img">img2img</button>
      </div></div>`;
    if (effective.mode === 'img2img') {
      html += `<div class="gen-params-section-title">Input image</div>`;
      html += _inputImageHtml('image', effective, 'source image');
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Denoise</span>
        <input type="range" min="0" max="1" step="0.05" value="${effective.denoise}" data-field="denoise" style="flex:1;">
        <span class="gen-params-lora-weight">${Number(effective.denoise).toFixed(2)}</span></div>`;
    }
    html += `<div class="gen-params-section-title">LoRAs <span class="gen-params-hint">(defaults to the two LIVE entries)</span></div>`;
    html += _loraRowsHtml('image', effective);
    html += `<div class="gen-params-section-title">Size</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Preset</span>
      <select class="settings-select" data-field="size_preset">${_sizeOptionsHtml('image', effective.size_preset)}</select></div>`;
    if (effective.size_preset === 'custom') {
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Width</span><input type="number" class="settings-select" min="64" step="8" value="${effective.width}" data-field="width"></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Height</span><input type="number" class="settings-select" min="64" step="8" value="${effective.height}" data-field="height"></div>`;
    }
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Batch</span><input type="number" class="settings-select" min="1" max="8" step="1" value="${effective.batch}" data-field="batch"></div>`;
  } else {
    html += `<div class="gen-params-row settings-row" style="align-items:flex-start;"><span class="settings-label">Negative</span><textarea class="settings-select" rows="2" data-field="negative_prompt">${esc(effective.negative_prompt)}</textarea></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Seed</span>
      <input type="number" class="settings-select" value="${effective.seed}" data-field="seed" ${effective.randomize_seed ? 'disabled' : ''}>
      <label style="display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;"><input type="checkbox" data-field="randomize_seed" ${effective.randomize_seed ? 'checked' : ''}> Randomize</label></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Steps</span><input type="number" class="settings-select" min="1" max="150" value="${effective.steps}" data-field="steps"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">CFG</span><input type="number" class="settings-select" min="0" max="30" step="0.1" value="${effective.cfg}" data-field="cfg"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Sampler</span><select class="settings-select" data-field="sampler">${_selectHtml(opts.samplers, effective.sampler)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Scheduler</span><select class="settings-select" data-field="scheduler">${_selectHtml(opts.schedulers, effective.scheduler)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Shift</span><input type="number" class="settings-select" min="0" max="20" step="0.1" value="${effective.shift}" data-field="shift"></div>`;
    html += `<div class="gen-params-section-title">Model overrides</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">UNET</span><select class="settings-select" data-field="unet">${_selectHtml(opts.unets, effective.unet, true)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">CLIP</span><select class="settings-select" data-field="clip">${_selectHtml(opts.clips, effective.clip, true)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">VAE</span><select class="settings-select" data-field="vae">${_selectHtml(opts.vaes, effective.vae, true)}</select></div>`;
    html += _heartbeatRowHtml();
  }
  return html;
}

function _renderVideoPanel(effective, advanced) {
  const opts = _optionsCache || {};
  const wf = (_workflowsCache && _workflowsCache.workflows || []).filter(w => w.kind === 'video' || w.kind === 'unknown');
  let html = '';
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Workflow</span>
    <select class="settings-select" data-field="workflow">
      <option value="Custom"${effective.workflow === 'Custom' ? ' selected' : ''}>Custom (defaults)</option>
      ${wf.map(w => `<option value="${esc(w.name)}"${effective.workflow === w.name ? ' selected' : ''}>${esc(w.name)}</option>`).join('')}
    </select></div>`;

  if (!advanced) {
    html += `<div class="gen-params-section-title">Start frame</div>`;
    html += _inputImageHtml('video', effective, 'start frame');
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Frames</span><input type="number" class="settings-select" min="9" max="241" value="${effective.frames}" data-field="frames"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">FPS</span><input type="number" class="settings-select" min="1" max="60" value="${effective.fps}" data-field="fps"></div>`;
    html += `<div class="gen-params-section-title">Size</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Preset</span>
      <select class="settings-select" data-field="size_preset">${_sizeOptionsHtml('video', effective.size_preset)}</select></div>`;
    if (effective.size_preset === 'custom') {
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Width</span><input type="number" class="settings-select" min="64" step="8" value="${effective.width}" data-field="width"></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Height</span><input type="number" class="settings-select" min="64" step="8" value="${effective.height}" data-field="height"></div>`;
    }
    html += `<div class="gen-params-hint">Video has no knowledge of the studio character/style LoRAs — it animates an already-on-model start frame, it does not generate style.</div>`;
  } else {
    html += `<div class="gen-params-row settings-row" style="align-items:flex-start;"><span class="settings-label">Negative</span><textarea class="settings-select" rows="2" data-field="negative_prompt">${esc(effective.negative_prompt)}</textarea></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Seed</span>
      <input type="number" class="settings-select" value="${effective.seed}" data-field="seed" ${effective.randomize_seed ? 'disabled' : ''}>
      <label style="display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;"><input type="checkbox" data-field="randomize_seed" ${effective.randomize_seed ? 'checked' : ''}> Randomize</label></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Total steps</span><input type="number" class="settings-select" min="1" max="40" value="${effective.steps}" data-field="steps"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">CFG</span><input type="number" class="settings-select" min="0" max="20" step="0.1" value="${effective.cfg}" data-field="cfg"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Sampler</span><select class="settings-select" data-field="sampler">${_selectHtml(opts.samplers, effective.sampler)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Scheduler</span><select class="settings-select" data-field="scheduler">${_selectHtml(opts.schedulers, effective.scheduler)}</select></div>`;
    html += `<div class="gen-params-section-title">Speed LoRA strengths</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">High-noise</span><input type="range" min="0" max="2" step="0.05" value="${effective.lora_high_weight}" data-field="lora_high_weight" style="flex:1;"><span class="gen-params-lora-weight">${Number(effective.lora_high_weight).toFixed(2)}</span></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Low-noise</span><input type="range" min="0" max="2" step="0.05" value="${effective.lora_low_weight}" data-field="lora_low_weight" style="flex:1;"><span class="gen-params-lora-weight">${Number(effective.lora_low_weight).toFixed(2)}</span></div>`;
    html += _heartbeatRowHtml();
  }
  return html;
}

function _heartbeatRowHtml() {
  return `<div class="gen-params-section-title">Notifications</div>
    <div class="gen-params-row settings-row">
      <label style="display:flex;align-items:center;gap:6px;font-size:11px;"><input type="checkbox" data-field="_heartbeat" ${_heartbeatEnabled() ? 'checked' : ''}> Post a chat line every 20s with percent + elapsed (useful when this tab is backgrounded)</label>
    </div>`;
}

function _renderPanel() {
  if (!_bodyEl) return;
  const sid = _currentSessionId || sessionModule.getCurrentSessionId();
  const effective = _effective(_currentKind, sid);
  const advanced = _currentTab === 'advanced';
  const scopeRow = `<div class="gen-params-row" style="justify-content:space-between;margin-bottom:2px;">
      <span class="gen-params-hint">Editing:</span>
      <div class="gen-params-tabs" style="margin:0;border:0;padding:0;">
        <button type="button" class="gen-params-tab${_editScope === 'session' ? ' active' : ''}" data-action="set-scope" data-scope-value="session">This chat</button>
        <button type="button" class="gen-params-tab${_editScope === 'default' ? ' active' : ''}" data-action="set-scope" data-scope-value="default">Global default</button>
      </div>
    </div>`;
  const body = _currentKind === 'video' ? _renderVideoPanel(effective, advanced) : _renderImagePanel(effective, advanced);
  _bodyEl.innerHTML = scopeRow + body;
}

// ── Delegated event handlers for the rendered body ──

function _onBodyClick(e) {
  const scopeBtn = e.target.closest('[data-action="set-scope"]');
  if (scopeBtn) { _editScope = scopeBtn.dataset.scopeValue; _renderPanel(); return; }

  const modeBtn = e.target.closest('[data-action="set-mode"]');
  if (modeBtn) { _setField('image', 'mode', modeBtn.dataset.modeValue); _renderPanel(); return; }

  const uploadBtn = e.target.closest('[data-action="upload-input-image"]');
  if (uploadBtn) {
    const input = _bodyEl.querySelector('[data-role="gen-params-file-input"]');
    if (input) {
      input.onchange = () => { if (input.files && input.files[0]) _uploadInputImage(_currentKind, input.files[0]); };
      input.click();
    }
    return;
  }
  const galleryBtn = e.target.closest('[data-action="pick-gallery-image"]');
  if (galleryBtn) { _openGalleryPicker(_currentKind); return; }
  const clearBtn = e.target.closest('[data-action="clear-input-image"]');
  if (clearBtn) { _clearInputImage(_currentKind); return; }
}

function _onBodyChange(e) {
  const field = e.target.dataset ? e.target.dataset.field : null;
  if (field === 'workflow') { _applyWorkflow(_currentKind, e.target.value); return; }
  if (field === '_heartbeat') { _setHeartbeat(e.target.checked); return; }
  if (field) {
    let value = e.target.value;
    if (e.target.type === 'checkbox') value = e.target.checked;
    else if (e.target.type === 'number' || e.target.type === 'range') value = Number(value);
    _setField(_currentKind, field, value);
    if (field === 'size_preset' || field === 'randomize_seed') _renderPanel();
    return;
  }
  const loraToggle = e.target.dataset.loraToggle;
  if (loraToggle) {
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    const effective = _effective('image', sid);
    const current = effective.loras === null
      ? _liveLoraDefaults(_optionsCache?.loras || [])
      : { ...effective.loras };
    const entry = (_optionsCache?.loras || []).find(l => l.key === loraToggle);
    // The checkbox is already `disabled` in the DOM when unavailable (see
    // _loraRowsHtml), so this only matters as a defensive backstop — never
    // let an unavailable entry (DEFECT 1) get selected into the payload.
    if (e.target.checked && _isLoraAvailable(entry)) {
      current[loraToggle] = entry ? (entry.default_weight ?? 1.0) : 1.0;
    } else {
      delete current[loraToggle];
    }
    _setField('image', 'loras', current);
    _renderPanel();
    return;
  }
}

function _onBodyInput(e) {
  const field = e.target.dataset ? e.target.dataset.field : null;
  if (field === 'negative_prompt') { _setField(_currentKind, field, e.target.value); return; }
  if (e.target.type === 'range' && field) {
    const val = Number(e.target.value);
    _setField(_currentKind, field, val);
    const label = e.target.parentElement.querySelector('.gen-params-lora-weight');
    if (label) label.textContent = val.toFixed(2);
    return;
  }
  const loraWeight = e.target.dataset.loraWeight;
  if (loraWeight && e.target.type === 'range') {
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    const effective = _effective('image', sid);
    const current = effective.loras === null
      ? _liveLoraDefaults(_optionsCache?.loras || [])
      : { ...effective.loras };
    current[loraWeight] = Number(e.target.value);
    _setField('image', 'loras', current);
    const label = e.target.parentElement.querySelector('.gen-params-lora-weight');
    if (label) label.textContent = Number(e.target.value).toFixed(2);
  }
}

// ── Generation + progress (plan §5, §6.3) ──

const _activeJobs = new Map(); // sessionId -> job record

// ── DEFECT 3: progress reattach across session switches / page loads ──
// The bug: switching chat sessions mid-job orphans the progress bubble — the
// job keeps running server-side (and, in this tab, its EventSource keeps
// streaming) but the DOM node it was updating gets discarded whenever
// #chat-history is rebuilt for the newly-selected session. sessions.js's
// selectSession() calls genParamsModule.onSessionSwitch() (this module's
// hook) BEFORE it re-fetches/re-renders that session's history
// (static/js/sessions.js ~1858 vs ~1953) — there is no later hook available
// to this module to know precisely when the rebuild has finished.
//
// Fix, in two parts:
//  1. Persist enough in localStorage to reconnect a FRESH EventSource
//     against the same job_id later (ComfyUI-side work is entirely
//     server-driven — see routes/comfy_routes.py's _run_job(); GET
//     /api/comfy/stream/{job_id} just polls the shared in-memory job
//     record, so a brand-new client connecting to an already-running or
//     already-finished job_id is a completely normal, supported reconnect,
//     not a special case).
//  2. Self-heal the bubble's DOM node on every progress tick (~1/sec)
//     rather than trying to time a single mount precisely against that
//     unpredictable async rebuild — simpler and more robust than a
//     MutationObserver-based "wait for #chat-history to settle" guess, and
//     it also transparently covers a user switching sessions back and
//     forth mid-job.
const INFLIGHT_MAX_AGE_MS = 30 * 60 * 1000; // "clean up entries older than ~30 min"

function _pruneInflight(map) {
  const now = Date.now();
  let changed = false;
  Object.keys(map).forEach(jobId => {
    const rec = map[jobId];
    if (!rec || !rec.startedAt || (now - rec.startedAt) > INFLIGHT_MAX_AGE_MS) {
      delete map[jobId];
      changed = true;
    }
  });
  return changed;
}

function _rememberInflight(job) {
  if (!job || !job.jobId) return;
  const map = Storage.loadGenInflight();
  _pruneInflight(map);
  map[job.jobId] = {
    jobId: job.jobId, sessionId: job.sessionId, kind: job.kind,
    prompt: job.prompt, params: job.params, startedAt: job.startedAt,
  };
  Storage.saveGenInflight(map);
}

// "once terminal" — called from _finishJob(), which every terminal path
// (done / error / cancelled) already routes through.
function _forgetInflight(jobId) {
  if (!jobId) return;
  const map = Storage.loadGenInflight();
  if (map[jobId]) {
    delete map[jobId];
    Storage.saveGenInflight(map);
  }
}

function _isCurrentSession(sessionId) {
  const current = _currentSessionId || (sessionModule.getCurrentSessionId ? sessionModule.getCurrentSessionId() : null);
  return !!sessionId && sessionId === current;
}

// (Re)paint a job's progress bubble once its session is confirmed to be the
// one currently on screen. Called immediately after a reattach AND on every
// tick (job.tickTimer, ~1/sec, wired in _connectStream below) so a mount
// that's too early (the chat-history rebuild for a just-selected session
// hasn't landed yet) self-corrects within about a second, and a bubble
// orphaned by that same rebuild gets repainted from the job's last-known
// progress rather than left blank.
function _ensureHolderForVisibleJob(job) {
  if (!job || !_isCurrentSession(job.sessionId)) return;
  if (job.holder && document.body.contains(job.holder)) return;
  const holder = _mountProgressBubble(job.kind);
  job.holder = holder;
  if (holder) {
    const cancelBtn = holder.querySelector('.gen-progress-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', () => _cancelJob(job));
    _updateProgressDom(holder, {
      label: job.lastLabel || 'Working…',
      percent: job.lastPercent || 0,
      elapsedMs: Date.now() - job.startedAt,
    });
  }
}

// Reconnect the session currently being shown to its in-flight job, if any —
// called from onSessionSwitch() and from init() (page load). A job this tab
// already knows about (it started it itself, or an earlier reattach found
// it) just gets its bubble (re)painted; a job this tab has never seen (a
// fresh page load, or another browser tab started it) gets a brand-new job
// record + EventSource built from the localStorage record.
function _reattachSession(sessionId) {
  if (!sessionId) return;
  const existing = _activeJobs.get(sessionId);
  if (existing) { _ensureHolderForVisibleJob(existing); return; }

  const map = Storage.loadGenInflight();
  if (_pruneInflight(map)) Storage.saveGenInflight(map);
  const rec = Object.values(map).find(r => r.sessionId === sessionId);
  if (!rec || !rec.jobId) return;

  const job = {
    sessionId, kind: rec.kind, prompt: rec.prompt || '', params: rec.params || {},
    holder: null, startedAt: rec.startedAt || Date.now(), jobId: rec.jobId,
  };
  _activeJobs.set(sessionId, job);
  _ensureHolderForVisibleJob(job);
  _connectStream(job);
}

// An input_image can come from either source, and BOTH resolve server-side:
//   (a) `comfy_filename` — the upload flow, already pushed into ComfyUI's own
//       input/ dir by POST /api/comfy/upload, so LoadImage finds it directly.
//   (b) `gallery_filename` — a data/generated_images/ name; the backend's
//       _resolve_input_image() matches it against GENERATED_IMAGE_RE, reads
//       the bytes and forwards them to ComfyUI transparently.
// Prefer the freshly uploaded file if somehow both are set.
function _resolvedInputImageName(img) {
  if (!img) return null;
  if (img.comfy_filename) return img.comfy_filename;
  if (img.gallery_filename) return img.gallery_filename;
  return null;
}

function _buildParamsPayload(kind, effective, promptText) {
  const [w, h] = String(effective.size_preset === 'custom' ? `${effective.width}x${effective.height}` : effective.size_preset).split('x').map(Number);
  const seed = effective.randomize_seed ? Math.floor(Math.random() * 2147483647) : Number(effective.seed) || 0;
  const loraList = [];
  if (kind === 'image') {
    const sel = effective.loras === null
      ? _liveLoraDefaults(_optionsCache?.loras || [])
      : (effective.loras || {});
    Object.entries(sel).forEach(([key, weight]) => {
      const entry = (_optionsCache?.loras || []).find(l => l.key === key);
      if (!entry) return;
      // DEFECT 1: send the backend-RESOLVED name (the exact string ComfyUI's
      // live LoraLoaderModelOnly combo will accept — see
      // src/comfy_graphs.py's resolve_lora_entries()), never the registry's
      // raw comfy_name. The checkbox for an unavailable entry is disabled
      // (see _loraRowsHtml) so this is normally unreachable for one, but a
      // stale localStorage selection (a LoRA that WAS available and got
      // toggled on in a previous session, then went unavailable — e.g. a
      // renamed/removed file) could still reach here; skip it with a toast
      // rather than sending a name ComfyUI will 404 on.
      if (!entry.resolved_name) {
        showToast(`Skipped "${entry.name || key}" — not available on the ComfyUI server yet (needs a restart to pick up new LoRA files).`);
        return;
      }
      // `kind` ("style" | "character") is carried through from loras.json so
      // the backend can tell whether any STYLE LoRA is active. With none, it
      // skips the style trigger and you get a clean general-purpose render
      // (the ComfyUI-side equivalent of Z-Image-General). LoraParam declares
      // this field, so it survives Pydantic rather than being dropped.
      loraList.push({ comfy_name: entry.resolved_name, weight: Number(weight), kind: entry.kind || null });
    });
  } else {
    loraList.push({ comfy_name: 'wan_high_noise', weight: Number(effective.lora_high_weight), role: 'high_noise' });
    loraList.push({ comfy_name: 'wan_low_noise', weight: Number(effective.lora_low_weight), role: 'low_noise' });
  }
  const params = {
    prompt: promptText,
    negative_prompt: effective.negative_prompt,
    loras: loraList,
    width: w || (kind === 'video' ? 720 : 1216),
    height: h || (kind === 'video' ? 720 : 672),
    seed,
    randomize_seed: !!effective.randomize_seed,
    steps: Number(effective.steps),
    cfg: Number(effective.cfg),
    sampler: effective.sampler,
    scheduler: effective.scheduler,
  };
  if (kind === 'image') {
    params.batch = Number(effective.batch) || 1;
    params.shift = Number(effective.shift);
    if (effective.unet) params.unet = effective.unet;
    if (effective.clip) params.clip = effective.clip;
    if (effective.vae) params.vae = effective.vae;
    if (effective.mode === 'img2img') {
      params.denoise = Number(effective.denoise);
      // Both input_image sources resolve server-side — see
      // _resolvedInputImageName(). Upload goes through /api/comfy/upload into
      // ComfyUI's own input/ dir; gallery names go through the backend's
      // GENERATED_IMAGE_RE passthrough. Verified live 2026-08-03.
      const ref = _resolvedInputImageName(effective.input_image);
      if (ref) params.input_image = ref;
    }
  } else {
    params.frames = Number(effective.frames);
    params.fps = Number(effective.fps);
    // Same two sources as the img2img branch above.
    const ref = _resolvedInputImageName(effective.input_image);
    if (ref) params.input_image = ref;
  }
  return params;
}

function _buildVideoBubble(videoUrl, prompt) {
  const wrap = document.createElement('div');
  wrap.className = 'msg msg-ai generated-video-wrap';
  const role = document.createElement('div');
  role.className = 'role';
  role.textContent = 'video';
  wrap.appendChild(role);
  const body = document.createElement('div');
  body.className = 'body';
  const video = document.createElement('video');
  video.className = 'generated-video';
  video.src = videoUrl;
  video.controls = true;
  video.loop = true;
  body.appendChild(video);
  if (prompt) {
    const cap = document.createElement('div');
    cap.className = 'generated-image-caption';
    cap.textContent = prompt;
    body.appendChild(cap);
  }
  wrap.appendChild(body);
  return wrap;
}

function _mountProgressBubble(kind) {
  const box = document.getElementById('chat-history');
  if (!box) return null;
  const holder = document.createElement('div');
  holder.className = 'msg msg-ai gen-progress-msg';
  const roleTs = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  holder.innerHTML = `<div class="role">${kind === 'video' ? 'Video' : 'Image'} <span class="role-timestamp">${esc(roleTs)}</span></div>
    <div class="body">
      <div class="gen-progress">
        <div class="gen-progress-row">
          <span class="gen-progress-label">Queued…</span>
          <span class="gen-progress-elapsed">0:00</span>
          <button type="button" class="gen-progress-cancel">Cancel</button>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:0%"></div></div>
      </div>
    </div>`;
  box.appendChild(holder);
  scrollHistory();
  return holder;
}

function _updateProgressDom(holder, { label, percent, elapsedMs }) {
  if (!holder || !document.body.contains(holder)) return;
  const labelEl = holder.querySelector('.gen-progress-label');
  const fillEl = holder.querySelector('.progress-fill');
  const elapsedEl = holder.querySelector('.gen-progress-elapsed');
  if (labelEl && label != null) labelEl.textContent = label;
  if (fillEl && percent != null) fillEl.style.width = Math.max(0, Math.min(100, percent)) + '%';
  if (elapsedEl && elapsedMs != null) elapsedEl.textContent = formatElapsed(elapsedMs);
}

function _postHeartbeatLine(job) {
  // A background job for a session that isn't currently on screen has
  // nowhere sane to post a heartbeat line — #chat-history is showing a
  // DIFFERENT session's transcript right now, and posting into it would
  // wrongly attribute the line to that session (the same cross-session bleed
  // DEFECT 3's reattach fix is about).
  if (!_isCurrentSession(job.sessionId)) return;
  const box = document.getElementById('chat-history');
  if (!box) return;
  const line = document.createElement('div');
  line.className = 'msg msg-ai gen-heartbeat-msg';
  const pct = job.lastPercent != null ? Math.round(job.lastPercent) + '%' : '…';
  line.innerHTML = `<div class="body" style="opacity:0.65;font-size:0.85em;">Still generating (${esc(job.kind)}) — ${pct}, ${esc(formatElapsed(Date.now() - job.startedAt))} elapsed.</div>`;
  box.appendChild(line);
  scrollHistory();
}

function _finishJob(job) {
  if (job.es) { try { job.es.close(); } catch (_) {} }
  if (job.heartbeatTimer) clearInterval(job.heartbeatTimer);
  if (job.tickTimer) clearInterval(job.tickTimer);
  _activeJobs.delete(job.sessionId);
  _forgetInflight(job.jobId); // "once terminal" — every terminal path (done/error/cancelled) routes through here
}

async function _cancelJob(job) {
  _updateProgressDom(job.holder, { label: 'Cancelling…' });
  try {
    await fetch(`${API_BASE}/api/comfy/cancel/${encodeURIComponent(job.jobId)}`, { method: 'POST', credentials: 'same-origin' });
  } catch (_) {}
  if (job.holder && document.body.contains(job.holder)) {
    const row = job.holder.querySelector('.gen-progress-row');
    if (row) row.insertAdjacentHTML('afterend', '<div class="gen-progress-error">Cancelled.</div>');
    const cancelBtn = job.holder.querySelector('.gen-progress-cancel');
    if (cancelBtn) cancelBtn.disabled = true;
  }
  _finishJob(job);
}

function _onJobError(job, message) {
  if (job.holder && document.body.contains(job.holder)) {
    job.holder.querySelector('.body').insertAdjacentHTML('beforeend', `<div class="gen-progress-error">${esc(message || 'Generation failed')}</div>`);
    const cancelBtn = job.holder.querySelector('.gen-progress-cancel');
    if (cancelBtn) cancelBtn.style.display = 'none';
  } else {
    showToast('Generation failed: ' + (message || 'unknown error'));
  }
  _finishJob(job);
}

function _onJobDone(job, data) {
  const images = (data && data.images) || [];
  if (job.holder && document.body.contains(job.holder)) job.holder.remove();
  // Only paint result bubbles into #chat-history when job.sessionId is the
  // session actually on screen right now — a job that finishes while the
  // user has switched to a DIFFERENT session must not have its bubble land
  // in that other session's visible transcript (the DOM has no idea whose
  // history it's showing beyond "whatever was last rendered"). The image is
  // already durably saved to the Gallery (+ the DB row's session_id is
  // correct) regardless of this check — gallery-refresh always fires below.
  if (_isCurrentSession(job.sessionId)) {
    const box = document.getElementById('chat-history');
    images.forEach(img => {
      if (!img || !img.url) return;
      const isVideo = /\.(mp4|mov|webm|mkv|m4v)$/i.test(img.filename || img.url);
      const bubble = isVideo
        ? _buildVideoBubble(img.url, job.prompt)
        : chatRenderer.buildImageBubble(img.url, job.prompt, job.kind === 'video' ? 'video' : 'image', `${job.params.width}x${job.params.height}`, null, img.gallery_id);
      if (box) box.appendChild(bubble);
    });
    if (box) scrollHistory();
  }
  window.dispatchEvent(new CustomEvent('gallery-refresh', { detail: { source: 'genParams' } }));
  _finishJob(job);
}

function _connectStream(job) {
  // _connectStream can be called a second time on transport-error retry —
  // clear any timers from a prior attempt first so a reconnect doesn't
  // silently stack duplicate intervals that never get cleared.
  if (job.tickTimer) clearInterval(job.tickTimer);
  if (job.heartbeatTimer) clearInterval(job.heartbeatTimer);

  const es = new EventSource(`${API_BASE}/api/comfy/stream/${encodeURIComponent(job.jobId)}`);
  job.es = es;

  job.tickTimer = setInterval(() => {
    // DEFECT 3 self-heal: (re)paint the bubble if this job's session is the
    // one on screen but its DOM node is missing/detached — see
    // _ensureHolderForVisibleJob()'s docstring for why this runs on every
    // tick instead of once at reattach time.
    _ensureHolderForVisibleJob(job);
    _updateProgressDom(job.holder, { elapsedMs: Date.now() - job.startedAt });
  }, 1000);

  es.addEventListener('progress', (evt) => {
    try {
      const d = JSON.parse(evt.data);
      job.lastPercent = d.percent;
      const nodeLabel = d.node_title || d.node || 'Working';
      const stepLabel = (d.step != null && d.max != null) ? ` — step ${d.step}/${d.max}` : '';
      // Remembered so a bubble (re)mounted later by _ensureHolderForVisibleJob
      // (a fresh reattach, or a self-heal after an orphaning session switch)
      // can show the last-known state immediately instead of a blank
      // "Queued…" while waiting for the next SSE frame.
      job.lastLabel = `${nodeLabel}${stepLabel}`;
      // Elapsed always comes from the client-side clock (job.tickTimer below
      // updates it every second regardless) rather than d.elapsed — the
      // contract doesn't pin down that field's unit (ms vs s), and getting
      // it wrong would make the displayed time visibly jump/lag.
      _updateProgressDom(job.holder, {
        label: job.lastLabel,
        percent: d.percent,
      });
    } catch (_) {}
  });
  es.addEventListener('done', (evt) => {
    try { _onJobDone(job, JSON.parse(evt.data)); } catch (_) { _onJobDone(job, {}); }
  });
  es.addEventListener('error', (evt) => {
    if (evt && evt.data) {
      // Named "error" SSE event from the backend (job failed) — EventSource
      // dispatches this under the same DOM event type as a transport error,
      // disambiguated here by whether a payload came through.
      try { _onJobError(job, JSON.parse(evt.data).message); } catch (_) { _onJobError(job, 'Generation failed'); }
      return;
    }
    // Real transport error — retry once, then give up quietly (the job may
    // still finish server-side and land in the Gallery regardless).
    es.close();
    if (job._retried) { _onJobError(job, 'Lost connection to the generation stream.'); return; }
    job._retried = true;
    setTimeout(() => { if (_activeJobs.has(job.sessionId)) _connectStream(job); }, 3000);
  });

  if (_heartbeatEnabled()) {
    job.heartbeatTimer = setInterval(() => _postHeartbeatLine(job), 20000);
  }
}

export async function generate(kind, promptText, sessionId) {
  const sid = sessionId || sessionModule.getCurrentSessionId();
  if (!sid) { showToast('No active chat session.'); return; }
  if (_activeJobs.has(sid)) { showToast('A generation is already running in this chat.'); return; }
  if (!promptText || !promptText.trim()) { showToast('Type a prompt first.'); return; }

  await Promise.all([_fetchOptions(), _fetchWorkflows()]);
  const effective = _effective(kind, sid);
  const params = _buildParamsPayload(kind, effective, promptText.trim());

  const holder = _mountProgressBubble(kind);
  const job = { sessionId: sid, kind, prompt: promptText.trim(), params, holder, startedAt: Date.now() };
  _activeJobs.set(sid, job);

  if (holder) {
    const cancelBtn = holder.querySelector('.gen-progress-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', () => _cancelJob(job));
  }

  try {
    const res = await fetch(`${API_BASE}/api/comfy/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({ kind, workflow: effective.workflow || 'Custom', session_id: sid, params }),
    });
    const data = await res.json();
    if (!res.ok || !data || !data.job_id) {
      throw new Error((data && (data.error || data.detail)) || `HTTP ${res.status}`);
    }
    job.jobId = data.job_id;
    job.promptId = data.prompt_id;
    _rememberInflight(job);
    _connectStream(job);
  } catch (e) {
    // _onJobError already surfaces this (inline in the bubble, or a toast if
    // the bubble couldn't be mounted) — don't also reject the promise, or
    // chat.js's .catch() on generate() would show a second, redundant toast.
    _onJobError(job, e.message || 'Failed to start generation');
  }
}

export default { init, onModeChange, onSessionSwitch, generate, isBusy };
