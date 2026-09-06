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
//
// MODEL-FIRST REDESIGN (2026-08-04): the Image panel's Simple tab is now
// JUST the model picker (data/studio/scripts/models.json presets, via
// /api/comfy/options' `models`/`default_model`) + its description + a
// one-line style-trigger banner — "pick a model, type a prompt, hit
// Generate." Everything that used to live in Simple (workflow, mode/img2img,
// LoRA rows, size, batch) plus the old Advanced tab now all live under
// Advanced together. See _renderImagePanel(), _modelPickerHtml(), and the
// touched-field tracking in _setField()/_touchedSet() (a model switch
// reseeds steps/cfg/sampler/scheduler/shift/size/negative_prompt/loras from
// the new model's own defaults, but never clobbers a field the user already
// customized). The Video tab is deliberately untouched by any of this —
// models.json has no video entries and the backend ignores `model` for
// kind:video.

import Storage from './storage.js';
import uiModule from './ui.js';
import sessionModule from './sessions.js';
import chatRenderer from './chatRenderer.js?v=20260722emailfastindex1';
import { formatElapsed } from './research/jobs.js?v=20260630researchthumb';
import { bindMenuDismiss } from './escMenuStack.js';
import { gpuStatus, prepareGpuFor, GPU_SWITCH_MESSAGE } from './gpuTimeshare.js?v=20260815timeshare1';
import fileHandler from './fileHandler.js';

const { esc, showToast, scrollHistory } = uiModule;

// DEFECT 13: esc() (static/js/ui.js) assumes a string -- it calls
// String.prototype.replace on its argument, which throws for any truthy
// number (only falsy ones survive its own `s || ''` guard). Every numeric
// param field below is normally a real number, but _applyWorkflow() seeds
// these straight from a workflow JSON file that anyone with LAN access to
// the unauthenticated ComfyUI on :8188 can plant -- so a belt-and-braces
// escape at each interpolation site needs a string-safe wrapper, not a bare
// esc() call.
function _escNum(value) { return esc(String(value)); }

let API_BASE = '';
let _inited = false;

// ── Baseline parameter schemas (plan §7) — the floor every effective value
// falls back to. Global defaults (Storage.loadGenDefaults()) and per-session
// overrides (Storage.loadGenSessions()) are both PARTIAL objects layered on
// top of these, so adding a new field later never orphans an old saved chat.
const IMAGE_BASELINE = Object.freeze({
  workflow: 'Custom',
  model: 'studio_toei',       // data/studio/scripts/models.json preset key. Only the
                               // pre-options-load floor — see _effectiveModelKey() for
                               // the live fallback chain (also what makes a stale
                               // pre-model-picker localStorage entry degrade safely:
                               // it never has a `model` key, so this baseline supplies
                               // one via the plain object spread in _effective()).
  mode: 'txt2img',            // 'txt2img' | 'img2img'
  input_image: null,          // {comfy_filename, name} (upload) or {id, gallery_filename, name, previewUrl} (gallery pick) --
                               // DEFECT 18: an upload's own previewUrl is a page-load-local blob: URL, kept OUT of this
                               // persisted object on purpose (see _liveUploadState) so it never survives to localStorage.
  loras: null,                // null = "use the selected model's own bundled LoRAs" (_defaultLoraMapFor()); else {key: weight}
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
  clip_type: '',
  vae: '',
});
const VIDEO_BASELINE = Object.freeze({
  workflow: 'Custom',
  model: 'wan22_i2v',       // data/studio/scripts/models.json `video_models` preset key --
                             // same pre-options-load-floor / live-fallback-chain role as
                             // IMAGE_BASELINE.model (see _effectiveVideoModelKey()). Matches
                             // wan22_i2v's OWN defaults below (steps/cfg/sampler/frames/fps),
                             // same reasoning as IMAGE_BASELINE matching studio_toei's.
  input_image: null,        // {comfy_filename|gallery_filename, name, ...} -- START frame (both engines); see IMAGE_BASELINE.input_image re: DEFECT 18
  last_frame: null,         // same shape -- OPTIONAL end frame (MiniMax H3 only)
  frames: 81,                // Wan-only: WanImageToVideo's raw frame count
  seconds: 2,                 // H3-only: duration; converted server-side via _h3_length()
  fps: 16,
  size_preset: '720x720',
  width: 720,
  height: 720,
  negative_prompt: 'text, letters, lettering, logo, watermark, signature',
  seed: 0,
  randomize_seed: true,
  steps: 4,
  cfg: 1,                     // Wan-only -- H3 is CFG-free (BasicGuider is unguided)
  sampler: 'euler',
  scheduler: 'simple',
  lora_high_weight: 1.0,      // Wan-only speed-LoRA strengths
  lora_low_weight: 1.0,
  shift_video: 12.0,          // H3-only -- MiniMaxH3SigmaShift's two floats (its own defaults)
  shift_audio: 3.0,
});
const SIZE_PRESETS_IMAGE = ['1216x672', '1152x864', '1024x1024', 'custom'];
const SIZE_PRESETS_VIDEO = ['720x720', 'custom'];

const MUSIC_BASELINE = Object.freeze({
  backend: '',
  lyrics: '',
  seconds: 60,
  seed: 0,
  randomize_seed: true,
  instrumental: false,
});

function _baseline(kind) {
  if (kind === 'video') return VIDEO_BASELINE;
  if (kind === 'music') return MUSIC_BASELINE;
  return IMAGE_BASELINE;
}

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

// Fields a MODEL PRESET (data/studio/scripts/models.json) can reseed on
// selection -- mirrors src/comfy_graphs.py's apply_model_preset()/
// _PRESET_DEFAULT_KEYS field set, plus the two size components and the LoRA
// sentinel. Only these keys ever get a `_touched` flag recorded (see
// _setField below) -- tracking touched-ness for a field no model preset ever
// opines on (denoise, batch, seed, unet override, ...) would just be dead
// bookkeeping.
//
// 'fps'/'frames'/'seconds'/'shift_video'/'shift_audio' are VIDEO-only
// additions (mirrors _PRESET_DEFAULT_KEYS' own widening in
// src/comfy_graphs.py) -- purely additive for image: an image `effective`
// object never has these keys at all, so hasOwnProperty checks against them
// are always false there.
const _RESEED_FIELDS = new Set([
  'steps', 'cfg', 'sampler', 'scheduler', 'shift', 'negative_prompt', 'size_preset', 'width', 'height', 'loras',
  'fps', 'frames', 'seconds', 'shift_video', 'shift_audio',
]);

// Stamps `key` as touched in `layer._touched`. If `layer` has no `_touched`
// map yet, this is the FIRST touched-write this layer has recorded since the
// model-picker redesign shipped — migrate once, in place: every reseedable
// field the layer already had a value for (e.g. a steps/cfg customization
// made before per-field touched-tracking existed) is backfilled as touched
// too, so it isn't silently discarded the next time a model switch or a
// generate reads this layer's touched-set. Combined with _layerTouched()'s
// read-time fallback (for the window before any post-upgrade write has
// happened at all), pre-existing customizations survive this upgrade.
function _stampTouched(layer, key) {
  let base = layer._touched;
  if (!base) {
    base = {};
    _RESEED_FIELDS.forEach(k => { if (Object.prototype.hasOwnProperty.call(layer, k)) base[k] = true; });
  }
  return { ...base, [key]: true };
}

// Writes ONE field of the given kind's params into whichever layer
// `_editScope` currently points at, then re-renders.
//
// `touched` (default true) records this key in that same layer's `_touched`
// map when it's one of _RESEED_FIELDS — this is how a model switch (see
// _reseedFromModel below) knows NOT to clobber a value the user set on
// purpose. Callers pass `false` for a PROGRAMMATIC write (a model reseed, or
// the "Reset to model defaults" action) so they don't immediately mark the
// field they themselves just set as user-touched.
function _setField(kind, key, value, touched = true) {
  const markTouched = touched && _RESEED_FIELDS.has(key);
  if (_editScope === 'default') {
    const d = _loadDefaults();
    d[kind] = { ...(d[kind] || {}), [key]: value };
    if (markTouched) d[kind]._touched = _stampTouched(d[kind], key);
    _saveDefaults(d);
  } else {
    const sessions = _loadSessions();
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    if (!sid) { // No session yet (pre-first-message) — fall back to defaults.
      const d = _loadDefaults();
      d[kind] = { ...(d[kind] || {}), [key]: value };
      if (markTouched) d[kind]._touched = _stampTouched(d[kind], key);
      _saveDefaults(d);
      return;
    }
    sessions[sid] = { ...(sessions[sid] || {}) };
    sessions[sid][kind] = { ...(sessions[sid][kind] || {}), [key]: value };
    if (markTouched) sessions[sid][kind]._touched = _stampTouched(sessions[sid][kind], key);
    _saveSessions(sessions);
  }
}

function _setFields(kind, patch) {
  Object.entries(patch).forEach(([k, v]) => _setField(kind, k, v));
}

// A layer's real `_touched` map if it has one; otherwise (data saved before
// per-field touched-tracking existed) every reseedable field it already set
// counts as touched — see _stampTouched()'s docstring for why both the
// write-time and this read-time fallback exist together.
function _layerTouched(layer) {
  if (layer._touched) return layer._touched;
  const out = {};
  _RESEED_FIELDS.forEach(k => { if (Object.prototype.hasOwnProperty.call(layer, k)) out[k] = true; });
  return out;
}

// Union of the touched-flags recorded at both persistence layers (plan
// §6.5's "global defaults, overridden per session" — a field touched at
// EITHER layer is considered touched, since either one could be the layer
// that ends up winning in _effective()).
function _touchedSet(kind, sessionId) {
  const defaults = _loadDefaults();
  const sessions = _loadSessions();
  const dLayer = defaults[kind] || {};
  const sLayer = (sessionId && sessions[sessionId] && sessions[sessionId][kind]) || {};
  return { ..._layerTouched(dLayer), ..._layerTouched(sLayer) };
}

// Clears touched-flags for `keys` from whichever layer `_editScope` currently
// points at (mirrors _setField's own scope branching) — used by the
// "Reset to model defaults" action so a field it resets is immediately
// eligible to be reseeded again by a FUTURE model switch too.
function _clearTouched(kind, keys) {
  const clear = (obj) => {
    if (!obj || !obj[kind] || !obj[kind]._touched) return obj;
    const t = { ...obj[kind]._touched };
    keys.forEach(k => delete t[k]);
    return { ...obj, [kind]: { ...obj[kind], _touched: t } };
  };
  if (_editScope === 'default') {
    _saveDefaults(clear(_loadDefaults()) || {});
  } else {
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    if (!sid) { _saveDefaults(clear(_loadDefaults()) || {}); return; }
    const sessions = _loadSessions();
    sessions[sid] = clear(sessions[sid] || {}) || {};
    _saveSessions(sessions);
  }
}

// ── Options / workflows (fetched from the backend the other agent is
// building — cached briefly, tolerant of it not existing yet). ──
let _optionsCache = null;
let _optionsFetchedAt = 0;
let _workflowsCache = null;
let _workflowsFetchedAt = 0;
let _musicBackendsCache = null;
let _musicBackendsFetchedAt = 0;
const _CACHE_MS = 60000;

async function _fetchOptions(force) {
  if (!force && _optionsCache && Date.now() - _optionsFetchedAt < _CACHE_MS) return _optionsCache;
  try {
    const res = await fetch(`${API_BASE}/api/comfy/options`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _optionsCache = await res.json();
  } catch (e) {
    console.warn('[genParams] /api/comfy/options unavailable:', e.message || e);
    _optionsCache = { loras: [], loras_all: [], unets: [], vaes: [], clips: [], samplers: [], schedulers: [], models: [], default_model: null };
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

async function _fetchMusicBackends(force) {
  if (!force && _musicBackendsCache && Date.now() - _musicBackendsFetchedAt < _CACHE_MS) return _musicBackendsCache;
  try {
    const res = await fetch(`${API_BASE}/api/music/backends`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    _musicBackendsCache = await res.json();
  } catch (e) {
    console.warn('[genParams] /api/music/backends unavailable:', e.message || e);
    _musicBackendsCache = { backends: [], default: null };
  }
  _musicBackendsFetchedAt = Date.now();
  return _musicBackendsCache;
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
    // DEFECT 8: introspect_graph() (src/comfy_graphs.py) returns `loras` as
    // a LIST of {comfy_name, weight} and `input_image`/`last_frame` as a
    // bare ComfyUI filename STRING -- but this panel's schema
    // (IMAGE_BASELINE/VIDEO_BASELINE) declares `loras` as a
    // {registry_key: weight} MAP and the image fields as
    // {comfy_filename, name, ...} objects. Passing the introspected shapes
    // straight through left no LoRA row checked (Object.entries() on an
    // array walks numeric indices, which never match a registry key) and
    // silently dropped the workflow's source image
    // (_resolvedInputImageName() only reads .comfy_filename/
    // .gallery_filename, never a bare string). Normalise both here, once,
    // before they ever reach the schema-shaped `patch` below.
    //
    // Make sure _optionsCache.loras (the comfy_name -> registry-key lookup)
    // is actually populated before using it -- _open() already kicks this
    // off, but _applyWorkflow can in principle run before that resolves.
    await _fetchOptions();
    const schema = _baseline(kind);
    const patch = {};
    Object.keys(data.params || {}).forEach(k => {
      if (!Object.prototype.hasOwnProperty.call(schema, k)) return;
      if (k === 'loras') {
        const registry = (_optionsCache && _optionsCache.loras) || [];
        const map = {};
        (Array.isArray(data.params.loras) ? data.params.loras : []).forEach(l => {
          if (!l || !l.comfy_name) return;
          const entry = registry.find(r => r.comfy_name === l.comfy_name || r.resolved_name === l.comfy_name);
          if (entry) map[entry.key] = (l.weight != null ? Number(l.weight) : 1.0);
        });
        patch.loras = map;
        return;
      }
      if (k === 'input_image' || k === 'last_frame') {
        const value = data.params[k];
        patch[k] = value ? { comfy_filename: String(value), name: String(value) } : null;
        return;
      }
      // DEFECT 13: a workflow JSON is attacker-reachable (anyone with LAN
      // access to the unauthenticated ComfyUI on :8188 can plant one), and
      // ui_to_api()/introspect_graph() apply no type validation server-side
      // -- a numeric schema field (width, seed, steps, ...) could arrive as
      // a crafted string. Number()-coerce every field the schema declares
      // as a number, dropping (not seeding) a value that doesn't actually
      // parse as one, rather than trusting the workflow file's JSON types.
      // This is the primary fix; the esc() calls at each render site are
      // belt-and-braces.
      let value = data.params[k];
      if (typeof schema[k] === 'number') {
        value = Number(value);
        if (Number.isNaN(value)) return;
      }
      patch[k] = value;
    });
    // A workflow that seeded a source image but never mentions `mode`
    // (introspect_graph() has no such concept) would otherwise stay on
    // txt2img, where _buildParamsPayload() never sends input_image at all --
    // silently dropping it all over again despite the shape fix above.
    if (kind === 'image' && patch.input_image) patch.mode = 'img2img';
    // DEFECT 9 (part 2): _buildParamsPayload() only trusts raw width/height
    // over the size_preset STRING when size_preset === 'custom' -- introspect_
    // graph() has no size_preset concept at all, so without this a seeded
    // custom size (e.g. Wan's 832x480) would resolve back to whatever named
    // preset happened to already be selected, even with the touched-field
    // fix below in place.
    if (Object.prototype.hasOwnProperty.call(patch, 'width') || Object.prototype.hasOwnProperty.call(patch, 'height')) {
      patch.size_preset = 'custom';
    }
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
// `field` defaults to 'input_image' -- same "every existing call site
// untouched" reasoning as _inputImageHtml() above.
// DEFECT 18: a blob: URL (URL.createObjectURL) is only valid for the page
// load that created it -- persisting one to localStorage (which _setField
// always does, synchronously) produces a broken thumbnail forever after any
// reload. Persisting `uploading: true` has the same class of problem: if the
// tab closes/crashes between the two _setField calls below, nothing is ever
// left to flip it back to false. Both are kept in this same-page-load-only
// side table instead (keyed by "kind:field"), so the instant local
// preview/progress text during THIS session is unaffected, but neither value
// ever reaches localStorage. A gallery-picked image's previewUrl (a real,
// durable server URL -- see _openGalleryPicker) is unaffected; it's still
// persisted as part of the field's own object, which _inputImageHtml below
// still prefers when present.
const _liveUploadState = new Map();

async function _uploadInputImage(kind, file, field = 'input_image') {
  if (!file) return;
  const cacheKey = `${kind}:${field}`;
  const previewUrl = URL.createObjectURL(file);
  _liveUploadState.set(cacheKey, { previewUrl, uploading: true });
  _setField(kind, field, { comfy_filename: null, name: file.name });
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
    _liveUploadState.set(cacheKey, { previewUrl, uploading: false });
    _setField(kind, field, { comfy_filename: ref, name: file.name });
  } catch (e) {
    showToast('Image upload failed: ' + (e.message || 'network error'));
    _liveUploadState.delete(cacheKey);
    _setField(kind, field, null);
  }
  _renderPanel();
}

function _clearInputImage(kind, field = 'input_image') {
  const cacheKey = `${kind}:${field}`;
  const live = _liveUploadState.get(cacheKey);
  if (live && live.previewUrl) { try { URL.revokeObjectURL(live.previewUrl); } catch (_) {} }
  _liveUploadState.delete(cacheKey);
  _setField(kind, field, null);
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
  const isGen = mode === 'image' || mode === 'video' || mode === 'music';
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
    if (mode === 'music') _fetchMusicBackends().catch(() => {});
    else {
      _fetchOptions().catch(() => {});
      _fetchWorkflows().catch(() => {});
    }
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

// `modelEntry` (image kind only) swaps in that model's own `sizes` list
// (e.g. Qwen's official aspect-ratio table vs Z-Image's studio framings, or
// MiniMax H3's 1344x768 vs Wan's 720x720 -- task item 2/4: "Selecting a
// model reseeds the panel from its defaults and sizes") plus a 'custom'
// escape hatch; omitted/null falls back to the hardcoded per-kind floor
// (pre-options-load, or a model with no `sizes` of its own). `modelEntry`
// now takes priority for BOTH kinds -- previously video ignored it entirely
// and always showed Wan's 720x720 preset even when H3 (1344x768) was
// selected, since the models.json `video_models` section didn't exist yet
// when this function was first written.
function _sizeOptionsHtml(kind, current, modelEntry) {
  let presets;
  if (modelEntry && Array.isArray(modelEntry.sizes) && modelEntry.sizes.length) {
    presets = [...modelEntry.sizes, 'custom'];
  } else if (kind === 'video') {
    presets = SIZE_PRESETS_VIDEO;
  } else {
    presets = SIZE_PRESETS_IMAGE;
  }
  return presets.map(p => `<option value="${esc(p)}"${p === current ? ' selected' : ''}>${p === 'custom' ? 'Custom' : esc(p.replace('x', ' × '))}</option>`).join('');
}

function _selectHtml(list, current, allowEmpty) {
  let html = allowEmpty ? `<option value=""${!current ? ' selected' : ''}>(model default)</option>` : '';
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

// ── Model presets (data/studio/scripts/models.json, via /api/comfy/options'
// `models`/`default_model`) — the new primary control. See
// routes/comfy_routes.py's comfy_generate()/apply_model_preset()/
// resolve_model_entries() for the backend half of this contract: "any field
// you also send explicitly WINS" over the preset's own default. ──

function _modelEntry(key) {
  return (_optionsCache && _optionsCache.models || []).find(m => m.key === key) || null;
}

// Resolution order: this session/kind's stored choice -> the backend's own
// registry default -> the hardcoded floor also baked into IMAGE_BASELINE.
// A stale pre-model-picker localStorage entry never had a `model` key at
// all, so the IMAGE_BASELINE spread in _effective() already supplies
// 'studio_toei' on its own — this function additionally prefers the LIVE
// registry default once options have loaded, in case that ever drifts from
// the hardcoded floor (task item 6).
function _effectiveModelKey(effective) {
  return (effective && effective.model) || (_optionsCache && _optionsCache.default_model) || 'studio_toei';
}

// Video model presets (data/studio/scripts/models.json's `video_models`
// section, via /api/comfy/options' `video_models`/`default_video_model`) --
// same "resolution order" contract as the image pair above, just a
// different cache key and hardcoded floor ('wan22_i2v', the lighter-weight,
// no-huge-download engine -- matches src/comfy_graphs.py's
// _FALLBACK_MODEL_REGISTRY choice for the same reason).
function _videoModelEntry(key) {
  return (_optionsCache && _optionsCache.video_models || []).find(m => m.key === key) || null;
}

function _effectiveVideoModelKey(effective) {
  return (effective && effective.model) || (_optionsCache && _optionsCache.default_video_model) || 'wan22_i2v';
}

// A model preset's OWN loras (models.json's `[{key, weight}, ...]` form,
// keyed into loras.json — same "key" namespace as _optionsCache.loras[].key)
// turned into the {key: weight} map shape the LoRA-row checkboxes and
// _buildParamsPayload already understand.
function _modelDefaultLoraMap(modelEntry) {
  if (!modelEntry || !Array.isArray(modelEntry.loras)) return {};
  const out = {};
  modelEntry.loras.forEach(l => { if (l && l.key) out[l.key] = (l.weight != null ? l.weight : 1.0); });
  return out;
}

// What the LoRA rows should show as "on" when the user hasn't touched
// `loras` at all. Prefers the SELECTED MODEL's own bundled LoRAs; falls back
// to the old registry-wide "status===LIVE" heuristic only when the model
// list hasn't loaded yet / the stored model key doesn't resolve (keeps the
// panel non-empty during the brief window before /api/comfy/options answers).
function _defaultLoraMapFor(effective) {
  const modelEntry = _modelEntry(_effectiveModelKey(effective));
  if (modelEntry) return _modelDefaultLoraMap(modelEntry);
  return _liveLoraDefaults(_optionsCache?.loras || []);
}

// The {steps, cfg, sampler, scheduler, shift, negative_prompt, size_preset,
// width, height, ...} patch a model preset wants to seed — mirrors
// src/comfy_graphs.py's apply_model_preset()/_PRESET_DEFAULT_KEYS field set
// exactly, so the panel previews what the backend will actually fill in.
//
// Shared by BOTH the image `models` registry and the video `video_models`
// registry (task item 4: "give the Video tab the same ... reseeding ...
// the Image tab has" -- reuse, don't reinvent) -- 'fps'/'frames'/'seconds'/
// 'shift_video'/'shift_audio' are video-only keys an IMAGE preset's
// `defaults` never declares, so this is purely additive for the image path.
function _modelDefaultsPatch(modelEntry) {
  const d = (modelEntry && modelEntry.defaults) || {};
  const patch = {};
  ['steps', 'cfg', 'sampler', 'scheduler', 'shift', 'negative_prompt',
   'fps', 'frames', 'seconds', 'shift_video', 'shift_audio'].forEach(k => {
    if (d[k] !== undefined) patch[k] = d[k];
  });
  if (typeof d.size === 'string' && d.size.includes('x')) {
    const [w, h] = d.size.split('x').map(Number);
    if (w && h) { patch.size_preset = d.size; patch.width = w; patch.height = h; }
  }
  return patch;
}

// Reseeds every _RESEED_FIELDS field the user has NOT already touched (see
// _setField) from `modelEntry`'s own defaults — "a user who set steps=45
// shouldn't lose it on every model switch, but a user who never touched
// steps should get each model's correct default" (task item 2). width/
// height are only reseeded alongside size_preset (not independently) so a
// still-untouched custom width/height pair doesn't drift out of sync with a
// touched size_preset. Shared by BOTH _reseedFromModel (image) and
// _reseedFromVideoModel (video) below -- everything past "which registry did
// modelEntry come from" is identical between the two kinds.
function _reseedCommonFields(kind, modelEntry, touched) {
  const patch = _modelDefaultsPatch(modelEntry);
  Object.entries(patch).forEach(([k, v]) => {
    if (k === 'width' || k === 'height') {
      if (!touched.size_preset && !touched[k]) _setField(kind, k, v, false);
    } else if (!touched[k]) {
      _setField(kind, k, v, false);
    }
  });
}

function _reseedFromModel(kind, modelKey, touched) {
  const modelEntry = _modelEntry(modelKey);
  if (!modelEntry) return;
  _reseedCommonFields(kind, modelEntry, touched);
  if (!touched.loras) _setField(kind, 'loras', null, false);
}

// Video counterpart of _reseedFromModel() -- no `loras` tail: video's LoRA
// handling (Wan's two fixed-name weight sliders, H3 has none at all) is a
// completely different, simpler mechanism than image's registry-backed
// {key: weight} map, and VIDEO_BASELINE has no `loras` key at all (see that
// baseline's own comments) -- there is nothing here to reset.
function _reseedFromVideoModel(kind, modelKey, touched) {
  const modelEntry = _videoModelEntry(modelKey);
  if (!modelEntry) return;
  _reseedCommonFields(kind, modelEntry, touched);
}

// Fired when the user picks a different Model in the dropdown.
function _onModelChange(kind, newKey) {
  const sid = _currentSessionId || sessionModule.getCurrentSessionId();
  if (kind === 'video') {
    // No "arch"/cross-architecture LoRA-clearing concept for video (Wan and
    // H3 don't share loras.json's registry-LoRA selection at all -- see
    // _reseedFromVideoModel()'s own comment), so this is simpler than the
    // image path below.
    _setField(kind, 'model', newKey, false);
    _reseedFromVideoModel(kind, newKey, _touchedSet(kind, sid));
    _renderPanel();
    return;
  }
  const oldEntry = _modelEntry(_effectiveModelKey(_effective(kind, sid)));
  const newEntry = _modelEntry(newKey);
  _setField(kind, 'model', newKey, false);
  if (oldEntry && newEntry && oldEntry.arch !== newEntry.arch) {
    // A LoRA trained for one base architecture (z_image vs qwen_image) is
    // not portable to the other — clear ANY touched LoRA selection rather
    // than risk sending an incompatible adapter into a graph built for a
    // different UNET. Touched-tracking exists to preserve INTENTIONAL
    // customizations across a model switch; a cross-architecture LoRA pick
    // was never a valid intention to carry forward.
    _clearTouched(kind, ['loras']);
    _setField(kind, 'loras', null, false);
  }
  _reseedFromModel(kind, newKey, _touchedSet(kind, sid));
  _renderPanel();
}

// "Reset to model defaults" (task item 2's small affordance) — unlike a
// model SWITCH, this is a deliberate blank-slate action: it clears the
// touched-flags too (so a LATER model switch can reseed these fields again).
// For image it ALSO resets the Advanced-only model-override fields (unet/
// clip/clip_type/vae) and the LoRA selection back to "let the model decide"
// -- video has no such controls in this panel (task item 4's H3 Advanced
// list is just seed/steps/sampler/scheduler/shift_video/shift_audio), so
// there is nothing extra to reset there.
function _resetToModelDefaults(kind) {
  const sid = _currentSessionId || sessionModule.getCurrentSessionId();
  if (kind === 'video') {
    const modelEntry = _videoModelEntry(_effectiveVideoModelKey(_effective(kind, sid)));
    if (!modelEntry) { showToast('Model info not loaded yet.'); return; }
    _clearTouched(kind, Array.from(_RESEED_FIELDS));
    const patch = _modelDefaultsPatch(modelEntry);
    Object.entries(patch).forEach(([k, v]) => _setField(kind, k, v, false));
    showToast('Reset to "' + (modelEntry.name || kind) + '" defaults.');
    _renderPanel();
    return;
  }
  const modelEntry = _modelEntry(_effectiveModelKey(_effective(kind, sid)));
  if (!modelEntry) { showToast('Model info not loaded yet.'); return; }
  _clearTouched(kind, Array.from(_RESEED_FIELDS).concat(['mode', 'input_image']));
  const patch = _modelDefaultsPatch(modelEntry);
  Object.entries(patch).forEach(([k, v]) => _setField(kind, k, v, false));
  ['unet', 'clip', 'clip_type', 'vae'].forEach(k => _setField(kind, k, '', false));
  _setField(kind, 'loras', null, false);
  _setField(kind, 'mode', 'txt2img', false);
  _setField(kind, 'input_image', null, false);
  showToast('Reset to "' + (modelEntry.name || kind) + '" defaults.');
  _renderPanel();
}

// Task item 5: the mode banner now reads the SELECTED MODEL's own
// `style_trigger` flag (the backend's authority — see routes/comfy_routes.py
// comfy_generate()'s "NEW AUTHORITY" comment) rather than inferring it from
// which LoRA checkboxes happen to be checked. Lives in the always-visible
// top section now (both Simple and Advanced), not just alongside the LoRA
// rows, since Simple no longer shows LoRA rows at all.
function _modelModeBannerHtml(modelEntry) {
  if (!modelEntry) return '';
  return modelEntry.style_trigger
    ? `<div class="gen-params-hint">Studio mode &mdash; the style trigger is added to your prompt automatically.</div>`
    : `<div class="gen-params-hint">General mode &mdash; this model has no style LoRA, so your prompt goes to it unchanged.</div>`;
}

// Video counterpart -- task item 4: "Show a one-line note that H3 generates
// synchronized audio." `has_audio` (data/studio/scripts/models.json) is true
// only for MiniMax H3 today; Wan shows no banner at all (nothing to say).
function _videoModelBannerHtml(modelEntry) {
  if (!modelEntry || !modelEntry.has_audio) return '';
  return `<div class="gen-params-hint">This model generates synchronized audio along with the video.</div>`;
}

// Task item 4: style LoRAs are always offered under Advanced (a user may
// want to stack a second one, or swap out the model preset's default for
// another); character LoRAs only when the selected model declares
// `characters_allowed: true`. A registry entry with no/unknown `kind` still
// shows (in its own bucket) rather than silently vanishing.
function _loraRowsHtml(kind, effective, modelEntry) {
  const opts = _optionsCache || { loras: [] };
  const all = (opts.loras || []);
  const charactersAllowed = !!(modelEntry && modelEntry.characters_allowed);
  const relevant = all.filter(l => (l.kind || '').toLowerCase() !== 'character' || charactersAllowed);
  if (!all.length) {
    return `<div class="gen-params-hint">No LoRA registry loaded yet (needs /api/comfy/options from the backend).</div>`;
  }
  if (!relevant.length) {
    return `<div class="gen-params-hint">This model has no style LoRAs registered, and character LoRAs aren't offered for it.</div>`;
  }
  const selected = effective.loras === null ? _defaultLoraMapFor(effective) : effective.loras;
  const row = (l) => {
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
  };
  const styleRows = relevant.filter(l => (l.kind || '').toLowerCase() === 'style');
  const characterRows = relevant.filter(l => (l.kind || '').toLowerCase() === 'character');
  const otherRows = relevant.filter(l => !['style', 'character'].includes((l.kind || '').toLowerCase()));
  let html = '';
  if (styleRows.length) html += `<div class="gen-params-section-title">Style LoRAs</div>` + styleRows.map(row).join('');
  if (characterRows.length) html += `<div class="gen-params-section-title">Characters</div>` + characterRows.map(row).join('');
  if (otherRows.length) html += `<div class="gen-params-section-title">Other LoRAs</div>` + otherRows.map(row).join('');
  return html;
}

// `field` defaults to 'input_image' (the start/first-frame slot every kind
// already had) so every EXISTING call site (image's img2img source, video's
// start frame -- both called with just 3 args) is untouched. MiniMax H3's
// OPTIONAL end frame reuses this same function with field='last_frame' --
// a SECOND, independent image slot, not a second copy of this markup.
function _inputImageHtml(kind, effective, label, field = 'input_image') {
  const img = effective[field];
  // DEFECT 18: prefer the persisted object's OWN previewUrl (a gallery pick
  // -- a real, durable server URL) when present; otherwise fall back to this
  // same page load's live-only upload preview/progress (see
  // _liveUploadState above) rather than a stale, dead blob: URL that would
  // have been read back from localStorage before this fix.
  const live = img ? _liveUploadState.get(`${kind}:${field}`) : null;
  const previewUrl = (img && img.previewUrl) || (live && live.previewUrl) || null;
  const uploading = !!(img && (img.uploading || (live && live.uploading)));
  return `
    <div class="gen-params-imgrow">
      <button type="button" class="gen-params-btn" data-action="upload-input-image" data-image-field="${esc(field)}">${img ? 'Replace' : 'Upload'} ${esc(label)}</button>
      <button type="button" class="gen-params-btn" data-action="pick-gallery-image" data-image-field="${esc(field)}">Pick from Gallery</button>
      <span class="gen-params-hint">${img ? esc(img.name) + (uploading ? ' (uploading...)' : '') : 'None selected'}</span>
      ${img ? `<button type="button" class="gen-params-btn" data-action="clear-input-image" data-image-field="${esc(field)}">Clear</button>` : ''}
    </div>
    ${img && previewUrl ? `<img class="gen-params-thumb" src="${esc(previewUrl)}" alt="">` : ''}
    <input type="file" accept="image/*" data-role="gen-params-file-input" data-image-field="${esc(field)}" style="display:none">
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

// `field` defaults to 'input_image' -- same "every existing call site
// untouched" reasoning as _inputImageHtml() above; MiniMax H3's end frame
// passes field='last_frame' so a picked thumbnail lands in the right slot.
async function _openGalleryPicker(kind, field = 'input_image') {
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
        _setField(kind, field, {
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

// Task item 1: the primary control, grouped by `group` ('Studio' | 'General'
// | 'Video' | ...) via <optgroup>, default to the resolved model key,
// description shown underneath. An `available: false` entry (routes/
// comfy_routes.py's resolve_model_entries() — model files not on disk /
// ComfyUI hasn't been restarted to see them yet) is offered but disabled,
// with a tooltip naming what's missing; its label also gets a plain-text
// "(unavailable)" suffix since a disabled <option>'s title tooltip is not
// reliably hoverable in every browser.
//
// Shared by BOTH the Image tab (`_optionsCache.models`) and the Video tab
// (`_optionsCache.video_models`, task item 4: "the same model-first picker
// the Image tab now has") — everything below the model-list/current-key
// resolution is identical between the two, so only that resolution branches
// on `kind`.
function _modelPickerHtml(kind, effective) {
  const isVideo = kind === 'video';
  const models = (_optionsCache && (isVideo ? _optionsCache.video_models : _optionsCache.models)) || [];
  const modelKey = isVideo ? _effectiveVideoModelKey(effective) : _effectiveModelKey(effective);
  if (!models.length) {
    return `<div class="gen-params-row settings-row"><span class="settings-label">Model</span>
      <select class="settings-select" disabled><option>Loading models…</option></select></div>`;
  }
  const groupOrder = [];
  const byGroup = new Map();
  models.forEach(m => {
    const g = m.group || 'Other';
    if (!byGroup.has(g)) { byGroup.set(g, []); groupOrder.push(g); }
    byGroup.get(g).push(m);
  });
  let optionsHtml = '';
  groupOrder.forEach(g => {
    optionsHtml += `<optgroup label="${esc(g)}">`;
    byGroup.get(g).forEach(m => {
      const unavailable = m.available === false;
      const missing = (m.missing || []).join(', ');
      const title = unavailable
        ? `${m.name} — model files not downloaded yet${missing ? ` (missing: ${missing})` : ''}`
        : (m.description || '');
      optionsHtml += `<option value="${esc(m.key)}"${m.key === modelKey ? ' selected' : ''}${unavailable ? ' disabled' : ''}${title ? ` title="${esc(title)}"` : ''}>${esc(m.name)}${unavailable ? ' (unavailable)' : ''}</option>`;
    });
    optionsHtml += `</optgroup>`;
  });
  const current = isVideo ? _videoModelEntry(modelKey) : _modelEntry(modelKey);
  return `<div class="gen-params-row settings-row"><span class="settings-label">Model</span>
      <select class="settings-select" data-field="model">${optionsHtml}</select></div>
      ${current && current.description ? `<div class="gen-params-hint">${esc(current.description)}</div>` : ''}`;
}

function _resetButtonHtml() {
  return `<button type="button" class="gen-params-btn" data-action="reset-model-defaults" style="align-self:flex-start;">Reset to model defaults</button>`;
}

function _renderImagePanel(effective, advanced) {
  const opts = _optionsCache || {};
  const modelKey = _effectiveModelKey(effective);
  const modelEntry = _modelEntry(modelKey);

  // "Model + prompt + Generate is the ENTIRE default surface" — this top
  // section renders on BOTH tabs (prompt/Generate themselves are the main
  // composer, outside this popup entirely); Simple stops right here.
  let html = '';
  html += _modelPickerHtml('image', effective);
  html += _modelModeBannerHtml(modelEntry);
  html += _resetButtonHtml();

  if (!advanced) {
    return html;
  }

  // Advanced — everything that used to be split across Simple/Advanced now
  // lives here together: workflow, mode/img2img, LoRAs, size/batch, then the
  // old numeric Advanced knobs, then model overrides.
  const wf = (_workflowsCache && _workflowsCache.workflows || []).filter(w => w.kind === 'image' || w.kind === 'unknown');
  html += `<div class="gen-params-section-title">Workflow</div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Workflow</span>
    <select class="settings-select" data-field="workflow">
      <option value="Custom"${effective.workflow === 'Custom' ? ' selected' : ''}>Custom (defaults)</option>
      ${wf.map(w => `<option value="${esc(w.name)}"${effective.workflow === w.name ? ' selected' : ''}>${esc(w.name)}</option>`).join('')}
    </select></div>`;

  html += `<div class="gen-params-section-title">Mode</div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Mode</span>
    <div class="gen-params-tabs" style="margin:0;border:0;padding:0;flex:1;">
      <button type="button" class="gen-params-tab${effective.mode !== 'img2img' ? ' active' : ''}" data-action="set-mode" data-mode-value="txt2img">txt2img</button>
      <button type="button" class="gen-params-tab${effective.mode === 'img2img' ? ' active' : ''}" data-action="set-mode" data-mode-value="img2img">img2img</button>
    </div></div>`;
  if (effective.mode === 'img2img') {
    html += _inputImageHtml('image', effective, 'source image');
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Denoise</span>
      <input type="range" min="0" max="1" step="0.05" value="${_escNum(effective.denoise)}" data-field="denoise" style="flex:1;">
      <span class="gen-params-lora-weight">${Number(effective.denoise).toFixed(2)}</span></div>`;
  }

  html += `<div class="gen-params-section-title">LoRAs <span class="gen-params-hint">(defaults to the selected model's own LoRAs)</span></div>`;
  html += _loraRowsHtml('image', effective, modelEntry);

  html += `<div class="gen-params-section-title">Size</div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Preset</span>
    <select class="settings-select" data-field="size_preset">${_sizeOptionsHtml('image', effective.size_preset, modelEntry)}</select></div>`;
  if (effective.size_preset === 'custom') {
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Width</span><input type="number" class="settings-select" min="64" step="8" value="${_escNum(effective.width)}" data-field="width"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Height</span><input type="number" class="settings-select" min="64" step="8" value="${_escNum(effective.height)}" data-field="height"></div>`;
  }
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Batch</span><input type="number" class="settings-select" min="1" max="8" step="1" value="${_escNum(effective.batch)}" data-field="batch"></div>`;

  html += `<div class="gen-params-section-title">Prompt &amp; sampling</div>`;
  html += `<div class="gen-params-row settings-row" style="align-items:flex-start;"><span class="settings-label">Negative</span><textarea class="settings-select" rows="2" data-field="negative_prompt">${esc(effective.negative_prompt)}</textarea></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Seed</span>
    <input type="number" class="settings-select" value="${_escNum(effective.seed)}" data-field="seed" ${effective.randomize_seed ? 'disabled' : ''}>
    <label style="display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;"><input type="checkbox" data-field="randomize_seed" ${effective.randomize_seed ? 'checked' : ''}> Randomize</label></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Steps</span><input type="number" class="settings-select" min="1" max="150" value="${_escNum(effective.steps)}" data-field="steps"></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">CFG</span><input type="number" class="settings-select" min="0" max="30" step="0.1" value="${_escNum(effective.cfg)}" data-field="cfg"></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Sampler</span><select class="settings-select" data-field="sampler">${_selectHtml(opts.samplers, effective.sampler)}</select></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Scheduler</span><select class="settings-select" data-field="scheduler">${_selectHtml(opts.schedulers, effective.scheduler)}</select></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">Shift</span><input type="number" class="settings-select" min="0" max="20" step="0.1" value="${_escNum(effective.shift)}" data-field="shift"></div>`;

  html += `<div class="gen-params-section-title">Model overrides</div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">UNET</span><select class="settings-select" data-field="unet">${_selectHtml(opts.unets, effective.unet, true)}</select></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">CLIP</span><select class="settings-select" data-field="clip">${_selectHtml(opts.clips, effective.clip, true)}</select></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">CLIP type</span><input type="text" class="settings-select" placeholder="auto" value="${esc(effective.clip_type)}" data-field="clip_type"></div>`;
  html += `<div class="gen-params-row settings-row"><span class="settings-label">VAE</span><select class="settings-select" data-field="vae">${_selectHtml(opts.vaes, effective.vae, true)}</select></div>`;
  html += _heartbeatRowHtml();
  return html;
}

// MODEL-FIRST, same redesign as _renderImagePanel() (task item 4: "give the
// Video tab the same model-first picker the Image tab now has"). Simple =
// model + start/end frame + duration + size + fps; Advanced = workflow +
// seed/steps/sampler/scheduler + whichever engine-specific knobs apply.
// `isH3` decides which fields show -- "Hide Wan-only fields (the two
// lightning-LoRA weights, cfg, negative prompt) when H3 is selected, and
// vice versa" (task item 4): the if/else split below IS that hiding, in
// both directions, for every field that differs between the two engines.
function _renderVideoPanel(effective, advanced) {
  const opts = _optionsCache || {};
  const modelKey = _effectiveVideoModelKey(effective);
  const modelEntry = _videoModelEntry(modelKey);
  const isH3 = !!(modelEntry && modelEntry.engine === 'minimax_h3');

  let html = '';
  html += _modelPickerHtml('video', effective);
  html += _videoModelBannerHtml(modelEntry);
  html += _resetButtonHtml();

  if (!advanced) {
    html += `<div class="gen-params-section-title">Start frame</div>`;
    html += _inputImageHtml('video', effective, 'start frame', 'input_image');
    if (isH3) {
      html += `<div class="gen-params-section-title">End frame <span class="gen-params-hint">(optional)</span></div>`;
      html += _inputImageHtml('video', effective, 'end frame', 'last_frame');
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Duration (s)</span><input type="number" class="settings-select" min="1" max="30" step="0.5" value="${_escNum(effective.seconds)}" data-field="seconds"></div>`;
    } else {
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Frames</span><input type="number" class="settings-select" min="9" max="241" value="${_escNum(effective.frames)}" data-field="frames"></div>`;
    }
    html += `<div class="gen-params-row settings-row"><span class="settings-label">FPS</span><input type="number" class="settings-select" min="1" max="60" value="${_escNum(effective.fps)}" data-field="fps"></div>`;
    html += `<div class="gen-params-section-title">Size</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Preset</span>
      <select class="settings-select" data-field="size_preset">${_sizeOptionsHtml('video', effective.size_preset, modelEntry)}</select></div>`;
    if (effective.size_preset === 'custom') {
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Width</span><input type="number" class="settings-select" min="64" step="8" value="${_escNum(effective.width)}" data-field="width"></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Height</span><input type="number" class="settings-select" min="64" step="8" value="${_escNum(effective.height)}" data-field="height"></div>`;
    }
    html += `<div class="gen-params-hint">Video has no knowledge of the studio character/style LoRAs — it animates an already-on-model start frame, it does not generate style.</div>`;
  } else {
    const wf = (_workflowsCache && _workflowsCache.workflows || []).filter(w => w.kind === 'video' || w.kind === 'unknown');
    html += `<div class="gen-params-section-title">Workflow</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Workflow</span>
      <select class="settings-select" data-field="workflow">
        <option value="Custom"${effective.workflow === 'Custom' ? ' selected' : ''}>Custom (defaults)</option>
        ${wf.map(w => `<option value="${esc(w.name)}"${effective.workflow === w.name ? ' selected' : ''}>${esc(w.name)}</option>`).join('')}
      </select></div>`;

    html += `<div class="gen-params-section-title">Sampling</div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Seed</span>
      <input type="number" class="settings-select" value="${_escNum(effective.seed)}" data-field="seed" ${effective.randomize_seed ? 'disabled' : ''}>
      <label style="display:flex;align-items:center;gap:4px;font-size:11px;white-space:nowrap;"><input type="checkbox" data-field="randomize_seed" ${effective.randomize_seed ? 'checked' : ''}> Randomize</label></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">${isH3 ? 'Steps' : 'Total steps'}</span><input type="number" class="settings-select" min="1" max="40" value="${_escNum(effective.steps)}" data-field="steps"></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Sampler</span><select class="settings-select" data-field="sampler">${_selectHtml(opts.samplers, effective.sampler)}</select></div>`;
    html += `<div class="gen-params-row settings-row"><span class="settings-label">Scheduler</span><select class="settings-select" data-field="scheduler">${_selectHtml(opts.schedulers, effective.scheduler)}</select></div>`;

    if (isH3) {
      html += `<div class="gen-params-section-title">Sigma shift</div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Shift (video)</span><input type="number" class="settings-select" min="0" max="30" step="0.1" value="${_escNum(effective.shift_video)}" data-field="shift_video"></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Shift (audio)</span><input type="number" class="settings-select" min="0" max="30" step="0.1" value="${_escNum(effective.shift_audio)}" data-field="shift_audio"></div>`;
    } else {
      html += `<div class="gen-params-row settings-row" style="align-items:flex-start;"><span class="settings-label">Negative</span><textarea class="settings-select" rows="2" data-field="negative_prompt">${esc(effective.negative_prompt)}</textarea></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">CFG</span><input type="number" class="settings-select" min="0" max="20" step="0.1" value="${_escNum(effective.cfg)}" data-field="cfg"></div>`;
      html += `<div class="gen-params-section-title">Speed LoRA strengths</div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">High-noise</span><input type="range" min="0" max="2" step="0.05" value="${_escNum(effective.lora_high_weight)}" data-field="lora_high_weight" style="flex:1;"><span class="gen-params-lora-weight">${Number(effective.lora_high_weight).toFixed(2)}</span></div>`;
      html += `<div class="gen-params-row settings-row"><span class="settings-label">Low-noise</span><input type="range" min="0" max="2" step="0.05" value="${_escNum(effective.lora_low_weight)}" data-field="lora_low_weight" style="flex:1;"><span class="gen-params-lora-weight">${Number(effective.lora_low_weight).toFixed(2)}</span></div>`;
    }
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
  let body;
  if (_currentKind === 'music') body = _renderMusicPanel(effective, advanced);
  else if (_currentKind === 'video') body = _renderVideoPanel(effective, advanced);
  else body = _renderImagePanel(effective, advanced);
  _bodyEl.innerHTML = scopeRow + body;
}

// ── Delegated event handlers for the rendered body ──

function _renderMusicPanel(effective, advanced) {
  const backends = (_musicBackendsCache && _musicBackendsCache.backends) || [];
  const available = backends.filter(b => b.available);
  const current = effective.backend || (_musicBackendsCache && _musicBackendsCache.default) || '';
  const opts = available.map(b =>
    `<option value="${esc(b.key)}"${b.key === current ? ' selected' : ''}>${esc(b.name || b.key)}</option>`
  ).join('');
  const chosen = available.find(b => b.key === current) || available[0];
  return `
    <div class="gen-params-section-title">Backend</div>
    <div class="gen-params-row settings-row">
      <label class="settings-label">Engine</label>
      <select class="settings-select" data-field="backend">${opts || '<option value="">(none available)</option>'}</select>
    </div>
    <div class="gen-params-hint">${esc((chosen && chosen.description) || 'MiniMax Music 3 — local Comfy weights or hosted API. Not MiniMax H3 video.')}</div>
    <div class="gen-params-section-title">Song</div>
    <div class="gen-params-hint">The send bar is the caption (style / mood / arrangement). Optional lyrics with section tags like [Verse] and [Chorus].</div>
    <textarea class="settings-textarea" data-field="lyrics" rows="5" placeholder="[Verse]&#10;optional lyrics">${esc(effective.lyrics || '')}</textarea>
    <div class="gen-params-row settings-row">
      <label class="settings-label">Seconds</label>
      <input type="number" class="settings-input" data-field="seconds" min="5" max="300" step="1" value="${_escNum(effective.seconds || 60)}">
    </div>
    <div class="gen-params-row settings-row">
      <label style="display:flex;align-items:center;gap:6px;font-size:11px;"><input type="checkbox" data-field="instrumental" ${effective.instrumental ? 'checked' : ''}> Instrumental (no vocals)</label>
    </div>
    ${advanced ? `<div class="gen-params-row settings-row">
      <label class="settings-label">Seed</label>
      <input type="number" class="settings-input" data-field="seed" value="${_escNum(effective.seed || 0)}">
    </div>
    <div class="gen-params-row settings-row">
      <label style="display:flex;align-items:center;gap:6px;font-size:11px;"><input type="checkbox" data-field="randomize_seed" ${effective.randomize_seed ? 'checked' : ''}> Randomize seed</label>
    </div>` : ''}
    ${_heartbeatRowHtml()}`;
}

function _onBodyClick(e) {
  const scopeBtn = e.target.closest('[data-action="set-scope"]');
  if (scopeBtn) { _editScope = scopeBtn.dataset.scopeValue; _renderPanel(); return; }

  const resetBtn = e.target.closest('[data-action="reset-model-defaults"]');
  if (resetBtn) { _resetToModelDefaults(_currentKind); return; }

  const modeBtn = e.target.closest('[data-action="set-mode"]');
  if (modeBtn) { _setField('image', 'mode', modeBtn.dataset.modeValue); _renderPanel(); return; }

  const uploadBtn = e.target.closest('[data-action="upload-input-image"]');
  if (uploadBtn) {
    // data-image-field disambiguates WHICH image slot this button is for --
    // the video panel can render two of these at once (start frame /
    // MiniMax H3's optional end frame), so a bare, unqualified query would
    // always find the FIRST file input in the DOM regardless of which
    // button was actually clicked.
    const field = uploadBtn.dataset.imageField || 'input_image';
    const input = _bodyEl.querySelector(`[data-role="gen-params-file-input"][data-image-field="${field}"]`);
    if (input) {
      input.onchange = () => { if (input.files && input.files[0]) _uploadInputImage(_currentKind, input.files[0], field); };
      input.click();
    }
    return;
  }
  const galleryBtn = e.target.closest('[data-action="pick-gallery-image"]');
  if (galleryBtn) { _openGalleryPicker(_currentKind, galleryBtn.dataset.imageField || 'input_image'); return; }
  const clearBtn = e.target.closest('[data-action="clear-input-image"]');
  if (clearBtn) { _clearInputImage(_currentKind, clearBtn.dataset.imageField || 'input_image'); return; }
}

function _onBodyChange(e) {
  const field = e.target.dataset ? e.target.dataset.field : null;
  if (field === 'model') { _onModelChange(_currentKind, e.target.value); return; }
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
      ? _defaultLoraMapFor(effective)
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
      ? _defaultLoraMapFor(effective)
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
  // Ask the session module FIRST and use our own cached id only as a fallback.
  //
  // This order matters and used to be reversed. `_currentSessionId` is only
  // written by onSessionSwitch(), which fires when you SWITCH sessions -- not
  // when a new chat is created. So starting a fresh chat and generating in it
  // left `_currentSessionId` pointing at the previous session, this returned
  // false for a job whose sessionId was perfectly correct, and _onJobDone()
  // silently skipped painting the result. Combined with the unconditional
  // holder.remove() above it, the progress bubble just vanished and nothing
  // replaced it -- the render was fine, in the Gallery and in the DB, but the
  // UI showed no completion at all. Reported live 2026-08-07.
  //
  // sessionModule.getCurrentSessionId() is the authority; the cache is only
  // there for the case where the module hasn't exposed the getter.
  const live = sessionModule.getCurrentSessionId ? sessionModule.getCurrentSessionId() : null;
  const current = live || _currentSessionId;
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
    _showRewriteNote(job);
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
  const seed = effective.randomize_seed ? Math.floor(Math.random() * 2147483647) : Number(effective.seed) || 0;

  if (kind === 'music') {
    return {
      prompt: promptText,
      caption: promptText,
      lyrics: effective.lyrics || '',
      seconds: Number(effective.seconds) || 60,
      backend: effective.backend || '',
      instrumental: !!effective.instrumental,
      seed,
      randomize_seed: !!effective.randomize_seed,
    };
  }

  if (kind === 'video') {
    // Video: model-preset-aware (task item 4), mirroring the image
    // branch below field-for-field: only TOUCHED fields (_touchedSet()) are
    // sent explicitly, everything else omitted so the backend's
    // apply_model_preset() fills it from the selected video model's own
    // recipe. `model` is always sent (both engines resolve through
    // data/studio/scripts/models.json's `video_models` section now -- no
    // more "video is untouched by the model-preset system").
    const sid = _currentSessionId || sessionModule.getCurrentSessionId();
    const touched = _touchedSet('video', sid);
    const modelKey = _effectiveVideoModelKey(effective);
    const modelEntry = _videoModelEntry(modelKey);
    const isH3 = !!(modelEntry && modelEntry.engine === 'minimax_h3');

    const params = {
      prompt: promptText,
      model: modelKey,
      seed,
      randomize_seed: !!effective.randomize_seed,
    };
    if (touched.steps) params.steps = Number(effective.steps);
    if (touched.sampler) params.sampler = effective.sampler;
    if (touched.scheduler) params.scheduler = effective.scheduler;
    if (touched.fps) params.fps = Number(effective.fps);
    // DEFECT 9: _applyWorkflow() marks width/height touched directly (it has
    // no `size_preset` concept to seed -- introspect_graph() only returns raw
    // width/height) but never touches size_preset itself, so a
    // workflow-seeded custom size was silently omitted from the payload
    // despite being marked touched.
    if (touched.size_preset || touched.width || touched.height) {
      const [w, h] = String(effective.size_preset === 'custom' ? `${effective.width}x${effective.height}` : effective.size_preset).split('x').map(Number);
      if (w) params.width = w;
      if (h) params.height = h;
    }
    // Start frame -- both engines, same two sources as the image img2img
    // branch below (upload vs gallery-pick, resolved server-side).
    const startRef = _resolvedInputImageName(effective.input_image);
    if (startRef) params.input_image = startRef;

    if (isH3) {
      if (touched.seconds) params.seconds = Number(effective.seconds);
      if (touched.shift_video) params.shift_video = Number(effective.shift_video);
      if (touched.shift_audio) params.shift_audio = Number(effective.shift_audio);
      // OPTIONAL end frame -- H3 only; Wan has no such slot.
      const endRef = _resolvedInputImageName(effective.last_frame);
      if (endRef) params.last_frame = endRef;
    } else {
      if (touched.negative_prompt) params.negative_prompt = effective.negative_prompt;
      if (touched.cfg) params.cfg = Number(effective.cfg);
      if (touched.frames) params.frames = Number(effective.frames);
      // Wan's two fixed-name speed-LoRA weights -- always sent (there is no
      // "untouched -> let the backend decide" state for these; the sentinel
      // comfy_name values are what _extract_wan_lora_weights() on the
      // backend keys off of, per src/comfy_graphs.py).
      params.loras = [
        { comfy_name: 'wan_high_noise', weight: Number(effective.lora_high_weight), role: 'high_noise' },
        { comfy_name: 'wan_low_noise', weight: Number(effective.lora_low_weight), role: 'low_noise' },
      ];
    }
    return params;
  }

  // ── Image: model-preset-aware (task item 3). Only fields the user
  // actually TOUCHED (_touchedSet()) are sent explicitly — everything else
  // is omitted so the backend's apply_model_preset() fills it from the
  // selected model's own recipe ("any field you also send explicitly
  // WINS" — routes/comfy_routes.py). This is what makes picking a model in
  // the dropdown actually change steps/cfg/sampler/etc instead of always
  // sending this panel's last-touched numbers to every model. ──
  const sid = _currentSessionId || sessionModule.getCurrentSessionId();
  const touched = _touchedSet('image', sid);
  const modelKey = _effectiveModelKey(effective);
  const modelEntry = _modelEntry(modelKey);
  const charactersAllowed = !!(modelEntry && modelEntry.characters_allowed);

  const params = {
    prompt: promptText,
    model: modelKey,
    seed,
    randomize_seed: !!effective.randomize_seed,
    batch: Number(effective.batch) || 1,
  };
  if (touched.negative_prompt) params.negative_prompt = effective.negative_prompt;
  if (touched.steps) params.steps = Number(effective.steps);
  if (touched.cfg) params.cfg = Number(effective.cfg);
  if (touched.sampler) params.sampler = effective.sampler;
  if (touched.scheduler) params.scheduler = effective.scheduler;
  if (touched.shift) params.shift = Number(effective.shift);
  // DEFECT 9: _applyWorkflow() marks width/height touched directly (it also
  // now forces size_preset to 'custom' when it does -- see that function),
  // but check all three here defensively so a custom size set any other way
  // that leaves size_preset untouched still reaches the payload.
  if (touched.size_preset || touched.width || touched.height) {
    const [w, h] = String(effective.size_preset === 'custom' ? `${effective.width}x${effective.height}` : effective.size_preset).split('x').map(Number);
    if (w) params.width = w;
    if (h) params.height = h;
  }
  if (effective.unet) params.unet = effective.unet;
  if (effective.clip) params.clip = effective.clip;
  if (effective.clip_type) params.clip_type = effective.clip_type;
  if (effective.vae) params.vae = effective.vae;

  if (effective.loras !== null) {
    const loraList = [];
    Object.entries(effective.loras || {}).forEach(([key, weight]) => {
      const entry = (_optionsCache?.loras || []).find(l => l.key === key);
      if (!entry) return;
      // A LoRA trained for the OTHER base architecture, or a character LoRA
      // left over from a model that allowed characters, can't just ride
      // along after a model switch — _onModelChange() already clears
      // `loras` on an architecture change, so this is defense-in-depth for
      // the narrower "same arch, but this model doesn't allow characters"
      // case (see _loraRowsHtml's own filtering, which this mirrors).
      if ((entry.kind || '').toLowerCase() === 'character' && !charactersAllowed) return;
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
      // the backend can tell whether any STYLE LoRA is active. LoraParam
      // declares this field, so it survives Pydantic rather than being
      // dropped.
      loraList.push({ comfy_name: entry.resolved_name, weight: Number(weight), kind: entry.kind || null });
    });
    params.loras = loraList;
  }
  // else: untouched — omit `loras` entirely so the backend fills it from the
  // selected model's OWN bundled LoRAs (apply_model_preset()); this is the
  // ComfyUI-side mechanism that makes "General" models like Qwen-Image or
  // plain Z-Image render with NO studio LoRAs at all without this panel
  // having to special-case them.

  if (effective.mode === 'img2img') {
    params.denoise = Number(effective.denoise);
    // Both input_image sources resolve server-side — see
    // _resolvedInputImageName(). Upload goes through /api/comfy/upload into
    // ComfyUI's own input/ dir; gallery names go through the backend's
    // GENERATED_IMAGE_RE passthrough. Verified live 2026-08-03.
    const ref = _resolvedInputImageName(effective.input_image);
    if (ref) params.input_image = ref;
  }
  return params;
}

function _mountProgressBubble(kind) {
  const box = document.getElementById('chat-history');
  if (!box) return null;
  const holder = document.createElement('div');
  holder.className = 'msg msg-ai gen-progress-msg';
  const roleTs = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  holder.innerHTML = `<div class="role">${kind === 'music' ? 'Music' : (kind === 'video' ? 'Video' : 'Image')} <span class="role-timestamp">${esc(roleTs)}</span></div>
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
    const cancelBase = job.kind === 'music' ? '/api/music/cancel' : '/api/comfy/cancel';
    await fetch(`${API_BASE}${cancelBase}/${encodeURIComponent(job.jobId)}`, { method: 'POST', credentials: 'same-origin' });
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
  const audio = (data && data.audio) || [];
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
    audio.forEach(item => {
      if (!item || !item.url) return;
      const bubble = chatRenderer.buildAudioBubble(item.url, job.prompt, job.params && job.params.backend);
      if (box) box.appendChild(bubble);
    });
    images.forEach(img => {
      if (!img || !img.url) return;
      const isVideo = /\.(mp4|mov|webm|mkv|m4v)$/i.test(img.filename || img.url);
      const bubble = isVideo
        ? chatRenderer.buildVideoBubble(img.url, job.prompt)
        : chatRenderer.buildImageBubble(img.url, job.prompt, job.kind === 'video' ? 'video' : 'image', `${job.params.width}x${job.params.height}`, null, img.gallery_id);
      if (bubble && job.usedRewrite && job.rewrittenPrompt) {
        const note = document.createElement('div');
        note.className = 'gen-rewrite-note';
        note.style.cssText = 'opacity:0.75;font-size:0.85em;margin-top:0.35em;';
        note.textContent = 'Rewrote: ' + job.rewrittenPrompt;
        const body = bubble.querySelector('.body') || bubble;
        body.appendChild(note);
      }
      if (box) box.appendChild(bubble);
    });
    if (box) scrollHistory();
  } else {
    // Belt-and-braces: the holder was already removed unconditionally above,
    // so if we get here the user sees the progress bubble disappear with
    // NOTHING replacing it -- which reads as "the render silently died" even
    // though it succeeded. That exact confusion was reported 2026-08-07.
    // Never fail silently: say where the result went. (This is now the
    // genuine cross-session case only -- the stale-cache false negative that
    // used to land here is fixed in _isCurrentSession().)
    const n = images.length + audio.length;
    const label = job.kind === 'music' ? 'Music' : (job.kind === 'video' ? 'Video' : 'Image');
    showToast(n
      ? `${label} finished in another chat - saved to the Gallery.`
      : 'Generation finished - saved to the Gallery.');
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

  const streamBase = job.kind === 'music' ? '/api/music/stream' : '/api/comfy/stream';
  const es = new EventSource(`${API_BASE}${streamBase}/${encodeURIComponent(job.jobId)}`);
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

// Paperclip attach on Image tab used to be ignored: chat.js hands generate()
// only the prompt text, and _buildParamsPayload() only sends input_image when
// mode is already img2img with a panel upload. If a pending image is sitting
// on the composer and there is no source yet, upload it to Comfy and flip
// to img2img so Generate restyles the photo.
async function _adoptPendingImageIfNeeded(kind, sid) {
  if (kind !== 'image') return;
  const already = _resolvedInputImageName(_effective(kind, sid).input_image);
  if (already) return;
  let files = [];
  try { files = fileHandler.getPendingRaw() || []; } catch (_) { return; }
  const file = files.find((f) => f && (((f.type || '').startsWith('image/')) || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name || '')));
  if (!file) return;
  _setField(kind, 'mode', 'img2img');
  await _uploadInputImage(kind, file, 'input_image');
  try { fileHandler.clearPending(); } catch (_) {}
  if (!_resolvedInputImageName(_effective(kind, sid).input_image)) {
    throw new Error('Could not use the attached image as an img2img source.');
  }
}


// Image rewrite opt-out. Tokens are stripped from the CLIP prompt.
// -force / -raw : send skip_rewrite so the CPU rewriter does not run.
function _stripRewriteForce(text) {
  const src = String(text || '');
  const re = /(?:^|\s)(-force|-raw)\b/gi;
  let skip = false;
  const prompt = src.replace(re, () => { skip = true; return ' '; }).replace(/\s+/g, ' ').trim();
  return { prompt, skipRewrite: skip };
}

export async function generate(kind, promptText, sessionId) {
  const sid = sessionId || sessionModule.getCurrentSessionId();
  if (!sid) { showToast('No active chat session.'); return; }
  if (_activeJobs.has(sid)) { showToast('A generation is already running in this chat.'); return; }
  if (!promptText || !promptText.trim()) { showToast('Type a prompt first.'); return; }

  const _force = kind === 'image' ? _stripRewriteForce(promptText) : { prompt: promptText.trim(), skipRewrite: false };
  if (kind === 'image') promptText = _force.prompt;
  if (!promptText || !promptText.trim()) { showToast('Type a prompt first.'); return; }

  if (kind === 'music') await _fetchMusicBackends();
  else await Promise.all([_fetchOptions(), _fetchWorkflows()]);
  if (kind === 'image') await _adoptPendingImageIfNeeded(kind, sid);
  const effective = _effective(kind, sid);
  const params = _buildParamsPayload(kind, effective, promptText.trim());

  // Render the user's own turn immediately, same as a normal chat message
  // (chat.js's handleChatSubmit() branches to this function BEFORE ever
  // calling addMessage('user', ...), so without this the prompt was never
  // shown at all, live or otherwise). The persisted version written by
  // routes/comfy_routes.py's _write_user_turn() carries a small "(via Image
  // tab · model X)" note too; kept off the live bubble here to avoid
  // guessing at backend formatting -- it appears on next reload.
  chatRenderer.addMessage('user', promptText.trim());

  // Register the job before any GPU wait so a second generate cannot start.
  const holder = _mountProgressBubble(kind);
  const job = { sessionId: sid, kind, prompt: promptText.trim(), params, holder, startedAt: Date.now() };
  _activeJobs.set(sid, job);

  // GPU time-share: only on generate, never on tab click / onModeChange.
  try {
    const st = await gpuStatus(kind);
    if (st && st.switch_needed) {
      _updateProgressDom(holder, { label: GPU_SWITCH_MESSAGE });
      await prepareGpuFor(kind, { onSwitching: () => {} });
    }
  } catch (_) {}
  _updateProgressDom(holder, { label: 'Queued…' });

  if (holder) {
    const cancelBtn = holder.querySelector('.gen-progress-cancel');
    if (cancelBtn) cancelBtn.addEventListener('click', () => _cancelJob(job));
  }

  try {
    const isMusic = kind === 'music';
    const url = isMusic ? `${API_BASE}/api/music/generate` : `${API_BASE}/api/comfy/generate`;
    const body = isMusic
      ? { session_id: sid, params: { ...params, prompt: promptText.trim() } }
      : { kind, workflow: effective.workflow || 'Custom', session_id: sid, params, skip_rewrite: !!(kind === 'image' && _force.skipRewrite) };
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok || !data || !data.job_id) {
      throw new Error((data && (data.error || data.detail)) || `HTTP ${res.status}`);
    }
    job.jobId = data.job_id;
    job.promptId = data.prompt_id;
    if (kind === 'image') {
      job.originalPrompt = data.original_prompt || job.prompt;
      job.rewrittenPrompt = data.rewritten_prompt || job.prompt;
      job.usedRewrite = !!data.used_rewrite;
      job.skipRewrite = !!_force.skipRewrite;
      _showRewriteNote(job);
    }
    _rememberInflight(job);
    _connectStream(job);
  } catch (e) {
    // _onJobError already surfaces this (inline in the bubble, or a toast if
    // the bubble couldn't be mounted) — don't also reject the promise, or
    // chat.js's .catch() on generate() would show a second, redundant toast.
    _onJobError(job, e.message || 'Failed to start generation');
  }
}

function _showRewriteNote(job) {
  if (!job) return;
  const holder = job.holder;
  if (!holder || !document.body.contains(holder)) return;
  const body = holder.querySelector('.body');
  if (!body || body.querySelector('.gen-rewrite-note')) return;
  const note = document.createElement('div');
  note.className = 'gen-rewrite-note';
  note.style.cssText = 'opacity:0.75;font-size:0.85em;margin-top:0.4em;';
  if (job.skipRewrite) note.textContent = 'Rewrite off (-force)';
  else if (job.usedRewrite && job.rewrittenPrompt) note.textContent = 'Rewrote: ' + job.rewrittenPrompt;
  else return;
  body.appendChild(note);
}

export default { init, onModeChange, onSessionSwitch, generate, isBusy };
