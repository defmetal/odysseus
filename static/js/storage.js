// static/js/storage.js
// Centralized localStorage access with key constants and JSON parse safety

// ── Key constants ──
export const KEYS = {
  THEME: 'odysseus-theme',
  TOGGLES: 'odysseus-toggles',
  SIDEBAR_COLLAPSED: 'sidebar-collapsed',
  SIDEBAR_WIDTH: 'sidebar-width',
  SIDEBAR_SIDE: 'sidebar-side',
  CURRENT_SESSION: 'currentSessionId',
  COMPARE_SAVE: 'compare-save-results',
  COMPARE_CHAT: 'compare-continue-chat',
  COMPARE_BLIND: 'compare-blind',
  COMPARE_RANDOM: 'compare-randomize',
  MODELS_EXPANDED: 'odysseus-model-expanded',
  MODEL_ENDPOINTS: 'odysseus-model-endpoints',
  MODEL_SELECTED: 'odysseus-selected-model',
  SORT_ORDER: 'odysseus-sessions-sort',
  CHAT_SEARCH_SCOPE: 'odysseus-search-scope',
  INCOGNITO: 'odysseus-incognito',
  RAG_ACTIVE: 'odysseus-rag-active',
  MCP_ACTIVE: 'odysseus-mcp-active',
  SECTION_ORDER: 'sidebar-section-order',
  ADMIN_LAST_TAB: 'admin-last-tab',
  DENSITY: 'odysseus-density',
  UI_SCALE: 'odysseus-ui-scale',
  WORKSPACE: 'odysseus-workspace',
  // Image/Video tab generation parameters (static/js/genParams.js). Global
  // defaults, overridden per chat session — see loadGenDefaults/loadGenSessions.
  GEN_DEFAULTS: 'odysseus-gen-defaults',
  GEN_SESSIONS: 'odysseus-gen-sessions',
  // In-flight ComfyUI job bookkeeping (job_id -> {sessionId, startedAt, ...})
  // so a session switch or page reload can re-open the SSE progress stream
  // instead of orphaning the bubble — see loadGenInflight/saveGenInflight.
  GEN_INFLIGHT: 'odysseus-gen-inflight'
};

/**
 * Safely get and parse a JSON value from localStorage.
 * Returns fallback on any error.
 */
export function getJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback !== undefined ? fallback : null;
    return JSON.parse(raw);
  } catch (e) {
    console.warn('[Storage] Failed to parse key "' + key + '":', e.message);
    return fallback !== undefined ? fallback : null;
  }
}

/**
 * Set a JSON-serialized value in localStorage.
 */
export function setJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (e) {
    console.warn('[Storage] Failed to set key "' + key + '":', e.message);
  }
}

/**
 * Get a raw string value from localStorage.
 */
export function get(key, fallback) {
  try {
    const val = localStorage.getItem(key);
    return val !== null ? val : (fallback !== undefined ? fallback : null);
  } catch (e) {
    return fallback !== undefined ? fallback : null;
  }
}

/**
 * Set a raw string value in localStorage.
 */
export function set(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch (e) {
    console.warn('[Storage] Failed to set key "' + key + '":', e.message);
  }
}

/**
 * Remove a key from localStorage.
 */
export function remove(key) {
  try {
    localStorage.removeItem(key);
  } catch (e) {
    // Ignore removal errors
  }
}

// ── Toggle state helpers ──

export function loadToggleState() {
  return getJSON(KEYS.TOGGLES, {});
}

export function saveToggleState(state) {
  setJSON(KEYS.TOGGLES, state);
}

export function getToggle(name, fallback) {
  const state = loadToggleState();
  return state[name] !== undefined ? state[name] : (fallback !== undefined ? fallback : false);
}

export function setToggle(name, value) {
  const state = loadToggleState();
  state[name] = value;
  saveToggleState(state);
}

// ── Image/Video generation-parameter helpers ──
// Global defaults live under GEN_DEFAULTS; a chat session may override any
// subset of fields under GEN_SESSIONS[sessionId]. Callers compute the
// effective params themselves as {...defaults, ...sessionOverride} (per-kind,
// e.g. defaults.image / sessions[id].image) — kept dumb here on purpose, same
// division of responsibility as loadToggleState/saveToggleState above.

export function loadGenDefaults() {
  return getJSON(KEYS.GEN_DEFAULTS, {});
}

export function saveGenDefaults(state) {
  setJSON(KEYS.GEN_DEFAULTS, state);
}

export function loadGenSessions() {
  return getJSON(KEYS.GEN_SESSIONS, {});
}

export function saveGenSessions(state) {
  setJSON(KEYS.GEN_SESSIONS, state);
}

// ── In-flight generation job bookkeeping (progress reattach) ──
// Shape: { [jobId]: { jobId, sessionId, kind, prompt, params, startedAt } }.
// Kept dumb here too — genParams.js owns pruning/staleness rules.

export function loadGenInflight() {
  return getJSON(KEYS.GEN_INFLIGHT, {});
}

export function saveGenInflight(state) {
  setJSON(KEYS.GEN_INFLIGHT, state);
}

const Storage = {
  KEYS,
  getJSON,
  setJSON,
  get,
  set,
  remove,
  loadToggleState,
  saveToggleState,
  getToggle,
  setToggle,
  loadGenDefaults,
  saveGenDefaults,
  loadGenSessions,
  saveGenSessions,
  loadGenInflight,
  saveGenInflight
};

export default Storage;
