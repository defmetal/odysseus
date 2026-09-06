// Studio-local GPU time-share. Call ONLY from generate / chat-send.
// Tab click / setMode must never import-call prepareGpuFor.

const SWITCH_MSG = 'Switching models, please wait';

export function showGpuSwitchBanner(message) {
  let el = document.getElementById('gpu-switch-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'gpu-switch-banner';
    el.className = 'gpu-switch-banner';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    document.body.appendChild(el);
  }
  el.textContent = message || SWITCH_MSG;
  el.hidden = false;
}

export function hideGpuSwitchBanner() {
  const el = document.getElementById('gpu-switch-banner');
  if (el) el.hidden = true;
}

export async function gpuStatus(purpose, { endpointUrl, model } = {}) {
  const q = new URLSearchParams();
  if (purpose) q.set('purpose', purpose);
  if (endpointUrl) q.set('endpoint_url', endpointUrl);
  if (model) q.set('model', model);
  const res = await fetch(`/api/gpu/status?${q}`, { credentials: 'same-origin' });
  if (res.status === 404) return null;
  if (!res.ok) return null;
  return res.json();
}

export async function prepareGpuFor(purpose, opts = {}) {
  const { endpointUrl, model, onSwitching } = opts;
  let st = null;
  try {
    st = await gpuStatus(purpose, { endpointUrl, model });
  } catch (_) {
    return { switched: false, skipped: true };
  }
  if (!st || !st.switch_needed) {
    return { switched: false, occupant: st && st.occupant, skipped: !st };
  }
  let showedBanner = false;
  try {
    if (typeof onSwitching === 'function') onSwitching(true, SWITCH_MSG);
    else {
      showGpuSwitchBanner(SWITCH_MSG);
      showedBanner = true;
    }
    const res = await fetch('/api/gpu/prepare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        purpose,
        endpoint_url: endpointUrl || '',
        model: model || '',
      }),
    });
    if (!res.ok) {
      return { switched: true, ok: false, error: `HTTP ${res.status}` };
    }
    return await res.json();
  } catch (e) {
    return { switched: true, ok: false, error: (e && e.message) || 'prepare failed' };
  } finally {
    if (typeof onSwitching === 'function') onSwitching(false, SWITCH_MSG);
    if (showedBanner) hideGpuSwitchBanner();
  }
}

export const GPU_SWITCH_MESSAGE = SWITCH_MSG;
export default { gpuStatus, prepareGpuFor, showGpuSwitchBanner, hideGpuSwitchBanner, GPU_SWITCH_MESSAGE };
