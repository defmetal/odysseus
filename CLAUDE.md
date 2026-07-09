# CLAUDE.md — orientation for any Claude/agent working in this repo

This repo is a self-hosted **Odysseus** AI-workspace install, and on top of it
Austin (`defmetal@gmail.com`) + his wife are building an **animation studio**:
an original film/comics in the **1990s Toei cel style** (Sailor Moon S1/S2 +
Cutie Honey Flash as *style references only* — all characters/story are theirs).

**This file is just the MAP. The real knowledge is in `data/studio/` — READ the
one(s) relevant to your task** (don't load them all every time):

| Doc | Read it when… |
|---|---|
| `data/studio/ARCHITECTURE.md` | working on image gen/edit/training — the stack, components, data flow, what we built, the VRAM budget |
| `data/studio/ODYSSEUS.md` | working on the platform — Docker/GPU/Ollama/auth/LAN config, the agent/tool/skill system, the codebase patches, the big gotchas |
| `data/studio/TRAINING-GUIDE.md` | curating datasets / training LoRAs — faces, hair, backdrops, character LoRAs, outfit-changeable characters, caption discipline |
| `data/studio/OPERATIONS.md` | running it day-to-day — generate, reboot, serve, sync the repo, troubleshoot |
| `data/studio/V2-WISHLIST.md` | gathering more training stills (what's underrepresented) |
| `data/studio/UPGRADES-2026.md` | planning future upgrades — ComfyUI/ControlNet, video (Wan 2.2/LTX-2), web-search/RAG. Research-backed w/ sources (by Cowork); has a 2026-06-20 status header on what's already done |
| `data/skills/studio/animation-studio-pipeline/SKILL.md` | what the in-app Odysseus agent itself knows |

(`data/` is gitignored — these exist on this machine but not in a fresh clone.
`HANDOFF.md`/`README.md` are earlier drafts; the table above is authoritative.)

Suggested first read for a fresh agent: ARCHITECTURE.md then ODYSSEUS.md.

## The current stack (as of 2026-06-25)
- **Image model: Z-Image-Base, FP8, served by `scripts/diffusion_server.py` on
  port 8100** with the Base-native style LoRA `data/studio/training/
  toei90s_zbase_v2/toei90s_zbase_v2.safetensors`, launched `--guidance 4.5
  --steps 30 --quantize-fp8 --style-config data/studio/scripts/styles.json
  --idle-unload-seconds 300`. (v2 = v1 + 65 hand-made backgrounds → clean,
  coherent scenery; v1 kept as fallback. Beat Turbo in an A/B; Turbo deleted.)
  The server AUTO-STARTS with the container (docker/studio.yml overlay
  → `data/studio/scripts/start-studio.sh`) — no manual launch needed.
- **Style trigger is automatic** — the server auto-prepends `toei90s style,
  smoon` (or maps "90s anime"/"cutie honey"/etc. via `styles.json`). Never type
  the trigger; just describe the scene.
- **Two model ids, one server (general mode):** `/v1/models` advertises
  `Z-Image` (styled: LoRA + auto-trigger) AND `Z-Image-General` (plain Base, no
  LoRA, no trigger — for non-studio/general images). Pick either in the Chat-tab
  model dropdown; the server swap-LOADS the chosen variant on demand. Because FP8
  fuses the LoRA at load, the two can't co-reside in one copy, so switching
  styled↔general costs a ~1-2 min reload (same as the LLM/ControlNet GPU swap);
  within a variant it's instant. Logic lives in `scripts/diffusion_server.py`
  (`_resolve_variant`/`_with_model` swap + `_variant`-gated `_styled`/`load_model`).
  Endpoint `bc38130a` `cached_models` must list both ids for the dropdown to show
  `Z-Image-General`.
- **LLMs (all Ollama @ host.docker.internal:11434, wired in data/settings.json):**
  default chat = `huihui_ai/Qwen3.6-abliterated:27b` (UNCENSORED — no fiction
  refusals), utility = `qwen3.5:4b`, vision/agent = `huihui_ai/qwen3-vl-abliterated:8b`
  (UNCENSORED), research/deep-research = `qwen3.6:35B-A3b` (kept STOCK — MoE
  abliteration hurts reasoning). Fiction specialist `cydonia-24b` (best prose) is
  selectable in the model dropdown. `teacher_enabled=false` (stops the loop that
  auto-wrote junk image skills). Image gen itself has NO content filter.
- **GPU auto-swap (32 GB can't hold the image model + a big LLM at once):** the
  image server idle-unloads after 5 min (or `POST :8100/admin/unload` to free
  VRAM now); Ollama auto-swaps its own models. So art and heavy-LLM (35B/Cydonia)
  time-share the GPU — expect a ~1-3 min model-load when switching modes.
- **Image editing via TOOLS:** `generate_image`, `restyle_image` (img2img on the
  latest upload), `inpaint_region` (fix a region — "fix her left hand"),
  `controlnet` (sketch/reference → on-model frame that follows its composition;
  bf16 swap-in, ~2 min). For RELIABLE one-shot gen, Chat-tab → Z-Image direct beats the agent
  (the abliterated 8B is a flaky tool-driver).
- Odysseus runs in Docker (`odysseus-odysseus-1`); LAN at `http://alienwaretv:7000`.

## How generation works (for explaining to the user)
- **Agent chat** (uncensored `qwen3-vl-abliterated:8b`): "generate 3 images of …",
  "fix her left hand", "restyle this" (attach an image). Slower + flakier (a
  thinking model); best for conversational edits, not bulk one-shot gen.
- **Chat tab** with image model `Z-Image` selected: type the scene directly; each
  message = one image; resend (↑) to re-roll. FASTEST + most reliable path.
- **Model dropdown**: pick `cydonia-24b` for serious fiction prose; the 27B
  default already won't refuse for everyday chat.
- Results land in the Gallery. No style trigger needed.

## Operating notes / gotchas (the short list — full list in HANDOFF.md)
- After reboot: run `data/studio/resume-after-reboot.bat` (brings the stack up +
  resumes captioning; the IMAGE SERVER auto-starts with the container). After any
  container restart give it ~3-5 min — the entrypoint ownership-repair walks the
  big bind-mounted `data/` before uvicorn, then the model loads.
- **Container now runs the BAKED current code** (post-merge HEAD + studio fixes),
  rebuilt 2026-06-25 — NO more cp-patch fragility, and `docker compose down/up`
  no longer reverts patches. **Dockerfile is PINNED to `python:3.12-slim`** (the
  `/app/.local` torch stack is cp312 — a 3.14 bump makes it unimportable and
  breaks image gen). The torch stack lives in the persisted mount `/app/.local`,
  NOT the image. origin = `defmetal/odysseus` fork, upstream =
  `pewdiepie-archdaemon/odysseus`, branch `dev`. `.env` COMPOSE_FILE =
  `docker-compose.yml;docker/gpu.nvidia.yml;docker/studio.yml`. After any
  `docker compose build`, retest image gen + the agent.
- VRAM (32 GB): see GPU auto-swap above — the image model and a big LLM can't
  co-reside; they time-share via idle-unload + Ollama swap.
- Ollama LAN API on :11434 with `OLLAMA_ORIGINS=*` (so desktop clients like the
  NousResearch Hermes app can reach it from a browser/Electron origin). Do NOT
  put `null` in OLLAMA_ORIGINS — it panics Ollama on boot; `*` alone covers it.

## Roadmap (each ~one focused session)
1. ✅ v2 style LoRA SERVED (backgrounds). v3 (bigger sailor-moon set) was TRAINED +
   A/B'd but KEPT v2 — v3 regressed backgrounds (dilution) + didn't improve hands;
   v3 lives at training/toei90s_zbase_v3/ as a fallback only.
2. Character LoRAs — IN PROGRESS (2026-07-09). Round 1 = Tetsuya on line art: a rough
   proof, UNDER-CONVERGED (line art is too thin a signal). BIG LEARNINGS in
   TRAINING-GUIDE.md — read that section before touching character LoRAs. Short
   version: COLORED images are the dense fuel (line art alone doesn't lock identity);
   trigger = `<name>_oc`; the production setup is character LoRA + style LoRA TOGETHER;
   train the COLOR-RICH characters FIRST. Cast is ~35-40 chars (property "Marzipan";
   mains Marzipan/Bon bon/Tetsuya/Drossel + Naomi). NEXT: train a character who HAS
   colored sheets (Tetsuya is color-poor → parked).
3. ✅ ControlNet DONE (2026-06-26) — native (NO ComfyUI) via
   `data/studio/scripts/controlnet.py` + the `controlnet` agent tool: bf16
   ZImageControlNetPipeline + alibaba-pai Union controlnet + the v2 style LoRA,
   canny/scribble, conditioning ~0.45. Heavier bf16 swap-in mode (~2 min, pauses
   FP8 gen). NEXT: pose/depth via `controlnet_aux` on the same pipeline.
4. Video (Wan 2.2) and voices (Chatterbox TTS) — later.

## In-flight / uncommitted (2026-07-09) — READ before a rebuild or `git` op
- **UNCOMMITTED src patches** (cp'd into the container, live; survive restart but a
  `docker compose down/up` recreate or `git stash/reset` would REVERT them — commit +
  rebuild to bake): `src/agent_loop.py` (agent now DETECTS attached images →
  "fix this"/"make her hair red" route to restyle/inpaint instead of a new image),
  `src/tool_implementations.py` (restyle_image optional strength line: light/medium/full
  or 0-1), `scripts/diffusion_server.py` (Z-Image-General variant — MOUNTED via
  studio.yml so it's live without a rebuild). Not yet user-verified end-to-end in the UI.
- **New machine-local scripts** in `data/studio/scripts/` (data/ is GITIGNORED, so
  they are NOT in the repo / a fresh clone): `dedup_dataset.py`, `controlnet_batch.py`,
  `test_char_style.py` (char+style combined render), `ab_compare.py` (fixed-seed LoRA
  A/B), `cutover-general-mode.sh`, and `dataset/characters/tetsuya/_gen_captions.py`.
- **COWORK CAVEAT:** the real knowledge lives in `data/studio/*.md` which is GITIGNORED —
  a fresh cloud clone WON'T have it. A Cowork session must run where `data/` exists (this
  machine) to see ARCHITECTURE/ODYSSEUS/TRAINING-GUIDE/OPERATIONS, or it's flying blind.

CLAUDE.md and AGENTS.md are IDENTICAL copies (keep in sync manually — a Windows
symlink needs admin). AGENTS.md is for non-Claude-Code tools (e.g. Hermes).
