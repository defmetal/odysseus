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
  toei90s_zbase_v1/toei90s_zbase_v1.safetensors`, launched `--guidance 4.5
  --steps 30 --quantize-fp8 --style-config data/studio/scripts/styles.json
  --idle-unload-seconds 300`. (Beat Turbo in an A/B; Turbo + its base model were
  deleted.) The server AUTO-STARTS with the container (docker/studio.yml overlay
  → `data/studio/scripts/start-studio.sh`) — no manual launch needed.
- **Style trigger is automatic** — the server auto-prepends `toei90s style,
  smoon` (or maps "90s anime"/"cutie honey"/etc. via `styles.json`). Never type
  the trigger; just describe the scene.
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
  latest upload), `inpaint_region` (fix a region by plain words — "fix her left
  hand"). For RELIABLE one-shot gen, Chat-tab → Z-Image direct beats the agent
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
1. v2 dataset + Base-native retrain (backgrounds, hands, more Cutie Honey Flash).
2. Character LoRAs (Fumiko first — identity locked, outfits promptable; see
   TRAINING-GUIDE.md). Bootstrap via img2img from her design sheets.
3. ControlNet (storyboard sketch → posed on-model frame) — the one capability
   not yet built; may use ComfyUI headless (parked at `C:\Users\austi\ComfyUI`).
4. Video (Wan 2.2) and voices (Chatterbox TTS) — later.

CLAUDE.md and AGENTS.md are IDENTICAL copies (keep in sync manually — a Windows
symlink needs admin). AGENTS.md is for non-Claude-Code tools (e.g. Hermes).
