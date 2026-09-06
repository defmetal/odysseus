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
| `data/studio/CHARACTER-PIPELINE-DECISION.md` | **THE reconciled character plan (2026-07-09) — READ THIS FIRST for character work.** Merges Hermes + Fable: Fable's QIE-2511 dataset-factory backbone + adapter-interference fix + Hermes's acceptance gate; rejects the ControlNet-2.1 swap. Supersedes the two proposals below for execution |
| `data/studio/PROPOSAL-CHARACTER-CONSISTENCY.md` | Hermes's proposal (reasoning/guardrail record) — why consistency fails, acceptance gate, "what not to do." Superseded for execution by CHARACTER-PIPELINE-DECISION.md |
| `data/studio/CHARACTER-PIPELINE-PLAN.md` | Fable/Cowork's proposal (citation record, ~18 sources) — QIE-2511 factory, adapter-interference mechanism, deep audit. Superseded for execution by CHARACTER-PIPELINE-DECISION.md |
| `data/studio/ODYSSEUS.md` | working on the platform — Docker/GPU/Ollama/auth/LAN config, the agent/tool/skill system, the codebase patches, the big gotchas |
| `data/studio/TRAINING-GUIDE.md` | curating datasets / training LoRAs — faces, hair, backdrops, character LoRAs, outfit-changeable characters, caption discipline |
| `data/studio/OPERATIONS.md` | running it day-to-day — generate, reboot, serve, sync the repo, troubleshoot |
| `data/studio/LIVE-STACK.md` | **current live ports/LoRAs/locks (2026-09-06)** — read before generate/park/train |
| `data/studio/V2-WISHLIST.md` | gathering more training stills (what's underrepresented) |
| `data/studio/UPGRADES-2026.md` | planning future upgrades — ComfyUI/ControlNet, video (Wan 2.2/LTX-2), web-search/RAG. Research-backed w/ sources (by Cowork); has a 2026-06-20 status header on what's already done |
| `data/studio/UPSTREAM-MERGE-PENDING.md` | doing the big upstream sync — fork is 344 commits behind (mostly security/stability + a `tool_implementations.py`→`src/tools/` refactor). The merge playbook: changelog, the 4 conflicts, which studio patches re-home where, the Python-3.14 track, execution steps + rollback, and who-does-it (Cowork preps, Claude Code executes) |
| `data/skills/studio/animation-studio-pipeline/SKILL.md` | what the in-app Odysseus agent itself knows |

(`data/` is gitignored — these exist on this machine but not in a fresh clone.
`HANDOFF.md`/`README.md` are earlier drafts; the table above is authoritative.)

Suggested first read for a fresh agent: LIVE-STACK.md, then ARCHITECTURE.md / ODYSSEUS.md.

## Current live stack (as of 2026-09-06)
One-pager truth: `data/studio/LIVE-STACK.md`. Day-to-day park/restore:
`data/studio/OPERATIONS.md` → **Live generate (:8101)**.

- **Diffusion is on `:8101`** (Z-Image Base + style + characters). **`:8100` is
  ChromaDB** — never treat it as the image server.
- **Launch (live):** Z-Image Base + `toei90s_zbase_v4_1` @ **0.75**, guidance
  **4.5**, steps **40**, `--quantize-fp8`, idle-unload **86400**,
  `--characters-config` → `data/studio/scripts/characters.json`,
  `--style-config` → `data/studio/scripts/styles.json`. Template:
  `data/studio/scripts/start-studio.sh` (auto-starts with the container).
  **Caveat:** if diffusion is killed, the start-studio restart loop dies —
  restore **manually as user 1000** with
  `PYTHONPATH=/app/.local/lib/python3.12/site-packages` (root `python3` has no
  torch). Exact commands: OPERATIONS.md + skill
  [Odysseus park restore 8101](sand-workflow:odysseus-park-restore-8101).
- **Style trigger is automatic** — server auto-prepends `toei90s style, smoon`
  (or maps aliases via `styles.json`). Never type the trigger; describe the scene.
- **Live characters (do not cut over without Austin/Ada):**
  `yuki_oc` → `yuki_char_v5`, `tetsuya_oc` → `tetsuya_char_v9` in
  `characters.json`. Verify paths on disk before generating.
- **UI `:7000`**, **Comfy `:8188`** (Music 3 + Wan). Music: **MiniMax Music 3
  local**; 30s one-shots can noise-collapse — use Mixkit/licensed beds for ads.
- **GPU timeshare:** tab click does **NOT** unload; swap happens on actual
  generate. Daily coding stays on **Mac 27B** — never park coding LLMs on the
  5090.
- **Dual-load identity LoRAs leak faces** — never fuse two character LoRAs on
  one denoise; use a **three-pass** for real duos.
- **Tetsuya kit lock:** always biker jacket; tee sometimes; jeans most of the
  time; when the leg shows, prompt wood with exact phrase
  `wooden prosthetic right leg visible with brass knee joint` ONLY (character-
  right; no left / viewer-left). Shorts = stress test only — no more shorts
  continues / no v12 of the same 3 keepers after v11 two-flesh fail. Live stays
  **v9**. Full lock: [Marzipan Tetsuya locks](sand-workflow:marzipan-tetsuya-locks).
- **Julie drop zones:** `dataset\characters\tetsuya\settei_inbox\` (Ada keep/drop
  first); COLOR_KEY updates → `dataset\color_keys\` (existing per-char keys also
  exist). LAN `\\AlienwareTV\studio`. Skill:
  [Marzipan settei inbox](sand-workflow:marzipan-settei-inbox).
- **Marzipan Slack = Leo only** — skill [Leo Slack](sand-workflow:leo-slack).
  Renders → `#marzipan-renders`. Never Austin’s Slack connector for Marzipan.
- **Role ownership:** Ada = character locks/scoring; Odysseus = stack / generate /
  train; Game Art Director = sheets; Projects Manager = Notion/Slack wiring;
  Tech Support = Windows/LAN.
- **Two model ids, one server (general mode):** `/v1/models` advertises `Z-Image`
  (styled) and `Z-Image-General` (plain Base). Switching styled↔general costs a
  ~1–2 min reload (FP8 fuses LoRA at load).
- **LLMs:** Ollama @ host.docker.internal:11434 (and Mac for daily coding). Image
  gen itself has NO content filter. `teacher_enabled=false`.
- **Image editing tools:** `generate_image`, `restyle_image`, `inpaint_region`,
  `controlnet`. Reliable one-shot gen: UI Image/Chat → Z-Image (or skill
  [Odysseus generate image](sand-workflow:odysseus-generate-image)).
- Odysseus Docker (`odysseus-odysseus-1`); LAN `http://alienwaretv:7000`.
- **ComfyUI** (`odysseus-comfyui` on `:8188`): Wan video + MiniMax Music 3; studio
  workflows under `data/studio/comfy/`. Template tiles with an "API" badge are
  PAID CLOUD — never use. Comfy output → `data/studio/comfy/output/`.
- **Live transparency:** `data/studio/RUNNING.md`, `data/studio/_bridge/BRIDGE.md`.

### Shared Grok Bot skills (sand-workflow)
Use these instead of chat memory for ops:

| Skill id | When |
|---|---|
| `leo-slack` | Marzipan Slack as Leo (`/post`, `/upload`) |
| `odysseus-generate-image` | Stills on `:8101` (seeds, dumps) |
| `odysseus-park-restore-8101` | Park/restore diffusion before Music/train |
| `odysseus-lora-continue-train` | Continue-train LoRAs (no cutover until Ada/Austin) |
| `odysseus-style-only-img2img` | Toei style-only restyle (no identity LoRA) |
| `marzipan-tetsuya-locks` | Tetsuya kit + wood phrase lock |
| `marzipan-settei-inbox` | Julie settei/COLOR_KEY drop zones |


## How generation works (for explaining to the user)
- **Agent chat** (uncensored `qwen3-vl-abliterated:8b`): "generate 3 images of …",
  "fix her left hand", "restyle this" (attach an image). Slower + flakier (a
  thinking model); best for conversational edits, not bulk one-shot gen.
- **Chat tab** with image model `Z-Image` selected: type the scene directly; each
  message = one image; resend (↑) to re-roll. FASTEST + most reliable path.
- **Model dropdown**: pick `cydonia-24b` for serious fiction prose; the 27B
  default already won't refuse for everyday chat.
- Results land in the Gallery. No style trigger needed.

## Working style for Claude 5-era models (Opus 5 / Fable 5) — differs from Opus 4.x
Opus 5 self-verifies, self-corrects, and completes whole tasks by default; it also
runs longer, narrates more, and will widen scope on its own judgment. Instruction
changes that get the best results here (per Anthropic's Opus 5 prompting guide):
- **Scope rule (the house default):** Deliver what was asked, at the scope
  intended. Make routine judgment calls yourself; check in only when different
  readings of the request would lead to materially different work. If the request
  seems mistaken or a better approach exists, SAY SO IN A SENTENCE and continue
  with the task as asked — don't quietly narrow, widen, or transform it. Finish
  the whole task; stop short of actions clearly beyond it.
- **DON'T add "verify your work" / "double-check" instructions** — Opus 5 already
  does this; explicit verification steps cause expensive over-verification. Same
  for re-check scaffolding in prompts. (Checklists of WHAT to test are fine —
  e.g. "after rebuild: image gen both variants, agent tools load".)
- **Narration cadence:** one sentence before the first tool call; brief updates
  only on findings or direction changes; finish outcome-first ("what happened"
  in the first sentence, detail after).
- **Written deliverables:** match length to substance; no filler sections,
  boilerplate, or redundant summaries.
- **Subagents:** delegate only large, genuinely independent parallel tracks
  (e.g. wide multi-file investigations). Don't delegate what fits in a handful
  of tool calls; never spawn subagents just to verify own work; prefer one over
  several. (Sonnet workers for parallel implementation remain the house pattern —
  this rule is about not letting Opus 5 over-delegate small things.)
- **Give the whole spec up front** and let it run — Opus 5 is strongest on
  complete task specifications with full autonomy (our plan-first-markdown habit
  is exactly right; keep doing it).
- **Code review asks:** never say "only report high-severity" — it will comply
  literally and under-report. Ask for everything, filter afterward.
- Effort (Claude Code setting): default `high`; reserve `xhigh` for the hardest
  agentic/coding work — lower effort on Opus 5 outperforms xhigh on prior models.

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

## Post-upstream-merge state (2026-07-09) — SYNCED to upstream/dev
- **Merged upstream/dev (344 commits) — committed `2cfb5b3`, pushed, and REBUILT/BAKED
  (py3.12).** All studio patches survived: the studio image tools (`do_restyle_image`
  +strength, `do_inpaint_region`, `do_controlnet`, `_run_studio_image_script`) were
  RE-HOMED into `src/tools/image.py` (upstream split tool_implementations.py into the
  `src/tools/` package, #4423) and re-exported via the `src.tool_implementations` facade;
  the attachment edit-routing (`_latest_user_has_image`/`edit_only`) + Z-Image-General
  are intact. Kept `FROM python:3.12-slim` (dropped upstream's 3.14 + Real-ESRGAN builder
  — the cp312 torch stack in `/app/.local` requires 3.12). VERIFIED: 4520 tests pass (8
  are container-env/docs artifacts, not code), image gen works for both variants, agent
  edit-routing fires, `manage_bg_jobs` (new) loads. Rollback: branch `dev-premerge-backup`
  + images `pre-merge-2026-07-09` / `merged-2026-07-09`. **Every FUTURE upstream sync will
  re-conflict on the Dockerfile (keep 3.12 — 1 line) + must re-home any NEW studio edits
  to tool files into `src/tools/`.** See `data/studio/UPSTREAM-MERGE-PENDING.md` (marked
  DONE for this round; it stays as the playbook for the next sync + the Python-3.14 track).
- **New machine-local scripts** in `data/studio/scripts/` (data/ is GITIGNORED, so
  they are NOT in the repo / a fresh clone): `dedup_dataset.py`, `controlnet_batch.py`,
  `test_char_style.py` (char+style combined render), `ab_compare.py` (fixed-seed LoRA
  A/B), `cutover-general-mode.sh`, and `dataset/characters/tetsuya/_gen_captions.py`.
- **COWORK CAVEAT:** the real knowledge lives in `data/studio/*.md` which is GITIGNORED —
  a fresh cloud clone WON'T have it. A Cowork session must run where `data/` exists (this
  machine) to see ARCHITECTURE/ODYSSEUS/TRAINING-GUIDE/OPERATIONS, or it's flying blind.

CLAUDE.md and AGENTS.md are kept in sync (identical). AGENTS.md is for non-Claude-Code tools (e.g. Hermes / Grok Bot). Live facts also live in `data/studio/LIVE-STACK.md`.