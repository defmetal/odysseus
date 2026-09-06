# Image, Video, and Music generation

Odysseus can add **Image**, **Video**, and **Music** chips next to Agent / Chat. They stay hidden until a backend is actually available. This is a generic, discovery-driven composer — connect **your** ComfyUI and/or a MiniMax Music 3 key. No custom LoRAs, character packs, or style prefixes are required.

An optional compose overlay is **not** in the default stack.

## Absence rules

| Mode | Shown when |
|---|---|
| Image / Video | Settings `comfy_base_url` is set, passes SSRF checks, and ComfyUI answers |
| Music | `music_gen_enabled` is on **and** at least one music backend is available |

No `comfy_base_url`, or Comfy unreachable → Image and Video stay hidden. No Music 3 weights **and** no `MINIMAX_API_KEY` → Music stays hidden.

## Connect your own ComfyUI (Image / Video)

1. Run [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on a machine you control.
2. In Odysseus **Settings → ComfyUI / Music**, set **Comfy URL** (for example `http://127.0.0.1:8188`).
3. Save. Odysseus SSRF-checks the URL (loopback and LAN are allowed; link-local / metadata addresses are not).
4. Reload the chat composer. Image and Video chips appear if `/api/comfy/status` reports `available`.

The Image/Video popup lists models from the **connected Comfy catalog** (`/object_info`), with optional recipes in `config/comfy_models.json`. Drop whatever checkpoints you already use. No custom LoRAs are required.

Image and Video POST to `/api/comfy/generate` with `kind` `image` or `video`. MiniMax **H3** is a video engine, not Music 3.

## Music (MiniMax Music 3)

Music is a separate surface: `/api/music/*`. Do not send music jobs as a Comfy `kind`.

Backends are registered in `config/music_backends.json` with an `engine` key:

| Engine | Needs | Notes |
|---|---|---|
| `minimax_music3_comfy` | `comfy_base_url` + Music 3 weights on that Comfy | No API key |
| `minimax_music3_api` | `MINIMAX_API_KEY` or Settings → MiniMax key | Hosted Music 3 (`music-3.0`) |

### Local Comfy Music 3

Drop the official MiniMax Music 3 weights into your Comfy models dirs (DiT, text encoder, DAV). Typical filenames:

- `minimax_music3_dit_fp16.safetensors`
- `minimax_music3_text_encoder_pruned_int8_convrot.safetensors`
- `minimax_music3_dav.safetensors`

You also need the Comfy Music 3 nodes (`MiniMaxMusic3TextEncode`, `EmptyMiniMaxMusic3LatentAudio`). Then set `comfy_base_url` as above. No API key.

### Hosted MiniMax API

Set `MINIMAX_API_KEY` in the environment, or paste the key in Settings (stored in settings, never committed). Leave the field blank on later saves to keep the existing key.

The agent tool `generate_music` uses the same `/api/music` backends. It is gated off when music is disabled or no backend is available.

## Settings keys

| Key | Default | Meaning |
|---|---|---|
| `comfy_base_url` | `""` | Your ComfyUI origin. Empty hides Image/Video. |
| `music_gen_enabled` | `true` | Master switch for the Music chip and `generate_music`. |
| `minimax_api_key` | `""` | Hosted Music 3 only. Prefer `MINIMAX_API_KEY`. |

## Testers

1. Fresh clone, no Comfy URL, no MiniMax key → only Agent and Chat.
2. Set a reachable `comfy_base_url` → Image and Video appear; pick a catalog model and generate.
3. Drop Music 3 weights **or** set `MINIMAX_API_KEY` → Music appears; generate a short caption-only clip.
4. Unset the URL / stop Comfy / clear the key → the matching chips hide again.
