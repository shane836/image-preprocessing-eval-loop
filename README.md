# Image Preprocessing Eval Loop

Python tools for masked image material experiments and video texture screening. Project paused; no new generation is scheduled.

## Included

- `asset_loop.py`: fal FLUX inpainting strength sweeps, fixed seeds, cached replay, mask compositing and screening.
- `harness.py`: MCP preflight and streaming OpenCV evaluation of texture, highlights, flow, flicker and shimmer.
- Unit tests, pinned dependencies and blank configuration templates.

`examples/kitchen-source.png` is the unapproved kitchen source image, included with the project owner’s permission. Generated candidates, masks, private prompts, provider receipts and production run logs are excluded. Supply your own experiment configuration. The live pilot did not establish a successful material treatment. Metrics are provisional and require visual calibration.

## Setup

```sh
uv sync --locked --group dev
uv run pytest -q
uv run asset_loop.py --help
uv run harness.py --help
```

Credentials are read at runtime. On macOS the fal client checks the login Keychain generic password service `soft-cactus.fal`, account `FAL_KEY`. This is a lookup label, not a credential. If absent, the client accepts `FAL_KEY` from the environment or a local ignored `.env`. Never commit populated environment files. MCP transport settings use the blank `.env.example` fields.

Preparation and local evaluation are offline. Generation sends the configured images, masks and prompts to the provider and incurs charges. Review configuration and budgets before execution. The original strict Higgsfield pipeline currently fails catalog preflight: the inspected Sunburst schema did not expose strength/seed and Seedance did not expose `1080p_pro`. Do not assume those controls exist or silently replace them.

The fal client does not automatically resubmit an unknown submission. Hosted fixed-seed generation is distinct from exact replay of cached output. Numerical edit strength does not directly represent physical roughness.

## How this was built

Built with OpenAI Codex assistance. Tests verify implementation behavior; they do not establish perceptual success on real footage.
