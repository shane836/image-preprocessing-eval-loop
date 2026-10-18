# Image Preprocessing Eval Loop

Python tools for masked image material experiments and video texture screening. Project paused; no new generation is scheduled.

## Background and provenance

This project grew out of Soft Cactus's animated indoor-scene workflow. The draft kitchen had the composition we wanted to explore, but its smooth surfaces looked plastic. We needed a controlled way to improve material texture while preserving the layout, then check whether that texture survived animation in Seedance.

The starting point was a Gemini-assisted proposal for a Higgsfield MCP pipeline: Soul for composition, Sunburst for surface detail, and Seedance for animation. It suggested numerical edit strength, seeded retries and computer-vision checks for flat highlights, missing grain and background drift. That proposal became the specification to investigate, rather than a verified description of the available APIs.

The October 18, 2026 catalog inspection exposed the blocker: Higgsfield's published Sunburst interface did not offer the proposed strength or seed controls. Seedance exposed `1080p`, not the proposed `1080p_pro`; the claim that a special resolution mode would eliminate smoothing was unverified. The original strict runner retains preflight checks so those gaps cannot silently become ignored settings.

We selected fal's `fal-ai/flux-lora/inpainting` endpoint for the image-preparation experiment because it exposed the controls needed for a measured comparison: an edit mask, numerical strength, seed, inference steps and guidance. This offered more precise experimental control than the Higgsfield Sunburst interface we inspected. It did not establish that fal produces better images overall. Higgsfield/Seedance remained the intended downstream animation route.

## Project goals

- Reduce the plastic appearance through believable material detail, rather than adding noise to satisfy a score.
- Compare bounded strength settings while holding the source, mask, prompt and seed fixed. Preserve unmasked pixels through local compositing.
- Save settings, receipts and cached outputs so experiments can be audited and reconstructed without paying to regenerate them.
- Screen highlights, spatial texture, temporal shimmer and geometric drift, with thresholds calibrated against human-approved examples.
- Carry an approved image into the existing shot-frame and Seedance workflow. Evaluate intentional motion and lighting with shot-specific masks; a static source plate does not require a motionless finished scene.

## First experiment and current limits

The first pilot edited a cabinet patch at strengths 0.15, 0.20, 0.25, 0.30 and 0.35 with seed 42. All five composites preserved the unmasked source pixels, but all failed the provisional fine-texture screen and showed subtle smoothing. None was approved, and no Seedance animation test followed. The harness works as an experiment runner; eliminating the plastic look remains an open goal.

Numerical strength controls edit freedom, not physical surface roughness. Fixed seeds support controlled comparisons but do not guarantee identical future hosted generations. Exact reconstruction uses cached outputs. The next useful experiment needs a better source or material reference and calibrated evaluation thresholds before another animation trial.

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
