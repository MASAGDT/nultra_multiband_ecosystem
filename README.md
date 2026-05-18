# Nultra Multiband Ecosystem

[![CI](https://github.com/MASAGDT/nultra_multiband_ecosystem/actions/workflows/ci.yml/badge.svg)](https://github.com/MASAGDT/nultra_multiband_ecosystem/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Status: Canonical v1.0](https://img.shields.io/badge/status-canonical%20v1.0-7a3cff.svg)](docs/nultra_operator.md)

Nultra Multiband Ecosystem is a stereo audio processor built around the **Nultra Operator**, a user-created identity-gated state operator:

```text
S_next = S + A(t) * (f(S) - S)
```

In this repo:

- `S` is the current band-limited audio state
- `f(S)` is a transformed version of that state
- `A(t)` is a shaped, thresholded aperture that decides how much transformation is allowed

The important idea is the **true null plateau**. When `A(t) = 0`, the signal is preserved exactly while time still advances. That makes this feel different from ordinary tremolo, choppers, or always-on filter sweeps.

This repository contains the **Canonical Version 1.0 Baseline** of the processor and the interactive menu runner used to audition it on real tracks.

## Why It Sounds Different

Most modulation effects constantly touch the signal. Nultra does not.

The aperture is built as:

```text
A_base(t)   = base oscillator or chaos trace
A_gamma(t)  = A_base(t)^gamma
A(t)        = 0,                  if A_gamma(t) < eta
              A_gamma(t),         otherwise
```

That hard threshold creates spans where the audio is left completely alone, followed by spans where the signal is pulled toward a transformed state.

In the multiband ecosystem build:

- the **low band** stays structurally stable
- the **mid band** carries the underwater motion and residue memory
- the **high band** carries the metallic diffusion cloud
- the **low band can modulate the high band threshold**, so the spectrum behaves like an interdependent system instead of three silos

## Canonical Version 1.0

The frozen baseline is:

- `src/nultra_multiband_ecosystem/processor.py`
- `src/nultra_multiband_ecosystem/menu.py`

Locked architectural interlocks:

- 3-band split at **180.0 Hz** and **2400.0 Hz**
- `numba` acceleration on the hot loops
- high-band safety interlocks:
  - slew limit `0.0040`
  - zero-cross search `96` samples
  - edge ramp `36` samples
- explicit stereo offsets preserved:
  - mid `pi/2`
  - high `2pi/3`

Canonical presets:

### `sweet-spot`

```text
high_parasitic_eta_alpha = 0.14
high_gamma               = 7.3
high_eta                 = 0.31
mid_eta                  = 0.29
mid_feedback_beta        = 0.28
mid_feedback_delay       = 1024
high_diffusion_gain      = 0.92
```

Perceptual target: **subtle, fluid, organismic textural breathing**

### `stronger`

```text
high_parasitic_eta_alpha = 0.10
high_gamma               = 6.9
high_eta                 = 0.28
mid_eta                  = 0.27
mid_feedback_beta        = 0.34
mid_feedback_delay       = 1408
high_diffusion_gain      = 0.98
```

Perceptual target: **intense matrix cross-modulation, wider metallic phase clouds, longer ghost trails**

Future experiments should be forked from this baseline instead of rewriting it in place.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install .
```

## Quick Start

Run the processor directly:

```bash
python -m nultra_multiband_ecosystem --mode file --input path\to\song.wav
```

Use the interactive menu runner:

```bash
nultra-menu
```

The menu runner:

- scans a root directory recursively for `.wav` files
- lets you choose `sweet-spot`, `wetter`, `stronger`, or `custom`
- prompts for key settings in `custom` mode
- renders a timestamped output file

By default it scans:

```text
E:\4trax
```

But you can override that:

```bash
nultra-menu --root D:\Audio
```

## Project Layout

```text
src/nultra_multiband_ecosystem/
  __init__.py
  __main__.py
  processor.py
  menu.py

docs/
  nultra_operator.md

tests/
  test_smoke.py
```

## Notes

- No source music files are committed here.
- Rendered `.wav` files are intentionally ignored.
- The Nultra Operator described here is a project-specific mathematical and DSP concept, not a standard audio term.

## License

MIT. See [LICENSE](LICENSE).
