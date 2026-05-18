"""
Interactive runner for the refined Nultra multiband ecosystem processor.

Browse E:\\4trax recursively, pick a WAV, choose a preset or custom mode,
and render without editing Python every time.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

try:
    from . import processor as core
except ImportError:  # pragma: no cover - direct script fallback
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from nultra_multiband_ecosystem import processor as core


MUSIC_ROOT = Path(r"E:\4trax")
SCRIPT_DIR = Path(__file__).resolve().parent


def prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_float(label: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = float(raw)
            except ValueError:
                print("Enter a number.")
                continue
        if minimum is not None and value < minimum:
            print(f"Must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Must be <= {maximum}.")
            continue
        return value


def prompt_int(label: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Enter an integer.")
                continue
        if minimum is not None and value < minimum:
            print(f"Must be >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Must be <= {maximum}.")
            continue
        return value


def prompt_choice(label: str, choices: list[str], default: str) -> str:
    normalized = {choice.lower(): choice for choice in choices}
    while True:
        raw = input(f"{label} {choices} [{default}]: ").strip().lower()
        if not raw:
            return default
        if raw in normalized:
            return normalized[raw]
        print("Pick one of the listed choices.")


def scan_wavs(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Music root not found: {root}")
    return sorted(root.rglob("*.wav"))


def choose_wav(root: Path) -> Path:
    files = scan_wavs(root)
    print(f"Found {len(files)} WAV files under {root}")
    filtered = files
    page = 0
    page_size = 20

    while True:
        if not filtered:
            print("No matches. Enter a new search.")
        else:
            start = page * page_size
            stop = min(len(filtered), start + page_size)
            print("")
            print(f"Showing {start + 1}-{stop} of {len(filtered)}")
            for idx, path in enumerate(filtered[start:stop], start=1):
                rel = path.relative_to(root)
                print(f"{idx:2d}. {rel}")

        print("")
        command = input("Choose number, /search text, n, p, or q: ").strip()
        if not command:
            continue
        if command.lower() == "q":
            raise KeyboardInterrupt
        if command.lower() == "n":
            if filtered and (page + 1) * page_size < len(filtered):
                page += 1
            continue
        if command.lower() == "p":
            page = max(0, page - 1)
            continue
        if command.startswith("/"):
            terms = [term.lower() for term in command[1:].split() if term]
            if not terms:
                filtered = files
            else:
                filtered = [path for path in files if all(term in str(path.relative_to(root)).lower() for term in terms)]
            page = 0
            continue
        try:
            choice = int(command)
        except ValueError:
            print("Enter a number or a command.")
            continue
        start = page * page_size
        index = start + choice - 1
        if filtered and 0 <= index < len(filtered):
            return filtered[index]
        print("Out of range.")


def choose_profile() -> str:
    print("")
    print("Profiles")
    print("1. sweet-spot   (current refined defaults)")
    print("2. wetter       (same behavior, lower wet/dry)")
    print("3. stronger     (more audible without major surgery)")
    print("4. custom       (prompt for key settings)")
    while True:
        choice = input("Choose profile [1]: ").strip() or "1"
        mapping = {"1": "sweet-spot", "2": "wetter", "3": "stronger", "4": "custom"}
        if choice in mapping:
            return mapping[choice]
        print("Pick 1-4.")


def snapshot_core_state() -> dict[str, object]:
    return {
        "LOW_CROSSOVER_HZ": core.LOW_CROSSOVER_HZ,
        "HIGH_CROSSOVER_HZ": core.HIGH_CROSSOVER_HZ,
        "WET_DRY": core.WET_DRY,
        "CHAOS_SYSTEM": core.CHAOS_SYSTEM,
        "HIGH_PARASITIC_ETA_ALPHA": core.HIGH_PARASITIC_ETA_ALPHA,
        "HIGH_DIFFUSION_OUTPUT_GAIN": core.HIGH_DIFFUSION_OUTPUT_GAIN,
        "MID_RESIDUE_FEEDBACK_BETA": core.MID_RESIDUE_FEEDBACK_BETA,
        "MID_RESIDUE_FEEDBACK_DELAY_SAMPLES": core.MID_RESIDUE_FEEDBACK_DELAY_SAMPLES,
        "LOW_BAND": core.LOW_BAND,
        "MID_BAND": core.MID_BAND,
        "HIGH_BAND": core.HIGH_BAND,
    }


def restore_core_state(state: dict[str, object]) -> None:
    for key, value in state.items():
        setattr(core, key, value)


def apply_profile(profile: str) -> dict[str, object]:
    config = {
        "label": profile.replace("-", "_"),
        "low_xover": core.LOW_CROSSOVER_HZ,
        "high_xover": core.HIGH_CROSSOVER_HZ,
        "wet_dry": 1.0,
        "chaos_system": core.CHAOS_SYSTEM,
        "high_parasitic_eta_alpha": core.HIGH_PARASITIC_ETA_ALPHA,
        "high_diffusion_output_gain": core.HIGH_DIFFUSION_OUTPUT_GAIN,
        "mid_residue_feedback_beta": core.MID_RESIDUE_FEEDBACK_BETA,
        "mid_residue_feedback_delay_samples": core.MID_RESIDUE_FEEDBACK_DELAY_SAMPLES,
        "mid_eta": core.MID_BAND.eta,
        "high_gamma": core.HIGH_BAND.gamma,
        "high_eta": core.HIGH_BAND.eta,
    }

    if profile == "wetter":
        config["wet_dry"] = 0.78
    elif profile == "stronger":
        config["high_parasitic_eta_alpha"] = 0.10
        config["high_diffusion_output_gain"] = 0.98
        config["mid_residue_feedback_beta"] = 0.34
        config["mid_residue_feedback_delay_samples"] = 1408
        config["mid_eta"] = 0.27
        config["high_gamma"] = 6.9
        config["high_eta"] = 0.28
    elif profile == "custom":
        print("")
        print("Custom settings")
        config["label"] = prompt_text("Output label", "custom")
        config["wet_dry"] = prompt_float("Wet/dry", 1.0, 0.0, 1.0)
        config["low_xover"] = prompt_float("Low crossover Hz", core.LOW_CROSSOVER_HZ, 20.0)
        config["high_xover"] = prompt_float("High crossover Hz", core.HIGH_CROSSOVER_HZ, 100.0)
        config["chaos_system"] = prompt_choice("Chaos system", ["lorenz", "rossler"], core.CHAOS_SYSTEM)
        config["high_parasitic_eta_alpha"] = prompt_float("High parasitic eta alpha", core.HIGH_PARASITIC_ETA_ALPHA, 0.0, 1.0)
        config["high_diffusion_output_gain"] = prompt_float("High diffusion output gain", core.HIGH_DIFFUSION_OUTPUT_GAIN, 0.1, 2.0)
        config["mid_residue_feedback_beta"] = prompt_float("Mid residue feedback beta", core.MID_RESIDUE_FEEDBACK_BETA, 0.0, 1.0)
        config["mid_residue_feedback_delay_samples"] = prompt_int("Mid residue delay samples", core.MID_RESIDUE_FEEDBACK_DELAY_SAMPLES, 0, 32768)
        config["mid_eta"] = prompt_float("Mid eta", core.MID_BAND.eta, 0.0, 0.99)
        config["high_gamma"] = prompt_float("High gamma", core.HIGH_BAND.gamma, 0.1, 20.0)
        config["high_eta"] = prompt_float("High eta", core.HIGH_BAND.eta, 0.0, 0.99)

    return config


def configure_core(config: dict[str, object]) -> None:
    core.WET_DRY = float(config["wet_dry"])
    core.CHAOS_SYSTEM = str(config["chaos_system"])
    core.HIGH_PARASITIC_ETA_ALPHA = float(config["high_parasitic_eta_alpha"])
    core.HIGH_DIFFUSION_OUTPUT_GAIN = float(config["high_diffusion_output_gain"])
    core.MID_RESIDUE_FEEDBACK_BETA = float(config["mid_residue_feedback_beta"])
    core.MID_RESIDUE_FEEDBACK_DELAY_SAMPLES = int(config["mid_residue_feedback_delay_samples"])
    core.MID_BAND = replace(core.MID_BAND, eta=float(config["mid_eta"]))
    core.HIGH_BAND = replace(core.HIGH_BAND, gamma=float(config["high_gamma"]), eta=float(config["high_eta"]))


def output_path_for(input_path: Path, label: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label).strip("_") or "run"
    return SCRIPT_DIR / f"{input_path.stem}_nultra_{safe_label}_{stamp}.wav"


def render_file(input_path: Path, config: dict[str, object]) -> tuple[Path, dict[str, dict[str, float]], float]:
    low_xover = float(config["low_xover"])
    high_xover = float(config["high_xover"])
    wet_dry = float(config["wet_dry"])

    sample_rate, audio = core.load_audio(input_path)
    if not 20.0 <= low_xover < high_xover < (sample_rate * 0.5 - 100.0):
        raise ValueError("Crossover values must be ordered and remain below Nyquist.")

    started = time.perf_counter()
    processed, stats = core.render_multiband(audio, sample_rate, low_xover, high_xover, wet_dry)
    elapsed = time.perf_counter() - started

    output_path = output_path_for(input_path, str(config["label"]))
    wavfile = core.wavfile
    wavfile.write(str(output_path), sample_rate, core.float32_to_int16(processed))
    return output_path, stats, elapsed


def print_stats(stats: dict[str, dict[str, float]], config: dict[str, object], elapsed: float) -> None:
    print("")
    print("Run summary")
    print(f"wet/dry={float(config['wet_dry']):.2f}, chaos={config['chaos_system']}, render_time={elapsed:.2f}s")
    print(
        "high:"
        f" parasitic_alpha={float(config['high_parasitic_eta_alpha']):.3f},"
        f" diffusion_gain={float(config['high_diffusion_output_gain']):.3f},"
        f" gamma={float(config['high_gamma']):.2f},"
        f" eta={float(config['high_eta']):.2f}"
    )
    print(
        "mid:"
        f" eta={float(config['mid_eta']):.2f},"
        f" feedback_beta={float(config['mid_residue_feedback_beta']):.3f},"
        f" feedback_delay={int(config['mid_residue_feedback_delay_samples'])}"
    )
    for band_name, band_stats in stats.items():
        extra = ""
        if band_name == "mid":
            extra = f", feedback_rms={band_stats['feedback_rms']:.6f}"
        if band_name == "high":
            extra = f", eta_span={band_stats['eta_min']:.3f}-{band_stats['eta_max']:.3f}"
        print(
            f"{band_name}:"
            f" null={band_stats['null_ratio']:.2%},"
            f" freq={band_stats['freq_min']:.2f}-{band_stats['freq_max']:.2f} Hz"
            f"{extra}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive menu for the refined Nultra processor.")
    parser.add_argument("--input-file", help="Optional WAV path to skip the file browser")
    parser.add_argument("--profile", choices=("sweet-spot", "wetter", "stronger", "custom"), help="Optional profile to skip the profile menu")
    parser.add_argument("--root", default=str(MUSIC_ROOT), help="Root directory to scan for WAVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_state = snapshot_core_state()

    try:
        input_path = Path(args.input_file) if args.input_file else choose_wav(Path(args.root))
        profile = args.profile or choose_profile()
        config = apply_profile(profile)
        configure_core(config)
        output_path, stats, elapsed = render_file(input_path, config)
        print("")
        print(f"Input : {input_path}")
        print(f"Output: {output_path}")
        print_stats(stats, config, elapsed)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    finally:
        restore_core_state(base_state)


if __name__ == "__main__":
    main()
