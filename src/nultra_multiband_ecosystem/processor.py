"""
Canonical Version 1.0 of the Nultra Multiband Ecosystem processor.

This module contains the frozen baseline DSP engine:
- 3-band crossover matrix (180 Hz / 2400 Hz)
- numba-accelerated hot loops
- chaotic aperture generators
- parasitic cross-band modulation
- high-band diffusion morphing
- residual feedback into the mid band
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_MODE = "file"
INPUT_WAV_PATH = r"input.wav"
OUTPUT_WAV_PATH = SCRIPT_DIR / "nultra_multiband_ecosystem_v1.wav"

GENERATED_FREQUENCY_HZ = 440.0
GENERATED_DURATION_SECONDS = 6.0
GENERATED_SAMPLE_RATE = 48000
GENERATED_AMPLITUDE = 0.7
GENERATED_CHANNELS = 2

LOW_CROSSOVER_HZ = 180.0
HIGH_CROSSOVER_HZ = 2400.0
WET_DRY = 1.0

CHAOS_SYSTEM = "lorenz"  # "lorenz" or "rossler"
CHAOS_BUFFER_LENGTH = 1048576
CHAOS_BURN_IN = 8192
CHAOS_ORBIT_SECONDS = 18.0
CHAOS_DT = 0.0035

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0

ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7

HIGH_PARASITIC_ETA_ALPHA = 0.14

HIGH_APERTURE_SLEW_ENABLED = True
HIGH_APERTURE_MAX_DELTA_PER_SAMPLE = 0.0040
HIGH_ZERO_ALIGN_SEARCH_SAMPLES = 96
HIGH_ZERO_ALIGN_RAMP_SAMPLES = 36
HIGH_ZERO_FLOOR = 1e-7

HIGH_DIFFUSION_BASE_DELAYS = (7, 19, 41)
HIGH_DIFFUSION_MORPH_DELAYS = (8, 14, 28)
HIGH_DIFFUSION_GAINS = (0.73, 0.59, 0.47)
HIGH_DIFFUSION_OUTPUT_GAIN = 0.92
HIGH_DIFFUSION_CONTEXT_SAMPLES = 96

MID_RESIDUE_FEEDBACK_BETA = 0.28
MID_RESIDUE_FEEDBACK_LOWPASS_HZ = 1850.0
MID_RESIDUE_FEEDBACK_DELAY_SAMPLES = 1024


@dataclass(frozen=True)
class BandSettings:
    name: str
    chaos_axis: str
    frequency_hz: float
    phase_radians: float
    gamma: float
    eta: float
    transform_mode: str
    transform_gain: float
    lowpass_cutoff_hz: float
    stereo_phase_offset_radians: float
    env_follow_frequency: bool
    env_attack_ms: float
    env_release_ms: float
    env_frequency_min_hz: float
    env_frequency_max_hz: float
    env_sensitivity: float


LOW_BAND = BandSettings(
    name="low",
    chaos_axis="x",
    frequency_hz=1.6,
    phase_radians=0.0,
    gamma=3.0,
    eta=0.60,
    transform_mode="gain",
    transform_gain=0.88,
    lowpass_cutoff_hz=250.0,
    stereo_phase_offset_radians=0.0,
    env_follow_frequency=True,
    env_attack_ms=20.0,
    env_release_ms=180.0,
    env_frequency_min_hz=0.9,
    env_frequency_max_hz=2.8,
    env_sensitivity=1.2,
)

MID_BAND = BandSettings(
    name="mid",
    chaos_axis="y",
    frequency_hz=5.0,
    phase_radians=np.pi / 7.0,
    gamma=7.0,
    eta=0.29,
    transform_mode="lowpass",
    transform_gain=0.05,
    lowpass_cutoff_hz=850.0,
    stereo_phase_offset_radians=np.pi / 2.0,
    env_follow_frequency=True,
    env_attack_ms=8.0,
    env_release_ms=140.0,
    env_frequency_min_hz=3.2,
    env_frequency_max_hz=12.5,
    env_sensitivity=1.6,
)

HIGH_BAND = BandSettings(
    name="high",
    chaos_axis="z",
    frequency_hz=8.4,
    phase_radians=2.0 * np.pi / 10.0,
    gamma=7.3,
    eta=0.31,
    transform_mode="diffusion",
    transform_gain=HIGH_DIFFUSION_OUTPUT_GAIN,
    lowpass_cutoff_hz=3200.0,
    stereo_phase_offset_radians=2.0 * np.pi / 3.0,
    env_follow_frequency=True,
    env_attack_ms=4.0,
    env_release_ms=90.0,
    env_frequency_min_hz=5.2,
    env_frequency_max_hz=16.0,
    env_sensitivity=1.35,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the refined Nultra multiband ecosystem processor.")
    parser.add_argument("--mode", choices=("generate", "file"), default=INPUT_MODE)
    parser.add_argument("--input", default=INPUT_WAV_PATH, help="Input WAV path when using --mode file")
    parser.add_argument("--output", default=str(OUTPUT_WAV_PATH), help="Output WAV path")
    parser.add_argument("--low-xover", type=float, default=LOW_CROSSOVER_HZ, help="Low/mid crossover in Hz")
    parser.add_argument("--high-xover", type=float, default=HIGH_CROSSOVER_HZ, help="Mid/high crossover in Hz")
    parser.add_argument("--wet-dry", type=float, default=WET_DRY, help="Final wet/dry mix from 0.0 to 1.0")
    parser.add_argument("--tone-frequency", type=float, default=GENERATED_FREQUENCY_HZ, help="Synthetic test tone frequency in Hz")
    parser.add_argument("--duration", type=float, default=GENERATED_DURATION_SECONDS, help="Synthetic tone duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=GENERATED_SAMPLE_RATE, help="Synthetic tone sample rate")
    parser.add_argument("--amplitude", type=float, default=GENERATED_AMPLITUDE, help="Synthetic tone amplitude")
    parser.add_argument("--channels", type=int, default=GENERATED_CHANNELS, choices=(1, 2), help="Synthetic tone channels")
    return parser.parse_args()


@njit(cache=True)
def integrate_lorenz(total_steps: int, dt: float, sigma: float, rho: float, beta: float, x0: float, y0: float, z0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x0
    y = y0
    z = z0
    xs = np.empty(total_steps, dtype=np.float64)
    ys = np.empty(total_steps, dtype=np.float64)
    zs = np.empty(total_steps, dtype=np.float64)
    for i in range(total_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx
        y += dt * dy
        z += dt * dz
        xs[i] = x
        ys[i] = y
        zs[i] = z
    return xs, ys, zs


@njit(cache=True)
def integrate_rossler(total_steps: int, dt: float, a: float, b: float, c: float, x0: float, y0: float, z0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = x0
    y = y0
    z = z0
    xs = np.empty(total_steps, dtype=np.float64)
    ys = np.empty(total_steps, dtype=np.float64)
    zs = np.empty(total_steps, dtype=np.float64)
    for i in range(total_steps):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dt * dx
        y += dt * dy
        z += dt * dz
        xs[i] = x
        ys[i] = y
        zs[i] = z
    return xs, ys, zs


@njit(cache=True)
def envelope_follower_1d(drive: np.ndarray, attack_coeff: float, release_coeff: float) -> np.ndarray:
    env = np.empty(drive.shape[0], dtype=np.float32)
    prev = 0.0
    for i in range(drive.shape[0]):
        sample = float(drive[i])
        coeff = attack_coeff if sample > prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * sample
        env[i] = prev
    return env


@njit(cache=True)
def one_pole_lowpass_1d(signal: np.ndarray, alpha: float) -> np.ndarray:
    output = np.empty(signal.shape[0], dtype=np.float32)
    prev = 0.0
    for i in range(signal.shape[0]):
        sample = float(signal[i])
        if i == 0:
            prev = sample
        else:
            prev = prev + alpha * (sample - prev)
        output[i] = prev
    return output


@njit(cache=True)
def slew_limit_segment_1d(segment: np.ndarray, max_delta: float) -> np.ndarray:
    output = np.empty(segment.shape[0], dtype=np.float32)
    prev = 0.0
    for i in range(segment.shape[0]):
        target = float(segment[i])
        delta = target - prev
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        prev += delta
        output[i] = prev
    return output


@njit(cache=True)
def allpass_delay_stage_dynamic_1d(signal: np.ndarray, delay_trace: np.ndarray, gain: float) -> np.ndarray:
    x = signal.astype(np.float64)
    y = np.zeros(signal.shape[0], dtype=np.float64)
    for i in range(signal.shape[0]):
        delay = int(delay_trace[i])
        x_delay = x[i - delay] if i >= delay else 0.0
        y_delay = y[i - delay] if i >= delay else 0.0
        y[i] = (-gain * x[i]) + x_delay + (gain * y_delay)
    return y.astype(np.float32)


def generate_sine_wave(
    frequency_hz: float,
    duration_seconds: float,
    sample_rate: int,
    amplitude: float,
    channels: int,
) -> tuple[int, np.ndarray]:
    num_samples = int(round(duration_seconds * sample_rate))
    t = np.arange(num_samples, dtype=np.float64) / sample_rate
    tone = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    audio = tone.astype(np.float32)
    if channels == 2:
        audio = np.column_stack((audio, audio))
    return sample_rate, audio


def wav_to_float32(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return np.asarray(data, dtype=np.float32)
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).clip(-1.0, 1.0)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        scale = max(abs(info.min), info.max)
        return (data.astype(np.float32) / float(scale)).clip(-1.0, 1.0)
    raise TypeError(f"Unsupported WAV dtype: {data.dtype}")


def float32_to_int16(data: np.ndarray) -> np.ndarray:
    return np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)


def load_audio(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(str(path))
    return sample_rate, wav_to_float32(data)


def normalize_if_needed(audio: np.ndarray, headroom: float = 0.999) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > headroom and peak > 0.0:
        audio = audio * (headroom / peak)
    return audio.astype(np.float32)


def envelope_follower(audio: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float) -> np.ndarray:
    drive = np.abs(audio) if audio.ndim == 1 else np.mean(np.abs(audio), axis=1)
    attack_coeff = 0.0 if attack_ms <= 0.0 else np.exp(-1.0 / (sample_rate * attack_ms * 0.001))
    release_coeff = 0.0 if release_ms <= 0.0 else np.exp(-1.0 / (sample_rate * release_ms * 0.001))
    env = envelope_follower_1d(np.asarray(drive, dtype=np.float32), float(attack_coeff), float(release_coeff))
    peak = float(np.max(env)) if env.size else 0.0
    if peak > 0.0:
        env = env / peak
    return env.astype(np.float32)


def map_envelope_to_frequency(
    envelope: np.ndarray,
    base_frequency_hz: float,
    follow_enabled: bool,
    min_frequency_hz: float,
    max_frequency_hz: float,
    sensitivity: float,
) -> np.ndarray | float:
    if not follow_enabled:
        return base_frequency_hz
    low = min(min_frequency_hz, max_frequency_hz)
    high = max(min_frequency_hz, max_frequency_hz)
    shaped = np.power(np.clip(envelope, 0.0, 1.0), max(sensitivity, 1e-6))
    return (low + (high - low) * shaped).astype(np.float32)


def generate_chaos_buffer() -> dict[str, np.ndarray]:
    total_steps = CHAOS_BUFFER_LENGTH + CHAOS_BURN_IN
    if CHAOS_SYSTEM == "lorenz":
        xs, ys, zs = integrate_lorenz(total_steps, CHAOS_DT, LORENZ_SIGMA, LORENZ_RHO, LORENZ_BETA, 0.11, 0.0, 0.0)
    elif CHAOS_SYSTEM == "rossler":
        xs, ys, zs = integrate_rossler(total_steps, CHAOS_DT, ROSSLER_A, ROSSLER_B, ROSSLER_C, 0.11, 0.0, 0.0)
    else:
        raise ValueError("CHAOS_SYSTEM must be 'lorenz' or 'rossler'.")

    def normalize_trace(trace: np.ndarray) -> np.ndarray:
        trimmed = trace[CHAOS_BURN_IN:]
        lo, hi = np.percentile(trimmed, [1.0, 99.0])
        if hi <= lo:
            return np.zeros_like(trimmed, dtype=np.float32)
        return np.clip((trimmed - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    return {"x": normalize_trace(xs), "y": normalize_trace(ys), "z": normalize_trace(zs)}


def build_chaos_base(
    coord: np.ndarray,
    num_samples: int,
    sample_rate: int,
    frequency_hz: np.ndarray | float,
    phase_radians: float,
) -> np.ndarray:
    coord_len = coord.shape[0]
    start_offset = (phase_radians / (2.0 * np.pi)) * coord_len
    if np.isscalar(frequency_hz):
        step = float(frequency_hz) * coord_len / (sample_rate * CHAOS_ORBIT_SECONDS)
        phase = start_offset + step * np.arange(num_samples, dtype=np.float64)
    else:
        freq = np.asarray(frequency_hz, dtype=np.float64)
        step = freq * coord_len / (sample_rate * CHAOS_ORBIT_SECONDS)
        phase = start_offset + np.concatenate(([0.0], np.cumsum(step[:-1], dtype=np.float64)))
    phase_mod = np.mod(phase, coord_len)
    idx0 = np.floor(phase_mod).astype(np.int64)
    idx1 = (idx0 + 1) % coord_len
    frac = (phase_mod - idx0).astype(np.float32)
    return (coord[idx0] * (1.0 - frac) + coord[idx1] * frac).astype(np.float32)


def build_nultra_aperture(base_trace: np.ndarray, gamma: float, eta: np.ndarray | float) -> np.ndarray:
    a_gamma = np.power(np.clip(base_trace, 0.0, 1.0), gamma)
    if np.isscalar(eta):
        return np.where(a_gamma < eta, 0.0, a_gamma).astype(np.float32)
    eta_trace = np.asarray(eta, dtype=np.float32)
    return np.where(a_gamma < eta_trace, 0.0, a_gamma).astype(np.float32)


def one_pole_lowpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    cutoff = max(float(cutoff_hz), 1.0)
    alpha = float(1.0 - np.exp(-2.0 * np.pi * cutoff / sample_rate))
    if audio.ndim == 1:
        return one_pole_lowpass_1d(np.asarray(audio, dtype=np.float32), alpha)
    left = one_pole_lowpass_1d(np.asarray(audio[:, 0], dtype=np.float32), alpha)
    right = one_pole_lowpass_1d(np.asarray(audio[:, 1], dtype=np.float32), alpha)
    return np.column_stack((left, right)).astype(np.float32)


def zero_phase_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float, btype: str) -> np.ndarray:
    nyquist = sample_rate * 0.5
    normalized = min(max(cutoff_hz / nyquist, 1e-6), 0.999)
    sos = butter(4, normalized, btype=btype, output="sos")
    return sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def split_bands(audio: np.ndarray, sample_rate: int, low_xover_hz: float, high_xover_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = zero_phase_filter(audio, sample_rate, low_xover_hz, "lowpass")
    high = zero_phase_filter(audio, sample_rate, high_xover_hz, "highpass")
    mid = (audio - low - high).astype(np.float32)
    return low, mid, high


def active_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, mask.size))
    return segments


def nearest_quiet_index(signal: np.ndarray, edge_index: int, search_samples: int, forward: bool) -> int:
    if forward:
        start = max(0, edge_index)
        stop = min(signal.size, edge_index + search_samples + 1)
    else:
        start = max(0, edge_index - search_samples)
        stop = min(signal.size, edge_index + 1)
    window = signal[start:stop]
    if window.size == 0:
        return int(np.clip(edge_index, 0, signal.size - 1))
    return int(start + np.argmin(np.abs(window)))


def slew_limit_segment(segment: np.ndarray, max_delta: float) -> np.ndarray:
    if segment.size == 0 or max_delta <= 0.0:
        return segment.astype(np.float32, copy=True)
    return slew_limit_segment_1d(np.asarray(segment, dtype=np.float32), float(max_delta))


def smooth_high_aperture_channel(audio: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    smoothed = np.zeros_like(aperture, dtype=np.float32)
    for start, end in active_segments(aperture > HIGH_ZERO_FLOOR):
        rise_index = nearest_quiet_index(audio, start, HIGH_ZERO_ALIGN_SEARCH_SAMPLES, forward=True)
        fall_index = nearest_quiet_index(audio, end - 1, HIGH_ZERO_ALIGN_SEARCH_SAMPLES, forward=False)
        if fall_index <= rise_index:
            rise_index = start
            fall_index = end - 1
        segment = aperture[rise_index : fall_index + 1].astype(np.float32, copy=True)
        if segment.size == 0:
            continue
        ramp = min(HIGH_ZERO_ALIGN_RAMP_SAMPLES, max(1, segment.size // 2))
        fade = np.linspace(0.0, 1.0, ramp + 1, dtype=np.float32)[:-1]
        segment[:ramp] *= fade
        segment[-ramp:] *= fade[::-1]
        segment[0] = 0.0
        segment[-1] = 0.0
        if HIGH_APERTURE_SLEW_ENABLED:
            segment = slew_limit_segment(segment, HIGH_APERTURE_MAX_DELTA_PER_SAMPLE)
            segment = slew_limit_segment(segment[::-1].copy(), HIGH_APERTURE_MAX_DELTA_PER_SAMPLE)[::-1]
        smoothed[rise_index : fall_index + 1] = segment
    return smoothed


def smooth_high_aperture(audio: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    if aperture.ndim == 1:
        return smooth_high_aperture_channel(audio, aperture)
    smoothed = np.zeros_like(aperture, dtype=np.float32)
    smoothed[:, 0] = smooth_high_aperture_channel(audio[:, 0], aperture[:, 0])
    smoothed[:, 1] = smooth_high_aperture_channel(audio[:, 1], aperture[:, 1])
    return smoothed


def diffuse_segment_dynamic(signal: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    diffused = signal.astype(np.float32, copy=True)
    for base_delay, morph_delay, gain in zip(HIGH_DIFFUSION_BASE_DELAYS, HIGH_DIFFUSION_MORPH_DELAYS, HIGH_DIFFUSION_GAINS):
        delay_trace = base_delay + np.round(morph_delay * np.clip(aperture, 0.0, 1.0)).astype(np.int32)
        delay_trace = np.maximum(delay_trace, 1)
        diffused = allpass_delay_stage_dynamic_1d(diffused, delay_trace, float(gain))
    return (diffused * HIGH_DIFFUSION_OUTPUT_GAIN).astype(np.float32)


def diffuse_high_band(audio: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    transformed = audio.astype(np.float32, copy=True)
    if audio.ndim == 1:
        for start, end in active_segments(aperture > HIGH_ZERO_FLOOR):
            ctx_start = max(0, start - HIGH_DIFFUSION_CONTEXT_SAMPLES)
            segment = diffuse_segment_dynamic(audio[ctx_start:end], aperture[ctx_start:end])
            transformed[start:end] = segment[start - ctx_start : end - ctx_start]
        return transformed
    for ch in range(audio.shape[1]):
        for start, end in active_segments(aperture[:, ch] > HIGH_ZERO_FLOOR):
            ctx_start = max(0, start - HIGH_DIFFUSION_CONTEXT_SAMPLES)
            segment = diffuse_segment_dynamic(audio[ctx_start:end, ch], aperture[ctx_start:end, ch])
            transformed[start:end, ch] = segment[start - ctx_start : end - ctx_start]
    return transformed


def apply_gain_transform(audio: np.ndarray, gain: float) -> np.ndarray:
    return (audio * gain).astype(np.float32)


def bleed_residue_to_mid(high_band: np.ndarray, high_transformed: np.ndarray, aperture: np.ndarray, sample_rate: int) -> np.ndarray:
    aperture_view = aperture[:, None] if (high_band.ndim == 2 and aperture.ndim == 1) else aperture
    residue = (1.0 - aperture_view) * (high_transformed - high_band)
    filtered = one_pole_lowpass(residue, sample_rate, MID_RESIDUE_FEEDBACK_LOWPASS_HZ)
    delayed = np.zeros_like(filtered, dtype=np.float32)
    d = MID_RESIDUE_FEEDBACK_DELAY_SAMPLES
    if filtered.ndim == 1:
        if filtered.shape[0] > d:
            delayed[d:] = filtered[:-d]
    else:
        if filtered.shape[0] > d:
            delayed[d:, :] = filtered[:-d, :]
    return (delayed * MID_RESIDUE_FEEDBACK_BETA).astype(np.float32)


def build_band_aperture(
    audio: np.ndarray,
    sample_rate: int,
    settings: BandSettings,
    chaos_buffer: dict[str, np.ndarray],
    eta_override: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray | float]:
    envelope = envelope_follower(audio, sample_rate, settings.env_attack_ms, settings.env_release_ms)
    frequency_trace = map_envelope_to_frequency(
        envelope=envelope,
        base_frequency_hz=settings.frequency_hz,
        follow_enabled=settings.env_follow_frequency,
        min_frequency_hz=settings.env_frequency_min_hz,
        max_frequency_hz=settings.env_frequency_max_hz,
        sensitivity=settings.env_sensitivity,
    )
    left_base = build_chaos_base(
        coord=chaos_buffer[settings.chaos_axis],
        num_samples=audio.shape[0],
        sample_rate=sample_rate,
        frequency_hz=frequency_trace,
        phase_radians=settings.phase_radians,
    )
    eta_value = settings.eta if eta_override is None else eta_override
    left_aperture = build_nultra_aperture(left_base, settings.gamma, eta_value)
    if audio.ndim == 2 and audio.shape[1] == 2 and settings.stereo_phase_offset_radians != 0.0:
        right_base = build_chaos_base(
            coord=chaos_buffer[settings.chaos_axis],
            num_samples=audio.shape[0],
            sample_rate=sample_rate,
            frequency_hz=frequency_trace,
            phase_radians=settings.phase_radians + settings.stereo_phase_offset_radians,
        )
        right_aperture = build_nultra_aperture(right_base, settings.gamma, eta_value)
        aperture = np.column_stack((left_aperture, right_aperture)).astype(np.float32)
    else:
        aperture = left_aperture
    return aperture, frequency_trace


def apply_nultra_band(
    audio: np.ndarray,
    sample_rate: int,
    settings: BandSettings,
    chaos_buffer: dict[str, np.ndarray],
    eta_override: np.ndarray | float | None = None,
    feedback_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | float]:
    state = audio if feedback_in is None else (audio + feedback_in).astype(np.float32)
    aperture, frequency_trace = build_band_aperture(state, sample_rate, settings, chaos_buffer, eta_override)
    if settings.name == "high":
        aperture = smooth_high_aperture(state, aperture)
        transformed = diffuse_high_band(state, aperture)
    elif settings.transform_mode == "gain":
        transformed = apply_gain_transform(state, settings.transform_gain)
    elif settings.transform_mode == "lowpass":
        transformed = one_pole_lowpass(state, sample_rate, settings.lowpass_cutoff_hz)
    else:
        raise ValueError(f"Unsupported transform mode: {settings.transform_mode}")
    aperture_view = aperture[:, None] if (state.ndim == 2 and aperture.ndim == 1) else aperture
    output = state + aperture_view * (transformed - state)
    return output.astype(np.float32), transformed.astype(np.float32), aperture, frequency_trace


def render_multiband(
    audio: np.ndarray,
    sample_rate: int,
    low_xover_hz: float,
    high_xover_hz: float,
    wet_dry: float,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    chaos_buffer = generate_chaos_buffer()
    low_band, mid_band, high_band = split_bands(audio, sample_rate, low_xover_hz, high_xover_hz)

    low_out, _, low_aperture, low_freq = apply_nultra_band(low_band, sample_rate, LOW_BAND, chaos_buffer)
    low_control = low_aperture if low_aperture.ndim == 1 else np.mean(low_aperture, axis=1)
    high_eta = np.clip(HIGH_BAND.eta + HIGH_PARASITIC_ETA_ALPHA * low_control, 0.0, 0.98).astype(np.float32)

    high_out, high_transformed, high_aperture, high_freq = apply_nultra_band(
        high_band,
        sample_rate,
        HIGH_BAND,
        chaos_buffer,
        eta_override=high_eta,
    )

    mid_feedback = bleed_residue_to_mid(high_band, high_transformed, high_aperture, sample_rate)
    mid_out, _, mid_aperture, mid_freq = apply_nultra_band(
        mid_band,
        sample_rate,
        MID_BAND,
        chaos_buffer,
        feedback_in=mid_feedback,
    )

    processed = low_out + mid_out + high_out
    wet = float(np.clip(wet_dry, 0.0, 1.0))
    combined = wet * processed + (1.0 - wet) * audio
    combined = normalize_if_needed(combined)

    stats = {
        "low": {
            "null_ratio": float(np.mean(low_aperture == 0.0)),
            "freq_min": float(np.min(low_freq)) if not np.isscalar(low_freq) else float(low_freq),
            "freq_max": float(np.max(low_freq)) if not np.isscalar(low_freq) else float(low_freq),
        },
        "mid": {
            "null_ratio": float(np.mean(mid_aperture == 0.0)),
            "freq_min": float(np.min(mid_freq)) if not np.isscalar(mid_freq) else float(mid_freq),
            "freq_max": float(np.max(mid_freq)) if not np.isscalar(mid_freq) else float(mid_freq),
            "feedback_rms": float(np.sqrt(np.mean(np.square(mid_feedback)))) if mid_feedback.size else 0.0,
        },
        "high": {
            "null_ratio": float(np.mean(high_aperture == 0.0)),
            "freq_min": float(np.min(high_freq)) if not np.isscalar(high_freq) else float(high_freq),
            "freq_max": float(np.max(high_freq)) if not np.isscalar(high_freq) else float(high_freq),
            "eta_min": float(np.min(high_eta)),
            "eta_max": float(np.max(high_eta)),
        },
    }
    return combined, stats


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    if args.mode == "generate":
        sample_rate, audio = generate_sine_wave(
            frequency_hz=args.tone_frequency,
            duration_seconds=args.duration,
            sample_rate=args.sample_rate,
            amplitude=args.amplitude,
            channels=args.channels,
        )
        source_label = f"generated {args.tone_frequency:.1f} Hz tone"
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input WAV not found: {input_path}")
        sample_rate, audio = load_audio(input_path)
        source_label = str(input_path)

    if not 20.0 <= args.low_xover < args.high_xover < (sample_rate * 0.5 - 100.0):
        raise ValueError("Crossover values must be ordered and remain below Nyquist.")

    processed, stats = render_multiband(audio, sample_rate, args.low_xover, args.high_xover, args.wet_dry)
    output_path = Path(args.output)
    wavfile.write(str(output_path), sample_rate, float32_to_int16(processed))

    elapsed = time.perf_counter() - start_time
    print(f"Source: {source_label}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {1 if processed.ndim == 1 else processed.shape[1]}")
    print(f"Output: {output_path.resolve()}")
    print(f"Crossovers: low={args.low_xover} Hz, high={args.high_xover} Hz")
    print(f"Wet/Dry: {float(np.clip(args.wet_dry, 0.0, 1.0)):.2f}")
    print(
        "Chaos:"
        f" system={CHAOS_SYSTEM},"
        f" buffer={CHAOS_BUFFER_LENGTH},"
        f" orbit_seconds={CHAOS_ORBIT_SECONDS},"
        f" dt={CHAOS_DT}"
    )
    print(
        "High ecosystem:"
        f" parasitic_alpha={HIGH_PARASITIC_ETA_ALPHA},"
        f" slew={'on' if HIGH_APERTURE_SLEW_ENABLED else 'off'},"
        f" max_delta={HIGH_APERTURE_MAX_DELTA_PER_SAMPLE},"
        f" base_delays={HIGH_DIFFUSION_BASE_DELAYS},"
        f" morph_delays={HIGH_DIFFUSION_MORPH_DELAYS}"
    )
    print(
        "Mid feedback:"
        f" beta={MID_RESIDUE_FEEDBACK_BETA},"
        f" cutoff={MID_RESIDUE_FEEDBACK_LOWPASS_HZ} Hz,"
        f" delay={MID_RESIDUE_FEEDBACK_DELAY_SAMPLES} samples"
    )
    for band_name, band_stats in stats.items():
        extra = ""
        if band_name == "mid":
            extra = f", feedback_rms={band_stats['feedback_rms']:.6f}"
        if band_name == "high":
            extra = f", eta_span={band_stats['eta_min']:.3f}-{band_stats['eta_max']:.3f}"
        print(
            f"{band_name.capitalize()} band:"
            f" null_coverage={band_stats['null_ratio']:.2%},"
            f" freq_span={band_stats['freq_min']:.2f}-{band_stats['freq_max']:.2f} Hz"
            f"{extra}"
        )
    print(f"Render time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
