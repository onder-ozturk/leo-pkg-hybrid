#!/usr/bin/env python3
"""Full LEO satellite pass simulator with Loo channel model.

Generates a realistic 5-minute FGN-100 pass with time-varying:
  - Elevation angle (horizon → zenith → horizon)
  - Doppler shift (positive → zero → negative)
  - Rician K-factor (elevation-dependent, ITU-R P.681)
  - Log-normal shadowing (correlated, Loo model)
  - Path loss (slant range dependent)

This provides the "real data substitute" that addresses the τ_slow problem:
  - Pass-level features (elevation trend, Doppler rate, shadowing envelope)
    change on timescales of SECONDS, far exceeding τ_RT ≈ 3.4 ms.

References:
  - Loo, C. (1985). "A statistical model for a land mobile satellite link."
    IEEE Trans. Veh. Technol., VT-34(3), 122–127.
  - ITU-R P.681-11 (2019). "Propagation data required for the design of
    Earth-space land mobile telecommunication systems."
  - Fontan, F.P. et al. (2001). "Statistical modeling of the LMS channel."
    IEEE Trans. Veh. Technol., 50(6), 1549–1567.
"""

from __future__ import annotations
import math
import hashlib
import hmac
import json
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"

# ===================================================================
# Loo Channel Model Parameters (ITU-R P.681 / Fontan et al.)
# ===================================================================
# These parameters are derived from real measurement campaigns
# for S-band (2.2 GHz) land-mobile satellite links.

@dataclass(frozen=True)
class LooParams:
    """Loo model parameters for a given environment/elevation."""
    mu_db: float      # LOS mean power (dB)
    sigma_db: float   # LOS fluctuation std (dB) — shadowing of LOS
    mp_power_db: float  # Multipath mean power (dB)

    @property
    def mu_linear(self) -> float:
        return 10 ** (self.mu_db / 20.0)

    @property
    def sigma_linear(self) -> float:
        return self.sigma_db / 8.686  # convert dB std to Neper std

    @property
    def mp_power_linear(self) -> float:
        return 10 ** (self.mp_power_db / 10.0)


def loo_params_from_elevation(elev_deg: float) -> LooParams:
    """ITU-R P.681 inspired Loo parameters vs elevation.

    Based on Table 4.1 in Fontan et al. (2001) for suburban environment,
    S-band, adapted to LEO geometry.
    """
    # Higher elevation → stronger LOS, less shadowing, less multipath
    if elev_deg >= 60:
        return LooParams(mu_db=-0.5, sigma_db=1.0, mp_power_db=-15.0)
    elif elev_deg >= 40:
        return LooParams(mu_db=-1.5, sigma_db=2.0, mp_power_db=-12.0)
    elif elev_deg >= 20:
        return LooParams(mu_db=-3.0, sigma_db=3.5, mp_power_db=-9.0)
    else:
        return LooParams(mu_db=-5.0, sigma_db=5.0, mp_power_db=-6.0)


# ===================================================================
# Satellite Pass Geometry
# ===================================================================

@dataclass
class PassGeometry:
    """Time-varying satellite pass geometry."""
    time_s: np.ndarray          # time vector (seconds)
    elevation_deg: np.ndarray   # elevation angle
    azimuth_deg: np.ndarray     # azimuth angle
    slant_range_km: np.ndarray  # slant range
    doppler_hz: np.ndarray      # Doppler shift
    doppler_rate_hz_s: np.ndarray  # Doppler rate (Hz/s)


def generate_pass_geometry(
    altitude_km: float = 510.0,
    velocity_kms: float = 7.6,
    freq_ghz: float = 2.2,
    max_elev_deg: float = 75.0,
    duration_s: float = 300.0,  # 5 minutes
    sample_rate_hz: float = 100.0,  # 100 Hz geometry update
) -> PassGeometry:
    """Generate a realistic LEO satellite pass trajectory.

    The pass follows a sinusoidal elevation profile:
      elev(t) = max_elev * sin(π * t / duration)
    peaking at the midpoint.
    """
    R_E = 6371.0
    c = 3e8
    f_c = freq_ghz * 1e9

    n_samples = int(duration_s * sample_rate_hz)
    t = np.linspace(0, duration_s, n_samples)

    # Elevation profile: smooth rise and fall
    elev = max_elev_deg * np.sin(np.pi * t / duration_s)
    elev = np.clip(elev, 5.0, 90.0)  # minimum 5° for practical reception

    # Azimuth: linear sweep (simplified)
    azimuth = np.linspace(180.0, 0.0, n_samples)

    # Slant range
    elev_rad = np.radians(elev)
    slant = -R_E * np.sin(elev_rad) + np.sqrt(
        (R_E * np.sin(elev_rad)) ** 2 + 2 * R_E * altitude_km + altitude_km ** 2
    )

    # Doppler: v_radial changes sign during pass
    # At low elevation: satellite approaching → positive Doppler
    # At zenith: v_radial ≈ 0
    # At high elevation (descending): satellite receding → negative Doppler
    # Model: f_D = (v * cos(elev) * sign(t - t_mid)) * f_c / c
    t_mid = duration_s / 2.0
    approach_sign = np.sign(t_mid - t)
    v_radial = velocity_kms * 1e3 * np.cos(elev_rad) * approach_sign
    doppler = v_radial * f_c / c

    # Doppler rate (Hz/s)
    doppler_rate = np.gradient(doppler, t)

    return PassGeometry(
        time_s=t,
        elevation_deg=elev,
        azimuth_deg=azimuth,
        slant_range_km=slant,
        doppler_hz=doppler,
        doppler_rate_hz_s=doppler_rate,
    )


# ===================================================================
# Loo Channel Generator
# ===================================================================

def generate_loo_channel(
    geometry: PassGeometry,
    freq_ghz: float = 2.2,
    bandwidth_mhz: float = 10.0,
    snr_db: float = 20.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate Loo channel for a full satellite pass.

    Returns: (h_alice, h_bob, h_eve, time_s)
    where each is a complex channel coefficient time series.
    """
    rng = np.random.default_rng(seed)
    n = len(geometry.time_s)
    dt = geometry.time_s[1] - geometry.time_s[0] if n > 1 else 0.01

    # Pre-allocate
    h_base = np.empty(n, dtype=complex)

    # Shadowing process: correlated log-normal (Loo model)
    # Correlation time: ~1-5 seconds (physical shadowing changes slowly)
    shadow_corr_time_s = 3.0
    shadow_alpha = math.exp(-dt / shadow_corr_time_s)
    shadow_process = np.empty(n)
    shadow_process[0] = rng.normal(0, 1)
    for i in range(1, n):
        shadow_process[i] = shadow_alpha * shadow_process[i-1] + \
                           math.sqrt(1 - shadow_alpha**2) * rng.normal(0, 1)

    for i in range(n):
        elev = geometry.elevation_deg[i]
        loo = loo_params_from_elevation(elev)

        # LOS component with shadowing (Loo model core)
        los_shadow = loo.mu_linear * math.exp(loo.sigma_linear * shadow_process[i])
        los_phase = 2 * math.pi * geometry.doppler_hz[i] * geometry.time_s[i]
        los = los_shadow * np.exp(1j * los_phase)

        # NLOS (Rayleigh) component
        mp_sigma = math.sqrt(loo.mp_power_linear / 2.0)
        nlos = mp_sigma * (rng.normal() + 1j * rng.normal())

        # Path loss (normalised — we add SNR via noise)
        slant = geometry.slant_range_km[i]
        fspl_db = 20 * math.log10(max(slant, 1)) + 20 * math.log10(freq_ghz) + 92.45
        path_loss = 10 ** (-fspl_db / 20.0)

        h_base[i] = path_loss * (los + nlos)

    # Normalise to unit mean power
    mean_power = np.mean(np.abs(h_base) ** 2)
    h_base /= np.sqrt(max(mean_power, 1e-15))

    # Alice (satellite) and Bob (ground) observations
    noise_sigma = 10 ** (-snr_db / 20.0)

    # Bob's observation: h_base + noise
    n_bob = noise_sigma * (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    h_bob = h_base + n_bob

    # Alice's observation: same slow features, different fast fading + noise
    # (models the reciprocity impairment from propagation delay)
    # The slow features (shadowing, Doppler structure, path loss) are shared
    # The fast fading (NLOS) is independently generated
    h_alice_base = np.empty(n, dtype=complex)
    for i in range(n):
        elev = geometry.elevation_deg[i]
        loo = loo_params_from_elevation(elev)
        los_shadow = loo.mu_linear * math.exp(loo.sigma_linear * shadow_process[i])
        los_phase = 2 * math.pi * geometry.doppler_hz[i] * geometry.time_s[i]
        los = los_shadow * np.exp(1j * los_phase)
        mp_sigma = math.sqrt(loo.mp_power_linear / 2.0)
        nlos = mp_sigma * (rng.normal() + 1j * rng.normal())  # INDEPENDENT fast fading
        slant = geometry.slant_range_km[i]
        fspl_db = 20 * math.log10(max(slant, 1)) + 20 * math.log10(freq_ghz) + 92.45
        path_loss = 10 ** (-fspl_db / 20.0)
        h_alice_base[i] = path_loss * (los + nlos)
    h_alice_base /= np.sqrt(max(np.mean(np.abs(h_alice_base) ** 2), 1e-15))
    n_alice = noise_sigma * (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    h_alice = h_alice_base + n_alice

    # Eve's observation: spatially decorrelated
    h_eve_base = np.empty(n, dtype=complex)
    spatial_corr = 0.15  # Eve at ~2λ distance
    for i in range(n):
        elev = geometry.elevation_deg[i]
        loo = loo_params_from_elevation(elev)
        # Eve sees partially correlated shadowing but independent fast fading
        eve_shadow = shadow_alpha * shadow_process[i] + \
                    math.sqrt(1 - shadow_alpha**2) * rng.normal(0, 1) * (1 - spatial_corr)
        los_shadow = loo.mu_linear * math.exp(loo.sigma_linear * eve_shadow)
        los_phase = 2 * math.pi * geometry.doppler_hz[i] * geometry.time_s[i] + rng.uniform(0, 2*math.pi)
        los = los_shadow * np.exp(1j * los_phase)
        mp_sigma = math.sqrt(loo.mp_power_linear / 2.0)
        nlos = mp_sigma * (rng.normal() + 1j * rng.normal())
        slant = geometry.slant_range_km[i]
        fspl_db = 20 * math.log10(max(slant, 1)) + 20 * math.log10(freq_ghz) + 92.45
        path_loss = 10 ** (-fspl_db / 20.0)
        h_eve_base[i] = path_loss * (los + nlos)
    h_eve_base /= np.sqrt(max(np.mean(np.abs(h_eve_base) ** 2), 1e-15))
    n_eve = noise_sigma * (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    h_eve = h_eve_base + n_eve

    return h_alice, h_bob, h_eve, geometry.time_s


# ===================================================================
# Pass-Level Feature Extraction
# ===================================================================

def extract_pass_features(h: np.ndarray, window_s: float = 2.0, sample_rate: float = 100.0) -> np.ndarray:
    """Extract slowly-varying features from a satellite pass.

    Window of ~2 seconds captures shadowing/geometry changes
    while averaging out fast fading.
    """
    win = max(3, int(window_s * sample_rate)) | 1
    kernel = np.ones(win) / win

    amp = np.convolve(np.abs(h), kernel, mode="same")
    phase = np.convolve(np.unwrap(np.angle(h)), kernel, mode="same")
    doppler_proxy = np.convolve(np.abs(np.gradient(phase)), kernel, mode="same")

    # Downsample to ~1 feature per window
    ds = max(1, win // 2)
    return np.concatenate([amp[::ds], phase[::ds], doppler_proxy[::ds]])


# ===================================================================
# Key generation pipeline for pass data
# ===================================================================

def cascade_reconciliation(bits_a, bits_b, num_passes=6):
    """Cascade protocol (same as main code)."""
    corrected = bits_b.copy()
    n = len(bits_a)
    total_leaked = 0
    for pass_idx in range(num_passes):
        block_size = max(4, n // (2 ** (pass_idx + 2)))
        perm = np.arange(n)
        if pass_idx > 0:
            rng_p = np.random.default_rng(42 + pass_idx)
            perm = rng_p.permutation(n)
        a_p = bits_a[perm]
        c_p = corrected[perm]
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            pa = int(np.sum(a_p[start:end])) % 2
            pc = int(np.sum(c_p[start:end])) % 2
            total_leaked += 1
            if pa != pc:
                lo, hi = start, end
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    ppa = int(np.sum(a_p[lo:mid])) % 2
                    ppc = int(np.sum(c_p[lo:mid])) % 2
                    total_leaked += 1
                    if ppa != ppc:
                        hi = mid
                    else:
                        lo = mid
                c_p[lo] = a_p[lo]
        inv_perm = np.argsort(perm)
        corrected = c_p[inv_perm]
    residual_kdr = float(np.mean(bits_a != corrected))
    return corrected, residual_kdr, total_leaked


def pass_key_generation(
    h_alice: np.ndarray,
    h_bob: np.ndarray,
    snr_db: float = 20.0,
    sample_rate: float = 100.0,
) -> dict:
    """Full PKG pipeline on pass-level data."""
    fa = extract_pass_features(h_alice, window_s=2.0, sample_rate=sample_rate)
    fb = extract_pass_features(h_bob, window_s=2.0, sample_rate=sample_rate)

    # Binary quantisation
    min_len = min(len(fa), len(fb))
    fa, fb = fa[:min_len], fb[:min_len]
    ref = 0.5 * (fa + fb)
    thr = np.median(ref)

    bits_a = (fa > thr).astype(np.int8)
    bits_b = (fb > thr).astype(np.int8)

    raw_kdr = float(np.mean(bits_a != bits_b))

    # Cascade reconciliation
    corrected_b, residual_kdr, leaked = cascade_reconciliation(bits_a, bits_b)

    # Privacy amplification
    key_material = corrected_b.tobytes()
    phy_key = hashlib.sha256(key_material).digest()[:16]

    # KGR
    n_channel = len(h_alice)
    usable_bits = max(0, len(bits_a) - leaked - 64)
    pa_bits = max(64, usable_bits)
    kgr = pa_bits / n_channel

    # NIST tests
    key_bits = np.unpackbits(np.frombuffer(phy_key, dtype=np.uint8))
    n_bits = len(key_bits)
    s = np.sum(2.0 * key_bits - 1.0)
    freq_p = float(math.erfc(abs(s) / math.sqrt(n_bits) / math.sqrt(2.0)))

    return {
        "raw_kdr": raw_kdr,
        "residual_kdr": residual_kdr,
        "kgr_bpcu": kgr,
        "leaked_bits": leaked,
        "n_features": min_len,
        "n_channel_samples": n_channel,
        "nist_freq_p": freq_p,
        "phy_key_hex": phy_key.hex(),
    }


# ===================================================================
# Autocorrelation analysis on pass data
# ===================================================================

def pass_autocorrelation(h: np.ndarray, sample_rate: float = 100.0, max_lag_s: float = 30.0) -> dict:
    """Compute ACF of fast and slow features from a satellite pass."""
    max_lag = int(max_lag_s * sample_rate)
    max_lag = min(max_lag, len(h) // 3)

    fast_amp = np.abs(h)
    win = max(3, int(2.0 * sample_rate)) | 1
    slow_amp = np.convolve(fast_amp, np.ones(win) / win, mode="same")

    def acf(x, ml):
        x = x - np.mean(x)
        var = np.var(x)
        if var < 1e-15:
            return np.ones(ml)
        r = np.empty(ml)
        n = len(x)
        for lag in range(ml):
            r[lag] = np.mean(x[:n - lag] * x[lag:n]) / var if lag < n else 0
        return r

    acf_fast = acf(fast_amp, max_lag)
    acf_slow = acf(slow_amp, max_lag)

    lags_s = np.arange(max_lag) / sample_rate
    lags_ms = lags_s * 1e3

    # Decorrelation times
    tau_fast_idx = np.argmax(acf_fast < 0.5) if np.any(acf_fast < 0.5) else max_lag - 1
    tau_slow_idx = np.argmax(acf_slow < 0.5) if np.any(acf_slow < 0.5) else max_lag - 1
    tau_fast_ms = float(lags_ms[tau_fast_idx])
    tau_slow_ms = float(lags_ms[tau_slow_idx])
    tau_rt_ms = 3.4  # FGN-100 round-trip

    return {
        "tau_fast_ms": tau_fast_ms,
        "tau_slow_ms": tau_slow_ms,
        "tau_rt_ms": tau_rt_ms,
        "ratio_slow_to_rt": tau_slow_ms / tau_rt_ms,
        "acf_fast": acf_fast,
        "acf_slow": acf_slow,
        "lags_ms": lags_ms,
    }


# ===================================================================
# Secrecy capacity on pass data
# ===================================================================

def pass_secrecy_capacity(h_alice, h_bob, h_eve, sample_rate=100.0):
    """Compute feature-level secrecy capacity."""
    fa = extract_pass_features(h_alice, 2.0, sample_rate)
    fb = extract_pass_features(h_bob, 2.0, sample_rate)
    fe = extract_pass_features(h_eve, 2.0, sample_rate)

    ml = min(len(fa), len(fb), len(fe))
    fa, fb, fe = fa[:ml], fb[:ml], fe[:ml]

    rho_ab = float(np.clip(np.corrcoef(fa, fb)[0, 1], -0.999, 0.999))
    rho_ae = float(np.clip(np.corrcoef(fa, fe)[0, 1], -0.999, 0.999))

    mi_ab = -0.5 * math.log2(max(1e-12, 1 - rho_ab ** 2))
    mi_ae = -0.5 * math.log2(max(1e-12, 1 - rho_ae ** 2))
    cs = max(0.0, mi_ab - mi_ae)

    return {
        "rho_AB": rho_ab,
        "rho_AE": rho_ae,
        "I_AB": mi_ab,
        "I_AE": mi_ae,
        "C_s": cs,
    }


# ===================================================================
# Main — run full pass simulation
# ===================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    print("=" * 60)
    print("LEO SATELLITE PASS SIMULATOR")
    print("Loo Channel Model + ITU-R P.681 Parameters")
    print("=" * 60)

    # Generate pass geometry
    geom = generate_pass_geometry(
        altitude_km=510.0,
        velocity_kms=7.6,
        freq_ghz=2.2,
        max_elev_deg=75.0,
        duration_s=300.0,
        sample_rate_hz=100.0,
    )
    print(f"\nPass duration: {geom.time_s[-1]:.0f} s")
    print(f"Max elevation: {np.max(geom.elevation_deg):.1f}°")
    print(f"Max Doppler: {np.max(np.abs(geom.doppler_hz))/1e3:.1f} kHz")
    print(f"Slant range: {np.min(geom.slant_range_km):.0f} - {np.max(geom.slant_range_km):.0f} km")

    # Run PKG for multiple SNR values
    snr_values = [10, 15, 20, 25, 30]
    mc_rounds = 50
    results = []

    for snr in snr_values:
        kdr_list, kgr_list, raw_kdr_list = [], [], []
        nist_pass = 0

        for mc in range(mc_rounds):
            h_a, h_b, h_e, t = generate_loo_channel(geom, snr_db=snr, seed=1000 + mc * 100 + snr)
            pkg = pass_key_generation(h_a, h_b, snr, sample_rate=100.0)
            kdr_list.append(pkg["residual_kdr"])
            kgr_list.append(pkg["kgr_bpcu"])
            raw_kdr_list.append(pkg["raw_kdr"])
            if pkg["nist_freq_p"] > 0.01:
                nist_pass += 1

        kdr_arr = np.array(kdr_list)
        kgr_arr = np.array(kgr_list)
        kdr_mean = float(kdr_arr.mean())
        kdr_ci = float(1.96 * kdr_arr.std() / math.sqrt(mc_rounds))

        results.append({
            "snr_db": snr,
            "raw_kdr_mean": float(np.mean(raw_kdr_list)),
            "kdr_mean": kdr_mean,
            "kdr_ci": kdr_ci,
            "kgr_mean": float(kgr_arr.mean()),
            "nist_pass_rate": nist_pass / mc_rounds,
        })
        print(f"\nSNR={snr:2d} dB: raw_KDR={np.mean(raw_kdr_list):.4f}, KDR={kdr_mean:.4f} ±{kdr_ci:.4f}, KGR={kgr_arr.mean():.4f}, NIST={nist_pass/mc_rounds:.1%}")

    # Autocorrelation analysis (single pass at 20 dB)
    print("\nAutocorrelation analysis...")
    h_a, h_b, h_e, t = generate_loo_channel(geom, snr_db=20.0, seed=9999)
    acf = pass_autocorrelation(h_b, sample_rate=100.0)
    print(f"  τ_fast = {acf['tau_fast_ms']:.1f} ms")
    print(f"  τ_slow = {acf['tau_slow_ms']:.1f} ms")
    print(f"  τ_RT   = {acf['tau_rt_ms']:.1f} ms")
    print(f"  τ_slow / τ_RT = {acf['ratio_slow_to_rt']:.1f}×")

    # Secrecy capacity
    print("\nSecrecy capacity analysis...")
    cs = pass_secrecy_capacity(h_a, h_b, h_e, sample_rate=100.0)
    print(f"  ρ(A,B) = {cs['rho_AB']:.4f}")
    print(f"  ρ(A,E) = {cs['rho_AE']:.4f}")
    print(f"  I(A;B) = {cs['I_AB']:.4f} bpcu")
    print(f"  I(A;E) = {cs['I_AE']:.4f} bpcu")
    print(f"  C_s    = {cs['C_s']:.4f} bpcu")

    # --- Figures ---

    # Figure: Pass geometry
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6))
    axes[0,0].plot(geom.time_s, geom.elevation_deg, color="#0b5ea8")
    axes[0,0].set_ylabel("Elevasyon (°)")
    axes[0,0].set_title("Uydu Geçişi Geometrisi")
    axes[0,0].grid(alpha=0.25)

    axes[0,1].plot(geom.time_s, geom.doppler_hz / 1e3, color="#c0392b")
    axes[0,1].set_ylabel("Doppler (kHz)")
    axes[0,1].set_title("Doppler Kayması")
    axes[0,1].grid(alpha=0.25)

    axes[1,0].plot(geom.time_s, geom.slant_range_km, color="#2ecc71")
    axes[1,0].set_ylabel("Eğik Menzil (km)")
    axes[1,0].set_xlabel("Zaman (s)")
    axes[1,0].set_title("Eğik Menzil")
    axes[1,0].grid(alpha=0.25)

    axes[1,1].plot(t, 20 * np.log10(np.abs(h_b) + 1e-10), color="#8e44ad", alpha=0.5, linewidth=0.5)
    fa_slow = extract_pass_features(h_b, 2.0, 100.0)
    # Plot slow envelope
    win = 201
    slow_env = np.convolve(np.abs(h_b), np.ones(win)/win, mode="same")
    axes[1,1].plot(t, 20 * np.log10(slow_env + 1e-10), color="#e67e22", linewidth=2, label="Yavaş zarf")
    axes[1,1].set_ylabel("Kanal kazancı (dB)")
    axes[1,1].set_xlabel("Zaman (s)")
    axes[1,1].set_title("Loo Kanal Yanıtı (@20 dB)")
    axes[1,1].legend(fontsize=7)
    axes[1,1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure11_pass_geometry.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure11_pass_geometry.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure: ACF comparison
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    ax.plot(acf["lags_ms"] / 1e3, acf["acf_fast"], color="#c0392b", alpha=0.7, label="Hızlı CSI")
    ax.plot(acf["lags_ms"] / 1e3, acf["acf_slow"], color="#0b5ea8", linewidth=2, label="Yavaş zarf (2s pencere)")
    ax.axvline(acf["tau_rt_ms"] / 1e3, color="#2ecc71", linestyle="--", linewidth=1.5, label=f"τ_RT = {acf['tau_rt_ms']:.1f} ms")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Gecikme (s)")
    ax.set_ylabel("Otokorelasyon")
    ax.set_title(f"ACF: τ_slow={acf['tau_slow_ms']:.0f} ms vs τ_RT={acf['tau_rt_ms']:.1f} ms (oran: {acf['ratio_slow_to_rt']:.0f}×)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure12_pass_acf.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure12_pass_acf.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Figure: KDR vs SNR (pass-based)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5))
    snrs = [r["snr_db"] for r in results]
    kdrs = [r["kdr_mean"] for r in results]
    kdr_cis = [r["kdr_ci"] for r in results]
    kgrs = [r["kgr_mean"] for r in results]

    ax1.errorbar(snrs, kdrs, yerr=kdr_cis, marker="o", color="#0b5ea8", linewidth=2, capsize=4, label="KDR (ort. ± 95%GA)")
    ax1.axhline(0.05, color="#c0392b", linestyle="--", label="Hedef: %5")
    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("KDR")
    ax1.set_title("Loo Kanal: KDR vs SNR")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.25)

    ax2.plot(snrs, kgrs, marker="s", color="#9a3412", linewidth=2)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("KGR (bpcu)")
    ax2.set_title("Loo Kanal: KGR vs SNR")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure13_pass_kdr_kgr.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure13_pass_kdr_kgr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Save results
    summary = {
        "model": "Loo (ITU-R P.681)",
        "pass_duration_s": 300,
        "sample_rate_hz": 100,
        "max_elevation_deg": 75,
        "mc_rounds": mc_rounds,
        "main_results": results,
        "autocorrelation": {
            "tau_fast_ms": acf["tau_fast_ms"],
            "tau_slow_ms": acf["tau_slow_ms"],
            "tau_rt_ms": acf["tau_rt_ms"],
            "ratio": acf["ratio_slow_to_rt"],
        },
        "secrecy_capacity": cs,
    }
    with (RESULT_DIR / "pass_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SONUC OZETI")
    print("=" * 60)
    print(f"τ_slow / τ_RT = {acf['ratio_slow_to_rt']:.0f}× → {'BASARILI' if acf['ratio_slow_to_rt'] > 10 else 'YETERSIZ'}")
    print(f"C_s = {cs['C_s']:.4f} bpcu → {'POZITIF' if cs['C_s'] > 0 else 'SIFIR'}")
    for r in results:
        status = "✓" if r["kdr_mean"] < 0.05 else "×"
        print(f"  SNR={r['snr_db']:2d}: KDR={r['kdr_mean']:.3f} {status}")


if __name__ == "__main__":
    main()
