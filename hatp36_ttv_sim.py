#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAT-P-36b TTV Simulation and Noise/Starspot Effects (batman version)
====================================================================

This module:
- Uses `batman` to generate transit light curves for HAT-P-36b.
- Simulates white/red noise, starspot occultations, stellar modulation,
  and instrumental trends.
- Fits global transit parameters and per-transit mid-times (TTVs).
- Includes a 2D visualisation of star, transit chord, and starspots.

Starspots are fully geometric:
- Each spot has latitude, longitude, radius, temperature.
- For each transit, longitude evolves with stellar rotation.
- We project spots to (x, y) on the stellar disk and determine if/when
  the planet actually occults them.
- Bump times are derived from the geometry (no hand-tuned phases).

Dependencies
------------
numpy, scipy, matplotlib, pandas, batman-package
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import batman


# ----------------------------------------------------------------------
# 1. System parameters for HAT-P-36 / HAT-P-36b (approximate literature)
# ----------------------------------------------------------------------
def get_default_hatp36_params():
    """
    Return a dictionary with default system parameters for HAT-P-36b.

    Values are approximate and suitable for TESS-style transit simulations.
    Times are in days, angles in degrees.
    """
    params = dict()

    # Stellar parameters (rough)
    params["M_star"] = 0.96        # [Msun]
    params["R_star"] = 1.05        # [Rsun]
    params["T_star"] = 5600.0      # [K] approximate Teff

    # Planet parameters (rough)
    params["M_p"] = 1.7            # [Mjup]
    params["R_p"] = 1.3            # [Rjup]

    # Orbital / transit parameters (circular orbit assumed)
    params["P"] = 1.327346702      # [days] orbital period
    params["t0"] = 0.0             # [days] reference mid-transit time
    params["a_rs"] = 4.77          # a/R_star
    params["b"] = 0.34             # impact parameter
    params["inc"] = 85.9           # [deg] inclination

    # Transit depth δ ≈ 0.0158 → Rp/Rs ≈ sqrt(δ) ≈ 0.126
    params["rp_rs"] = np.sqrt(0.01580)

    # Limb-darkening coefficients (TESS band, approximate)
    params["u1"] = 0.392
    params["u2"] = 0.257

    return params


# ----------------------------------------------------------------------
# 2. Time array generation
# ----------------------------------------------------------------------
def generate_time_array(P, n_transits=10, cadence_sec=120.0,
                        t0=0.0, t_window=0.20):
    """
    Generate a time array for N consecutive transits with given cadence.
    """
    dt = cadence_sec / 86400.0  # seconds -> days
    t_start = t0 - t_window
    t_end = t0 + (n_transits - 1) * P + t_window
    time = np.arange(t_start, t_end, dt)
    return time


# ----------------------------------------------------------------------
# 3. Transit model using batman
# ----------------------------------------------------------------------
def make_batman_params(rp_rs, a_rs, inc_deg, t0, period,
                       u1, u2, ecc=0.0, w=90.0):
    """
    Create a batman.TransitParams object for a planet on a circular orbit.
    """
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inc_deg
    params.ecc = ecc
    params.w = w
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    return params


def transit_flux(time, rp_rs, a_rs, inc_deg, t0, period, u1, u2):
    """
    Compute model transit flux for given parameters and time grid.
    """
    params = make_batman_params(rp_rs, a_rs, inc_deg, t0, period, u1, u2)
    model = batman.TransitModel(params, time)
    flux = model.light_curve(params)
    return flux


def simulate_clean_transit_series(time, system_params):
    """
    Generate a clean (noiseless) transit time series for HAT-P-36b.
    """
    rp_rs = system_params["rp_rs"]
    a_rs = system_params["a_rs"]
    inc = system_params["inc"]
    t0 = system_params["t0"]
    P = system_params["P"]
    u1 = system_params["u1"]
    u2 = system_params["u2"]

    flux_clean = transit_flux(time, rp_rs, a_rs, inc, t0, P, u1, u2)
    return flux_clean


# ----------------------------------------------------------------------
# 4. Noise and systematics injection
# ----------------------------------------------------------------------
def inject_white_noise(flux, sigma_ppm, rng=None):
    """
    Inject Gaussian white noise into a flux array.
    """
    if sigma_ppm <= 0:
        return flux.copy()
    if rng is None:
        rng = np.random.default_rng()
    sigma = sigma_ppm * 1e-6
    noise = rng.normal(0.0, sigma, size=len(flux))
    return flux + noise


def inject_red_noise_ar1(time, flux, sigma_ppm, tau_days, rng=None):
    """
    Inject simple AR(1) red noise (time-correlated noise).
    """
    if sigma_ppm <= 0 or tau_days <= 0:
        return flux.copy()
    if rng is None:
        rng = np.random.default_rng()

    sigma = sigma_ppm * 1e-6
    dt = np.median(np.diff(time))
    rho = np.exp(-dt / tau_days)  # AR(1) coefficient
    innovation_std = sigma * np.sqrt(1 - rho**2)

    red = np.zeros_like(flux)
    red[0] = rng.normal(0.0, sigma)
    for i in range(1, len(flux)):
        red[i] = rho * red[i - 1] + rng.normal(0.0, innovation_std)
    return flux + red


def inject_stellar_modulation(time, flux, amp_ppm, Prot_days, phase=0.0):
    """
    Inject sinusoidal stellar modulation (starspot rotation signal).
    """
    if amp_ppm <= 0 or Prot_days <= 0:
        return flux.copy()
    amp = amp_ppm * 1e-6
    modulation = amp * np.sin(2 * np.pi * time / Prot_days + phase)
    return flux + modulation


def inject_systematics(time, flux, trend_amp_ppm=0.0,
                       step_amp_ppm=0.0, step_time=None):
    """
    Inject slow baseline trend and an optional step-like jump.
    """
    flux_sys = flux.copy()
    t_min, t_max = np.min(time), np.max(time)
    T = t_max - t_min

    if trend_amp_ppm != 0 and T > 0:
        amp = trend_amp_ppm * 1e-6
        trend = amp * ((time - t_min) / T - 0.5) * 2.0
        flux_sys += trend

    if step_amp_ppm != 0 and step_time is not None:
        amp_step = step_amp_ppm * 1e-6
        flux_sys[time >= step_time] += amp_step

    return flux_sys


# ----------------------------------------------------------------------
# 5. Starspot geometry & occultations
# ----------------------------------------------------------------------
def transit_duration_T14(system_params):
    """
    Approximate total transit duration T14 [days] using standard formula
    for a circular orbit.
    """
    P = system_params["P"]
    a_rs = system_params["a_rs"]
    b = system_params["b"]
    rp_rs = system_params["rp_rs"]
    inc = np.deg2rad(system_params["inc"])

    term = ((1 + rp_rs)**2 - b**2) / (np.sin(inc)**2 * a_rs**2)
    term = np.clip(term, 0.0, 1.0)
    T14 = (P / np.pi) * np.arcsin(np.sqrt(term))
    return T14


def _project_spot_xy(lat_deg, lon_deg):
    """
    Project a spot at (latitude, longitude) on the stellar surface onto the
    sky-plane (x, y), assuming:
    - observer along +z,
    - latitude measured from equator (-90..+90),
    - longitude measured from central meridian (0 = disk center at mid-transit).

    Returns
    -------
    x, y, visible
        x, y in units of stellar radius, visible=True if on the front side.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    # Simple spherical → Cartesian projection
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    visible = (z > 0) and (x**2 + y**2 <= 1.0 + 1e-6)
    return x, y, visible


def build_spot_list_from_ui(spot_dicts, system_params, amp_scale=1.0):
    """
    From a list of UI spot settings (each a dict with keys:
        - lat_deg
        - lon_deg
        - r_spot_rs
        - T_spot

    produce a list of internal spot definitions with:
        - lat_deg, lon0_deg, r_spot_rs, T_spot
        - amp_ppm, sigma_phase

    Geometry (projection and occultation) is computed per-transit later.
    """
    T_star = system_params.get("T_star", 5600.0)
    rp_rs = system_params["rp_rs"]

    spots = []
    for s in spot_dicts:
        lat_deg = float(s["lat_deg"])
        lon_deg = float(s["lon_deg"])
        r_spot_rs = float(s["r_spot_rs"])
        T_spot = float(s["T_spot"])

        # Temperature contrast → intensity contrast
        contrast = 1.0 - (T_spot / T_star) ** 4  # positive for cooler spots
        contrast = max(0.0, contrast)

        # Approximate bump amplitude:
        depth = rp_rs**2
        area_ratio = (r_spot_rs**2) / max(rp_rs**2, 1e-6)
        amp_rel = depth * contrast * area_ratio
        amp_ppm = amp_rel * 1e6 * amp_scale

        # Width in phase ∝ spot size
        sigma_phase = max(0.01, 0.25 * r_spot_rs / max(rp_rs, 0.05))

        spots.append(
            dict(
                lat_deg=lat_deg,
                lon0_deg=lon_deg,
                r_spot_rs=r_spot_rs,
                T_spot=T_spot,
                amp_ppm=amp_ppm,
                sigma_phase=sigma_phase,
            )
        )

    return spots


def compute_spot_geometry_for_transit(spot, system_params, epoch_index,
                                      Prot_days=None):
    """
    For a given spot and transit epoch, compute its projected geometry:

    Inputs
    ------
    spot : dict
        lat_deg, lon0_deg, r_spot_rs, amp_ppm, sigma_phase
    system_params : dict
        includes b, rp_rs, P, etc.
    epoch_index : int
        transit index (0,1,2,...)
    Prot_days : float or None
        stellar rotation period [days]. If given, the spot longitude
        drifts by 360*(epoch_index*P/Prot) degrees.

    Returns
    -------
    geom : dict
        {
          "x": x_spot,
          "y": y_spot,
          "visible": True/False,
          "occultation": True/False,
          "phase_T14": phase in units of T14 (-0.5..0.5),
          "overlap_factor": [0..1]
        }
    """
    b = system_params["b"]
    rp_rs = system_params["rp_rs"]
    P = system_params["P"]
    T14 = transit_duration_T14(system_params)
    L = np.sqrt(max(1.0 - b**2, 1e-6))  # half chord length in x

    lat_deg = spot["lat_deg"]
    lon0_deg = spot["lon0_deg"]

    if Prot_days is None or Prot_days <= 0:
        lon_eff = lon0_deg
    else:
        # how many stellar rotations between epoch 0 and epoch_index?
        delta_t = epoch_index * P
        delta_rot = delta_t / Prot_days
        lon_eff = lon0_deg + 360.0 * delta_rot

    # wrap longitude to [-180,180]
    lon_eff = ((lon_eff + 180.0) % 360.0) - 180.0

    x_spot, y_spot, visible = _project_spot_xy(lat_deg, lon_eff)

    occultation = False
    phase_T14 = 0.0
    overlap_factor = 0.0

    if visible:
        r_spot_rs = spot["r_spot_rs"]
        # distance to transit chord y=b
        d_line = abs(y_spot - b)
        max_sep = rp_rs + r_spot_rs

        if d_line <= max_sep:
            # require x roughly within track + margins
            if abs(x_spot) <= (L + max_sep):
                # time of closest approach when planet centre at x = x_spot
                delta_x = x_spot / L
                if abs(delta_x) <= 1.5:  # allow some slack
                    phase_T14 = 0.5 * delta_x
                    occultation = True

                    # crude overlap factor (0..1)
                    overlap_factor = 1.0 - d_line / max_sep
                    overlap_factor = float(np.clip(overlap_factor, 0.0, 1.0))

    geom = dict(
        x=x_spot,
        y=y_spot,
        visible=visible,
        occultation=occultation,
        phase_T14=phase_T14,
        overlap_factor=overlap_factor,
    )
    return geom


def inject_starspot_occultations(time, flux, system_params, spot_list,
                                 Prot_days=None):
    """
    Inject Gaussian starspot occultation bumps into transits.

    For each transit epoch and each spot:
    - compute spot geometry (x,y,occultation,phase_T14,overlap_factor)
    - if occultation=True, add a Gaussian bump around
        t_c + phase_T14 * T14
      with amplitude amp_ppm * overlap_factor.

    Prot_days
    ---------
    Stellar rotation period [days]. If not None, spot longitudes drift
    between epochs according to Prot_days.
    """
    if not spot_list:
        return flux.copy()

    P = system_params["P"]
    t0 = system_params["t0"]
    T14 = transit_duration_T14(system_params)

    flux_spotted = flux.copy()
    n_est = int((time.max() - t0) / P) + 2

    for n in range(-1, n_est + 1):
        t_c = t0 + n * P
        dt = time - t_c

        for s in spot_list:
            geom = compute_spot_geometry_for_transit(
                s, system_params, epoch_index=n, Prot_days=Prot_days
            )
            if not geom["occultation"]:
                continue

            phase = geom["phase_T14"]
            overlap = geom["overlap_factor"]

            amp = s["amp_ppm"] * overlap * 1e-6
            sigma_t = s["sigma_phase"] * T14
            t_center = phase * T14

            bump = amp * np.exp(-0.5 * ((dt - t_center) / sigma_t) ** 2)
            in_transit = flux_spotted < 0.9999
            flux_spotted[in_transit] += bump[in_transit]

    return flux_spotted



# ----------------------------------------------------------------------
# 6. Fitting routines
# ----------------------------------------------------------------------
def fit_global_transit(time, flux, system_params, fit_period=True):
    """
    Fit a global transit model to the entire light curve.

    We fit:
        rp_rs, a_rs, inc_deg, t0, P (optionally fixing P).
    Limb darkening is fixed to system_params["u1"], ["u2"].
    """
    u1 = system_params["u1"]
    u2 = system_params["u2"]

    rp0 = system_params["rp_rs"]
    a0 = system_params["a_rs"]
    inc0 = system_params["inc"]
    t00 = system_params["t0"]
    P0 = system_params["P"]

    if not fit_period:
        def model_fixed_P(t, rp_rs, a_rs, inc_deg, t0):
            return transit_flux(t, rp_rs, a_rs, inc_deg, t0, P0, u1, u2)

        p0 = [rp0, a0, inc0, t00]
        bounds = ([0.01, 1.0, 70.0, t00 - 0.1],
                  [0.5, 20.0, 90.0, t00 + 0.1])

        popt, pcov = curve_fit(model_fixed_P, time, flux, p0=p0, bounds=bounds)
        diag = np.sqrt(np.diag(pcov))
        popt_full = np.array([popt[0], popt[1], popt[2], popt[3], P0])
        perr_full = np.zeros_like(popt_full)
        perr_full[:4] = diag
        return popt_full, perr_full

    def model_free_P(t, rp_rs, a_rs, inc_deg, t0, period):
        return transit_flux(t, rp_rs, a_rs, inc_deg, t0, period, u1, u2)

    p0 = [rp0, a0, inc0, t00, P0]
    bounds = ([0.01, 1.0, 70.0, t00 - 0.1, P0 * 0.9],
              [0.5, 20.0, 90.0, t00 + 0.1, P0 * 1.1])

    popt, pcov = curve_fit(model_free_P, time, flux, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def fit_single_transit_tc(time, flux, system_params, P,
                          rp_rs, a_rs, inc, t0_guess):
    """
    Fit a single transit mid-time (T_c) with all other parameters fixed.
    """
    u1 = system_params["u1"]
    u2 = system_params["u2"]

    def local_model(t, t_c, offset):
        return transit_flux(t, rp_rs, a_rs, inc, t_c, P, u1, u2) + offset

    p0 = [t0_guess, 0.0]
    bounds = ([t0_guess - 0.05, -0.01],
              [t0_guess + 0.05, 0.01])

    try:
        popt, _ = curve_fit(local_model, time, flux, p0=p0, bounds=bounds)
        t_c_best = popt[0]
    except RuntimeError:
        t_c_best = t0_guess

    return t_c_best


def measure_ttv_series(time, flux, global_fit_params, system_params,
                       n_transits=10, t_window=0.15):
    """
    Measure TTVs by fitting mid-transit times for each individual transit.
    """
    t0_fit = global_fit_params["t0"]
    P_fit = global_fit_params["P"]
    rp_rs_fit = global_fit_params["rp_rs"]
    a_rs_fit = global_fit_params["a_rs"]
    inc_fit = global_fit_params["inc"]

    epochs = np.arange(n_transits, dtype=int)
    t_c_obs = np.zeros(n_transits)
    t_c_calc = np.zeros(n_transits)

    for i, n in enumerate(epochs):
        t_c_calc_n = t0_fit + n * P_fit
        t_c_calc[i] = t_c_calc_n

        mask = (time >= t_c_calc_n - t_window) & (time <= t_c_calc_n + t_window)
        t_seg = time[mask]
        f_seg = flux[mask]

        if len(t_seg) < 10:
            t_c_obs[i] = t_c_calc_n
            continue

        t_c_best = fit_single_transit_tc(
            t_seg,
            f_seg,
            system_params,
            P_fit,
            rp_rs_fit,
            a_rs_fit,
            inc_fit,
            t0_guess=t_c_calc_n,
        )
        t_c_obs[i] = t_c_best

    ttv = t_c_obs - t_c_calc
    return epochs, t_c_obs, t_c_calc, ttv


# ----------------------------------------------------------------------
# 7. 2D system view (star, planet chord, spots)
# ----------------------------------------------------------------------
def plot_starspot_system(system_params, spot_list,
                         Prot_days=None, epoch_index=0):
    """
    Create a 2D view of the stellar disk, transit chord, planet and spots.

    - Star: radius 1.
    - Planet: drawn at mid-transit with radius rp_rs.
    - Transit chord: horizontal line at y = b.
    - Spots: circles at projected (x,y). Spots that are occulted at the
      chosen epoch are drawn in red; others in grey.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    b = system_params["b"]
    rp_rs = system_params["rp_rs"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Star
    star = plt.Circle((0.0, 0.0), 1.0, color="#ffdd66", alpha=0.5, ec="k")
    ax.add_artist(star)

    # Transit chord
    L = np.sqrt(max(1.0 - b**2, 1e-6))
    x_track = np.linspace(-1.2, 1.2, 300)
    y_track = np.full_like(x_track, b)
    ax.plot(x_track, y_track, "k--", lw=1.5, label="Planet path")

    # Planet at mid-transit (centre of chord)
    planet = plt.Circle((0.0, b), rp_rs, color="k", alpha=0.8)
    ax.add_artist(planet)

    # Spots
    for i, s in enumerate(spot_list):
        geom = compute_spot_geometry_for_transit(
            s, system_params, epoch_index=epoch_index, Prot_days=Prot_days
        )
        x_spot = geom["x"]
        y_spot = geom["y"]
        r_spot = s["r_spot_rs"]
        occ = geom["occultation"]
        vis = geom["visible"]

        if not vis:
            # Behind the star – show faint grey outline
            color = "#999999"
            alpha = 0.3
        else:
            color = "#aa2222" if occ else "#444444"
            alpha = 0.95 if occ else 0.7

        circle = plt.Circle(
            (x_spot, y_spot),
            r_spot,
            color=color,
            alpha=alpha,
            ec="k",
            lw=0.6,
        )
        ax.add_artist(circle)
        ax.text(
            x_spot,
            y_spot + 0.06,
            f"S{i+1}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="k",
        )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("x / R★ (sky plane)")
    ax.set_ylabel("y / R★ (sky plane)")
    ax.set_title(
        f"Star, transit chord and starspots (epoch {epoch_index})\n"
        "Red = occulted, grey = visible but not occulted"
    )
    ax.legend(loc="upper right", fontsize=8)

    return fig


# ----------------------------------------------------------------------
# 8. TTV summary (for CLI use)
# ----------------------------------------------------------------------
def summarize_ttv(ttv_clean, ttv_noisy):
    """
    Print basic statistics of TTVs (RMS, max amplitude) for clean and noisy.
    """
    def stats(ttv_days):
        t_min = ttv_days * 24 * 60
        rms = np.sqrt(np.mean(t_min**2))
        max_amp = np.max(np.abs(t_min))
        return rms, max_amp

    rms_clean, max_clean = stats(ttv_clean)
    rms_noisy, max_noisy = stats(ttv_noisy)

    print("\n=== TTV statistics (minutes) ===")
    print(f"Clean: RMS = {rms_clean:.3f} min, max |TTV| = {max_clean:.3f} min")
    print(f"Noisy: RMS = {rms_noisy:.3f} min, max |TTV| = {max_noisy:.3f} min")


if __name__ == "__main__":
    # Minimal sanity test if you run this file directly
    params = get_default_hatp36_params()
    time = generate_time_array(params["P"], n_transits=5, cadence_sec=120.0)
    flux = simulate_clean_transit_series(time, params)
    print("Standalone test ran, N points =", len(time))
