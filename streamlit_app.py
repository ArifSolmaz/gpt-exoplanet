#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit app for HAT-P-36b TTV simulation and real-data analysis
==================================================================

Enhancements
------------
- One-column, larger plots for easy visual inspection.
- Toggles to show/hide:
    * clean / noisy full light curves
    * clean / noisy phase-folded curves
    * per-transit zoom panels
- Zoom window slider and y-axis padding slider.
- TESS-like plot styling.
- Full starspot control:
    * per-spot latitude, longitude, size (R_spot/R★), temperature
    * amplitude derived from geometry + temperature contrast
    * occultation determined from geometry; bumps only if intersected.
- 2D schematic of star, planet track and spots, with epoch selector.
- Upload mode for real data (time, flux, flux_err).

How to run
----------
1) Activate the environment where batman is installed (e.g. conda env).
2) Run:
       python -m streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import hatp36_ttv_sim as sim


# -------------------------------------------------------------------
# Helper: read uploaded data
# -------------------------------------------------------------------
def read_user_time_series(uploaded_files):
    frames = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, sep=None, engine="python", comment="#")
        except Exception:
            f.seek(0)
            df = pd.read_csv(f, delim_whitespace=True, comment="#", header=None)

        if df.shape[1] < 2:
            st.warning(f"File {f.name} has fewer than 2 columns; skipping.")
            continue

        if df.shape[1] == 2:
            df = df.iloc[:, :2]
            df.columns = ["time", "flux"]
            df["flux_err"] = np.nan
        else:
            df = df.iloc[:, :3]
            df.columns = ["time", "flux", "flux_err"]

        frames.append(df)

    if not frames:
        return None, None, None

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["time", "flux"])
    all_df = all_df.sort_values("time")

    return (all_df["time"].to_numpy(float),
            all_df["flux"].to_numpy(float),
            all_df["flux_err"].to_numpy(float))


# -------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------
def style_axes(ax):
    ax.grid(False)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def make_lightcurve_fig(
    time,
    flux,
    model=None,
    title="",
    y_padding_ppm=500.0,
    marker_label="data",
):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, flux, ".", ms=2, alpha=0.8, label=marker_label)
    if model is not None:
        ax.plot(time, model, "-", lw=1.5, label="model")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Flux")
    ax.set_title(title)
    style_axes(ax)

    # Y-limits based on min/max so full transit is visible
    y_min = np.nanmin(flux)
    y_max = np.nanmax(flux)
    pad = y_padding_ppm * 1e-6
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend()
    fig.tight_layout()
    return fig


def make_phase_fig(
    phase,
    flux,
    title="",
    y_padding_ppm=500.0,
    marker_label="data",
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(phase, flux, ".", ms=2, alpha=0.8, label=marker_label)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Flux")
    ax.set_title(title)
    style_axes(ax)
    y_min = np.nanmin(flux)
    y_max = np.nanmax(flux)
    pad = y_padding_ppm * 1e-6
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend()
    fig.tight_layout()
    return fig


def make_zoom_panels(time, flux, fit_params, system_params,
                     n_transits, zoom_half_window):
    """
    Per-transit zoom panels for a given light curve.
    """
    t0 = fit_params["t0"]
    P = fit_params["P"]

    n_cols = 3
    n_rows = int(np.ceil(n_transits / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.2 * n_rows),
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for i in range(n_transits):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        t_c = t0 + i * P
        mask = (time >= t_c - zoom_half_window) & (time <= t_c + zoom_half_window)
        t_seg = time[mask]
        f_seg = flux[mask]

        if len(t_seg) == 0:
            ax.axis("off")
            continue

        ax.plot(t_seg - t_c, f_seg, ".", ms=2)
        ax.set_title(f"Transit {i}")
        ax.set_xlabel("Time from mid-transit [days]")
        style_axes(ax)

    # Hide unused axes
    for j in range(n_transits, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()
    return fig


def make_ttv_fig(epochs, ttv_clean_days=None, ttv_noisy_days=None, title="TTV diagram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)

    if ttv_clean_days is not None:
        ax.plot(
            epochs,
            ttv_clean_days * 24 * 60,
            "o-",
            ms=6,
            label="Clean",
        )
    if ttv_noisy_days is not None:
        ax.plot(
            epochs,
            ttv_noisy_days * 24 * 60,
            "s-",
            ms=6,
            label="Noisy / observed",
        )
    ax.set_xlabel("Transit epoch")
    ax.set_ylabel("TTV [minutes]")
    ax.set_title(title)
    style_axes(ax)
    ax.legend()
    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="HAT-P-36b TTV Simulator", layout="wide")

    st.title("HAT-P-36b TTV Simulation & Real-data Analysis")

    true_params_default = sim.get_default_hatp36_params()

    # ------------------------------------------------------------------
    # Sidebar: dataset mode and basic transit params
    # ------------------------------------------------------------------
    st.sidebar.header("Dataset type")
    mode = st.sidebar.radio(
        "Choose dataset",
        ["Simulated HAT-P-36b", "Upload observational data"],
        key="mode_radio",
    )

    st.sidebar.header("Transit parameters (truth / initial guess)")
    P_user = st.sidebar.number_input(
        "Period P [days]",
        value=float(true_params_default["P"]),
        format="%.8f",
        key="P_user",
    )
    rp_rs_user = st.sidebar.number_input(
        "Rp/Rs",
        value=float(true_params_default["rp_rs"]),
        format="%.4f",
        key="rp_rs_user",
    )
    a_rs_user = st.sidebar.number_input(
        "a/Rs",
        value=float(true_params_default["a_rs"]),
        format="%.3f",
        key="a_rs_user",
    )
    inc_user = st.sidebar.number_input(
        "Inclination i [deg]",
        value=float(true_params_default["inc"]),
        min_value=70.0,
        max_value=90.0,
        format="%.3f",
        key="inc_user",
    )
    b_user = st.sidebar.number_input(
        "Impact parameter b",
        value=float(true_params_default["b"]),
        format="%.3f",
        key="b_user",
    )
    t0_user = st.sidebar.number_input(
        "Reference mid-transit T0 [days]",
        value=float(true_params_default["t0"]),
        format="%.5f",
        key="t0_user",
    )
    u1_user = st.sidebar.number_input(
        "Limb darkening u1 (TESS)",
        value=float(true_params_default["u1"]),
        format="%.3f",
        key="u1_user",
    )
    u2_user = st.sidebar.number_input(
        "Limb darkening u2 (TESS)",
        value=float(true_params_default["u2"]),
        format="%.3f",
        key="u2_user",
    )

    user_params = true_params_default.copy()
    user_params["P"] = P_user
    user_params["rp_rs"] = rp_rs_user
    user_params["a_rs"] = a_rs_user
    user_params["inc"] = inc_user
    user_params["b"] = b_user
    user_params["t0"] = t0_user
    user_params["u1"] = u1_user
    user_params["u2"] = u2_user

    # ------------------------------------------------------------------
    # Sidebar: simulation settings (only for simulated mode)
    # ------------------------------------------------------------------
    if mode == "Simulated HAT-P-36b":
        st.sidebar.header("Simulation setup")
        n_transits = st.sidebar.slider(
            "Number of transits",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
            key="n_transits_sim",
        )
        cadence_sec = st.sidebar.selectbox(
            "Cadence [seconds]",
            options=[20, 60, 120, 180],
            index=2,
            key="cadence_sim",
        )

        st.sidebar.subheader("Noise & systematics")
        use_white = st.sidebar.checkbox("Add white noise", value=True, key="white_on")
        white_ppm = st.sidebar.number_input(
            "White noise RMS [ppm]",
            value=300.0,
            min_value=0.0,
            step=50.0,
            key="white_ppm",
        )
        use_red = st.sidebar.checkbox("Add red noise (AR(1))", value=True, key="red_on")
        red_ppm = st.sidebar.number_input(
            "Red noise RMS [ppm]",
            value=300.0,
            min_value=0.0,
            step=50.0,
            key="red_ppm",
        )
        red_tau_hours = st.sidebar.number_input(
            "Red-noise correlation time [hours]",
            value=1.0,
            min_value=0.1,
            step=0.1,
            key="red_tau_hours",
        )

        st.sidebar.subheader("Stellar modulation")
        use_rot = st.sidebar.checkbox("Add rotational modulation", value=True, key="rot_on")
        rot_period_days = st.sidebar.number_input(
            "Rotation period [days]",
            value=15.0,
            min_value=0.1,
            step=0.5,
            key="rot_period_days",
        )
        rot_amp_ppm = st.sidebar.number_input(
            "Rotation amplitude [ppm]",
            value=1000.0,
            min_value=0.0,
            step=100.0,
            key="rot_amp_ppm",
        )

        st.sidebar.subheader("Instrumental systematics")
        use_sys = st.sidebar.checkbox("Add baseline trend + step", value=True, key="sys_on")
        trend_amp_ppm = st.sidebar.number_input(
            "Trend amplitude [ppm]",
            value=500.0,
            min_value=0.0,
            step=50.0,
            key="trend_amp_ppm",
        )
        step_amp_ppm = st.sidebar.number_input(
            "Step jump amplitude [ppm]",
            value=300.0,
            min_value=0.0,
            step=50.0,
            key="step_amp_ppm",
        )

        st.sidebar.subheader("Starspots (occulted)")
        use_spots = st.sidebar.checkbox("Add starspot bumps", value=True, key="spots_on")
        max_spots = st.sidebar.slider(
            "Number of spots",
            min_value=1,
            max_value=4,
            value=2,
            key="n_spots",
        )
        spot_amp_scale = st.sidebar.slider(
            "Global spot bump strength (scaling)",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            key="spot_amp_scale",
        )

        # Collect per-spot settings
        spot_dicts = []
        for i in range(max_spots):
            with st.sidebar.expander(f"Spot {i+1} settings", expanded=(i == 0)):
                lat_deg = st.number_input(
                    f"Spot {i+1} latitude [deg]",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(0.0),
                    step=5.0,
                    key=f"spot_lat_{i}",
                )
                lon_deg = st.number_input(
                    f"Spot {i+1} longitude [deg] (0 = central meridian)",
                    min_value=-180.0,
                    max_value=180.0,
                    value=float(-40.0 + 30.0 * i),
                    step=5.0,
                    key=f"spot_lon_{i}",
                )
                r_spot_rs = st.number_input(
                    f"Spot {i+1} radius R_spot/R★",
                    min_value=0.01,
                    max_value=0.5,
                    value=0.12,
                    step=0.01,
                    key=f"spot_r_{i}",
                )
                T_spot = st.number_input(
                    f"Spot {i+1} temperature [K]",
                    min_value=3000.0,
                    max_value=6000.0,
                    value=4800.0,
                    step=100.0,
                    key=f"spot_T_{i}",
                )
                spot_dicts.append(
                    dict(
                        lat_deg=lat_deg,
                        lon_deg=lon_deg,
                        r_spot_rs=r_spot_rs,
                        T_spot=T_spot,
                    )
                )

        # TTV fit settings
        st.sidebar.subheader("TTV fit settings")
        t_window_ttv = st.sidebar.number_input(
            "Per-transit fit half-window [days]",
            value=0.15,
            format="%.3f",
            key="t_window_ttv_sim",
        )

    else:
        # Upload mode
        st.sidebar.header("Real data / upload settings")
        n_transits = st.sidebar.slider(
            "Estimated number of transits",
            min_value=1,
            max_value=50,
            value=8,
            step=1,
            key="n_transits_upload",
        )
        t_window_ttv = st.sidebar.number_input(
            "Per-transit fit half-window [days]",
            value=0.15,
            format="%.3f",
            key="t_window_ttv_upload",
        )

    st.sidebar.markdown("---")
    st.sidebar.header("Plot controls")
    show_clean_full = st.sidebar.checkbox("Show clean full LC", value=True, key="show_clean_full")
    show_noisy_full = st.sidebar.checkbox("Show noisy full LC", value=True, key="show_noisy_full")
    show_clean_phase = st.sidebar.checkbox("Show clean phase-folded", value=True, key="show_clean_phase")
    show_noisy_phase = st.sidebar.checkbox("Show noisy phase-folded", value=True, key="show_noisy_phase")
    show_zoom = st.sidebar.checkbox("Show per-transit zoom panels", value=True, key="show_zoom")
    show_ttv = st.sidebar.checkbox("Show TTV diagram", value=True, key="show_ttv")
    show_2d = st.sidebar.checkbox("Show 2D system view (spots)", value=True, key="show_2d")

    y_pad_ppm = st.sidebar.slider(
        "Y-axis padding around min/max [ppm]",
        min_value=100.0,
        max_value=3000.0,
        value=800.0,
        step=100.0,
        key="y_pad_ppm",
    )
    zoom_half_window = st.sidebar.slider(
        "Zoom half-window per transit [days]",
        min_value=0.02,
        max_value=0.3,
        value=0.12,
        step=0.01,
        key="zoom_half_window",
    )

    # For the 2D system view: choose epoch (used only in simulated mode)
    epoch_for_2d = st.sidebar.slider(
        "Epoch index for 2D spot view",
        min_value=0,
        max_value=25,
        value=0,
        step=1,
        key="epoch_2d",
    )

    st.sidebar.markdown("---")
    run_button = st.sidebar.button("Run analysis", key="run_button")

    # IMPORTANT:
    # - In simulated mode: require Run button.
    # - In upload mode: run automatically when files exist.
    if mode == "Simulated HAT-P-36b" and not run_button:
        st.info("Adjust settings and click **Run analysis** in the sidebar.")
        return

    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # SIMULATED MODE
    # ------------------------------------------------------------------
    if mode == "Simulated HAT-P-36b":
        st.subheader("Simulation: HAT-P-36b")

        # Generate clean light curve
        with st.spinner("Generating clean transit series..."):
            time = sim.generate_time_array(
                user_params["P"],
                n_transits=n_transits,
                cadence_sec=cadence_sec,
                t0=user_params["t0"],
                t_window=0.2,
            )
            flux_clean = sim.simulate_clean_transit_series(time, user_params)
            time_clean = time.copy()

        # Build noisy light curve
        flux_noisy = flux_clean.copy()
        with st.spinner("Injecting noise, starspots and systematics..."):
            if use_white and white_ppm > 0:
                flux_noisy = sim.inject_white_noise(flux_noisy, white_ppm, rng=rng)
            if use_red and red_ppm > 0:
                flux_noisy = sim.inject_red_noise_ar1(
                    time, flux_noisy, red_ppm,
                    tau_days=red_tau_hours / 24.0, rng=rng
                )

            if use_spots:
                spot_list = sim.build_spot_list_from_ui(
                    spot_dicts, user_params, amp_scale=spot_amp_scale
                )
                Prot_for_spots = rot_period_days if use_rot else None
                flux_noisy = sim.inject_starspot_occultations(
                    time, flux_noisy, user_params, spot_list,
                    Prot_days=Prot_for_spots,
                )
            else:
                spot_list = []
                Prot_for_spots = None

            if use_rot and rot_amp_ppm > 0:
                flux_noisy = sim.inject_stellar_modulation(
                    time, flux_noisy, rot_amp_ppm,
                    Prot_days=rot_period_days,
                )
            if use_sys and (trend_amp_ppm > 0 or step_amp_ppm > 0):
                step_time = user_params["t0"] + 0.5 * n_transits * user_params["P"]
                flux_noisy = sim.inject_systematics(
                    time,
                    flux_noisy,
                    trend_amp_ppm=trend_amp_ppm,
                    step_amp_ppm=step_amp_ppm,
                    step_time=step_time,
                )

        time_noisy = time

        # Global fits
        st.subheader("Global transit fitting")

        with st.spinner("Fitting CLEAN light curve..."):
            popt_clean, _ = sim.fit_global_transit(time_clean, flux_clean, user_params)
        rp_c, a_c, inc_c, t0_c, P_c = popt_clean
        fit_clean = dict(rp_rs=rp_c, a_rs=a_c, inc=inc_c, t0=t0_c, P=P_c)

        with st.spinner("Fitting NOISY light curve..."):
            popt_noisy, _ = sim.fit_global_transit(time_noisy, flux_noisy, user_params)
        rp_n, a_n, inc_n, t0_n, P_n = popt_noisy
        fit_noisy = dict(rp_rs=rp_n, a_rs=a_n, inc=inc_n, t0=t0_n, P=P_n)

        model_clean = sim.transit_flux(
            time_clean, rp_c, a_c, inc_c, t0_c, P_c,
            user_params["u1"], user_params["u2"]
        )
        model_noisy = sim.transit_flux(
            time_noisy, rp_n, a_n, inc_n, t0_n, P_n,
            user_params["u1"], user_params["u2"]
        )

        # Parameter summary table
        st.markdown("#### Transit parameter comparison")
        def b_from(a_rs, inc_deg):
            return a_rs * np.cos(np.deg2rad(inc_deg))

        df_params = pd.DataFrame(
            {
                "Parameter": ["rp_rs", "a_rs", "inc_deg", "P_days", "t0_days", "b"],
                "True": [
                    user_params["rp_rs"],
                    user_params["a_rs"],
                    user_params["inc"],
                    user_params["P"],
                    user_params["t0"],
                    b_from(user_params["a_rs"], user_params["inc"]),
                ],
                "Fit (clean)": [
                    rp_c,
                    a_c,
                    inc_c,
                    P_c,
                    t0_c,
                    b_from(a_c, inc_c),
                ],
                "Fit (noisy)": [
                    rp_n,
                    a_n,
                    inc_n,
                    P_n,
                    t0_n,
                    b_from(a_n, inc_n),
                ],
            }
        )
        st.dataframe(df_params, use_container_width=True)

        # TTVs
        st.subheader("TTV analysis")
        with st.spinner("Measuring TTVs for CLEAN data..."):
            epochs, t_c_clean, t_calc_clean, ttv_clean = sim.measure_ttv_series(
                time_clean,
                flux_clean,
                fit_clean,
                user_params,
                n_transits=n_transits,
                t_window=t_window_ttv,
            )
        with st.spinner("Measuring TTVs for NOISY data..."):
            epochs2, t_c_noisy, t_calc_noisy, ttv_noisy = sim.measure_ttv_series(
                time_noisy,
                flux_noisy,
                fit_noisy,
                user_params,
                n_transits=n_transits,
                t_window=t_window_ttv,
            )

        # TTV statistics
        def ttv_stats(ttv_days):
            m = ttv_days * 24 * 60
            return float(np.sqrt(np.mean(m**2))), float(np.max(np.abs(m)))

        rms_c, max_c = ttv_stats(ttv_clean)
        rms_n, max_n = ttv_stats(ttv_noisy)
        st.write(f"**Clean TTVs**: RMS = {rms_c:.3f} min, max |TTV| = {max_c:.3f} min")
        st.write(f"**Noisy TTVs**: RMS = {rms_n:.3f} min, max |TTV| = {max_n:.3f} min")

        # Light curves & phase folding
        st.markdown("### Light curves & phase-folded views")

        phase_clean = ((time_clean - t0_c) / P_c) % 1.0
        phase_clean[phase_clean > 0.5] -= 1.0
        phase_noisy = ((time_noisy - t0_n) / P_n) % 1.0
        phase_noisy[phase_noisy > 0.5] -= 1.0

        if show_clean_full:
            fig = make_lightcurve_fig(
                time_clean, flux_clean, model_clean,
                title="Clean light curve",
                y_padding_ppm=y_pad_ppm,
                marker_label="Clean",
            )
            st.pyplot(fig, clear_figure=True)

        if show_noisy_full:
            fig = make_lightcurve_fig(
                time_noisy, flux_noisy, model_noisy,
                title="Noisy / observed light curve",
                y_padding_ppm=y_pad_ppm,
                marker_label="Noisy",
            )
            st.pyplot(fig, clear_figure=True)

        if show_clean_phase:
            fig = make_phase_fig(
                phase_clean,
                flux_clean,
                title="Clean (phase-folded)",
                y_padding_ppm=y_pad_ppm,
                marker_label="Clean",
            )
            st.pyplot(fig, clear_figure=True)

        if show_noisy_phase:
            fig = make_phase_fig(
                phase_noisy,
                flux_noisy,
                title="Noisy / observed (phase-folded)",
                y_padding_ppm=y_pad_ppm,
                marker_label="Noisy / observed",
            )
            st.pyplot(fig, clear_figure=True)

        if show_zoom:
            st.markdown("### Per-transit zoom panels (noisy light curve)")
            fig_zoom = make_zoom_panels(
                time_noisy,
                flux_noisy,
                fit_noisy,
                user_params,
                n_transits=n_transits,
                zoom_half_window=zoom_half_window,
            )
            st.pyplot(fig_zoom, clear_figure=True)

        if show_ttv:
            st.markdown("### TTV diagram")
            fig_ttv = make_ttv_fig(epochs, ttv_clean, ttv_noisy)
            st.pyplot(fig_ttv, clear_figure=True)

        if show_2d:
            st.markdown("### 2D view of star, planet path, and starspots")
            fig_sys = sim.plot_starspot_system(
                user_params,
                spot_list,
                Prot_days=Prot_for_spots,
                epoch_index=epoch_for_2d,
            )
            st.pyplot(fig_sys, clear_figure=True)

    # ------------------------------------------------------------------
    # UPLOAD MODE
    # ------------------------------------------------------------------
    else:
        st.subheader("Upload observational data (time, flux, flux_err)")
        uploaded_files = st.file_uploader(
            "Upload one or more files with columns: time, flux, flux_err",
            type=["txt", "dat", "csv"],
            accept_multiple_files=True,
            key="upload_files",
        )
        if not uploaded_files:
            st.warning("Please upload at least one file.")
            return

        with st.spinner("Reading uploaded files..."):
            time, flux, flux_err = read_user_time_series(uploaded_files)
        if time is None:
            st.error("No usable data found in uploaded files.")
            return

        st.write(
            f"Loaded {len(time)} data points from {len(uploaded_files)} file(s). "
            f"Time span: {time.min():.5f} – {time.max():.5f} days."
        )

        flux_noisy = flux
        time_noisy = time

        st.subheader("Global transit fit for uploaded data")
        with st.spinner("Fitting transit model..."):
            popt_obs, _ = sim.fit_global_transit(time_noisy, flux_noisy, user_params)
        rp_o, a_o, inc_o, t0_o, P_o = popt_obs
        fit_obs = dict(rp_rs=rp_o, a_rs=a_o, inc=inc_o, t0=t0_o, P=P_o)

        model_obs = sim.transit_flux(
            time_noisy, rp_o, a_o, inc_o, t0_o, P_o,
            user_params["u1"], user_params["u2"]
        )

        st.markdown("#### Transit parameter comparison (initial guess vs fit)")
        def b_from(a_rs, inc_deg):
            return a_rs * np.cos(np.deg2rad(inc_deg))

        df_params = pd.DataFrame(
            {
                "Parameter": ["rp_rs", "a_rs", "inc_deg", "P_days", "t0_days", "b"],
                "Initial guess": [
                    user_params["rp_rs"],
                    user_params["a_rs"],
                    user_params["inc"],
                    user_params["P"],
                    user_params["t0"],
                    b_from(user_params["a_rs"], user_params["inc"]),
                ],
                "Fit (uploaded data)": [
                    rp_o,
                    a_o,
                    inc_o,
                    P_o,
                    t0_o,
                    b_from(a_o, inc_o),
                ],
            }
        )
        st.dataframe(df_params, use_container_width=True)

        st.subheader("TTV analysis (uploaded data)")
        with st.spinner("Measuring TTVs..."):
            epochs, t_c_obs, t_calc, ttv_obs = sim.measure_ttv_series(
                time_noisy,
                flux_noisy,
                fit_obs,
                user_params,
                n_transits=n_transits,
                t_window=t_window_ttv,
            )

        def ttv_stats_single(ttv_days):
            m = ttv_days * 24 * 60
            return float(np.sqrt(np.mean(m**2))), float(np.max(np.abs(m)))

        rms_u, max_u = ttv_stats_single(ttv_obs)
        st.write(
            f"TTV RMS = {rms_u:.3f} min, max |TTV| = {max_u:.3f} min "
            f"(over {len(epochs)} epochs)"
        )

        # Light curves & TTV plots for uploaded data
        st.markdown("### Light curves")
        if show_noisy_full:
            fig = make_lightcurve_fig(
                time_noisy,
                flux_noisy,
                model_obs,
                title="Uploaded light curve",
                y_padding_ppm=y_pad_ppm,
                marker_label="Observed",
            )
            st.pyplot(fig, clear_figure=True)

        if show_noisy_phase:
            phase_obs = ((time_noisy - t0_o) / P_o) % 1.0
            phase_obs[phase_obs > 0.5] -= 1.0
            fig = make_phase_fig(
                phase_obs,
                flux_noisy,
                title="Uploaded (phase-folded)",
                y_padding_ppm=y_pad_ppm,
                marker_label="Observed",
            )
            st.pyplot(fig, clear_figure=True)

        if show_zoom:
            st.markdown("### Per-transit zoom panels (uploaded data)")
            fig_zoom = make_zoom_panels(
                time_noisy,
                flux_noisy,
                fit_obs,
                user_params,
                n_transits=n_transits,
                zoom_half_window=zoom_half_window,
            )
            st.pyplot(fig_zoom, clear_figure=True)

        if show_ttv:
            st.markdown("### TTV diagram (uploaded data)")
            fig_ttv = make_ttv_fig(epochs, ttv_clean_days=None, ttv_noisy_days=ttv_obs)
            st.pyplot(fig_ttv, clear_figure=True)


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()