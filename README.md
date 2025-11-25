# HAT-P-36b TTV Simulator

Interactive Streamlit app to simulate and analyze transit timing variations (TTVs)
for the hot Jupiter **HAT-P-36b** as observed by TESS, including:

- Clean (noiseless) transit series with a batman transit model
- White & red noise injection
- Rotational stellar modulation
- Instrumental baseline trends & step-like jumps
- Geometric starspot occultations (latitude, longitude, size, temperature)
- 2D system view (star, transit chord, planet, spots)
- Per-transit mid-time fitting and TTV (O–C) diagrams
- Upload mode for real data (`time, flux, flux_err`)

## Installation

```bash
git clone https://github.com/ArifSolmaz/gpt-exoplanet.git
cd gpt-exoplanet

python -m venv .venv
source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt