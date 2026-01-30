from pathlib import Path
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Paths
# -----------------------
BASE = Path("artigos/vielas/insumos/favela/sao_remo/periodos/2020")
OUT = Path("artigos/vielas/figs")
OUT.mkdir(parents=True, exist_ok=True)

mds_path = BASE / "mds.tif"
vielas_path = BASE / "vielas.gpkg"

# -----------------------
# Load raster (MDS)
# -----------------------
with rasterio.open(mds_path) as src:
    mds = src.read(1, masked=True)
    extent = [
        src.bounds.left,
        src.bounds.right,
        src.bounds.bottom,
        src.bounds.top,
    ]

# Robust stretch
vmin, vmax = np.nanpercentile(mds, [2, 98])

# -----------------------
# Load vector (Vielas)
# -----------------------
vielas = gpd.read_file(vielas_path)

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(6, 6))

ax.imshow(
    mds,
    cmap="gray",
    extent=extent,
    vmin=vmin,
    vmax=vmax,
    origin="upper"
)

vielas.plot(
    ax=ax,
    color="#b30000",
    linewidth=0.7
)

ax.set_axis_off()

# -----------------------
# Save
# -----------------------
out_file = OUT / "fig_4_1_sao_remo_mds_vielas.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print(f"✔ Figura salva em {out_file}")
