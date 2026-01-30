from pathlib import Path
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("artigos/vielas/insumos/favela")
OUT = Path("artigos/vielas/figs/vvv_width")
OUT.mkdir(parents=True, exist_ok=True)

YEAR = "2020"

# --------------------------------------------------
# Visual parameters (global, fixed)
# --------------------------------------------------
FIG_SIZE = (6, 6)
LINE_WIDTH = 0.9
RASTER_GRAY = "#eeeeee"
COLORMAP = "viridis"
WIDTH_PERCENTILES = (5, 95)

# Scale bar parameters
SCALE_BAR_LENGTH_M = 50   # metros (ajustável)
SCALE_BAR_HEIGHT = 3      # espessura visual
SCALE_BAR_MARGIN = 0.05   # margem relativa da figura

# --------------------------------------------------
# Discover GLOBAL width scale
# --------------------------------------------------
all_widths = []

for favela_dir in BASE.iterdir():
    period_dir = favela_dir / "periodos" / YEAR
    vielas_path = period_dir / "vielas.gpkg"
    if vielas_path.exists():
        gdf = gpd.read_file(vielas_path)
        if "mean_width_m" in gdf.columns:
            all_widths.extend(gdf["mean_width_m"].dropna().values)

if len(all_widths) == 0:
    raise RuntimeError("Nenhum valor de mean_width_m encontrado.")

vmin, vmax = np.percentile(all_widths, WIDTH_PERCENTILES)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

print(f"Escala GLOBAL de largura (m): {vmin:.2f} – {vmax:.2f}")

# --------------------------------------------------
# Iterate over favelas
# --------------------------------------------------
for favela_dir in sorted(BASE.iterdir()):
    if not favela_dir.is_dir():
        continue

    period_dir = favela_dir / "periodos" / YEAR
    if not period_dir.exists():
        continue

    terrain_path = period_dir / "terrain_050.tif"
    vielas_path = period_dir / "vielas.gpkg"

    if not terrain_path.exists() or not vielas_path.exists():
        print(f"⚠️  Dados incompletos em {favela_dir.name}, pulando.")
        continue

    print(f"→ Gerando imagem B para {favela_dir.name}")

    # --------------------------------------------------
    # Load terrain raster
    # --------------------------------------------------
    with rasterio.open(terrain_path) as src:
        terrain = src.read(1).astype("float32")
        nodata = src.nodata

        xmin, ymin, xmax, ymax = src.bounds

    if nodata is not None:
        terrain[terrain == nodata] = np.nan

    # --------------------------------------------------
    # Load VVV network
    # --------------------------------------------------
    gdf = gpd.read_file(vielas_path).explode(ignore_index=True)

    if "mean_width_m" not in gdf.columns:
        print(f"⚠️  Campo mean_width_m ausente em {favela_dir.name}, pulando.")
        continue

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Terrain background (neutral)
    ax.imshow(
        terrain,
        cmap=colors.ListedColormap([RASTER_GRAY]),
        extent=[xmin, xmax, ymin, ymax],
        origin="upper",
        zorder=1
    )

    # VVV network colored by width
    gdf.plot(
        ax=ax,
        column="mean_width_m",
        cmap=COLORMAP,
        linewidth=LINE_WIDTH,
        norm=norm,
        zorder=2
    )

    ax.set_axis_off()

    # --------------------------------------------------
    # Colorbar
    # --------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("Mean width (m)", fontsize=8)

    # --------------------------------------------------
    # Scale bar (cartographic)
    # --------------------------------------------------
    width_map = xmax - xmin
    height_map = ymax - ymin

    x0 = xmin + width_map * SCALE_BAR_MARGIN
    y0 = ymin + height_map * SCALE_BAR_MARGIN

    x1 = x0 + SCALE_BAR_LENGTH_M

    # main bar
    ax.add_line(
        Line2D(
            [x0, x1],
            [y0, y0],
            linewidth=SCALE_BAR_HEIGHT,
            color="black",
            zorder=3
        )
    )

    # end ticks
    ax.add_line(Line2D([x0, x0], [y0, y0 + height_map * 0.01], color="black", linewidth=1))
    ax.add_line(Line2D([x1, x1], [y0, y0 + height_map * 0.01], color="black", linewidth=1))

    # label
    ax.text(
        (x0 + x1) / 2,
        y0 + height_map * 0.02,
        f"{SCALE_BAR_LENGTH_M} m",
        ha="center",
        va="bottom",
        fontsize=8
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_file = OUT / f"{favela_dir.name}_vvv_mean_width.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"✔ Salvo em {out_file}")

print("✅ Todas as imagens B geradas com escala gráfica.")
