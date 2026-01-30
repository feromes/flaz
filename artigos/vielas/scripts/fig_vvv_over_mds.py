from pathlib import Path
import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("artigos/vielas/insumos/favela")
OUT = Path("artigos/vielas/figs/vvv_mds")
OUT.mkdir(parents=True, exist_ok=True)

YEAR = "2020"

# --------------------------------------------------
# Global visual parameters (fixed for all favelas)
# --------------------------------------------------
LINE_COLOR = "#b30000"   # rede VVV
LINE_WIDTH = 0.6
FIG_SIZE = (6, 6)
PERCENTILES = (2, 98)

# Colormap com NoData explícito
cmap = plt.cm.gray
cmap.set_bad(color="white")

# --------------------------------------------------
# Iterate over favelas
# --------------------------------------------------
for favela_dir in sorted(BASE.iterdir()):
    if not favela_dir.is_dir():
        continue

    period_dir = favela_dir / "periodos" / YEAR
    if not period_dir.exists():
        continue

    mds_path = period_dir / "mds.tif"
    vielas_path = period_dir / "vielas.gpkg"

    if not mds_path.exists() or not vielas_path.exists():
        print(f"⚠️  Dados incompletos em {favela_dir.name}, pulando.")
        continue

    print(f"→ Gerando imagem para {favela_dir.name}")

    # --------------------------------------------------
    # Load raster (MDS) — CORRIGIDO
    # --------------------------------------------------
    with rasterio.open(mds_path) as src:
        mds = src.read(1).astype("float32")
        nodata = src.nodata

        extent = [
            src.bounds.left,
            src.bounds.right,
            src.bounds.bottom,
            src.bounds.top,
        ]

    # Converter NoData para NaN explicitamente
    if nodata is not None:
        mds[mds == nodata] = np.nan

    # Calcular contraste apenas sobre pixels válidos
    valid = np.isfinite(mds)
    if valid.sum() == 0:
        print(f"⚠️  Raster vazio em {favela_dir.name}, pulando.")
        continue

    vmin, vmax = np.percentile(mds[valid], PERCENTILES)

    # --------------------------------------------------
    # Load vector (VVV)
    # --------------------------------------------------
    gdf = gpd.read_file(vielas_path)
    gdf = gdf.explode(ignore_index=True)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # --- MDS (fundo)
    ax.imshow(
        mds,
        cmap=cmap,
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        origin="upper",
        zorder=1
    )

    # --- Rede VVV (overlay)
    gdf.plot(
        ax=ax,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=2
    )

    ax.set_axis_off()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_file = OUT / f"{favela_dir.name}_vvv_mds.png"
    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close()

    print(f"✔ Salvo em {out_file}")

print("✅ Todas as imagens A geradas com sucesso.")
