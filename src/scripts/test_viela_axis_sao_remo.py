from pathlib import Path
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt

from flaz import Favela


API_ROOT = Path("~/MacLab/fviz/apps/web/public/api").expanduser()


def main():
    print("🚀 Teste calc_viela_axis — São Remo")

    # -------------------------------------------------
    # 1. Carrega favela
    # -------------------------------------------------
    f = (
        Favela("São Remo")
        .periodo(2017)
        .set_api_path(API_ROOT)
    )

    period_dir = f.periodo_dir(2017)
    print(f"📁 Diretório: {period_dir}")

    terrain_path = period_dir / "terrain_025.tif"
    wall_path = period_dir / "wall_candidates_025.tif"

    assert terrain_path.exists(), "❌ terrain_025.tif não encontrado"
    assert wall_path.exists(), "❌ wall_candidates_025.tif não encontrado"

    # -------------------------------------------------
    # 2. Executa cálculo do eixo
    # -------------------------------------------------
    print("⚙️  Calculando eixo das vielas (skeleton)...")
    result = f.calc_viela_axis(
        cellsize=0.25,
        force=True,
    )

    axis = result["axis"]
    walkable = result["walkable"]
    transform = result["transform"]
    crs = result["crs"]

    print(f"✅ Pixels no eixo: {axis.sum()}")

    # -------------------------------------------------
    # 3. Visualização
    # -------------------------------------------------
    with rasterio.open(terrain_path) as src:
        terrain = src.read(1)

    # Campo de andabilidade
    plt.figure(figsize=(6, 6))
    plt.imshow(walkable, cmap="gray")
    plt.title("Campo de andabilidade")
    plt.axis("off")
    plt.show()

    # Skeleton puro
    plt.figure(figsize=(6, 6))
    plt.imshow(axis, cmap="hot")
    plt.title("Eixo das vielas (skeleton)")
    plt.axis("off")
    plt.show()

    # Overlay sobre o terreno
    fig, ax = plt.subplots(figsize=(8, 8))
    show(
        terrain,
        ax=ax,
        transform=transform,
        cmap="gray",
        title="Eixo das vielas sobre terreno",
    )

    ax.imshow(
        np.ma.masked_where(axis == 0, axis),
        cmap="autumn",
        alpha=0.9,
    )

    plt.show()

    # -------------------------------------------------
    # 4. Salva raster do eixo
    # -------------------------------------------------
    out_axis = period_dir / "viela_axis_025.tif"

    with rasterio.open(
        out_axis,
        "w",
        driver="GTiff",
        height=axis.shape[0],
        width=axis.shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(axis.astype("uint8"), 1)

    print(f"💾 Viela axis salvo em: {out_axis}")
    print("🎉 Teste concluído com sucesso.")


if __name__ == "__main__":
    main()
