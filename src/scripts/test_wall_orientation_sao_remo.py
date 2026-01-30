from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from flaz import Favela


API_ROOT = Path("~/MacLab/fviz/apps/web/public/api").expanduser()


def main():
    print("🧭 Teste campo direcional das paredes — São Remo")

    f = (
        Favela("São Remo")
        .periodo(2017)
        .set_api_path(API_ROOT)
    )

    period_dir = f.periodo_dir(2017)
    print(f"📁 Diretório: {period_dir}")

    # -------------------------------------------------
    # 🔑 PASSO 0 — garantir que a base LiDAR existe
    # -------------------------------------------------
    print("⚙️  Garantindo base LiDAR (wall candidates incluídos)...")

    f._build_favela_lidar_base(
        out_dir=period_dir,
        force=False  # coloque True se quiser recriar tudo
    )

    # -------------------------------------------------
    # Caminhos esperados
    # -------------------------------------------------
    wall_path = period_dir / "wall_candidates_025.tif"
    orient_path = period_dir / "wall_orientation_025.tif"
    coh_path = period_dir / "wall_coherence_025.tif"

    assert wall_path.exists(), "❌ wall_candidates_025.tif não encontrado"
    assert orient_path.exists(), "❌ wall_orientation_025.tif não encontrado"

    # -------------------------------------------------
    # 1. Leitura dos rasters
    # -------------------------------------------------
    with rasterio.open(wall_path) as src:
        walls = src.read(1).astype("float32")
        transform = src.transform

    with rasterio.open(orient_path) as src:
        orientation = src.read(1).astype("float32")

    coherence = None
    if coh_path.exists():
        with rasterio.open(coh_path) as src:
            coherence = src.read(1).astype("float32")

    # -------------------------------------------------
    # 2. Subamostragem para setas
    # -------------------------------------------------
    step = 10
    ys, xs = np.mgrid[0:walls.shape[0]:step, 0:walls.shape[1]:step]

    theta = orientation[ys, xs]

    u = np.cos(theta)
    v = np.sin(theta)

    if coherence is not None:
        alpha = coherence[ys, xs]
    else:
        alpha = np.ones_like(u)

    # -------------------------------------------------
    # 3. Plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 10))
    plt.imshow(walls, cmap="gray")
    plt.title("Campo direcional das paredes")
    plt.axis("off")

    plt.quiver(
        xs,
        ys,
        u,
        v,
        alpha=alpha,
        color="red",
        scale=30,
        width=0.003,
    )

    plt.show()

    print("✅ Visualização concluída.")


if __name__ == "__main__":
    main()
