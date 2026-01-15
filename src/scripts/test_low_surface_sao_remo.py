from pathlib import Path
import rasterio
import numpy as np

from flaz import Favela

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

API_PATH = Path("../fviz/apps/web/public/api").resolve()
ANO = 2017
FAVELA = "São Remo"

TERRAIN = "terrain_0125.tif"
LOW_SURFACE = "low_surface_0125.tif"



# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def inspect_raster(path: Path):
    print(f"\n📦 Raster: {path.name}")

    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        res = src.res

    valid = np.ones_like(data, dtype=bool)
    if nodata is not None:
        valid &= data != nodata

    print(f"  • shape        : {data.shape}")
    print(f"  • resolution   : {res}")
    print(f"  • nodata       : {nodata}")
    print(f"  • valid pixels : {valid.sum()} / {data.size}")

    if valid.any():
        print(f"  • min / max    : {data[valid].min()} / {data[valid].max()}")
        print(f"  • mean         : {data[valid].mean():.2f}")
    else:
        print("  ⚠️  nenhum pixel válido!")


# ------------------------------------------------------------
# TEST
# ------------------------------------------------------------

def main():
    print("🚀 Teste raster 12.5 cm — São Remo\n")

    # instancia favela
    f = (
        Favela(FAVELA)
        .set_api_path(API_PATH)
        .periodo(ANO)
    )

    # diretório de saída
    out_dir = f.periodo_dir()

    print(f"📁 Diretório de saída: {out_dir}")

    # força reconstrução da base LiDAR
    print("\n⚙️  Executando _build_favela_lidar_base(force=True)")
    f._build_favela_lidar_base(
        out_dir=out_dir,
        force=True,
    )

    # caminhos esperados
    terrain_path = out_dir / TERRAIN
    low_surface_path = out_dir / LOW_SURFACE

    # checagens básicas
    assert terrain_path.exists(), f"❌ {TERRAIN} não foi gerado"
    assert low_surface_path.exists(), f"❌ {LOW_SURFACE} não foi gerado"

    print("\n✅ Arquivos gerados com sucesso")

    # inspeção
    inspect_raster(terrain_path)
    inspect_raster(low_surface_path)

    print("\n🎉 Teste concluído com sucesso!")


if __name__ == "__main__":
    main()
