from pathlib import Path
from flaz import Favela

API_ROOT = Path("~/MacLab/fviz/apps/web/public/api").expanduser()


def main():
    print("🏗️  Gerando artefatos básicos de vielas — São Remo")

    f = (
        Favela("São Remo")
        .periodo(2017)
        .set_api_path(API_ROOT)
    )

    period_dir = f.periodo_dir(2017)
    print(f"📁 Diretório: {period_dir}")

    # -------------------------------------------------
    # 1. Executa SOMENTE a base LiDAR
    #    (terrain + walls)
    # -------------------------------------------------
    print("⚙️  Executando _build_favela_lidar_base()")

    result = f._build_favela_lidar_base(
        out_dir=period_dir,
        force=True,
    )

    # -------------------------------------------------
    # 2. Verificação dos artefatos
    # -------------------------------------------------
    terrain_path = period_dir / "terrain_025.tif"
    wall_path = period_dir / "wall_candidates_025.tif"

    if terrain_path.exists():
        print(f"✅ Terrain gerado: {terrain_path.name}")
    else:
        print("❌ Terrain NÃO gerado")

    if wall_path.exists():
        print(f"✅ Wall candidates gerado: {wall_path.name}")
    else:
        print("❌ Wall candidates NÃO gerado")

    print("🎯 Artefatos prontos para inspeção.")
    print("👉 Próximo passo: abrir no QGIS ou visualizar em script.")


if __name__ == "__main__":
    main()
