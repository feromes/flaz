from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import laspy

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE = Path("artigos/vielas/insumos/favela")
MAPS = Path("artigos/vielas/insumos/mapas")
OUT = Path("artigos/vielas/tables")
OUT.mkdir(parents=True, exist_ok=True)

YEAR = "2020"

# --------------------------------------------------
# Load favela polygons (authoritative geometry)
# --------------------------------------------------
favelas_gpkg = MAPS / "favelas.gpkg"
gdf_favelas = gpd.read_file(favelas_gpkg)

# Required fields check
required_fields = {"favela_id", "favela"}
missing = required_fields - set(gdf_favelas.columns)
if missing:
    raise RuntimeError(f"Campos ausentes em favelas.gpkg: {missing}")

# Garantir CRS em metros
if gdf_favelas.crs is None:
    raise RuntimeError("favelas.gpkg não possui CRS definido.")

gdf_favelas = gdf_favelas.to_crs(31983)

# Index by favela_id for robust join
gdf_favelas = gdf_favelas.set_index("favela_id")

rows = []

# --------------------------------------------------
# Iterate over favelas (directory name = favela_id)
# --------------------------------------------------
for idx, favela_dir in enumerate(sorted(BASE.iterdir()), start=1):
    if not favela_dir.is_dir():
        continue

    favela_id = favela_dir.name

    if favela_id not in gdf_favelas.index:
        print(f"⚠️  favela_id '{favela_id}' não encontrado em favelas.gpkg.")
        continue

    period_dir = favela_dir / "periodos" / YEAR
    copc_path = period_dir / "favela.copc.laz"
    mds_path = period_dir / "mds.tif"

    if not copc_path.exists() or not mds_path.exists():
        print(f"⚠️  Dados incompletos para favela_id '{favela_id}', pulando.")
        continue

    print(f"→ Processando favela_id '{favela_id}'")

    # --------------------------------------------------
    # Área vetorial da favela (autoridade)
    # --------------------------------------------------
    favela_row = gdf_favelas.loc[favela_id]
    geom = favela_row.geometry
    favela_nome = favela_row["favela"]

    area_m2 = float(geom.area)
    area_ha = area_m2 / 10_000

    # --------------------------------------------------
    # Número absoluto de pontos LiDAR
    # --------------------------------------------------
    with laspy.open(copc_path) as f:
        lidar_points = f.header.point_count

    lidar_density = lidar_points / area_m2 if area_m2 > 0 else np.nan

    # --------------------------------------------------
    # Estatísticas altimétricas (MDS)
    # --------------------------------------------------
    with rasterio.open(mds_path) as src:
        mds = src.read(1).astype("float32")
        nodata = src.nodata

    if nodata is not None:
        mds[mds == nodata] = np.nan

    mds_min = float(np.nanmin(mds))
    mds_max = float(np.nanmax(mds))
    mds_range = mds_max - mds_min

    # --------------------------------------------------
    # Row
    # --------------------------------------------------
    rows.append({
        "id": chr(64 + idx),          # A, B, C...
        "favela_id": favela_id,       # chave técnica
        "favela": favela_nome,        # nome completo (legível)
        "area_m2": round(area_m2, 1),
        "area_ha": round(area_ha, 2),
        "lidar_points": int(lidar_points),
        "lidar_density_pt_m2": round(lidar_density, 2),
        "mds_min_m": round(mds_min, 2),
        "mds_max_m": round(mds_max, 2),
        "mds_range_m": round(mds_range, 2),
    })

# --------------------------------------------------
# Save table
# --------------------------------------------------
if not rows:
    raise RuntimeError("Nenhuma favela foi processada. Verifique favela_id e diretórios.")

df = pd.DataFrame(rows).sort_values("area_ha", ascending=False)

out_csv = OUT / "tabela_favelas_geom_lidar.csv"
df.to_csv(out_csv, index=False)

print("✅ Tabela salva em:", out_csv)
print(df)
