from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_VVV = Path("artigos/vielas/insumos/por_favela")
MAPS = Path("artigos/vielas/insumos/mapas")
OUT = Path("artigos/vielas/tables")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load favela polygons (authoritative geometry)
# --------------------------------------------------
favelas_gpkg = MAPS / "favelas.gpkg"
gdf_favelas = gpd.read_file(favelas_gpkg)

required_fields = {"favela_id", "favela"}
missing = required_fields - set(gdf_favelas.columns)
if missing:
    raise RuntimeError(f"Campos ausentes em favelas.gpkg: {missing}")

gdf_favelas = gdf_favelas.to_crs(31983).set_index("favela_id")

rows = []

# --------------------------------------------------
# Iterate over favelas (directory = favela_id)
# --------------------------------------------------
for idx, favela_dir in enumerate(sorted(BASE_VVV.iterdir()), start=1):
    if not favela_dir.is_dir():
        continue

    favela_id = favela_dir.name

    if favela_id not in gdf_favelas.index:
        print(f"⚠️  favela_id '{favela_id}' não encontrado em favelas.gpkg.")
        continue

    vielas_path = favela_dir / "vielas.gpkg"
    if not vielas_path.exists():
        print(f"⚠️  vielas.gpkg ausente para '{favela_id}', pulando.")
        continue

    print(f"→ Processando VVV para favela_id '{favela_id}'")

    # --------------------------------------------------
    # Load favela geometry + name
    # --------------------------------------------------
    favela_row = gdf_favelas.loc[favela_id]
    area_m2 = float(favela_row.geometry.area)
    area_ha = area_m2 / 10_000
    favela_nome = favela_row["favela"]

    # --------------------------------------------------
    # Load VVV network
    # --------------------------------------------------
    gdf_vvv = gpd.read_file(vielas_path).explode(ignore_index=True)

    # Comprimento total da rede
    lengths_m = gdf_vvv.geometry.length
    total_length_m = float(lengths_m.sum())
    total_segments = int(len(gdf_vvv))

    # Estatísticas de largura
    if "mean_width_m" in gdf_vvv.columns:
        width_vals = gdf_vvv["mean_width_m"].dropna().values
        width_avg = float(np.mean(width_vals)) if len(width_vals) else np.nan
        width_std = float(np.std(width_vals)) if len(width_vals) else np.nan
    else:
        width_avg = np.nan
        width_std = np.nan

    # --------------------------------------------------
    # Row
    # --------------------------------------------------
    rows.append({
        "id": chr(64 + idx),
        "favela_id": favela_id,
        "favela": favela_nome,

        "vvv_length_m": round(total_length_m, 1),
        "vvv_segments": total_segments,

        "vvv_length_km_per_ha": round((total_length_m / 1000) / area_ha, 3),
        "vvv_segments_per_ha": round(total_segments / area_ha, 2),

        "mean_width_m_avg": round(width_avg, 2),
        "mean_width_m_std": round(width_std, 2),
    })

# --------------------------------------------------
# Save table
# --------------------------------------------------
if not rows:
    raise RuntimeError("Nenhuma favela processada na Tabela 2.")

df = pd.DataFrame(rows).sort_values(
    "vvv_length_km_per_ha", ascending=False
)

out_csv = OUT / "tabela_vvv_resumo_por_favela.csv"
df.to_csv(out_csv, index=False)

print("✅ Tabela 2 salva em:", out_csv)
print(df)
