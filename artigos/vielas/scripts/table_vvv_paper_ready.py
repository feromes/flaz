from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path("artigos/vielas")
TABLES_DIR = BASE_DIR / "tables"

INPUT_CSV = TABLES_DIR / "tabela_vvv_resumo_por_favela.csv"
OUTPUT_CSV = TABLES_DIR / "tabela_2_vvv_paper_ready.csv"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# ------------------------------------------------------------
# Build paper-ready table
# ------------------------------------------------------------
df_paper = pd.DataFrame({
    "ID": df["id"],
    "Settlement": df["favela"],
    "Total VVV length (km)": df["vvv_length_m"] / 1000.0,
    "VVV length density (km/ha)": df["vvv_length_km_per_ha"],
    "Number of segments": df["vvv_segments"],
    "Segment density (segments/ha)": df["vvv_segments_per_ha"],
    "Mean path width (m)": df["mean_width_m_avg"],
    "Path width variability (SD, m)": df["mean_width_m_std"],
})

# ------------------------------------------------------------
# Rounding (paper-friendly)
# ------------------------------------------------------------
df_paper["Total VVV length (km)"] = df_paper["Total VVV length (km)"].round(1)
df_paper["VVV length density (km/ha)"] = df_paper["VVV length density (km/ha)"].round(2)
df_paper["Segment density (segments/ha)"] = df_paper["Segment density (segments/ha)"].round(2)
df_paper["Mean path width (m)"] = df_paper["Mean path width (m)"].round(2)
df_paper["Path width variability (SD, m)"] = df_paper["Path width variability (SD, m)"].round(2)

# ------------------------------------------------------------
# Sorting: analytical order (most intense first)
# ------------------------------------------------------------
df_paper = df_paper.sort_values(
    by="VVV length density (km/ha)",
    ascending=False
).reset_index(drop=True)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df_paper.to_csv(OUTPUT_CSV, index=False)

print("✅ Tabela 2 (paper-ready) salva em:")
print(f"   {OUTPUT_CSV.resolve()}")
