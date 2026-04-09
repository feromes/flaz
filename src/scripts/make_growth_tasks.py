# scripts/make_growth_tasks.py

from pathlib import Path
from importlib.resources import files
import geopandas as gpd

YEARS = [2017, 2020, 2024]
OUT_PATH = Path("tasks.tsv")

def main():
    gpkg_path = files("flaz.data") / "SIRGAS_GPKG_favela.gpkg"

    gdf = gpd.read_file(gpkg_path)

    # mesma lógica da classe Favela:
    # dissolve por nome para tratar fragmentos da mesma favela como uma unidade
    gdf = gdf.dissolve(by="fv_nome").reset_index()

    # limpeza básica
    gdf["fv_nome"] = gdf["fv_nome"].astype(str).str.strip()
    nomes = sorted(n for n in gdf["fv_nome"].unique() if n)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for nome in nomes:
            for year in YEARS:
                f.write(f"{nome}\t{year}\n")

    print(f"[ok] {len(nomes)} favelas únicas")
    print(f"[ok] {len(nomes) * len(YEARS)} tarefas escritas em {OUT_PATH}")

if __name__ == "__main__":
    main()