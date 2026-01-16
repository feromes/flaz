from pathlib import Path
from flaz import Favela

def main():
    # -------------------------------------------------
    # Configurações
    # -------------------------------------------------
    API_PATH = Path("flaz_tmp")
    ANO = 2017

    # -------------------------------------------------
    # Favela
    # -------------------------------------------------
    favela = (
        Favela("São Remo")
        .set_api_path(API_PATH)
        .periodo(ANO)
    )

    print(f"▶ Gerando vielas vetoriais — {favela.nome} ({ANO})")

    # -------------------------------------------------
    # Geração do GPKG
    # -------------------------------------------------
    gdf = favela.calc_vielas_vector(
        cellsize=0.5,          # alinhado ao terrain_050.tif
        via_threshold_m=2.5,   # limiar simples via × viela
        force=False,
    )

    out = favela.periodo_dir(ANO) / "vielas.gpkg"

    print("✔ Arquivo gerado com sucesso:")
    print(out.resolve())
    print(f"✔ Número de segmentos: {len(gdf)}")

    # estatísticas rápidas (sanity check)
    if len(gdf) > 0:
        print("— largura média (m):", round(gdf["mean_width_m"].mean(), 2))
        print("— largura mínima (m):", round(gdf["min_width_m"].min(), 2))
        print("— largura máxima (m):", round(gdf["max_width_m"].max(), 2))

if __name__ == "__main__":
    main()
