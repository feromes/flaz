import typer
from pathlib import Path
import warnings
import json

import geopandas as gpd

from flaz import Favela, Favelas

warnings.filterwarnings(
    "ignore",
    message="Measured \\(M\\) geometry types are not supported.*",
    category=UserWarning,
    module="pyogrio"
)

app = typer.Typer(pretty_exceptions_enable=False)


def resolve_api_path(api: str) -> Path:
    """
    Resolve o caminho da API como Path gravável.
    Aceita caminho relativo ou absoluto.
    """
    return Path(api).expanduser().resolve()


# ------------------------------------------------------------------------------
# HAG — uma favela
# ------------------------------------------------------------------------------

@app.command()
def calc_hag(
    favela: str = typer.Option(..., "--favela", "-f", help="Nome da favela."),
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    api: str = typer.Option(
        "./flaz_api",
        "--api",
        help="Diretório raiz onde a API FLAZ será gravada."
    ),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Calcula a camada HAG para uma única favela.
    """
    api_path = resolve_api_path(api)

    typer.echo(f"→ API path: {api_path}")

    f = Favela(favela)
    f.periodo(ano).calc_flaz()
    f.persist(api_path)

    card = f.to_card()

    card_path = api_path / "favelas.json"
    card_path.write_text(
        json.dumps(card, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    typer.echo("✔ Concluído!")


# ------------------------------------------------------------------------------
# PROCESSAMENTO COMPLETO — múltiplas favelas (FLAZ → FVIZ)
# ------------------------------------------------------------------------------

@app.command()
def calc_more(
    ano: int = typer.Option(..., "--ano", "-a", help="Ano do processamento."),
    api: str = typer.Option(
        "./flaz_api",
        "--api",
        help="Diretório raiz onde a API FLAZ será gravada."
    ),
    force: bool = typer.Option(False, "--force", help="Ignora cache."),
):
    """
    Processa todas as favelas:
    - base LiDAR
    - flaz
    - HAG
    - classification
    - via / viela / vazio
    - persistência completa para FVIZ
    """

    favelas = Favelas()
    api_path = resolve_api_path(api)

    typer.echo(f"Processando {len(favelas)} favelas")
    typer.echo(f"→ API path: {api_path}")

    cards = []

    for f in favelas:
        typer.echo(f"\n→ {f} ({ano})")

        # -----------------------------
        # Configuração básica
        # -----------------------------
        f.set_api_path(api_path)
        f.periodo(ano)

        # -----------------------------
        # Base LiDAR (COPC, MDT, MDS, terrain)
        # -----------------------------
        typer.echo("  • Base LiDAR")
        f._build_favela_lidar_base(
            out_dir=f.periodo_dir(),
            force=force
        )

        # -----------------------------
        # Núcleo FLAZ
        # -----------------------------
        typer.echo("  • calc_flaz")
        f.calc_flaz(force_recalc=force)

        typer.echo("  • calc_hag")
        f.calc_hag(force_recalc=force)

        typer.echo("  • calc_classification")
        f.calc_classification(force_recalc=force)

        # -----------------------------
        # NOVO — Via / Viela / Vazio
        # -----------------------------
        typer.echo("  • calc_via_viela_vazio")
        f.calc_via_viela_vazio(force_recalc=force)

        # -----------------------------
        # Persistência API FVIZ
        # -----------------------------
        typer.echo("  • persist")
        f.persist(api_path)

        # -----------------------------
        # Card
        # -----------------------------
        cards.append(f.to_card())

    # ------------------------------------------------------------------
    # Atualiza catálogo de favelas
    # ------------------------------------------------------------------
    catalog_path = api_path / "favelas.json"

    catalog_path.write_text(
        json.dumps(cards, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    typer.echo("\n✔ Concluído processamento de todas as favelas!")



# ------------------------------------------------------------------------------
# H3 — grid + cor + índice de busca
# ------------------------------------------------------------------------------

@app.command("calc-h3")
def calc_h3(
    gpkg_path: Path = typer.Argument(
        Path("data/geoportal_subprefeitura_v2.gpkg"),
        exists=True,
        readable=True,
        help="GPKG com limites administrativos (default: data/geoportal_subprefeitura_v2.gpkg)",
    ),
    resolution: int = typer.Option(8, help="Resolução H3"),
    buffer_m: float = typer.Option(1200, help="Buffer em metros"),
    out_dir: Path = typer.Option(
        Path("data/derived/h3"),
        help="Diretório de saída",
    ),
):
    """
    Calcula o grid H3 do território:
    - gera hexágonos
    - colore apenas os que contêm favelas
    - cria índice H3 → favelas
    """

    typer.echo("🔷 Calculando grid H3 via Favelas.to_h3()...")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣ Domínio
    # ------------------------------------------------------------------
    favelas = Favelas(all=True)

    # ------------------------------------------------------------------
    # 2️⃣ Grid H3 (com cor geodésica base)
    # ------------------------------------------------------------------
    gdf_h3 = favelas.to_h3(
        gpkg_path=gpkg_path,
        resolution=resolution,
        buffer_m=buffer_m,
        out_dir=out_dir,
    )

    typer.echo(f"✔ {len(gdf_h3)} hexágonos gerados")

    # ------------------------------------------------------------------
    # 3️⃣ GDF das favelas
    # ------------------------------------------------------------------
    gdf_favelas = favelas.to_gdf()
    typer.echo(f"✔ {len(gdf_favelas)} favelas carregadas")

    # ------------------------------------------------------------------
    # 4️⃣ Índice H3 → favelas
    # ------------------------------------------------------------------
    h3_index = favelas.build_h3_index(
        gdf_h3=gdf_h3,
        gdf_favelas=gdf_favelas,
    )

    active_h3 = set(h3_index.keys())
    typer.echo(f"✔ {len(active_h3)} hexágonos contêm favelas")

    # ------------------------------------------------------------------
    # 4.5️⃣ Materializar lista de favelas por hexágono
    # ------------------------------------------------------------------

    def favela_list(h3_id):
        return h3_index.get(h3_id, [])

    gdf_h3["favelas"] = gdf_h3["h3"].apply(favela_list)

    # ------------------------------------------------------------------
    # 5️⃣ Aplicar máscara de cor
    # ------------------------------------------------------------------
    def mask_color(row):
        if row["h3"] in active_h3:
            return row["color"]          # mantém cor geodésica
        return "#EDEDED"                  # neutro / escuro

    gdf_h3["color"] = gdf_h3.apply(mask_color, axis=1)

    # (opcional, se quiser flag explícita)
    gdf_h3["has_favela"] = gdf_h3["h3"].isin(active_h3)

    def serialize_hexes(gdf, resolution, buffer_m):
        return {
            "resolution": resolution,
            "buffer_m": buffer_m,
            "count": len(gdf),
            "hexes": [
                {
                    "h3": row.h3,
                    "color": row.color,
                    "center": [row.geometry.centroid.x, row.geometry.centroid.y],
                    "has_favela": bool(row.has_favela),
                    "favelas": row.favelas,
                }
                for row in gdf.itertuples()
            ]
        }

    # ------------------------------------------------------------------
    # 6️⃣ Persistência
    # ------------------------------------------------------------------
    parquet_path = out_dir / f"h3_r{resolution}_buf{int(buffer_m)}.parquet"
    geojson_path = out_dir / f"h3_r{resolution}_buf{int(buffer_m)}.geojson"
    hexjson_path = out_dir / f"h3_r{resolution}_buf{int(buffer_m)}.json"
    index_path = out_dir / "h3_favela_index.json"

    # formatos pesados (debug / QGIS)
    gdf_h3.to_parquet(parquet_path)
    gdf_h3.to_file(geojson_path, driver="GeoJSON")

    # formato leve (API / FVIZ)
    hex_payload = serialize_hexes(gdf_h3, resolution, buffer_m)

    hexjson_path.write_text(
        json.dumps(hex_payload, ensure_ascii=False),
        encoding="utf-8"
    )


    typer.echo(f"📦 Parquet salvo em: {parquet_path}")
    typer.echo(f"🗺️ GeoJSON salvo em: {geojson_path}")
    typer.echo(f"🧊 Hex JSON salvo em: {hexjson_path}")
    typer.echo(f"🔗 Índice H3→Favelas salvo em: {index_path}")

if __name__ == "__main__":
    app()
