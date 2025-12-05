"""
Índice espacial de articulação LiDAR por ano.

Este módulo define a classe LiDARIndex, responsável por:
- carregar índices espaciais de tiles LAZ/COPC por ano
- selecionar automaticamente os tiles que intersectam uma geometria
- fornecer os URIs dos arquivos LAZ a serem consumidos pelo pipeline

Ele NÃO conhece Favela, Flaz, cálculos ou cache de resultados científicos.
Apenas resolve: geometria → lista de tiles.
"""

from __future__ import annotations

from pathlib import Path
import os
import geopandas as gpd
from importlib.resources import files


class LiDARIndex:
    """
    Índice espacial de articulação LiDAR por ano.

    Parameters
    ----------
    index_by_year : dict[int, Path]
        Mapeamento {ano -> caminho do GeoPackage de articulação}.

    """

    def __init__(self, index_by_year: dict[int, Path]):
        self.index_by_year = index_by_year
        self._cache: dict[int, gpd.GeoDataFrame] = {}

    # def tiles_for_geom(self, geom, ano: int) -> list[str]:
    #     """
    #     Retorna a lista de URIs LAZ/COPC que intersectam a geometria.

    #     Parameters
    #     ----------
    #     geom : shapely geometry
    #         Geometria da favela ou da área de interesse.

    #     ano : int
    #         Ano da campanha LiDAR.

    #     Returns
    #     -------
    #     list[str]
    #         Lista de URIs (ou caminhos locais) dos arquivos LAZ.
    #     """
    #     gdf = self._load_index(ano)
    #     sel = gdf[gdf.intersects(geom)]

    #     # if "uri" not in sel.columns:
    #     #     raise KeyError("O índice LiDAR precisa possuir uma coluna 'uri'.")

    #     # return sel["uri"].tolist()

    #     return sel

    # def tiles_for_geom(self, geom, ano: int):
    #     gdf = self._load_index(ano)

    #     recortes = gdf[gdf.intersects(geom)]

    #     if recortes.empty:
    #         raise ValueError(f"Nenhuma quadrícula LiDAR encontrada para a geometria no ano {ano}")

    #     laz_files = [
    #         self._resolve_laz_name(row, ano)
    #         for _, row in recortes.iterrows()
    #     ]

    #     return laz_files

    def tiles_for_geom(self, geom, ano: int):
        gdf = self._load_index(ano)
        recortes = gdf[gdf.intersects(geom)]

        if recortes.empty:
            raise ValueError(f"Nenhuma quadrícula LiDAR encontrada para a geometria no ano {ano}")

        raw_dir = _raw_dir_for_year(ano)

        laz_files = []
        for _, row in recortes.iterrows():
            laz_name = self._resolve_laz_name(row, ano)
            laz_path = raw_dir / laz_name

            if not laz_path.exists():
                raise FileNotFoundError(f"Arquivo LAZ não encontrado: {laz_path}")

            laz_files.append(laz_path)

        return laz_files

    def _load_index(self, ano: int) -> gpd.GeoDataFrame:
        """
        Carrega o índice espacial do ano solicitado com cache interno.
        """
        if ano not in self.index_by_year:
            raise ValueError(f"Ano {ano} não disponível no índice LiDAR.")

        if ano not in self._cache:
            path = self.index_by_year[ano]
            if not path.exists():
                raise FileNotFoundError(f"Índice LiDAR não encontrado em: {path}")

            self._cache[ano] = gpd.read_file(path)

        return self._cache[ano]
    
    def _resolve_laz_name(self, row, ano: int) -> str:
        try:
            rule = LIDAR_FILENAME_RULES[ano]
        except KeyError:
            raise KeyError(f"Não há regra de nomeação de LAZ para o ano {ano}")

        base_value = row.get(rule.base_field)

        if base_value is None:
            raise KeyError(
                f"Campo base '{rule.base_field}' não encontrado no índice LiDAR "
                f"para o ano {ano}. Colunas: {list(row.index)}"
            )

        base_value = str(base_value).strip()

        return f"{rule.prefix}{base_value}{rule.suffix}"



# ---------------------------------------------------------------------
# Instanciação padrão (configurável via variável de ambiente)
# ---------------------------------------------------------------------


def build_default_lidar_index() -> "LiDARIndex":
    """
    Constrói o índice LiDAR padrão a partir dos dados empacotados em flaz.data.
    """

    data_dir = files("flaz.data")

    index_by_year = {
        2017: Path(data_dir / "articulacao_2017.zip"),
        2020: Path(data_dir / "articulacao_2020.zip"),
        2024: Path(data_dir / "articulacao_2024.gpkg"),
    }

    return LiDARIndex(index_by_year=index_by_year)


from dataclasses import dataclass

@dataclass(frozen=True)
class LidarFilenameRule:
    base_field: str
    prefix: str = ""
    suffix: str = ".laz"


LIDAR_FILENAME_RULES = {
    2017: LidarFilenameRule(
        base_field="cd_quadric",
        prefix="MDS_color_",
        suffix=".laz",
    ),
    2020: LidarFilenameRule(
        base_field="cd_quadric",
        prefix="MDS_",
        suffix="_1000.laz",
    ),
    2024: LidarFilenameRule(
        base_field="nome_arquivo",
        prefix="",
        suffix=".laz",
    ),
}

def _resolve_lidar_base_dir() -> Path:
    env = os.getenv("FLAZ_LIDAR_RAW_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # fallback apenas para desenvolvimento local
    fallback = Path.home() / "dev" / "ogdc"
    if fallback.exists():
        return fallback.resolve()

    raise RuntimeError(
        "Diretório base dos arquivos LiDAR não encontrado. "
        "Defina a variável de ambiente FLAZ_LIDAR_RAW_DIR."
    )

LIDAR_BASE_DIR = _resolve_lidar_base_dir()

LIDAR_RAW_SUBDIRS = {
    2017: "LiDAR-Sampa-2017",
    2020: "LiDAR-Sampa-2020",
    2024: "LiDAR-Sampa-2024",
}


def _raw_dir_for_year(ano: int) -> Path:
    try:
        sub = LIDAR_RAW_SUBDIRS[ano]
    except KeyError:
        raise KeyError(f"Não há diretório bruto configurado para o ano {ano}")

    path = LIDAR_BASE_DIR / sub

    if not path.exists():
        raise FileNotFoundError(f"Diretório LiDAR bruto não encontrado: {path}")

    return path
