"""
Materialização mínima de um token FLaz a partir de uma nuvem de pontos.

Este módulo contém apenas a lógica de criação de um objeto FLaz a partir de
dados brutos. Não conhece Favela, cache, I/O ou camadas semânticas.
"""

from __future__ import annotations

import geopandas as gpd
from flaz.models.flaz_core import FLaz

def calc(
    cloud: gpd.GeoDataFrame,
    *,
    meta: dict | None = None,
) -> FLaz:
    """
    Cria um objeto FLaz mínimo a partir de uma nuvem de pontos.

    >>> from flaz import Favela
    >>> f = Favela("São Remo").periodo(2017)
    >>> points = f._load_points()
    >>> points.num_rows
    28142130

    Parameters
    ----------
    cloud : geopandas.GeoDataFrame
        Nuvem de pontos bruta (LAZ/COPC já convertida para GeoDataFrame).

    meta : dict, optional
        Metadados mínimos a serem associados ao token FLaz.

    Returns
    -------
    flaz.FLaz
        Instância mínima de FLaz materializada.

    """

    return FLaz(
        geom=cloud,
        meta=meta or {},
    )
