"""
Cálculo de Height Above Ground (HAG).

Este módulo contém apenas a lógica numérica pura do cálculo de HAG.
Não conhece Favela, Flaz, cache ou camadas semânticas.
Ele apenas recebe dados e devolve dados.

Uso típico:
    from flaz.compute.hag import calc as compute_hag
"""

from __future__ import annotations

import numpy as np
import geopandas as gpd


def calc(
    points: gpd.GeoDataFrame,
    ground: gpd.GeoDataFrame,
    *,
    z_col: str = "z",
    out_col: str = "hag",
) -> gpd.GeoDataFrame:
    """
    Calcula Height Above Ground (HAG) para uma nuvem de pontos.

    Parameters
    ----------
    points : geopandas.GeoDataFrame
        GeoDataFrame contendo os pontos 3D da nuvem.
        A coordenada Z deve estar disponível via `geometry.z`
        ou explicitamente em uma coluna informada em `z_col`.

    ground : geopandas.GeoDataFrame
        GeoDataFrame representando o solo (MDT/MDS filtrado).
        Neste esqueleto, assume-se que a cota do solo pode ser
        aproximada por uma estatística simples (ex: média).

    z_col : str, default="z"
        Nome da coluna que contém a cota Z, caso não esteja em geometry.z.

    out_col : str, default="hag"
        Nome da coluna onde o HAG será armazenado no resultado.

    Returns
    -------
    geopandas.GeoDataFrame
        Cópia de `points` com uma nova coluna `out_col` contendo
        a altura relativa ao solo (HAG).
    """

    # --- Extração da cota Z dos pontos -------------------------------
    if z_col in points.columns:
        z_points = points[z_col].to_numpy()
    else:
        # fallback: tenta pegar da geometria 3D
        z_points = np.array([geom.z for geom in points.geometry])

    # --- Extração da cota Z do solo ----------------------------------
    if z_col in ground.columns:
        z_ground = ground[z_col].to_numpy()
    else:
        z_ground = np.array([geom.z for geom in ground.geometry])

    # Aqui usamos uma aproximação simples (média do solo).
    # Em versões futuras isso pode virar:
    # - interpolação raster
    # - kNN local
    # - TIN / IDW / Kriging
    z0 = float(np.nanmean(z_ground))

    # --- Cálculo do HAG ----------------------------------------------
    hag = z_points - z0

    # --- Retorno -----------------------------------------------------
    result = points.copy()
    result[out_col] = hag

    return result
