"""
Cálculo de MDS — Modelo Digital de Superfície.

Recebe pontos em formato tabular (pyarrow ou numpy) e devolve
uma nuvem agregada representando a superfície superior.
"""

from __future__ import annotations
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


def calc(
    points: pa.Table,
    *,
    resolution: float = 0.5,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
) -> pa.Table:
    """
    Calcula o Modelo Digital de Superfície (MDS).

    Agrupa os pontos em uma malha regular e extrai o Z máximo por célula.

    Parameters
    ----------
    points : pyarrow.Table
        Tabela com colunas x, y, z.

    resolution : float
        Tamanho da célula do grid.

    Returns
    -------
    pyarrow.Table
        Tabela com colunas: x, y, z (z = máximo por célula).
    """

    x = points[x_col].to_numpy()
    y = points[y_col].to_numpy()
    z = points[z_col].to_numpy()

    # índices do grid
    ix = np.floor(x / resolution).astype(np.int64)
    iy = np.floor(y / resolution).astype(np.int64)

    # chave única por célula
    keys = ix.astype(str) + "_" + iy.astype(str)

    # criar estrutura de agregação
    result = {}

    for k, xx, yy, zz in zip(keys, ix, iy, z):
        if k not in result:
            result[k] = (xx, yy, zz)
        else:
            if zz > result[k][2]:
                result[k] = (xx, yy, zz)

    # reconstruir arrays
    xs = np.array([v[0] for v in result.values()]) * resolution
    ys = np.array([v[1] for v in result.values()]) * resolution
