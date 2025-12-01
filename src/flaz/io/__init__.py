"""
Camada de I/O do ecossistema FLAZ.

Este módulo centraliza:
- acesso a sistemas de arquivos (local, S3/B2)
- leitura e escrita de objetos .flaz
- índices espaciais auxiliares (ex: articulação LiDAR)

Ele **não** contém lógica científica de cálculo.
Apenas resolve **onde estão os dados**.
"""

# from .flaz_io import FlazIO
from .lidar_index import LiDARIndex, build_default_lidar_index

# ---------------------------------------------------------------------
# Índice LiDAR padrão do ambiente
# ---------------------------------------------------------------------

LIDAR_INDEX = build_default_lidar_index()

__all__ = [
    "FlazIO",
    "LiDARIndex",
    "LIDAR_INDEX",
]
