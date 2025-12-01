# flaz/__init__.py
from .models.favela import Favela
from .models.favelas import Favelas
from .models.flaz_core import FLaz
from .flaz_io import FlazIO  # import relativo dentro do pacote

__all__ = [
    "Favela",
    "Favelas",
    "FLaz",
    "FlazIO",
]
