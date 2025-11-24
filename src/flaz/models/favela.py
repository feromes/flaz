from pathlib import Path
import pyarrow as pa
from importlib.resources import files
import geopandas as gpd

class Favela:
    def __init__(self, nome: str, db_path: Path = None):
        self.nome = nome
        self._ano = None
        self.db_path = db_path or (files("flaz.data") / "SIRGAS_GPKG_favela.gpkg")

        # Procura a favela no banco de dados pelo nome
        gdf = gpd.read_file(self.db_path)
        gdf_favela = gdf[gdf["fv_nome"].str.lower() == nome.lower()]
        if gdf_favela.empty:
            raise ValueError(f"Favela '{nome}' não encontrada no banco de dados.")
        self.geometry = gdf_favela.geometry.values[0]
        self.bounds = self.geometry.bounds  # (minx, miny, maxx, maxy)


    def periodo(self, ano: int):
        self._ano = ano
        return self

    def load_points(self):
        # futura: autodetecta se é LAZ, COPC ou FLÁZ
        pass

    def calc_hag(self):
        from ..features.hag import calc_hag
        return calc_hag(self)

    def calc_vielas(self, w_max=6.0):
        from ..features.vielas import calc_vielas
        return calc_vielas(self)
