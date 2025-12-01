from pathlib import Path
from importlib.resources import files
import geopandas as gpd
from .favela import Favela
class Favelas:
    """
    Representa um conjunto de favelas, permitindo iteração e acesso a cada
    instância de :class:`Favela`.

    A classe de FAvelas representa inicialmente os dados baixados do GeoSampa com as
    geometrias das favelas da cidade de São Paulo.

    >>> from flaz import Favelas
    >>> favelas = Favelas(["São Remo", "Paraisópolis"])
    >>> len(favelas)
    2

    """

    FAVELAS_MORE = [
        "Heliópolis",
        "Cocaia I",
        "Paraisópolis",
        "Futuro Melhor",
        "São Remo"
    ]

    """
    >>> favelas = Favelas(favelas=Favelas.FAVELAS_MORE)
    >>> for favela in favelas:
    ...     print(favela.nome)
    Heliópolis
    Cocaia I
    Paraisópolis
    Futuro Melhor
    São Remo
    """

    def __init__(self, favelas=[], distritos=[], sub_prefeituras=[], db_path: Path = None):
        self.db_path = db_path or (files("flaz.data") / "SIRGAS_GPKG_favela.gpkg")
        gdf = gpd.read_file(self.db_path)

        gdf = gdf[gdf["fv_nome"].str.lower().isin([f.lower() for f in favelas])]

        self.favelas = []
        for _, row in gdf.iterrows():
            favela = Favela(nome=row["fv_nome"])
            self.favelas.append(favela)

    def __iter__(self):
        for favela in self.favelas:
            yield favela

    def __len__(self):
        return len(self.favelas)
    
    def __getitem__(self, index):
        return self.favelas[index]  
    
