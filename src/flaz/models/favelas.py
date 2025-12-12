from pathlib import Path
from importlib.resources import files
import geopandas as gpd
import pandas as pd
from .favela import Favela
import unicodedata
import json
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

    No caso de não especificar nenhuma favela, as 5 favelas Favelas.FAVELAS_MORE serão
    carregadas por padrão.

    >>> favelas = Favelas()
    >>> len(favelas)
    10
    >>> for favela in favelas:
    ...     print(favela.nome)
    Heliópolis
    Heliópolis - Viela Das Gaivotas
    Heliópolis L2 (Atílio Bartalini)
    Cocaia I
    Parque Cocaia II
    Parque Cocaia III
    Paraisópolis
    Futuro Melhor
    São Remo
    Abacateiro

    >>> favelas = Favelas()
    >>> json = favelas.to_json()

    """

    FAVELAS_MORE = [
        "Heliópolis",
        "Cocaia I",
        "Paraisópolis",
        "Futuro Melhor",
        "São Remo",
        "Abacateiro",
    ]

    def __init__(self, favelas=[], distritos=[], sub_prefeituras=[], db_path: Path = None):
        self.db_path = db_path or (files("flaz.data") / "SIRGAS_GPKG_favela.gpkg")
        gdf = gpd.read_file(self.db_path)
        gdf = gdf.dissolve(by="fv_nome").reset_index()

        # se nenhum nome de favela for fornecido, carrega as em Favelas.FAVELAS_MORE
        if not favelas and not distritos and not sub_prefeituras:
            favelas = Favelas.FAVELAS_MORE

        gdf = Favelas.filtrar(gdf, favelas)

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
    
    @staticmethod
    def _normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = s.encode("ascii", "ignore").decode()
        return s.lower().strip()

    @classmethod
    def filtrar(cls, gdf, nomes):
        """Retorna todas as favelas que contenham os nomes fornecidos,
        ignorando acentos, maiúsculas/minúsculas e variantes."""
        gdf = gdf.copy()
        gdf["__norm"] = gdf["fv_nome"].apply(cls._normalize)

        resultados = []

        for nome in nomes:
            query = cls._normalize(nome)
            subset = gdf[gdf["__norm"].str.contains(query)]

            if subset.empty:
                raise ValueError(f"Nenhuma favela encontrada contendo '{nome}'.")

            resultados.append(subset)

        return pd.concat(resultados).drop_duplicates()
    
    def to_cards(self) -> list[dict]:
        """
        Retorna a lista de cards de todas as favelas conhecidas.
        """
        cards = []
        for f in self.favelas:  # ou FAVELAS_MORE
            cards.append(f.to_card())
        return cards
    
    def to_json(self, **kwargs):
        """Retorna a representação JSON de todas as favelas no conjunto."""
        lista = []
        for favela in self.favelas:
            lista.append({
                "nome": favela.nome,
                "nome_secundario": favela.fv_nome_sec,
            })

        return json.dumps(lista, ensure_ascii=False, indent=2)
    

    
