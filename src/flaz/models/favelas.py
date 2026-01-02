from pathlib import Path
from importlib.resources import files
import geopandas as gpd
import pandas as pd
from .favela import Favela
import unicodedata
import json
import h3
from flaz.compute.calc_h3_grid import calc_h3_from_gpkg
from flaz.compute.geo_color import geo_color_from_point
from shapely.geometry import Polygon

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

        self.gdf = gdf

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
    
    def to_h3(
        self,
        gpkg_path,
        *,
        resolution=7,
        buffer_m=1200,
        out_dir=None
    ):
        # 1️⃣ calcula apenas os IDs H3
        h3_cells = calc_h3_from_gpkg(
            gpkg_path,
            resolution=resolution,
            buffer_m=buffer_m,
        )

        out_dir = Path(out_dir or self.base_dir / "derived" / "h3")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2️⃣ H3 → polígono (WGS84)
        def h3_to_polygon(h):
            boundary = h3.cell_to_boundary(h)  # [(lat, lon), ...]
            return Polygon([(lon, lat) for lat, lon in boundary])

        gdf = gpd.GeoDataFrame(
            {"h3": list(h3_cells)},
            geometry=[h3_to_polygon(h) for h in h3_cells],
            crs="EPSG:4326"
        )

        # 3️⃣ reprojeta para CRS métrico para calcular cor
        gdf_m = gdf.to_crs(31983)

        # 4️⃣ calcula a cor geodésica por centroide
        def hex_color(geom):
            if geom is None or geom.is_empty:
                return "#444444"
            cx, cy = geom.centroid.coords[0]
            return geo_color_from_point(cx, cy, mode="hex")

        gdf["color"] = gdf_m.geometry.apply(hex_color)

        # 5️⃣ (opcional) salvar
        # out_path = out_dir / f"h3_sp_r{resolution}_buf{buffer_m}.parquet"
        # gdf.to_parquet(out_path)

        return gdf
    
    def build_h3_index(
        self,
        gdf_h3: gpd.GeoDataFrame,
        gdf_favelas: gpd.GeoDataFrame,
    ) -> dict[str, list[str]]:
        """
        Constrói um índice H3 → lista de IDs de favelas que intersectam cada hexágono.

        Retorna apenas hexágonos que contêm ao menos uma favela.
        """

        # 1️⃣ Garantir CRS métrico comum
        if gdf_h3.crs is None or gdf_h3.crs.to_epsg() != 31983:
            gdf_h3 = gdf_h3.to_crs(31983)

        if gdf_favelas.crs is None or gdf_favelas.crs.to_epsg() != 31983:
            gdf_favelas = gdf_favelas.to_crs(31983)

        # 2️⃣ Spatial join: H3 ⨝ Favelas
        joined = gpd.sjoin(
            gdf_h3[["h3", "geometry"]],
            gdf_favelas[["fv_nome", "geometry"]],
            how="left",
            predicate="intersects",
        )

        # 3️⃣ Agregar IDs de favelas por hexágono
        h3_index = (
            joined
            .dropna(subset=["fv_nome"])
            .groupby("h3")["fv_nome"]
            .apply(lambda x: sorted(set(x)))
            .to_dict()
        )

        return h3_index
    
    def to_gdf(self) -> gpd.GeoDataFrame:
        """
        Retorna um GeoDataFrame com todas as favelas no conjunto.
        """
        return self.gdf