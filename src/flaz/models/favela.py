from pathlib import Path
import pyarrow as pa
from importlib.resources import files
import geopandas as gpd
from flaz.compute import compute_hag
from flaz.compute import compute_flaz
# from flaz.io.lidar_index import LiDARIndex, build_default_lidar_index
from flaz.io import LIDAR_INDEX
import laspy
import numpy as np


class Favela:
    """
    Representa uma favela específica, com funcionalidades para carregar dados
    e calcular atributos relevantes.

    >>> from flaz import Favela
    >>> f = Favela("São Remo")
    >>> f.geometry.area  # GeoDataFrame com a geometria da favela
    82629.24619930894
    """
    def __init__(self, nome: str):
        self.nome = nome
        self._ano = None
        self.geometry = self._load_geom_from_package()

    def calc_flaz(self, force_recalc: bool = False):

        if hasattr(self, "flaz") and not force_recalc:
            return self

        arrow_table = self._load_points()

        self.flaz = compute_flaz(
            arrow_table,
            meta=getattr(self, "meta", {}),
        )

        if hasattr(self, "_register_layer"):
            self._register_layer("flaz")

        return self

    def periodo(self, ano: int):
        self._ano = ano
        return self

    def calc_hag(self, force_recalc: bool = False):
        """
        Calcula a camada Height Above Ground (HAG) da favela.

        Após a execução, o resultado fica disponível em:
            self.hag
        """

        # 1. Cache
        if hasattr(self, "hag") and not force_recalc:
            return self

        # 2. Carrega insumos
        points = self.load_points()
        ground = self.load_ground()

        # 3. Cálculo puro
        self.hag = compute_hag(points, ground)

        # 4. Registro da camada (se existir)
        if hasattr(self, "_register_layer"):
            self._register_layer("hag")

        return self

    def calc_vielas(self, w_max=6.0):
        from ..features.vielas import calc_vielas
        return calc_vielas(self)

    def quadriculas_laz(self):

        """"
        Retorna as quadículas LAZ associadas à favela, dada uma articulação de arquivos LAZ/COPC/EPT.
        
        >>> from flaz import Favela
        >>> f = Favela("São Remo").periodo(2024)
        
        >>> len(f.quadriculas_laz())
        2

        >>> len(f.periodo(2017).quadriculas_laz())
        4

        """

        ano = self._ano
        return LIDAR_INDEX.tiles_for_geom(self.geometry, ano=ano)  # carrega o índice LiDAR do ano selecionado
    
    def _load_geom_from_package(self):
        """
        Carrega a geometria da favela a partir do GPKG embarcado no pacote.
        """

        path = files("flaz.data") / "SIRGAS_GPKG_favela.gpkg"

        gdf = gpd.read_file(path)

        # normalizar nome (opcional, mas altamente recomendado)
        gdf["fv_nome_norm"] = gdf["fv_nome"].str.strip().str.upper()
        nome_norm = self.nome.strip().upper()

        row = gdf[gdf["fv_nome_norm"] == nome_norm]

        if row.empty:
            raise ValueError(
                f"Favela '{self.nome}' não encontrada em "
                f"{path.name} (campo fv_nome)."
            )

        if len(row) > 1:
            raise ValueError(
                f"Mais de uma favela encontrada com nome '{self.nome}'."
            )

        # retorna apenas a geometria (shapely)
        geom = row.geometry.iloc[0]

        # também pode guardar crs, se quiser
        self.crs = gdf.crs

        return geom

    def _load_points(self):
        """
        Carrega os pontos LAZ da favela no período corrente
        e retorna um PyArrow Table com X, Y, Z e Classification.
        """

        paths = self.quadriculas_laz()  # já resolvidos com path absoluto

        xs, ys, zs, cls = [], [], [], []

        for path in paths:
            las = laspy.read(Path(path))

            # Conversão correta de ScaledArrayView -> numpy.ndarray
            xs.append(np.asarray(las.x, dtype="float64"))
            ys.append(np.asarray(las.y, dtype="float64"))
            zs.append(np.asarray(las.z, dtype="float64"))

            if "classification" in las.point_format.dimension_names:
                cls.append(np.asarray(las.classification, dtype="uint8"))
            else:
                cls.append(np.zeros(len(las.x), dtype="uint8"))

        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        Z = np.concatenate(zs)
        C = np.concatenate(cls)

        table = pa.Table.from_pydict(
            {
                "x": X,
                "y": Y,
                "z": Z,
                "classification": C,
            }
        )

        return table
