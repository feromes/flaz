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
from flaz.io import FlazIO


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

    def __str__(self):
        return f"Favela(nome='{self.nome}')"
    
    @property
    def num_points(self) -> int:
        """
        Número de pontos da nuvem associada à favela.
        Só existe após o calc_flaz().
        """
        if not hasattr(self, "flaz") or self.flaz is None:
            return 0  # ontologicamente: ainda não há nuvem materializada
        return self.table.num_rows

    def calc_flaz(self, force_recalc: bool = False):
        """
        Materializa a nuvem de pontos sob o formato de um token FLaz mínimo da favela.
        Este é o ponto de entrada ontológico de todo o pipeline.
        Após a execução, o resultado fica disponível em:
            self.flaz

        >>> from flaz import Favela
        >>> f = Favela("São Remo").periodo(2017)
        >>> f.calc_flaz().num_points
        28142130

        """
        if hasattr(self, "flaz") and not force_recalc:
            return self

        self.table = self._load_points()

        self.flaz = compute_flaz(
            self,
            meta=getattr(self, "meta", {}),
        )

        if hasattr(self, "_register_layer"):
            self._register_layer("flaz")

        return self

    def persist(self, uri: str):
        """
        Persiste o objeto Favela atual em disco, incluindo todas as camadas
        calculadas (mds, hag, vielas, flaz, etc).

        >>> from flaz import Favela
        >>> f = Favela("São Remo").periodo(2017)
        >>> uri = f.calc_flaz().persist("temp://sao_remo_2017.parquet")

        Parameters
        ----------
        path : Path
            Caminho do arquivo .parquet a ser salvo.
        """

        io = FlazIO()
        final_path = io.write_favela(self, uri)
        return final_path

    def periodo(self, ano: int):
        self._ano = ano
        return self
    
    def calc_mds(self, resolution: float = 0.5, force_recalc: bool = False):
        """
        Calcula o Modelo Digital de Superfície (MDS) da favela.

        Após a execução, o resultado fica disponível em:
            self.mds
        """

        # 1. Cache
        if hasattr(self, "mds") and not force_recalc:
            return self

        # 2. Carrega insumos
        points = self.load_points()

        # 3. Cálculo puro
        self.mds = compute_mds(
            points,
            resolution=resolution,
            x_col="x",
            y_col="y",
            z_col="z",
        )

        # 4. Registro da camada (se existir)
        if hasattr(self, "_register_layer"):
            self._register_layer("mds")

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
