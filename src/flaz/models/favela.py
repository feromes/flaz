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
from flaz.config import RESOLUCAO_MINIMA
import json
from pathlib import Path
from datetime import datetime
import flaz  # para obter a versão


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
        if hasattr(self, "flaz") and not force_recalc:
            return self

        # 1) Carrega pontos
        table = self._load_points()

        # 2) Normaliza coordenadas ANTES do Morton
        table = self._normalize_coordinates(table)
        self.table = table

        # 3) Computa FLaz sobre coordenadas já quantizadas
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
        >>> uri = f.calc_flaz().persist("temp://sao_remo_2017.arrow")

        Parameters
        ----------
        path : Path
            Caminho do arquivo .parquet a ser salvo.
        """

        io = FlazIO()
        final_path = io.write_favela(self, uri)
        self.write_metadata(self.table, final_path)

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
    
    def nome_normalizado(self):
        """
        Retorna o nome da favela normalizado (minúsculas, sem espaços extras, camelcase e sem acento).
        Útil para comparações.

        >>> from flaz import Favela
        >>> f = Favela("São Remo")
        >>> f.nome_normalizado()
        'sao_remo'
        """
        import unicodedata

        nome_norm = self.nome.strip().lower()
        nome_norm = unicodedata.normalize("NFKD", nome_norm).encode("ASCII", "ignore").decode("ASCII")
        nome_norm = "_".join(nome_norm.split())  # remove espaços extras
        # nome_norm = nome_norm.title()  # camelcase

        return nome_norm


    def write_metadata(self, table, arrow_path: str):
        """
        Escreve o metadata JSON da favela no mesmo diretório do arquivo .arrow.
        Estrutura: FLAZ-Metadata v1.0.
        """

        arrow_path = Path(arrow_path)
        out_dir = arrow_path.parent

        nome_json = f"{self.nome_normalizado()}_{self._ano}.json"
        json_path = out_dir / nome_json

        bb = self.compute_bounding_box(table)

        metadata = {
            "id": self.nome_normalizado(),
            "ano": self._ano,
            "bb_normalizado": bb,
            "resolucao": 12.5,             # constante atual do FLAZ
            "offset": [0, 0, 0],           # por enquanto fixo
            "src": "EPSG:31983",
            "point_count": table.num_rows,
            "atributos": [col for col in table.column_names],
            "versao_flaz": flaz.__version__,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return json_path

    def compute_bounding_box(self, table: pa.Table):
        """
        Calcula o bounding box normalizado da tabela Arrow.
        Retorna [xmin, xmax, ymin, ymax, zmin, zmax].

        >>> from flaz import Favela
        >>> f = Favela("São Remo").periodo(2017).calc_flaz()
        >>> bb = f.compute_bounding_box(f.table)
        >>> bb
        [0, 8651, 0, 9364, 0, 759]
        """

        xs = table["x"].to_numpy(zero_copy_only=False)
        ys = table["y"].to_numpy(zero_copy_only=False)
        zs = table["z"].to_numpy(zero_copy_only=False)

        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        zmin, zmax = int(zs.min()), int(zs.max())

        return [xmin, xmax, ymin, ymax, zmin, zmax]

    def _load_geom_from_package(self):
        """
        Carrega a geometria da favela a partir do GPKG embarcado no pacote.
        """

        path = files("flaz.data") / "SIRGAS_GPKG_favela.gpkg"

        gdf = gpd.read_file(path)

        gdf = gdf.dissolve(by="fv_nome").reset_index()

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

    def _normalize_coordinates(self, table):
        """
        Normaliza X,Y,Z para inteiros usando RESOLUCAO_MINIMA como step.
        """

        import pyarrow as pa
        import pyarrow.compute as pc

        # ---- extrai mínimos (float)
        xmin = pc.min(table["x"]).as_py()
        ymin = pc.min(table["y"]).as_py()
        zmin = pc.min(table["z"]).as_py()

        q = RESOLUCAO_MINIMA

        # ---- cria scalars Arrow dos valores mínimos
        xmin_sc = pa.scalar(xmin, pa.float64())
        ymin_sc = pa.scalar(ymin, pa.float64())
        zmin_sc = pa.scalar(zmin, pa.float64())
        q_sc = pa.scalar(q, pa.float64())

        # ---- operações todas via pyarrow.compute
        x_centered = pc.subtract(table["x"], xmin_sc)
        y_centered = pc.subtract(table["y"], ymin_sc)
        z_centered = pc.subtract(table["z"], zmin_sc)

        x_norm = pc.round(pc.divide(x_centered, q_sc)).cast(pa.int32())
        y_norm = pc.round(pc.divide(y_centered, q_sc)).cast(pa.int32())
        z_norm = pc.round(pc.divide(z_centered, q_sc)).cast(pa.int32())

        # ---- substituir colunas (Arrow é imutável → retorna nova tabela)
        table = table.set_column(
            table.schema.get_field_index("x"), "x", x_norm
        )
        table = table.set_column(
            table.schema.get_field_index("y"), "y", y_norm
        )
        table = table.set_column(
            table.schema.get_field_index("z"), "z", z_norm
        )

        # ---- registra metadados úteis para FVIZ
        self.meta = getattr(self, "meta", {})
        self.meta["offsets"] = {"xmin": xmin, "ymin": ymin, "zmin": zmin}
        self.meta["quantization"] = q

        return table
