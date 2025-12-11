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
        """
        Calcula o token .flaz e adiciona uma coluna flaz_colormap (uint8)
        baseada em uma variável contínua — por padrão o eixo Z normalizado.
        """
        if hasattr(self, "flaz") and not force_recalc:
            return self

        # 1) Carrega pontos brutos
        table = self._load_points()

        # 1.1) Constroi a máscara geometrica baseada na geometria da favela
        mask = self._build_geometry_mask(table, self.geometry)

        # 1.5) Corta a nuvem de pontos para ficar contida dentro da geometria da favela
        table = self._crop_points_to_geometry(table, mask)
        self.table = table  # atualiza tabela intermediária

        # 2) Normaliza coordenadas ANTES do Morton
        table = self._normalize_coordinates(table)
        self.table = table  # salva estado intermediário

        # 3) Computa o .flaz sobre coordenadas normalizadas
        self.flaz = compute_flaz(
            self,
            meta=getattr(self, "meta", {}),
        )

        # 4) Gera coluna flaz_colormap (uint8)
        # ------------------------------------------------------------
        # Por padrão usamos eixo Z normalizado para colorir os pontos.
        # Você pode trocar facilmente por 'hag', 'mds', etc.
        # ------------------------------------------------------------

        import numpy as np
        import pyarrow as pa

        # pega coluna z normalizada
        z = table["z"].to_numpy(zero_copy_only=False)

        # evita divisão por zero
        zmin = float(np.min(z))
        zmax = float(np.max(z))
        if zmax == zmin:
            cmap = np.zeros_like(z, dtype="uint8")
        else:
            cmap = ((z - zmin) / (zmax - zmin) * 255).astype("uint8")

        # adiciona coluna no Arrow
        table = table.append_column(
            "flaz_colormap",
            pa.array(cmap, type=pa.uint8())
        )

        self.table = table  # atualiza tabela final

        # 5) Registra layer caso exista o registry
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

        nome_json = f"{self.nome_normalizado()}.json"
        json_path = out_dir / nome_json

        bb = self.compute_bounding_box(table)

        metadata = {
            "id": self.nome_normalizado(),
            "nome": self.nome,
            "nome_secundario": self.fv_nome_sec if hasattr(self, "fv_nome_sec") else None,
            # "anos": self._ano, # aqui o ideal seria uma lista de anos
            "area_m2": self.geometry.area,
            "centroide": [
                float(self.geometry.centroid.x),
                float(self.geometry.centroid.y)
            ],
            "icone": "favela_nome.svg",
            "cor": "#FF5733",               # cor padrão (pode ser alterada)
            "data_geracao": datetime.now().strftime("%Y-%m-%d"),
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
        [0, 3407, 0, 3149, 0, 442]
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

        # campos uteis
        self.crs = gdf.crs
        self.fv_nome = row.get("fv_nome").iloc[0]
        self.fv_nome_sec = row.get("fv_nom_sec").iloc[0]   # ⭐ agora funciona!
        self.area = row.geometry.area               # se quiser

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
    
    def _build_geometry_mask(self, table, geometry, resolution=1024):
        """
        Constrói uma máscara rasterizada (numpy.ndarray bool)
        baseada na geometria da favela, no SRC ORIGINAL dos pontos.

        A resolução é adaptada ao bounding box da geometria.
        """

        import numpy as np
        import shapely

        if geometry is None:
            return None

        # Extrai bounding box da geometria (SRC original)
        minx, miny, maxx, maxy = geometry.bounds
        width = maxx - minx
        height = maxy - miny

        # Define resolução proporcional
        if width >= height:
            W = resolution
            H = max(1, int((height / width) * resolution))
        else:
            H = resolution
            W = max(1, int((width / height) * resolution))

        # Prepara grid raster (em coordenadas reais)
        xs = np.linspace(minx, maxx, W, endpoint=True)
        ys = np.linspace(miny, maxy, H, endpoint=True)

        # Cria meshgrid de coordenadas
        gx, gy = np.meshgrid(xs, ys[::-1])  
        coords = np.column_stack((gx.ravel(), gy.ravel()))

        # Teste vetorizado geometry.contains
        pts = shapely.points(coords[:, 0], coords[:, 1])
        inside = shapely.contains(geometry, pts)

        # Converte para raster booleano
        mask = inside.reshape((H, W))

        # Guarda metadados para uso no crop
        self._geom_mask = mask
        self._geom_mask_bounds = (minx, miny, maxx, maxy)
        self._geom_mask_res = (H, W)

        return mask
    
    def _crop_points_to_geometry(self, table, mask):
        """
        Usa a máscara rasterizada para recortar a nuvem de pontos.
        Nenhuma operação geométrica pesada acontece aqui.
        O crop vira apenas um lookup NumPy -> Arrow filter.
        """

        import numpy as np
        import pyarrow as pa

        # Se não houver máscara, retorna tabela original
        if mask is None:
            return table

        H, W = mask.shape
        minx, miny, maxx, maxy = self._geom_mask_bounds

        # Conversão coordenadas -> índice raster
        xs = table["x"].to_numpy(zero_copy_only=False)
        ys = table["y"].to_numpy(zero_copy_only=False)

        # Normaliza para [0, 1]
        tx = (xs - minx) / (maxx - minx + 1e-12)
        ty = (ys - miny) / (maxy - miny + 1e-12)

        # Converte para índices válidos
        xi = (tx * (W - 1)).astype(int)
        yi = ((1 - ty) * (H - 1)).astype(int)  # invertido por causa da linha superior

        # Mantém apenas pontos dentro dos limites
        valid = (
            (xi >= 0) & (xi < W) &
            (yi >= 0) & (yi < H)
        )

        # Inicializa máscara final de filtragem
        crop_mask = np.zeros(len(xs), dtype=bool)

        # Marca apenas os pontos que caíram dentro do polígono rasterizado
        crop_mask[valid] = mask[yi[valid], xi[valid]]

        # Se nenhum ponto passou, retorna tabela vazia
        if crop_mask.sum() == 0:
            return table.slice(0, 0)

        # Aplica filtro Arrow (zero-copy)
        crop_mask_arrow = pa.array(crop_mask)
        return table.filter(crop_mask_arrow)





