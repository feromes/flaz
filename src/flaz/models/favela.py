from pathlib import Path
import pyarrow as pa
import pyarrow.ipc as ipc
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
from datetime import datetime
import flaz  # para obter a versão
from shapely.affinity import scale as scale_geom, translate
from shapely.geometry import Polygon, MultiPolygon
import math
import colorsys
from flaz.compute.geo_color import geo_color_from_point
import subprocess
import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter
from skimage.graph import route_through_array
from shapely.geometry import LineString, MultiLineString
import geopandas as gpd


SE = (333060.9, 7392952.6)  # coordenadas do SE (m)

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
        Só existe após calc_flaz().
        """
        if not hasattr(self, "table") or self.table is None:
            return 0
        return self.table.num_rows

    def icone(self, size: int = 200, fill: str | None = None) -> str:
        """
        Retorna um SVG (texto) da geometria da favela normalizada
        em size×size, mantendo proporção e centralizada.
        """
        geom = self.geometry
        if geom is None or geom.is_empty:
            return f"""<svg xmlns="http://www.w3.org/2000/svg"
                width="{size}" height="{size}" viewBox="0 0 {size} {size}"></svg>"""

        if fill is None:
            fill = self.color(mode="hex")

        # --- bounds ---
        minx, miny, maxx, maxy = geom.bounds
        w = maxx - minx
        h = maxy - miny

        # --- normaliza para origem ---
        g = translate(geom, xoff=-minx, yoff=-miny)

        # --- escala proporcional ---
        s = size / max(w, h)
        g = scale_geom(g, xfact=s, yfact=s, origin=(0, 0))

        # --- centraliza ---
        dx = (size - w * s) / 2
        dy = (size - h * s) / 2
        g = translate(g, xoff=dx, yoff=dy)

        # --- inverte eixo Y (GIS → SVG) ---
        g = scale_geom(g, xfact=1, yfact=-1, origin=(0, 0))

        # após inverter, a geometria fica com Y negativo,
        # então precisamos transladar para cima
        g = translate(g, yoff=size)

        # --- geometry → SVG path ---
        def ring(coords):
            pts = list(coords)
            d = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
            d += [f"L {x:.2f} {y:.2f}" for x, y in pts[1:]]
            d.append("Z")
            return " ".join(d)

        def poly(p: Polygon):
            d = ring(p.exterior.coords)
            for hole in p.interiors:
                d += " " + ring(hole.coords)
            return d

        if isinstance(g, Polygon):
            d = poly(g)
        elif isinstance(g, MultiPolygon):
            d = " ".join(poly(p) for p in g.geoms)
        else:
            d = poly(g.convex_hull)

        return f"""\
    <svg xmlns="http://www.w3.org/2000/svg"
        width="{size}" height="{size}"
        viewBox="0 0 {size} {size}">
    <path d="{d}" fill="{fill}" fill-rule="evenodd"/>
    </svg>
    """

    def set_api_path(self, api_path: Path | str):
        """
        Define o diretório raiz da API FLAZ para esta instância.
        Deve ser chamado pelo orquestrador (CLI / API).
        """
        self.api_path = Path(api_path).expanduser().resolve()
        return self
    
    def favela_dir(self) -> Path:
        """
        Diretório base da favela dentro da API.
        Ex: {api_path}/favela/sao_remo
        """
        if not hasattr(self, "api_path"):
            raise RuntimeError(
                "api_path não definido. Use f.set_api_path(api_path) antes."
            )

        return self.api_path / "favela" / self.nome_normalizado()
    
    def periodo_dir(self, ano: int | None = None) -> Path:
        """
        Diretório do período da favela.
        Ex: {api_path}/favela/sao_remo/periodos/2017
        """
        ano = ano if ano is not None else self._ano

        if ano is None:
            raise ValueError("Ano não definido. Use .periodo(ano).")

        return self.favela_dir() / "periodos" / str(ano)

    def color(self, mode: str = "hex") -> str:
        geom = getattr(self, "geometry", None)
        if geom is None or geom.is_empty:
            return "#444444"

        cx, cy = geom.centroid.coords[0]

        return geo_color_from_point(
            cx,
            cy,
            mode=mode
        )

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

        # 🔹 estatísticas físicas (antes da normalização)
        self.elevation = self._compute_elevation_stats(table)

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

    def persist(self, root: Path | str) -> str:
        """
        Persiste a favela em um diretório local,
        seguindo a estrutura oficial da API FLAZ (v0).
        """
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)

        base = root / "favela" / self.nome_normalizado()

        # --- meta ---
        meta_dir = base / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "nome": self.nome,
            "nome_normalizado": self.nome_normalizado(),
            "entidade": "Favela"
        }

        (meta_dir / "favela.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # --- periodo ---
        if self._ano is not None:
            per_dir = base / "periodos" / str(self._ano)
            per_dir.mkdir(parents=True, exist_ok=True)

            # flaz
            if hasattr(self, "table"):
                arrow_path = per_dir / "flaz.arrow"
                self._write_arrow(arrow_path)

            # mdt
            if (per_dir / "mdt.tif").exists():
                self._persist_mdt_png_from_tif(per_dir)

            # mds
            if (per_dir / "mds.tif").exists():
                self._persist_mds_png_from_tif(per_dir)

            # HAG
            if hasattr(self, "hag_table"):
                hag_path = per_dir / "hag_flaz.arrow"
                self._write_hag_arrow(hag_path)

            # Classification
            if hasattr(self, "class_table"):
                class_path = per_dir / "class_flaz.arrow"
                self._write_class_arrow(class_path)

            # Via / Viela / Vazio
            if hasattr(self, "vvv_table"):
                vvv_path = per_dir / "via_viela_vazio_flaz.arrow"
                self._write_vvv_arrow(vvv_path)

            # Delta
            if hasattr(self, "delta_table"):
                delta_path = per_dir / "delta_flaz.arrow"
                with pa.OSFile(delta_path.as_posix(), "wb") as f:
                    with ipc.RecordBatchFileWriter(f, self.delta_table.schema) as writer:
                        writer.write_table(self.delta_table)

            # (futuro)
            # if hasattr(self, "mds"):
            #     self._persist_mds(per_dir)

            # if hasattr(self, "vielas"):
            #     self._persist_vielas(per_dir)

        # --- ícone da favela ---
        svg = self.icone()
        icon_path = base / f"{self.nome_normalizado()}.svg"
        icon_path.write_text(svg, encoding="utf-8")

        return root.as_posix()

    def periodo(self, ano: int):
        self._ano = ano
        return self

    def to_card(self) -> dict:

        bb = self.compute_bounding_box(self.table)
        geom = getattr(self, "geometry", None)

        if geom is not None and not geom.is_empty:
            minx, miny, maxx, maxy = geom.bounds
            cx, cy = geom.centroid.coords[0]

            bbox = [minx, miny, maxx, maxy]
            centroid = [cx, cy]

            # distância até a Sé (em metros)
            dx = cx - SE[0]
            dy = cy - SE[1]
            dist_se_m = math.sqrt(dx * dx + dy * dy)

        else:
            bbox = None
            centroid = None
            dist_se_m = None

        return {
            "id": self.nome_normalizado(),
            "nome": self.nome,
            "nome_secundario": self.fv_nome_sec if hasattr(self, "fv_nome_sec") else None,
            "entidade": "Favela",

            "icon": f"favela/{self.nome_normalizado()}/{self.nome_normalizado()}.svg",
            "color": self.color(),

            "bbox": bbox,
            "centroid": centroid,

            # 🔹 atributos escalares
            "hag": getattr(self, "hag_stats", None),
            "delta": getattr(self, "delta_stats", None),   # ✅ AQUI

            "dist_se_m": dist_se_m,
            "area_m2": self.geometry.area,

            "periodos": self.periodos if hasattr(self, "periodos") else (
                [self._ano] if self._ano is not None else []
            ),

            "data_geracao": datetime.now().strftime("%Y-%m-%d"),
            "bb_normalizado": bb,
            "resolucao": 12.5,
            "offset": [0, 0, 0],
            "src": "EPSG:31983",
            "point_count": self.table.num_rows,
            "elevation": getattr(self, "elevation", None),
            "versao_flaz": flaz.__version__,
        }

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

    def calc_viela_axis(
        self,
        cellsize: float = 0.50,
        closing_radius_m: float = 0.5,
        smooth_sigma_m: float | None = None,
        force: bool = False,
    ):
        """
        Extração do eixo das vielas a partir do raster de terreno,
        preservando gargalos e conexões mínimas.

        O distance transform é usado apenas como atributo,
        nunca como filtro de exclusão.
        """

        import numpy as np
        import rasterio
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        from skimage.morphology import closing, disk, skeletonize

        period_dir = self.periodo_dir(self._ano)
        terrain_path = period_dir / "terrain_050.tif"

        # 1. Leitura
        with rasterio.open(terrain_path) as src:
            terrain = src.read(1).astype("float32")
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

        # 2. Máscara válida
        valid = np.ones_like(terrain, dtype=bool)
        if nodata is not None:
            valid &= terrain != nodata

        # 3. Campo caminhável
        walkable = np.zeros_like(terrain, dtype=bool)
        walkable[valid & (terrain > 0)] = True

        # 4. Densificação morfológica
        closing_radius_px = max(1, int(round(closing_radius_m / cellsize)))
        walkable_dense = closing(walkable, disk(closing_radius_px))

        # 5. Suavização opcional
        if smooth_sigma_m is not None:
            sigma_px = smooth_sigma_m / cellsize
            smooth = gaussian_filter(walkable_dense.astype("float32"), sigma=sigma_px)
            walkable_dense = smooth > 0.5

        # 6. Distance transform (atributo, NÃO filtro)
        dist = distance_transform_edt(walkable_dense)

        # 7. Skeleton
        axis = skeletonize(walkable_dense).astype("uint8")

        return {
            "walkable_raw": walkable.astype("uint8"),
            "walkable_dense": walkable_dense.astype("uint8"),
            "distance": dist.astype("float32"),
            "axis": axis,
            "transform": transform,
            "crs": crs,
            "params": {
                "cellsize_m": cellsize,
                "closing_radius_m": closing_radius_m,
                "smooth_sigma_m": smooth_sigma_m,
            },
        }

    def calc_vielas_vector(
        self,
        cellsize: float = 0.5,
        via_threshold_m: float = 10000,
        force: bool = False,
    ):
        """
        Converte o skeleton raster das vielas em LineStrings vetoriais
        garantindo que TODO o skeleton seja vetorizado,
        incluindo ciclos e vias contínuas.
        """

        import numpy as np
        import geopandas as gpd
        from shapely.geometry import LineString
        from rasterio.transform import xy
        import networkx as nx

        # -------------------------------------------------
        # 1. Skeleton
        # -------------------------------------------------
        res = self.calc_viela_axis(cellsize=cellsize, force=force)

        axis = res["axis"].astype(bool)
        dist = res["distance"]
        transform = res["transform"]
        crs = res["crs"]

        H, W = axis.shape

        # -------------------------------------------------
        # 2. Grafo 8-conectado
        # -------------------------------------------------
        G = nx.Graph()

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1),
        ]

        for y in range(H):
            for x in range(W):
                if not axis[y, x]:
                    continue

                G.add_node((y, x))

                for dy, dx in neighbors:
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx_ < W:
                        if axis[ny, nx_]:
                            G.add_edge((y, x), (ny, nx_))

        degree = dict(G.degree())
        visited = set()
        lines = []

        # -------------------------------------------------
        # 3. Função de caminhada genérica
        # -------------------------------------------------
        def walk_path(start):
            path = [start]
            visited.add(start)

            prev = None
            curr = start

            while True:
                nbrs = [n for n in G.neighbors(curr) if n != prev]
                if not nbrs:
                    break

                # segue o primeiro não visitado
                nxt = None
                for n in nbrs:
                    if n not in visited:
                        nxt = n
                        break

                if nxt is None:
                    break

                path.append(nxt)
                visited.add(nxt)
                prev, curr = curr, nxt

            return path

        # -------------------------------------------------
        # 4. Passo A — caminhos iniciando em terminais
        # -------------------------------------------------
        terminals = [n for n, d in degree.items() if d != 2]

        for node in terminals:
            if node in visited:
                continue
            path = walk_path(node)
            if len(path) >= 2:
                lines.append(path)

        # -------------------------------------------------
        # 5. Passo B — cobre ciclos e vias contínuas
        # -------------------------------------------------
        for node in G.nodes():
            if node in visited:
                continue
            path = walk_path(node)
            if len(path) >= 2:
                lines.append(path)

        # -------------------------------------------------
        # 6. Converte em GeoDataFrame
        # -------------------------------------------------
        records = []

        for path in lines:
            coords = [xy(transform, y, x) for y, x in path]

            line = LineString(coords)

            widths = [2 * dist[y, x] * cellsize for y, x in path]

            mean_w = float(np.mean(widths))
            min_w = float(np.min(widths))
            max_w = float(np.max(widths))

            kind = "via" if mean_w >= via_threshold_m else "viela"

            records.append(
                {
                    "geometry": line,
                    "length_m": float(line.length),
                    "mean_width_m": mean_w,
                    "min_width_m": min_w,
                    "max_width_m": max_w,
                    "n_pixels": len(path),
                    "kind": kind,
                }
            )

        gdf = gpd.GeoDataFrame(records, crs=crs)

        # -------------------------------------------------
        # 7. Persistência
        # -------------------------------------------------
        out = self.periodo_dir(self._ano) / "vielas.gpkg"
        gdf.to_file(out, layer="vielas", driver="GPKG")

        self.vielas = gdf
        return gdf

    def calc_wall_orientation(
        wall_raster_path,
        out_orientation_path,
        out_coherence_path=None,
    ):
        """
        Calcula campo de orientação dominante das paredes
        a partir de um raster de candidatos a parede.
        """

        import numpy as np
        import rasterio
        from scipy.ndimage import sobel, gaussian_filter

        with rasterio.open(wall_raster_path) as src:
            walls = src.read(1).astype("float32")
            transform = src.transform
            crs = src.crs
            profile = src.profile

        # -------------------------------------------------
        # 1. Suavização leve (remove ruído de densidade)
        # -------------------------------------------------
        walls_smooth = gaussian_filter(walls, sigma=1)

        # -------------------------------------------------
        # 2. Gradientes
        # -------------------------------------------------
        gx = sobel(walls_smooth, axis=1)
        gy = sobel(walls_smooth, axis=0)

        magnitude = np.sqrt(gx**2 + gy**2)

        # -------------------------------------------------
        # 3. Orientação (ângulo da normal → direção da parede)
        # -------------------------------------------------
        # normal = atan2(gy, gx)
        # parede = normal + 90°
        orientation = np.arctan2(gy, gx) + np.pi / 2

        # normaliza para [0, π)
        orientation = np.mod(orientation, np.pi)

        # -------------------------------------------------
        # 4. Coerência (opcional, mas MUITO útil)
        # -------------------------------------------------
        if out_coherence_path is not None:
            coherence = magnitude / (magnitude.max() + 1e-6)
        else:
            coherence = None

        # -------------------------------------------------
        # 5. Escrita dos rasters
        # -------------------------------------------------
        profile.update(
            dtype="float32",
            count=1,
            nodata=-9999,
        )

        with rasterio.open(out_orientation_path, "w", **profile) as dst:
            dst.write(orientation.astype("float32"), 1)

        if coherence is not None:
            with rasterio.open(out_coherence_path, "w", **profile) as dst:
                dst.write(coherence.astype("float32"), 1)

        return {
            "orientation": out_orientation_path,
            "coherence": out_coherence_path,
        }

    def calc_hag(self, force_recalc: bool = False):
        """
        Calcula o HAG normalizado (uint8) e armazena em self.hag_table.

        ⚠️ Versão escalar:
        NÃO inclui x, y, z.
        """

        if hasattr(self, "hag_table") and not force_recalc:
            return self

        if not hasattr(self, "table"):
            raise RuntimeError("Execute calc_flaz() antes de calc_hag().")

        if "hag" not in self.table.column_names:
            raise ValueError("Coluna 'hag' não encontrada na tabela.")

        import numpy as np
        import pyarrow as pa

        hag = self.table["hag"].to_numpy(zero_copy_only=False)

        valid = ~np.isnan(hag)
        if valid.sum() == 0:
            raise ValueError("Todos os valores de HAG são NaN.")

        hmin = float(hag[valid].min())
        hmax = float(hag[valid].max())

        cmap = np.zeros(len(hag), dtype="uint8")
        if hmax > hmin:
            cmap[valid] = ((hag[valid] - hmin) / (hmax - hmin) * 255).astype("uint8")

        # stats físicos (para o card)
        self.hag_stats = {
            "min": hmin,
            "max": hmax,
            "unit": "m",
        }

        # 🔹 tabela escalar
        self.hag_table = pa.Table.from_pydict(
            {
                "hag_colormap": pa.array(cmap, type=pa.uint8())
            }
        )

        return self

    def calc_classification(self, force_recalc: bool = False):
        """
        Calcula a classificação simplificada (uint8) e armazena em self.class_table.

        ⚠️ Versão experimental:
        Este artefato NÃO carrega geometria (x,y,z),
        apenas a coluna escalar de classificação.
        """

        if hasattr(self, "class_table") and not force_recalc:
            return self

        if not hasattr(self, "table"):
            raise RuntimeError("Execute calc_flaz() antes de calc_classification().")

        if "classification" not in self.table.column_names:
            raise ValueError("Coluna 'classification' não encontrada na tabela.")

        import pyarrow as pa

        class_array = self.table["classification"]

        # 🔹 tabela escalar pura (mesmo número de linhas!)
        self.class_table = pa.Table.from_pydict(
            {
                "classification": class_array
            }
        )

        return self

    def calc_via_viela_vazio(self, force_recalc: bool = False):
        """
        Marca pontos pertencentes a vias, vielas ou vazios.

        Versão escalar:
        - 1 = via / viela / vazio
        - 0 = demais
        """

        if hasattr(self, "vvv_table") and not force_recalc:
            return self

        if not hasattr(self, "table"):
            raise RuntimeError("Execute calc_flaz() antes de calc_via_viela_vazio().")

        import numpy as np
        import pyarrow as pa

        table = self.table

        if "classification" not in table.column_names:
            raise ValueError("Coluna 'classification' não encontrada.")

        if "hag" not in table.column_names:
            raise ValueError("Coluna 'hag' não encontrada.")

        classification = table["classification"].to_numpy(zero_copy_only=False)
        hag = table["hag"].to_numpy(zero_copy_only=False)

        # máscaras semânticas
        is_vegetation = (classification >= 3) & (classification <= 5)
        is_building = classification == 6
        is_low = hag <= 1.5

        via_viela_vazio = (
            (~is_vegetation) &
            (~is_building) &
            is_low
        )

        vvv_uint8 = via_viela_vazio.astype("uint8")

        # 🔹 tabela escalar
        self.vvv_table = pa.Table.from_pydict(
            {
                "via_viela_vazio": pa.array(vvv_uint8, type=pa.uint8())
            }
        )

        self.vvv_stats = {
            "total_points": int(len(vvv_uint8)),
            "vvv_points": int(vvv_uint8.sum()),
            "ratio": float(vvv_uint8.sum() / max(len(vvv_uint8), 1)),
            "hag_threshold_m": 1.5,
        }

        return self

    def calc_delta(self, force_recalc: bool = False):
        """
        Calcula o Delta vertical entre o período atual e o período anterior,
        comparando os pontos do ano atual com o MDS (50 cm) do ano anterior.

        O resultado é um artefato escalar:
            - delta_flaz.arrow
            - coluna: delta_colormap (uint8)

        O Delta pertence SEMPRE ao período atual.
        """

        # -------------------------------------------------
        # 0. Cache
        # -------------------------------------------------
        if hasattr(self, "delta_table") and not force_recalc:
            return self

        if self._ano is None:
            raise ValueError("Ano não definido. Use .periodo(ano) antes de calc_delta().")

        # -------------------------------------------------
        # 1. Determinar período anterior
        # -------------------------------------------------
        ano_atual = self._ano

        base_dir = self.favela_dir() / "periodos"
        if not base_dir.exists():
            raise RuntimeError("Nenhum período anterior encontrado para Delta.")

        anos_disponiveis = sorted(
            int(p.name) for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()
        )

        anos_passados = [a for a in anos_disponiveis if a < ano_atual]

        if not anos_passados:
            raise RuntimeError(
                f"Delta exige período anterior. Nenhum ano < {ano_atual} encontrado."
            )

        ano_anterior = anos_passados[-1]

        # -------------------------------------------------
        # 2. Insumos obrigatórios
        # -------------------------------------------------
        if not hasattr(self, "table"):
            raise RuntimeError("Execute calc_flaz() antes de calc_delta().")

        # caminho do MDS anterior
        mds_path = self.periodo_dir(ano_anterior) / "mds.tif"

        if not mds_path.exists():
            raise FileNotFoundError(
                f"MDS do período anterior não encontrado: {mds_path}"
            )

        # -------------------------------------------------
        # 3. Abrir MDS do período anterior
        # -------------------------------------------------
        import rasterio
        import numpy as np
        import pyarrow as pa
        from rasterio.transform import rowcol

        with rasterio.open(mds_path) as src:
            Z_mds = src.read(1).astype("float32")
            transform = src.transform
            nodata = src.nodata
            height, width = Z_mds.shape

        # -------------------------------------------------
        # 4. Coordenadas reais dos pontos atuais
        # -------------------------------------------------
        # ⚠️ Usa offsets salvos na normalização
        meta = getattr(self, "meta", {})
        offsets = meta.get("offsets")

        if offsets is None:
            raise RuntimeError(
                "Offsets não encontrados. Delta exige calc_flaz() com normalização."
            )

        q = meta.get("quantization", None)
        if q is None:
            raise RuntimeError("Quantização não encontrada no meta.")

        # coordenadas normalizadas (int)
        xs = self.table["x"].to_numpy(zero_copy_only=False)
        ys = self.table["y"].to_numpy(zero_copy_only=False)
        zs = self.table["z"].to_numpy(zero_copy_only=False)

        # reconstrói coordenadas reais (metros)
        x_real = xs * q + offsets["xmin"]
        y_real = ys * q + offsets["ymin"]
        z_real = zs * q + offsets["zmin"]

        # -------------------------------------------------
        # 5. Lookup raster (ponto -> MDS)
        # -------------------------------------------------
        delta_z = np.full(len(z_real), np.nan, dtype="float32")

        rows, cols = rowcol(transform, x_real, y_real)

        valid = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
        )

        rows_v = rows[valid]
        cols_v = cols[valid]

        z_ref = Z_mds[rows_v, cols_v]

        if nodata is not None:
            ok = z_ref != nodata
            idx = np.where(valid)[0][ok]
            delta_z[idx] = z_real[idx] - z_ref[ok]
        else:
            idx = np.where(valid)[0]
            delta_z[idx] = z_real[idx] - z_ref

        # -------------------------------------------------
        # 6. Normalização para colormap (uint8)
        # -------------------------------------------------
        valid_delta = ~np.isnan(delta_z)

        if valid_delta.sum() == 0:
            raise RuntimeError("Delta inválido: nenhum valor válido calculado.")

        dmin = float(np.nanmin(delta_z))
        dmax = float(np.nanmax(delta_z))

        delta_colormap = np.zeros(len(delta_z), dtype="uint8")

        if dmax > dmin:
            delta_colormap[valid_delta] = (
                (delta_z[valid_delta] - dmin) / (dmax - dmin) * 255
            ).astype("uint8")

        # -------------------------------------------------
        # 7. Tabela Arrow escalar
        # -------------------------------------------------
        self.delta_table = pa.Table.from_pydict(
            {
                "delta_colormap": pa.array(delta_colormap, type=pa.uint8())
            }
        )

        # -------------------------------------------------
        # 8. Estatísticas (para card / debug)
        # -------------------------------------------------
        self.delta_stats = {
            "ref": f"{ano_anterior}-{ano_atual}",
            "min": dmin,
            "max": dmax,
            "unit": "m",
            "source": "mds",
            "resolution_m": 0.5,
        }

        return self

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
        Carrega os pontos LAZ/COPC canônicos da favela
        (gerados via PDAL em _build_favela_lidar_base)
        e retorna um PyArrow Table com X, Y, Z, Classification (+ HAG se existir).
        """

        if self._ano is None:
            raise ValueError("Ano não definido. Use .periodo(ano) antes.")

        # caminho esperado da base canônica
        base_dir = self.periodo_dir(self._ano) 

        copc_path = base_dir / "favela.copc.laz"

        if not copc_path.exists():
            raise FileNotFoundError(
                f"COPC canônico não encontrado em {copc_path}. "
                "Execute _build_favela_lidar_base antes."
            )

        las = laspy.read(copc_path)

        # coordenadas (já recortadas, alinhadas com MDT/MDS)
        X = np.asarray(las.x, dtype="float64")
        Y = np.asarray(las.y, dtype="float64")
        Z = np.asarray(las.z, dtype="float64")

        # classificação
        if "classification" in las.point_format.dimension_names:
            C = np.asarray(las.classification, dtype="uint8")
        else:
            C = np.zeros(len(X), dtype="uint8")

        data = {
            "x": X,
            "y": Y,
            "z": Z,
            "classification": C,
        }

        # HAG — se existir, entra automaticamente no Arrow
        if "HeightAboveGround" in las.point_format.dimension_names:
            data["hag"] = np.asarray(
                las.HeightAboveGround,
                dtype="float32"
            )

        table = pa.Table.from_pydict(data)

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
    
    def _write_arrow(self, dest: Path):
        """
        Escreve o flaz.arrow contendo apenas as colunas
        necessárias para visualização no FVIZ.
        """

        # colunas permitidas no Arrow final
        cols = ["x", "y", "z", "flaz_colormap"]

        # sanity check
        missing = [c for c in cols if c not in self.table.column_names]
        if missing:
            raise ValueError(
                f"Colunas ausentes na tabela para persistência: {missing}"
            )

        table_out = self.table.select(cols)

        with pa.OSFile(dest.as_posix(), "wb") as f:
            with ipc.RecordBatchFileWriter(f, table_out.schema) as writer:
                writer.write_table(table_out)

    def _compute_elevation_stats(self, table: pa.Table):
        z = table["z"].to_numpy(zero_copy_only=False)

        if len(z) == 0:
            return None

        return {
            "min": float(z.min()),
            "max": float(z.max()),
            "ref": "NMM",
        }

    def _build_favela_lidar_base(
        self,
        out_dir: Path | None = None,
        force: bool = False,
    ):
        """
        Constrói a base LiDAR da favela para o período corrente.

        Produtos:
        - favela.copc.laz  (canônico, com HAG)
        - mds.tif
        - mdt.tif
        - terrain_025.tif
        - wall_candidates_025.tif
        """

        if self._ano is None:
            raise ValueError("Ano não definido. Use .periodo(ano) antes.")

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # GRID CANÔNICO (alinhamento raster)
        # ----------------------------------------------------
        grid_025 = self._compute_aligned_grid(0.25)
        grid_050 = self._compute_aligned_grid(0.5)

        copc_path = out_dir / "favela.copc.laz"
        mds_path = out_dir / "mds.tif"
        mdt_path = out_dir / "mdt.tif"
        terrain_path = out_dir / "terrain_050.tif"
        wall_candidates_path = out_dir / "wall_candidates_025.tif"

        if copc_path.exists() and mds_path.exists() and mdt_path.exists() and not force:
            return {"copc": copc_path, "mds": mds_path, "mdt": mdt_path}

        polygon_wkt = self.geometry.wkt
        polygon_bb_wkt = self.geometry.envelope.wkt
        srs = "EPSG:31983"

        # ====================================================
        # PIPELINE MDT / MDS / TERRAIN
        # ====================================================
        pipeline_mdt = {
            "pipeline": (
                [
                    {
                        "type": "readers.las",
                        "filename": str(path),
                    }
                    for path in self.quadriculas_laz()
                ]
                + [
                    {"type": "filters.merge"},

                    {
                        "type": "filters.crop",
                        "polygon": polygon_bb_wkt,
                    },

                    # -----------------------------
                    # MDS (50 cm)
                    # -----------------------------
                    {
                        "type": "writers.gdal",
                        "filename": str(mds_path),
                        "resolution": grid_050["resolution"],
                        "origin_x": grid_050["origin_x"],
                        "origin_y": grid_050["origin_y"],
                        "width": grid_050["width"],
                        "height": grid_050["height"],
                        "output_type": "max",
                        "nodata": -9999,
                        "override_srs": srs,
                    },

                    # -----------------------------
                    # MDT via TIN
                    # -----------------------------
                    {
                        "type": "filters.range",
                        "limits": "Classification[2:2]",
                    },
                    {"type": "filters.delaunay"},
                    {
                        "type": "filters.faceraster",
                        "resolution": grid_050["resolution"],
                    },
                    {
                        "type": "writers.raster",
                        "filename": str(mdt_path),
                        "nodata": -9999,
                    },

                    # # -----------------------------
                    # # TERRAIN (50 cm, alinhado)
                    # # -----------------------------
                    # {
                    #     "type": "filters.crop",
                    #     "polygon": polygon_wkt,
                    # },
                    # {
                    #     "type": "writers.gdal",
                    #     "filename": str(terrain_path),
                    #     "resolution": grid_050["resolution"],
                    #     "origin_x": grid_050["origin_x"],
                    #     "origin_y": grid_050["origin_y"],
                    #     "width": grid_050["width"],
                    #     "height": grid_050["height"],
                    #     "output_type": "min",
                    #     "nodata": -9999,
                    #     "override_srs": srs,
                    # },
                ]
            )
        }

        pipeline_path_mdt = out_dir / "favela_lidar_base_pipeline_mdt.json"
        pipeline_path_mdt.write_text(json.dumps(pipeline_mdt, indent=2), encoding="utf-8")

        subprocess.run(
            ["pdal", "pipeline", str(pipeline_path_mdt)],
            check=True,
        )

        # ====================================================
        # PIPELINE COPC + WALL CANDIDATES
        # ====================================================
        pipeline = {
            "pipeline": (
                [
                    {
                        "type": "readers.las",
                        "filename": str(path),
                    }
                    for path in self.quadriculas_laz()
                ]
                + [
                    {"type": "filters.merge"},

                    {
                        "type": "filters.crop",
                        "polygon": polygon_wkt,
                    },

                    # -----------------------------
                    # HAG
                    # -----------------------------
                    {"type": "filters.hag_nn"},

                    # -----------------------------
                    # COPC canônico
                    # -----------------------------
                    {
                        "type": "writers.copc",
                        "filename": str(copc_path),
                        "extra_dims": "all",
                    },

                    # -----------------------------
                    # Remove vegetação
                    # -----------------------------
                    {
                        "type": "filters.expression",
                        "expression": "!(Classification >= 3 && Classification <= 5)",
                    },

                    # -----------------------------
                    # TERRAIN (50 cm, alinhado)
                    # -----------------------------
                    {
                        "type": "filters.expression",
                        "expression": "HeightAboveGround <= 1.5",
                    },
                    {
                        "type": "writers.gdal",
                        "filename": str(terrain_path),
                        "resolution": grid_050["resolution"],
                        "origin_x": grid_050["origin_x"],
                        "origin_y": grid_050["origin_y"],
                        "width": grid_050["width"],
                        "height": grid_050["height"],
                        "output_type": "min",
                        "nodata": -9999,
                        "override_srs": srs,
                    },

                    # # -----------------------------
                    # # Covariance features
                    # # -----------------------------
                    # {
                    #     "type": "filters.covariancefeatures",
                    #     "knn": 16,
                    # },

                    # # -----------------------------
                    # # Wall candidates (verticality)
                    # # -----------------------------
                    # {
                    #     "type": "filters.expression",
                    #     "expression": "Verticality > 0.66",
                    # },
                    # {
                    #     "type": "writers.gdal",
                    #     "filename": str(wall_candidates_path),
                    #     "resolution": grid_025["resolution"],
                    #     "origin_x": grid_025["origin_x"],
                    #     "origin_y": grid_025["origin_y"],
                    #     "width": grid_025["width"],
                    #     "height": grid_025["height"],
                    #     "output_type": "count",
                    #     "nodata": 0,
                    #     "override_srs": srs,
                    # },
                ]
            )
        }

        pipeline_path = out_dir / "favela_lidar_base_pipeline.json"
        pipeline_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")

        subprocess.run(
            ["pdal", "pipeline", str(pipeline_path)],
            check=True,
        )

        return {
            "copc": copc_path,
            "mds": mds_path,
            "mdt": mdt_path,
            "terrain": terrain_path,
            "walls": wall_candidates_path,
        }

    def _persist_mdt_png_from_tif(self, out_dir: Path):
        """
        Converte o MDT GeoTIFF em PNG RGBA + metadata JSON
        para consumo direto no FVIZ.
        """

        TARGET_RES = 2.0  # metros

        from PIL import Image
        import numpy as np
        import rasterio
        import json
        from rasterio.features import rasterize
        from rasterio.warp import reproject, calculate_default_transform
        from rasterio.enums import Resampling

        tif_path = out_dir / "mdt.tif"
        if not tif_path.exists():
            return

        png_path = out_dir / "mdt.png"
        meta_path = out_dir / "mdt.json"

        with rasterio.open(tif_path) as src:
            nodata = src.nodata
            bounds = src.bounds
            crs = src.crs
            transform_src = src.transform
            res_src = src.res

            scale = res_src[0] / TARGET_RES

            new_width = int(src.width * scale)
            new_height = int(src.height * scale)

            Z = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=Resampling.average
            ).astype("float32")

            transform = src.transform * src.transform.scale(
                src.width / new_width,
                src.height / new_height
            )

            res = (TARGET_RES, TARGET_RES)

        valid = ~np.isnan(Z)
        if nodata is not None:
            valid &= Z != nodata

        if valid.sum() == 0:
            return

        zmin = float(Z[valid].min())
        zmax = float(Z[valid].max())

        img = np.zeros_like(Z, dtype="uint8")
        img[valid] = ((Z[valid] - zmin) / (zmax - zmin) * 255).astype("uint8")

        alpha = (valid * 255).astype("uint8")

        rgba = np.dstack([img, img, img, alpha])

        Image.fromarray(rgba, mode="RGBA").save(png_path)

        meta = {
            "type": "MDT",
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "resolution": [TARGET_RES, TARGET_RES],
            "nodata": nodata,
            "stats": {
                "min": zmin,
                "max": zmax,
            },
            "crs": str(crs),
            "format": "PNG-RGBA",
            "recommended_colormap": "terrain"
        }

        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _persist_mds_png_from_tif(self, out_dir: Path):
        """
        Converte o MDS GeoTIFF em PNG RGBA + metadata JSON
        para consumo direto no FVIZ.
        """

        from PIL import Image
        import numpy as np
        import rasterio
        import json

        tif_path = out_dir / "mds.tif"
        if not tif_path.exists():
            return

        png_path = out_dir / "mds.png"
        meta_path = out_dir / "mds.json"

        with rasterio.open(tif_path) as src:
            Z = src.read(1).astype("float32")
            nodata = src.nodata
            bounds = src.bounds
            res = src.res
            crs = src.crs

        # máscara válida
        valid = ~np.isnan(Z)
        if nodata is not None:
            valid &= Z != nodata

        if valid.sum() == 0:
            return

        zmin = float(Z[valid].min())
        zmax = float(Z[valid].max())

        img = np.zeros_like(Z, dtype="uint8")
        img[valid] = ((Z[valid] - zmin) / (zmax - zmin) * 255).astype("uint8")

        alpha = (valid * 255).astype("uint8")

        rgba = np.dstack([img, img, img, alpha])

        Image.fromarray(rgba, mode="RGBA").save(png_path)

        meta = {
            "type": "MDS",
            "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
            "resolution": list(res),
            "nodata": nodata,
            "stats": {
                "min": zmin,
                "max": zmax,
            },
            "crs": str(crs),
            "format": "PNG-RGBA",
            "recommended_colormap": "gray"
        }

        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def _write_hag_arrow(self, dest: Path):
        if not hasattr(self, "hag_table"):
            raise RuntimeError("Execute calc_hag() antes de persistir.")

        with pa.OSFile(dest.as_posix(), "wb") as f:
            with ipc.RecordBatchFileWriter(f, self.hag_table.schema) as writer:
                writer.write_table(self.hag_table)

    def _write_class_arrow(self, dest: Path):
        if not hasattr(self, "class_table"):
            raise RuntimeError("Execute calc_classification() antes de persistir.")

        with pa.OSFile(dest.as_posix(), "wb") as f:
            with ipc.RecordBatchFileWriter(f, self.class_table.schema) as writer:
                writer.write_table(self.class_table)

    def _write_vvv_arrow(self, dest: Path):
        """
        Escreve o artefato via_viela_vazio_flaz.arrow
        """

        if not hasattr(self, "vvv_table"):
            raise RuntimeError("Execute calc_via_viela_vazio() antes de persistir.")

        import pyarrow.ipc as ipc
        import pyarrow as pa

        with pa.OSFile(dest.as_posix(), "wb") as f:
            with ipc.RecordBatchFileWriter(f, self.vvv_table.schema) as writer:
                writer.write_table(self.vvv_table)

    def _compute_aligned_grid(self, resolution: float):
        """
        Computa um grid raster alinhado (origem, largura, altura)
        a partir do envelope da geometria da favela.

        Esse grid DEVE ser reutilizado por todos os writers.gdal
        que precisem alinhar célula-a-célula.
        """

        if self.geometry is None or self.geometry.is_empty:
            raise ValueError("Geometria da favela não disponível.")

        minx, miny, maxx, maxy = self.geometry.bounds

        # 🔹 Quantiza origem para baixo (garante alinhamento global)
        origin_x = math.floor(minx / resolution) * resolution
        origin_y = math.floor(miny / resolution) * resolution

        # 🔹 Calcula dimensões do grid
        width = math.ceil((maxx - origin_x) / resolution)
        height = math.ceil((maxy - origin_y) / resolution)

        grid = {
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "width": int(width),
            "height": int(height),
            "crs": "EPSG:31983",
        }

        # guarda para inspeção / debug / metadata
        self._aligned_grids = getattr(self, "_aligned_grids", {})
        self._aligned_grids[resolution] = grid

        return grid
