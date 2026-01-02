import geopandas as gpd
from shapely.geometry import Polygon
import h3

from flaz.compute.calc_h3_grid import calc_h3_from_gpkg

cells = calc_h3_from_gpkg(
    "data/geoportal_subprefeitura_v2.gpkg",
    resolution=7,
    buffer_m=1200
)

def h3_to_polygon(h):
    boundary = h3.cell_to_boundary(h)  # [(lat, lon), ...]
    return Polygon([(lon, lat) for lat, lon in boundary])

gdf = gpd.GeoDataFrame(
    {"h3": list(cells)},
    geometry=[h3_to_polygon(h) for h in cells],
    crs="EPSG:4326"
)

# salva onde você quiser
out_path = "data/resultado/h3_sp_r7_buf1200.geojson"
# gdf.to_parquet(out_path)
gdf.to_file(
    "data/resultado/h3_sp_r7_buf1200.geojson",
    driver="GeoJSON"
)

print("Arquivo salvo em:", out_path)
