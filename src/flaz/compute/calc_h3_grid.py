"""
Geração de grid H3 a partir de limites administrativos.

Exemplo
-------
>>> from flaz.compute.calc_h3_grid import calc_h3_from_gpkg
>>> cells = calc_h3_from_gpkg(
...     "data/geoportal_subprefeitura_v2.gpkg",
...     resolution=7,
...     buffer_m=1200
... )
>>> len(cells) > 0
True
"""


def calc_h3_from_gpkg(
    gpkg_path,
    layer=None,
    resolution=7,
    buffer_m=1200,
    crs_proj=31983,
):
    import geopandas as gpd
    import h3
    from shapely.geometry import Polygon

    gdf = gpd.read_file(gpkg_path, layer=layer).to_crs(crs_proj)
    geom = gdf.geometry.union_all().buffer(buffer_m)
    geom = gpd.GeoSeries([geom], crs=crs_proj).to_crs(4326).iloc[0]

    def polyfill(geom):
        coords = list(geom.exterior.coords)
        geojson = {
            "type": "Polygon",
            "coordinates": [[(x, y) for x, y in coords]]
        }
        return set(h3.geo_to_cells(geojson, resolution))

    cells = set()
    if geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            cells |= polyfill(g)
    else:
        cells |= polyfill(geom)

    return sorted(cells)

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

