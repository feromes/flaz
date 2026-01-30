from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString

# -----------------------
# Paths
# -----------------------
BASE = Path("artigos/vielas/insumos/favela/sao_remo/periodos/2020")
OUT = Path("artigos/vielas/figs")
OUT.mkdir(parents=True, exist_ok=True)

vielas_path = BASE / "vielas.gpkg"
out_file = OUT / "fig_4_2_sao_remo_coherence_gaps.png"

# -----------------------
# Load data
# -----------------------
gdf = gpd.read_file(vielas_path)
gdf = gdf.explode(ignore_index=True)

# -----------------------
# Build graph
# -----------------------
G = nx.Graph()

def endpoints(line: LineString):
    coords = list(line.coords)
    return tuple(coords[0]), tuple(coords[-1])

for idx, geom in enumerate(gdf.geometry):
    if not isinstance(geom, LineString):
        continue
    u, v = endpoints(geom)
    G.add_edge(u, v, idx=idx)

# -----------------------
# Connected components
# -----------------------
components = list(nx.connected_components(G))

# Identify dominant component (by total length)
component_length = {}

for comp in components:
    length = 0.0
    for u, v, data in G.subgraph(comp).edges(data=True):
        geom = gdf.loc[data["idx"], "geometry"]
        length += geom.length
    component_length[frozenset(comp)] = length

dominant_nodes = max(component_length, key=component_length.get)

# Map each segment to dominant / gap
is_gap = []

for idx, geom in enumerate(gdf.geometry):
    u, v = endpoints(geom)
    is_gap.append(not (u in dominant_nodes and v in dominant_nodes))

gdf["is_gap"] = is_gap

# -----------------------
# Plot
# -----------------------
fig, ax = plt.subplots(figsize=(6, 6))

# Dominant network (coherent structure)
gdf[~gdf["is_gap"]].plot(
    ax=ax,
    color="#666666",
    linewidth=0.6,
    zorder=1
)

# Local disconnections (gaps)
gdf[gdf["is_gap"]].plot(
    ax=ax,
    color="#000000",
    linewidth=1.1,
    zorder=2
)

ax.set_axis_off()

plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close()

print("✔ Figura salva em:", out_file)
print("Total segments:", len(gdf))
print("Gap segments:", gdf['is_gap'].sum())
