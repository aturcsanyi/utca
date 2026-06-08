import geopandas as gpd
import utca
from pathlib import Path
import osmnx as ox
import networkx as nx
import pandas as pd
from tqdm import tqdm

folder = Path("output/neat_20260103_171612")
pop = utca.load_population()
towns = pop[pop[2011] > 2000].index.to_list()


def process_stats(cityname):
    # streets = gpd.read_file(folder / f'{cityname}.geojson')
    # streets['length'] = streets.geometry.length
    # nodes, edges = utca.rebuild_neat_graph(streets)
    # G = ox.graph_from_gdfs(nodes, edges)
    G = ox.load_graphml(folder / f"{cityname}_simplified.graphml")
    G = utca.prepare_graph(G)
    # ! largest cc ----
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    # ! -----
    G = utca.remove_all_roundabouts(G)

    return utca.graph_stats(G)


results = []
for town in tqdm(towns):
    try:
        stats = process_stats(town)
        stats.update({"cityname": town})
        results.append(stats)
    except Exception as e:
        print(f"Error processing {town}: {e}")

all_graph_stats = pd.DataFrame(results)
# all_graph_stats.set_index('cityname')

all_graph_stats.to_csv("output/all_graph_stats_maj8.csv", index=False)
