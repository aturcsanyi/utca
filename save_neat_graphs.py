import utca
import neatnet
import osmnx as ox
from pathlib import Path
from tqdm import tqdm
import traceback
import datetime
import networkx as nx


def process_town(path: Path, output_path: Path, dry_run=False):
    cityname = path.name
    path = path.joinpath(f"{cityname}_osm.graphml")
    G = ox.load_graphml(path)
    G = ox.project_graph(G, to_crs=utca.params.crs)
    streets = ox.graph_to_gdfs(G, node_geometry=False, nodes=False, edges=True)
    neat = neatnet.neatify(streets)
    neat["length"] = neat.geometry.length
    nodes, edges = utca.rebuild_neat_graph(neat)
    G = ox.graph_from_gdfs(nodes, edges)
    G = G.to_undirected()
    G = utca.prepare_graph(G)
    # ! largest cc ----
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    # ! -----
    G = utca.remove_all_roundabouts(G)
    G = utca.prepare_graph(G)
    if not dry_run:
        # neat.to_file(output_path / f'{cityname}.geojson', driver='GeoJSON')
        ox.save_graphml(G, output_path / f"{cityname}_simplified.graphml")


batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Output root
output_root = Path("output") / f"neat_{batch_id}"
output_root.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Created batch folder: {output_root}")

folder_path = Path("output/20251203_184441")
town_folders = [d for d in folder_path.iterdir() if d.is_dir()]  # [:10]
error_logs = []
# town_folders = [folder_path.joinpath("Szigetmonostor")]

for town_path in tqdm(town_folders, desc="Processing towns", ascii=True):
    cityname = town_path.name
    print(cityname)
    try:
        process_town(town_path, output_root)
    except Exception as e:
        error_message = (
            f"ERROR in {cityname}:\n"
            f"{str(e)}\n"
            f"{traceback.format_exc()}\n"
            "------------------------------------------\n"
        )
        error_logs.append(error_message)
        print(f"[ERROR] Problem with {cityname}, continuing...")

if error_logs:
    log_path = output_root / "error_log.txt"
    with open(log_path, "w") as f:
        f.writelines(error_logs)
    print(f"[INFO] Error log saved to {log_path}")
print("\n[INFO] Batch completed.")
