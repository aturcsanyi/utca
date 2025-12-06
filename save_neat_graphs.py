import utca
import neatnet
import osmnx as ox
from pathlib import Path
from tqdm import tqdm
import traceback
import datetime


def process_town(path: Path, output_path: Path, dry_run=False):
    cityname = path.name
    path = path.joinpath(f"{cityname}_osm.graphml")
    G = ox.load_graphml(path)
    G = ox.project_graph(G, to_crs=utca.params.crs)
    streets = ox.graph_to_gdfs(G, node_geometry=False, nodes=False, edges=True)
    neat = neatnet.neatify(streets)
    if not dry_run:
        neat.to_file(output_path / f"{cityname}.geojson", driver="GeoJSON")


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
