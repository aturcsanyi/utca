import utca
import importlib

importlib.reload(utca)
import geopandas as gpd
import osmnx as ox
import momepy as mm
import neatnet
import pandas as pd
import os
import datetime
from pathlib import Path
import networkx as nx
import traceback
from tqdm import tqdm

my_filter = (
    '["highway"]["area"!~"yes"]["access"!~"private"]'
    '["highway"!~"abandoned|bridleway|bus_guideway|construction|corridor|'
    #'cycleway|elevator|escalator|footway|no|path|pedestrian|planned|platform|' # no pedestrian
    "cycleway|elevator|escalator|footway|no|path|planned|platform|"  # yes pedestrian
    "proposed|raceway|razed|rest_area|service|services|steps|track|"  # no services
    #'proposed|raceway|razed|rest_area|services|steps|track|' # yes services
    'motorway|motorway_link|trunk|trunk_link|primary_link|secondary_link"]'
    #'["motor_vehicle"!~"no"]["motorcar"!~"no"]' # no car
    '["service"!~"alley|driveway|emergency_access|parking|parking_aisle|private"]'
)
crs = utca.params.crs  #'EPSG:3035'
pop = utca.load_population()
towns = pop[pop[2011] > 2000].index.to_list()


def process_town(cityname: str, output_path: Path, dry_run=False):
    # place_name = f'{cityname}, Hungary'
    place_name = {"city": cityname, "country": "Hungary"}
    G = ox.graph_from_place(place_name, custom_filter=my_filter, truncate_by_edge=False)
    G = ox.project_graph(G, to_crs=crs)  # crs from dem raster (elevation)
    G = utca.prepare_graph(
        G
    )  # better not to prepare?, list and dict attributes problematic
    stats = utca.graph_stats(G)
    for node in G.nodes:
        del G.nodes[node]["bearings"]
        del G.nodes[node]["angles"]
        del G.nodes[node]["adj_sorted"]
        del G.nodes[node]["type"]
    if not dry_run:
        ox.io.save_graphml(G, output_path / f"{cityname}_osm.graphml")
    return stats


# towns = ['Páty']
# towns = towns[:10]
batch_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Output root
output_root = Path("output") / batch_id
output_root.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Created batch folder: {output_root}")
stats_list = []
error_logs = []
# Process each town
for town in tqdm(towns):
    try:
        town_output = output_root / town
        town_output.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing {town}...")
        stats = process_town(town, town_output, dry_run=False)
        stats["town"] = town  # inject town name
        stats_list.append(stats)
    except Exception as e:
        error_message = (
            f"ERROR in {town}:\n"
            f"{str(e)}\n"
            f"{traceback.format_exc()}\n"
            "------------------------------------------\n"
        )
        error_logs.append(error_message)
        print(f"[ERROR] Problem with {town}, continuing...")
print("\n[INFO] All towns processed successfully!")

if stats_list:
    df = pd.DataFrame(stats_list)
    df_path = output_root / "town_statistics.csv"
    df.to_csv(df_path, index=False)
    print(f"\n[INFO] Statistics saved to {df_path}")
else:
    print("\n[WARN] No statistics were generated.")
# Save log file
if error_logs:
    log_path = output_root / "error_log.txt"
    with open(log_path, "w") as f:
        f.writelines(error_logs)
    print(f"[INFO] Error log saved to {log_path}")
print("\n[INFO] Batch completed.")
