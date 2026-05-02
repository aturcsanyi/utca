# script to calculate and save the standard deviation of node elevations for all city street networks

import utca
from tqdm import tqdm
import pandas as pd
from pathlib import Path

pop = utca.load_population()
towns = pop[pop[2011] > 2000].index.to_list()

folder = Path("output/neat_20251207_001303")
elevation_path = [f"data/GEO/eu_dem_v11_E{x}0N20.tif" for x in [4, 5]]

results = []
for town in tqdm(towns):
    try:
        elev_dev = utca.town_elev_deviation(
            town, graphs_folder=folder, elevation_path=elevation_path
        )
        results.append({"cityname": town, "elev_std": elev_dev})
    except Exception as e:
        print(f"Error processing {town}: {e}")

elev_stdevs = pd.DataFrame(results)

elev_stdevs.to_csv("output/elev_stdevs_maj2.csv", index=False)
