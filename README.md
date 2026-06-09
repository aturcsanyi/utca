# utca

Tools and notebooks for analyzing city street networks in the framework of mosaics. Code for my thesis work
> The Analysis of the Evolution of the City Street Network

## Background

The main aim of the thesis was to apply the theory of convex mosaics to street networks.
A mosaic is made up of nodes, edges, and cells; the key metrics in this theory are the corner degrees of nodes (number of sharp corners at the node) and cells (number of vertices of the cell).
The theory has found success in describing crack networks in rocks and their evolution, see the article [Plato’s cube and the natural geometry of fragmentation](https://doi.org/10.1073/pnas.2001037117)

## Repository contents

* `utca.py`: main utilities module for analyzing street networks
* `/dash_app/`: folder containing an interactive visualization app, complete with data
* `/data/`: folder containing key input data files
* `/output/`: folder containing saved street networks for convenience (they can also be generated using the scripts below)

Notebooks

* `00_city_demo.ipynb`: notebook demonstrating the basic usage of the `utca` module
* `10_all_stats.ipynb`: processing the mosaic statistics of Hungarian towns
* `20_elevation.ipynb`: about the connection between topography and street structure
* `30_population.ipynb`: looking at the link between population dynamics and street mosaics
* `40_bp_hist_map.ipynb`: generating a map of Budapest indicating the ages of streets
* `41_bp_districts_hist.ipynb`: mosaic metrics of the districts of Budapest, and their historical evolution
* `42_node_hist.ipynb`: looking at how intersections form and evolve in different districts
* `50_radial.ipynb`: modeling street network evolution as radial growth
* `51_radial_vs_hist.ipynb`: comparing the radial model with historical data
* `60_thresholding_polygons.ipynb`: different methods of determining a threshold for too large, non-meaningful polygons

Utility scripts

* `calculate_all_graph_stats.py`: calculate the mosaic metrics for all Hungarian towns
* `calculate_elevation_stdev.py`: calculate the standard deviation of the elevation of all nodes for every Hungarian town
* `save_bp_districts.py`: simplify and save the street network of the districts of Budapest
* `save_bp_simplified.py`: save the street network of Budapest after applying artefact removal
* `save_neat_graphs.py`: script to simplify the street graphs of Hungarian towns with neatnet
* `save_osm_graphs.py`: script to save locally the street graphs of Hungarian towns from OpenStreetMap
