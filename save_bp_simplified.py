import utca
import osmnx as ox
import neatnet
import networkx as nx

place_name = f"Budapest, Hungary"
G = ox.graph_from_place(place_name, custom_filter=utca.params.filter, simplify=True)
G = ox.project_graph(G, to_crs=utca.params.crs)
G = utca.prepare_graph(G)

streets = ox.graph_to_gdfs(G, node_geometry=False, nodes=False, edges=True)
neat = neatnet.neatify(streets)

streets["length"] = streets.geometry.length
nodes, edges = utca.rebuild_neat_graph(neat)
G = ox.graph_from_gdfs(nodes, edges)
G = G.to_undirected()
G = utca.prepare_graph(G)
# ! largest cc ----
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
# ! -----
G = utca.remove_all_roundabouts(G)

ox.io.save_graphml(G, f"output/bp_simplified.graphml")
