# Adam Turcsanyi
# diploma thesis work, 2025

# core module for methods used in notebooks

import geopandas as gpd
import pandas as pd
import itertools
import numpy as np
import networkx as nx
import shapely
import osmnx as ox
import matplotlib.pyplot as plt
import folium
import neatnet


# Global parameters
class Parameters:
    def __init__(self):
        self.tolerance_180 = 15
        self.tolerance_x = 15
        self.crs = "EPSG:3035"


params = Parameters()

# Old approach for historical data


def load_streets(kerulet=None, streets=None):
    """load streets from csv"""
    if kerulet is not None:
        query = f"cityname='Budapest {kerulet}. kerület'"
        gdf = gpd.read_file(
            "data/street_hist.csv",
            where=f"cityname='Budapest {kerulet}. kerület'",
        )
    else:
        gdf = gpd.read_file("data/street_hist.csv")
    if streets is not None:
        gdf = gdf[gdf["id"].isin(streets)]
    from_date_na = "1800-01-01"
    to_date_na = "3000-01-01"
    gdf.replace(
        to_replace="",
        value={"from_date": from_date_na, "to_date": to_date_na},
        inplace=True,
    )
    gdf = gdf.sort_values(["id", "from_date"])
    grouped = gdf.groupby("id", sort=False).agg(
        {
            "from_date": "min",
            "to_date": "max",
            "utca": "last",
            "cityid": "last",
            "cityname": "last",
            "geom": "last",
        }
    )
    geom_series = gpd.GeoSeries.from_wkb(grouped["geom"])
    grouped["geometry"] = geom_series
    grouped["geom_type"] = geom_series.geom_type
    grouped = gpd.GeoDataFrame(grouped, crs="EPSG:4326")
    return grouped


def basic_visu(gdf: gpd.GeoDataFrame, m=None, column=None):
    return gdf.explore(
        column=column,
        m=m,
        highlight_kwds={"color": "red"},
        tiles="CartoDB positron",
        cmap="viridis",
        popup=6,
        tooltip=6,
        missing_kwds={"color": "k"},
    )


def get_intersections(gdf: gpd.GeoDataFrame):
    intersections = []
    egyik_utca = []
    masik_utca = []
    utcak = []

    for (idx_a, a), (idx_b, b) in itertools.combinations(gdf.iterrows(), 2):
        if a.geometry.intersects(b.geometry):
            intersections.append(a.geometry.intersection(b.geometry))
            egyik_utca.append(idx_a)
            masik_utca.append(idx_b)
            utcak.append({idx_a, idx_b})

    intersections_gdf = gpd.GeoDataFrame(geometry=intersections, crs="EPSG:4326")
    intersections_gdf["egyik_utca_id"] = egyik_utca
    intersections_gdf["masik_utca_id"] = masik_utca
    intersections_gdf["utcak"] = utcak
    return intersections_gdf.explode()


def group_intersections(gdf: gpd.GeoDataFrame):
    return gdf.groupby("geometry").agg(
        {"utcak": lambda x: set().union(*x)}
    )  # .reset_index()


# osmnx based workflow

# Intersection edges angle calc


def atan_local(edge):
    """
    edge: nx edge with `data='geometry'`
    """
    # print(edge)
    geom = edge[-1]
    if geom is None:
        return -1  # should be exception
    start_node = np.array(geom.coords[0])  # [x, y]
    first_node = np.array(geom.coords[1])  # [x, y]
    # print(first_node)
    angle = np.arctan2(*(first_node - start_node)[::-1])
    return np.rad2deg(angle)


def atan_end2end(edge, G_proj: nx.MultiDiGraph):
    # print(edge)
    start_node = G_proj.nodes[edge[0]]  # {'y': 123, 'x': 123, ...}
    end_node = G_proj.nodes[edge[1]]
    # print(end_node)
    y = end_node["y"] - start_node["y"]
    x = end_node["x"] - start_node["x"]
    angle = np.arctan2(y, x)
    return np.rad2deg(angle)


def atan_unified(edge, G_proj):
    start_node = G_proj.nodes[edge[0]]  # {'y': 123, 'x': 123, ...}
    geom = edge[-1]
    if geom is None:
        # end2end method
        end_node = G_proj.nodes[edge[1]]
        y = end_node["y"] - start_node["y"]
        x = end_node["x"] - start_node["x"]
        angle = np.arctan2(y, x)
        return np.rad2deg(angle)
    else:
        # local method
        coords = list(geom.coords)
        line_start_node = coords[0]
        if (line_start_node[0] != start_node["x"]) or (
            line_start_node[1] != start_node["y"]
        ):
            coords = coords[::-1]
        y = coords[1][1] - coords[0][1]
        x = coords[1][0] - coords[0][0]
        angle = np.arctan2(y, x)
        return np.rad2deg(angle)


def intersection_bearings(node_id, G_proj):
    bearings = {}
    for edge in G_proj.edges([node_id], data="geometry", keys=True):
        bearings[edge[:3]] = atan_unified(edge, G_proj)
    return bearings


def intersection_angles(node_id, G_proj):
    bearings = intersection_bearings(node_id, G_proj)
    bearings = {k: v for k, v in sorted(bearings.items(), key=lambda item: item[1])}
    bearings_list = list(bearings.values())
    angles = {}
    for i, key in enumerate(bearings.keys()):
        angles[key] = (bearings_list[i] - bearings_list[i - 1]) % 360
    return angles


def angles_for_updating(G):
    """
    Returns a list of node, attribute dict pair tuples, with the intersection angles dict as the `angles` attribute
    """
    result = []  # contains node, attribute dict pair tuples
    for node in G:
        angles = intersection_angles(node, G)
        result.append((node, {"angles": angles, "adj_sorted": list(angles.keys())}))
    return result


# Test demo geometry


def test_geom():
    """
    Generates a simple synthetic test geometry
    """
    origin = shapely.Point(20, 20)
    point1 = shapely.Point(20, 40)
    point2 = shapely.Point(40, 0)
    point3 = shapely.Point(20, 0)
    point4 = shapely.Point(60, -20)
    points = [origin, point1, point2, point3, point4]
    nodes = gpd.GeoDataFrame(
        {"osmid": [0, 1, 2, 3, 4], "y": [20, 40, 0, 0, -20], "x": [20, 20, 40, 20, 60]},
        geometry=points,
        crs="EPSG:3857",
    )
    nodes.set_index("osmid", inplace=True)

    street2 = shapely.LineString([origin, [40, 20], point2])
    street3 = shapely.LineString([origin, [0, 20], [0, 0], point3])
    street7 = shapely.LineString([point1, [60, 40], point4])
    streets = [None, street2, street3, None, None, None, street7]
    edgelist = [
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
        (2, 3, 0),
        (0, 3, 1),
        (2, 4, 0),
        (1, 4, 0),
    ]
    edge_ids = [10, 20, 30, 40, 50, 60, 70]
    edge_index = pd.MultiIndex.from_tuples(tuples=edgelist, names=["u", "v", "key"])
    edges = gpd.GeoDataFrame(
        data={"osmid": edge_ids, "idx": edgelist},
        geometry=streets,
        index=edge_index,
        crs="EPSG:3857",
    )

    G = ox.graph_from_gdfs(gdf_nodes=nodes, gdf_edges=edges)
    return G


def fill_graph_attributes(G: nx.Graph):
    """Prepares graph for use with the utca module.
    In order: converts to undirected, removes self-loops, adds angle dicts for nodes, adds node types, adds node corner degrees.
    """
    G = ox.convert.to_undirected(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.add_nodes_from(angles_for_updating(G))
    G.add_nodes_from(node_types_list(G))
    G.add_nodes_from(node_n_corners_list(G))
    return G


def apply_on_nodes(G: nx.Graph, func, attr_name: str):
    """Creates list of node, attribute dict pair tuples, where the attribute is given by a function acting on a node.

    Parameters
    ----------
    G : nx.Graph
        Street graph.
    func: callable
        Function with signature (node, G).
    attr_name: str
        Key to use for the attribute.

    Returns
    -------
    list
        Node, attr dict pair tuples.
    """
    return [(node, {attr_name: func(node, G)}) for node in G]


def prepare_graph(G: nx.Graph) -> nx.Graph:
    """Prepares graph for use with the utca module.
    Replaces `fill_graph_attributes`.
    In order:
    - converts to undirected,
    - removes self-loops,
    - adds bearings of adjacent edges as dict `bearings`,
    - adds angle dicts for nodes `angles`,
    - adds list of adjacent edges sorted by angle `adj_sorted`,
    - adds node types `type`,
    - adds node corner degrees `n_corners`.

    Parameters
    ----------
    G : nx.Graph
        Street graph

    Returns
    -------
    nx.Graph
        Street graph updated with node attributes.
    """
    G = ox.convert.to_undirected(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    attributes = {
        "bearings": intersection_bearings,
        "angles": intersection_angles,
        "adj_sorted": lambda x, g: list(intersection_angles(x, g).keys()),
        "type": node_type,
        "n_corners": node_n_corners,
    }

    for attr, func in attributes.items():
        G.add_nodes_from(apply_on_nodes(G, func, attr))
    return G


def explore_graph(
    G,
    polygons: gpd.GeoDataFrame = None,
    node_column_to_plot=None,
    edge_column_to_plot=None,
    poly_column_to_plot=None,
):
    nodes, edges = ox.convert.graph_to_gdfs(G)
    if polygons is not None:
        m = polygons.explore(
            name="Polygons",
            column=poly_column_to_plot,
            tiles="CartoDB positron",
            highlight_kwds={"color": "red"},
            popup=True,
            style_kwds={"stroke=": False, "opacity": 0.2},
        )
    else:
        m = None
    m = edges.explore(
        m=m,
        name="Edges",
        column=edge_column_to_plot,
        tiles="CartoDB positron",
        highlight_kwds={"color": "red"},
        popup=True,
        style_kwds={"opacity": 0.6, "weight": 5},
    )
    nodes.explore(
        name="Nodes",
        column=node_column_to_plot,
        m=m,
        tiles="CartoDB positron",
        highlight_kwds={"color": "red"},
        popup=True,
        marker_kwds={"radius": 6},
    )
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# Intersection classification
# TODO: use params for tolerances


def t_intersection(angles_list):
    if np.abs(max(angles_list) - 180) <= params.tolerance_180 / 2:
        if min(angles_list) >= 90 - params.tolerance_180 * 2:
            return True
        else:
            return False
    elif np.abs(max(angles_list) - 180) <= params.tolerance_180:
        if min(angles_list) >= 90 - params.tolerance_180:
            return True
        else:
            return False
    else:
        return False


def x_intersection(angles_list):
    # incomplete! need to check also the other pair
    if np.abs(angles_list[0] - angles_list[2]) <= params.tolerance_x:
        return True
    return False


def node_type(node_id, G):
    angles = list(G.nodes[node_id]["angles"].values())
    degree = len(angles)
    if degree == 3:
        if t_intersection(angles):
            return "T"
        elif max(angles) > 180 + params.tolerance_180:
            return "other"
        return "Y"
    elif degree == 4:
        if x_intersection(angles):
            return "X"
        # elif t_intersection(angles):
        #    return "K"
        return "other"
    else:
        return "other"


def node_types_list(G):
    result = []  # contains node, attribute dict pair tuples
    for node in G:
        result.append((node, {"type": node_type(node, G)}))
    return result


def node_n_corners(node_id, G):
    angles = list(G.nodes[node_id]["angles"].values())
    n_corners = 0
    for angle in angles:
        if np.abs(180 - angle) >= params.tolerance_180:
            n_corners += 1
    return n_corners


def node_n_corners_list(G):
    result = []  # contains node, attribute dict pair tuples
    for node in G:
        result.append((node, {"n_corners": node_n_corners(node, G)}))
    return result


# Polygonizing, graph traversal


def polygonize(G: nx.Graph, iter_limit: int = 1000000):
    """
    Polygonizing by graph traversal, returns list of polygons, which are lists of the comprising edges.

    Parameters
    ----------
    G : nx.Graph
        Street graph with each node having an adjacency list sorted by angle `adj_sorted`.
    iter_limit : int
        Iteration limit for stepping infinite loop.

    Reutrns
    -------
    list
        List of polygons, which are lists of the comprising edges.
    """
    visited = set()
    polygons = []
    for e in G.edges:
        for ee in [e, (e[1], e[0], e[2])]:
            if ee in visited:
                continue
            current_edge = ee
            visited.add(current_edge)
            current_polygon = []
            counter = 0
            while True:  # each pass is one step in the traversal
                counter += 1
                if counter == iter_limit:
                    print("too many iterations, stopping")
                    break
                valid_next_step = False
                current_polygon.append(current_edge)
                visited.add(current_edge)
                current_adj_edges = G.nodes[current_edge[1]]["adj_sorted"]  # list
                incoming_edge = (current_edge[1], current_edge[0], current_edge[2])
                prev_edge_idx = current_adj_edges.index(incoming_edge)
                next_edge_idx = prev_edge_idx + 1 - len(current_adj_edges)
                for i in range(
                    len(current_adj_edges)
                ):  # iterate through possible next steps
                    next_edge_candidate = current_adj_edges[next_edge_idx]
                    if len(current_polygon) != 0:
                        if (
                            next_edge_candidate == current_polygon[0]
                        ):  # polygon got around, also implies visited
                            polygons.append(current_polygon)  # write out polygon
                            current_polygon = []  # new polygon
                            next_edge_idx += 1
                            continue  # try next candidate
                    if next_edge_candidate in visited:
                        if len(current_polygon) != 0:
                            print("edge already visited, but doesn't close polygon")
                        next_edge_idx += 1
                        continue  # try next candidate
                    else:  # valid next edge
                        current_edge = next_edge_candidate
                        valid_next_step = True
                        break  # stop looking for next step
                if not valid_next_step:  # end traversal
                    if len(current_polygon) != 0:
                        print("No valid next step, but last polygon is not closed")
                    break
    return polygons


def plot_lines_polygon(poly_edges, G):
    fig, ax = plt.subplots()
    for edge in poly_edges:
        ax.plot(*G.edges[edge]["geometry"].xy)
    return fig, ax


def edge_coords(edge, G):
    return [xy for xy in G.edges[edge]["geometry"].coords]


def create_polygon_old(poly_edges: list, G: nx.Graph):
    """Creates polygon geometry object from list of edges.
    Not guaranteed to be a valid polygon, uses `shapely.Polygon` with a list of coordinates from the edges.

    Parameters
    ----------
    poly_edges : list
        List of edge id tuples comprising the polygon.
    G : nx.Graph
        Street graph.

    Returns
    -------
    shapely.Polygon
        Geometric polygon object.
    """
    coord_list = []
    first = edge_coords(poly_edges[0], G)
    second = edge_coords(poly_edges[1], G)
    if (second[0] == first[-1]) or (second[-1] == first[-1]):
        coord_list += first
    else:
        coord_list += first[::-1]
    for i in range(1, len(poly_edges)):
        current_coords = edge_coords(poly_edges[i], G)
        if current_coords[0] == coord_list[-1]:
            coord_list += current_coords
        elif current_coords[-1] == coord_list[-1]:
            coord_list += current_coords[::-1]
        else:
            print("Polygon segment discontinuity error")
            # plot_lines_polygon(poly_edges, G)
    return shapely.Polygon(coord_list)


def create_polygon(poly_edges: list, G: nx.Graph):
    """Creates polygon geometry object from list of edges, using `shapely.polygonize`. If no geometry is created, then returns None.

    Parameters
    ----------
    poly_edges : list
        List of edge id tuples comprising the polygon.
    G : nx.Graph
        Street graph.

    Returns
    -------
    shapely.Polygon
        Geometric polygon object.
    """
    geom_list = [G.edges[edge]["geometry"] for edge in poly_edges]
    result = shapely.polygonize(geom_list).geoms
    if len(result) != 1:
        print(f"Polygon warning: {len(result)} geometries")
    if result:
        return max(result, key=lambda g: g.area)
    else:
        return None


def remove_dead_ends(poly_edges: list, full_output: bool = False):
    """Removes dead ends inside polygons from the edge sequence.

    Parameters
    ----------
    poly_edges : list
        List of edge id tuples comprising the polygon.
    full_output : bool, optional
        If true, returns the removed edges alongside the polygon.

    Returns
    -------
    list | (list, list) (ezt hogy kell?)
        New edge sequence of polygon.
    """
    stack = []
    cut_edges = []
    for edge in poly_edges:
        if stack and stack[-1] == (edge[1], edge[0], edge[2]):
            cut_edges.append(stack.pop())
            cut_edges.append(edge)
        else:
            stack.append(edge)
    # one last check with the first element
    while stack and stack[0] == (stack[-1][1], stack[-1][0], stack[-1][2]):
        cut_edges.append(stack.pop(-1))
        cut_edges.append(stack.pop(0))
    # no need for else, its already in stack[0]
    if full_output:
        return stack, cut_edges
    else:
        return stack


def is_roundabout(poly_edges, G):
    """Returns true if all edges of a polygon are tagged as roundabout."""
    junctions = [
        G.edges[edge].get("junction") for edge in poly_edges
    ]  # junction tag of edges
    return all(junction == "roundabout" for junction in junctions)


def poly_df(
    G: nx.Graph,
    polygons: list = None,
    drop_outer: bool = True,
    use_old_create_polygon: bool = False,
    dont_count_dead_ends: bool = True,
):
    """Creates GeoDataFrame of polygons from edgelists

    Parameters
    ----------
    G : nx.Graph
        Street graph.
    polygons : list, optional
        List of polygons, each of which is a list of edges comprising the polygon. If not passed, it's computed internally.
    drop_outer : bool
        Drop the outer polygon by finding the largest. Default True.
    use_old_create_polygon : bool, optional
        Use the old, handwritten polygon creator instead of the new one with shapely. Default False.
    dont_count_dead_ends : bool, optional
        Won't count dead ends when counting corner degree. Default True.

    Returns
    -------
    gpd.GeoDataFrame
    """
    if polygons is None:
        polygons = polygonize(G)
    if dont_count_dead_ends:
        polygons = [remove_dead_ends(poly) for poly in polygons]
        # maybe its better to remove at the n_sides calc step, so the angle at the cut edges can be calculated properly
    if use_old_create_polygon:
        geom = [create_polygon_old(poly, G) for poly in polygons]
    else:
        geom = [create_polygon(poly, G) for poly in polygons]
    crs = G.graph["crs"]
    if type(crs) is not str:
        crs = crs.srs
    result = gpd.GeoDataFrame(
        data={
            "n_sides": [poly_n_sides(poly, G) for poly in polygons],
            "edges": [str(poly) for poly in polygons],
            "area": [poly.area if poly else 0 for poly in geom],
            "roundabout": [is_roundabout(poly, G) for poly in polygons],
        },
        geometry=geom,
        crs=crs,
    ).reset_index()
    if drop_outer:
        result = result.drop(result["area"].idxmax())
    return result


def reverse_edge(edge: tuple[int, int, int]):
    """Gives the index of an edge in the reverse direction."""
    return (edge[1], edge[0], edge[2])


def poly_angles(polygon: list, G: nx.Graph) -> list:
    """Calculates the angles inside a polygon at the nodes.
    (Using bearings so works also when dead ends have been pruned.)

    Parameters
    ----------
    polygon : list
        Edge sequence of a polygon.
    G : nx.Graph
        Street graph with `bearings` attribute at nodes.

    Returns
    -------
    list
        Angles in order around the polygon, for each edge the angle at the starting node.
    """
    result = []
    for i in range(len(polygon)):
        current_edge = polygon[i]
        prev_edge = polygon[i - 1]
        node_bearings = G.nodes[current_edge[0]]["bearings"]
        result.append(
            (node_bearings[current_edge] - node_bearings[reverse_edge(prev_edge)]) % 360
        )
    return result


def poly_angles_old(polygon: list, G: nx.Graph):
    result = []
    for edge in polygon:
        start_node = edge[0]
        node_angles = G.nodes[start_node]["angles"]
        result.append(node_angles[edge])
        # if the actual next one is cut, then add the two angles?
    return result


def poly_n_sides(polygon: list, G: nx.Graph):
    angles = poly_angles(polygon, G)
    n_sides = 0
    for angle in angles:
        if np.abs(180 - angle) >= params.tolerance_180:
            n_sides += 1
    return n_sides


# roundabout removal


def _merge_2deg_node(node, G: nx.Graph, ref=None):
    G2 = G.copy()
    edges = list(G2.edges(node, data=True))
    geom_list = []
    for edge in edges:
        if edge[2].get("geometry") is None:
            node_xy = (G.nodes[node]["x"], G.nodes[node]["y"])
            other_xy = (G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"])
            geom_list.append(shapely.LineString([node_xy, other_xy]))
        else:
            geom_list.append(edge[2]["geometry"])  # ez biztos jo?
    # add new edge (result of merge)
    geom = shapely.line_merge(shapely.MultiLineString(geom_list))
    G2.add_edge(edges[0][1], edges[1][1], geometry=geom, osmid=f"node_{node}", ref=ref)
    G2.remove_node(node)  # also removes old edges
    return G2


def node_distance(idx1, idx2, G):
    x1 = G.nodes[idx1]["x"]
    y1 = G.nodes[idx1]["y"]
    x2 = G.nodes[idx2]["x"]
    y2 = G.nodes[idx2]["y"]
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def remove_roundabout(poly_df_row, G: nx.Graph, poly_edges: list) -> nx.Graph:
    idx = f"rab_{poly_df_row['index']}"
    geom = poly_df_row["geometry"]
    G2 = G.copy()
    # new node at center
    centroid = geom.centroid
    G2.add_node(
        idx, roundabout=True, geometry=geom.centroid, x=centroid.x, y=centroid.y
    )
    # nodes of the roundabout
    rab_nodes = set()
    for edge in poly_edges:
        rab_nodes.add(edge[0])
        rab_nodes.add(edge[1])
    # delete old rab edges
    G2.remove_edges_from(poly_edges)
    # new edges to the center
    new_edges = []
    rab_adj_nodes = set()
    for node in rab_nodes:
        for adj in G2[node]:
            if adj not in rab_adj_nodes and adj not in rab_nodes:
                rab_adj_nodes.add(adj)
                # calculate length
                length = node_distance(idx, adj, G2)
                new_edges.append(
                    (idx, adj, {"osmid": f"{idx}_{node}", "length": length})
                )
    G2.add_edges_from(new_edges)
    # remove old rab nodes
    G2.remove_nodes_from(rab_nodes)
    # remove deg2 rab adj nodes
    for node in rab_adj_nodes:
        if len(G2[node]) == 2:
            G2 = _merge_2deg_node(node, G2, ref=idx)
    # check if rab node is deg 2
    if len(G2[idx]) == 2:
        G2 = _merge_2deg_node(idx, G2, ref=idx)
    return G2


def remove_all_roundabouts(G):
    poly_sequence = polygonize(G)
    polygons = poly_df(G, poly_sequence)
    for idx, poly in polygons[polygons["roundabout"]].iterrows():
        G = remove_roundabout(poly, G, poly_sequence[idx])
    G = prepare_graph(G)
    return G


# graph stats


def graph_stats(G, threshold=True, no_deg1_nodes=True):
    polygons = poly_df(G)
    if threshold:
        threshold = polygons["area"].quantile(0.99)
        polygons = polygons[polygons["area"] < threshold]
    nodes = ox.convert.graph_to_gdfs(G, edges=False)
    if no_deg1_nodes:
        nodes = nodes[nodes["n_corners"] > 1]
    F = len(polygons)
    V = len(nodes)
    N_F = polygons["n_sides"].sum()
    N_V = nodes["n_corners"].sum()
    v = polygons["n_sides"].mean()
    n = nodes["n_corners"].mean()
    return {
        "F": F,
        "V": V,
        "N_F": N_F,
        "N_V": N_V,
        "v": v,
        "n": n,
    }


def rebuild_neat_graph(
    neat_streets: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Creates the nodes, edges geodataframes expected by osmnx from the output of neatnet.neatify."""
    nodes_arr = neatnet.nodes._nodes_from_edges(neat_streets.geometry)
    nodes = gpd.GeoDataFrame(geometry=nodes_arr, crs=params.crs)

    from_points = gpd.GeoDataFrame(
        geometry=neat_streets["geometry"].apply(shapely.get_point, args=(0,)),
        crs=params.crs,
    )  # shape of streets
    to_points = gpd.GeoDataFrame(
        neat_streets["geometry"].apply(shapely.get_point, args=(-1,)), crs=params.crs
    )
    joined_from = from_points.sjoin(nodes, how="left")
    joined_to = to_points.sjoin(nodes, how="left")

    edges = neat_streets.copy()
    edges["from"] = joined_from["index_right"]
    edges["to"] = joined_to["index_right"]
    edges["_key"] = edges.groupby(["from", "to"]).cumcount()
    idx = pd.MultiIndex.from_frame(
        df=edges[["from", "to", "_key"]], sortorder=0, names=["u", "v", "key"]
    )
    edges.index = idx

    nodes.index.name = "osmid"
    nodes["x"] = nodes.geometry.x
    nodes["y"] = nodes.geometry.y
    return nodes, edges


# population
def load_population(path="data/pop.dat"):
    """Loads population dataframe from `pop.dat`"""
    cols = [
        "cityname",
        1870,
        1880,
        1890,
        1900,
        1910,
        1920,
        1930,
        1941,
        1949,
        1960,
        1970,
        1980,
        1990,
        2001,
        2011,
    ]
    return pd.read_csv(path, sep=" ", names=cols, index_col="cityname")
