import json
from pathlib import Path

import dash
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback_context, dash_table, dcc, html
from dash_extensions.javascript import assign


APP_TITLE = "District Stats: Scatter + Ternary + GeoJSON Map"

STATS_PATH = Path("output/keruletek_stats_apr16.csv")
GEOJSON_DIR = Path("output/keruletek_geojson_apr17")

GEOJSON_BASE_STYLE = {
    "color": "#2b8cbe",
    "weight": 2,
    "fillColor": "#74a9cf",
    "fillOpacity": 0.28,
}

POLYGON_STYLE_BY_N_SIDES = assign(
    """
function(feature) {
    const raw = feature && feature.properties ? feature.properties.n_sides : null;
    const nSides = Number(raw);

    function getFillColor(v) {
        if (!Number.isFinite(v)) return "#969696";
        if (v <= 3) return "#2b8cbe";
        if (v === 4) return "#31a354";
        if (v === 5) return "#feb24c";
        return "#de2d26";
    }

    return {
        color: "#1f4e79",
        weight: 2,
        fillColor: getFillColor(nSides),
        fillOpacity: 0.32
    };
}
"""
)

GEOJSON_HOVER_STYLE = {
    "color": "#045a8d",
    "weight": 4,
    "fillOpacity": 0.45,
}

GEOJSON_ON_EACH_FEATURE = assign(
    """
function(feature, layer) {
    const props = (feature && feature.properties) ? feature.properties : {};
    const entries = Object.entries(props).filter(
        ([key, _value]) => !String(key).trim().toLowerCase().includes("edge")
    );
    const content = entries.length
        ? entries.map(([key, value]) => `<b>${key}</b>: ${value}`).join("<br>")
        : "No properties";
    layer.bindTooltip(content, {sticky: true, direction: "top", opacity: 0.95});
}
"""
)

NODES_POINT_TO_LAYER = assign(
    """
function(feature, latlng) {
    const rawType = feature && feature.properties ? feature.properties.type : null;
    const nodeType = rawType === null || rawType === undefined ? "other" : String(rawType).trim();

    function getColor(t) {
        if (t === "T") return "#d7301f";
        if (t === "X") return "#31a354";
        if (t === "Y") return "#3182bd";
        return "#808080";
    }

    const markerColor = getColor(nodeType);

    return L.circleMarker(latlng, {
        radius: 4,
        color: markerColor,
        weight: 1,
        fillColor: markerColor,
        fillOpacity: 0.9
    });
}
"""
)

EDGES_STYLE = {
    "color": "#4d4d4d",
    "weight": 1.5,
    "opacity": 0.75,
}

DISTRICT_FILES = {
    f"{x}. kerulet": str(GEOJSON_DIR / f"{x}ker_poly.geojson") for x in range(1, 24)
}

DISTRICT_NODE_FILES = {
    f"{x}. kerulet": str(GEOJSON_DIR / f"{x}ker_nodes.geojson") for x in range(1, 24)
}

DISTRICT_EDGE_FILES = {
    f"{x}. kerulet": str(GEOJSON_DIR / f"{x}ker_edges.geojson") for x in range(1, 24)
}


def district_label(raw_value) -> str | None:
    if raw_value is None or pd.isna(raw_value):
        return None
    try:
        return f"{int(float(raw_value))}. kerulet"
    except (TypeError, ValueError):
        return str(raw_value)


def load_geojson(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_scatter_figure(data: pd.DataFrame, selected_id: str | None) -> go.Figure:
    selected_idx = None
    if selected_id is not None:
        selected_idx = data.index[data["id"] == selected_id].tolist()

    fig = go.Figure(
        data=[
            go.Scatter(
                x=data["n"],
                y=data["v"],
                mode="markers",
                marker={"size": 8, "color": "#1f77b4"},
                customdata=data["id"],
                text=data["id"],
                hovertemplate=(
                    "District: %{customdata}<br>n: %{x}<br>v: %{y}<extra></extra>"
                ),
                selectedpoints=selected_idx,
                selected={"marker": {"color": "#d62728", "size": 9, "opacity": 1.0}},
                unselected={"marker": {"opacity": 0.2}},
            )
        ]
    )

    fig.update_layout(
        title="Scatter",
        clickmode="event+select",
        dragmode="zoom",
        xaxis_title="n",
        yaxis_title="v",
    )

    return fig


def build_ternary_figure(data: pd.DataFrame, selected_id: str | None) -> go.Figure:
    selected_idx = None
    if selected_id is not None:
        selected_idx = data.index[data["id"] == selected_id].tolist()

    fig = go.Figure(
        data=[
            go.Scatterternary(
                a=data["Y"],
                b=data["T"],
                c=data["X"],
                mode="markers",
                marker={"size": 8, "color": "#2ca02c"},
                customdata=data["id"],
                text=data["id"],
                hovertemplate=(
                    "District: %{customdata}<br>Y: %{a}<br>T: %{b}<br>X: %{c}<extra></extra>"
                ),
                selectedpoints=selected_idx,
                selected={"marker": {"color": "#d62728", "size": 9, "opacity": 1.0}},
                unselected={"marker": {"opacity": 0.2}},
            )
        ]
    )

    fig.update_layout(
        title="Ternary",
        clickmode="event+select",
        ternary={
            "sum": 100,
            "aaxis": {"title": "Y"},
            "baxis": {"title": "T"},
            "caxis": {"title": "X"},
        },
    )

    return fig


def row_for_id(data: pd.DataFrame, selected_id: str | None):
    if selected_id is None:
        return []

    row = data[data["id"] == selected_id]
    if row.empty:
        return []

    record = row.iloc[0].to_dict()
    return [{"field": key, "value": value} for key, value in record.items()]


DATA = pd.read_csv(STATS_PATH)
if "kerulet" not in DATA.columns:
    raise ValueError(f"Missing required 'kerulet' column in {STATS_PATH}")
DATA["id"] = DATA["kerulet"].apply(district_label)

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "20px auto", "fontFamily": "sans-serif"},
    children=[
        html.H3(APP_TITLE),
        html.Label("Select district"),
        dcc.Dropdown(
            id="id-dropdown",
            options=[{"label": value, "value": value} for value in DATA["id"]],
            value=None,
            placeholder="No district selected",
            clearable=True,
            style={"marginBottom": "16px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            children=[
                html.Div(
                    style={"flex": "1 1 440px", "minWidth": "320px"},
                    children=[
                        dcc.Graph(
                            id="scatter-plot", figure=build_scatter_figure(DATA, None)
                        )
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 440px", "minWidth": "320px"},
                    children=[
                        dcc.Graph(
                            id="ternary-plot", figure=build_ternary_figure(DATA, None)
                        )
                    ],
                ),
            ],
        ),
        html.Div(
            style={
                "display": "flex",
                "gap": "16px",
                "flexWrap": "wrap",
                "alignItems": "flex-start",
                "marginTop": "8px",
            },
            children=[
                html.Div(
                    style={
                        "flex": "0 1 300px",
                        "minWidth": "260px",
                        "maxWidth": "340px",
                    },
                    children=[
                        dash_table.DataTable(
                            id="details-table",
                            columns=[
                                {"name": "Field", "id": "field"},
                                {"name": "Value", "id": "value"},
                            ],
                            data=[],
                            style_table={
                                "overflowX": "auto",
                                "maxHeight": "360px",
                                "overflowY": "auto",
                            },
                            style_cell={
                                "textAlign": "left",
                                "padding": "6px",
                                "fontSize": "13px",
                            },
                            style_cell_conditional=[
                                {
                                    "if": {"column_id": "field"},
                                    "width": "40%",
                                    "fontWeight": "600",
                                },
                                {"if": {"column_id": "value"}, "width": "60%"},
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "2 1 560px", "minWidth": "420px"},
                    children=[
                        html.Div(
                            id="selected-district",
                            style={"marginBottom": "8px", "fontWeight": "600"},
                        ),
                        html.Div(
                            style={"position": "relative"},
                            children=[
                                dl.Map(
                                    id="district-map",
                                    center=[47.4979, 19.0402],
                                    zoom=11,
                                    style={"width": "100%", "height": "360px"},
                                    children=[
                                        dl.TileLayer(
                                            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                                        ),
                                        dl.Pane(
                                            id="polygon-pane",
                                            name="polygon-pane",
                                            style={"zIndex": 390},
                                            children=[
                                                dl.GeoJSON(
                                                    id="district-geojson",
                                                    zoomToBounds=True,
                                                    options={
                                                        "style": POLYGON_STYLE_BY_N_SIDES
                                                    },
                                                    hoverStyle=GEOJSON_HOVER_STYLE,
                                                    onEachFeature=GEOJSON_ON_EACH_FEATURE,
                                                )
                                            ],
                                        ),
                                        dl.Pane(
                                            id="nodes-pane",
                                            name="nodes-pane",
                                            style={"zIndex": 460},
                                            children=[
                                                dl.GeoJSON(
                                                    id="district-nodes-geojson",
                                                    options={
                                                        "pointToLayer": NODES_POINT_TO_LAYER
                                                    },
                                                    onEachFeature=GEOJSON_ON_EACH_FEATURE,
                                                )
                                            ],
                                        ),
                                        dl.Pane(
                                            id="edges-pane",
                                            name="edges-pane",
                                            style={"zIndex": 430},
                                            children=[
                                                dl.GeoJSON(
                                                    id="district-edges-geojson",
                                                    options={"style": EDGES_STYLE},
                                                    onEachFeature=GEOJSON_ON_EACH_FEATURE,
                                                )
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "position": "absolute",
                                        "right": "10px",
                                        "bottom": "10px",
                                        "zIndex": "1000",
                                        "background": "rgba(255,255,255,0.95)",
                                        "padding": "8px 10px",
                                        "border": "1px solid #c7c7c7",
                                        "borderRadius": "6px",
                                        "fontSize": "12px",
                                        "lineHeight": "1.4",
                                    },
                                    children=[
                                        html.Div(
                                            "Node type",
                                            style={
                                                "fontWeight": "600",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#2b8cbe",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "Y",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#31a354",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "X",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#d7301f",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "T",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#808080",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "other",
                                            ]
                                        ),
                                        html.Div(
                                            "Polygon n_sides",
                                            style={
                                                "fontWeight": "600",
                                                "marginTop": "8px",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#2b8cbe",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "<= 3",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#31a354",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "4",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#feb24c",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "5",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#de2d26",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                ">= 6",
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Span(
                                                    style={
                                                        "display": "inline-block",
                                                        "width": "10px",
                                                        "height": "10px",
                                                        "background": "#969696",
                                                        "borderRadius": "50%",
                                                        "marginRight": "6px",
                                                    }
                                                ),
                                                "missing/invalid",
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.H3("---Adam Turcsanyi, ELTE/BME, 2026---"),
    ],
)


@app.callback(
    Output("scatter-plot", "figure"),
    Output("ternary-plot", "figure"),
    Output("details-table", "data"),
    Output("district-geojson", "data"),
    Output("district-edges-geojson", "data"),
    Output("district-nodes-geojson", "data"),
    Output("selected-district", "children"),
    Output("id-dropdown", "value"),
    Input("scatter-plot", "selectedData"),
    Input("ternary-plot", "selectedData"),
    Input("id-dropdown", "value"),
)
def sync_selection(scatter_selected_data, ternary_selected_data, dropdown_id):
    selected_id = dropdown_id

    if callback_context.triggered:
        prop_id = callback_context.triggered[0]["prop_id"]
        trigger, prop = prop_id.split(".")
        if trigger == "scatter-plot" and prop == "selectedData":
            if scatter_selected_data and scatter_selected_data.get("points"):
                selected_id = scatter_selected_data["points"][0]["customdata"]
            else:
                selected_id = None
        elif trigger == "ternary-plot" and prop == "selectedData":
            if ternary_selected_data and ternary_selected_data.get("points"):
                selected_id = ternary_selected_data["points"][0]["customdata"]
            else:
                selected_id = None
        elif trigger == "id-dropdown":
            selected_id = dropdown_id

    empty_geojson = {"type": "FeatureCollection", "features": []}
    geojson_data = empty_geojson
    edges_geojson_data = empty_geojson
    nodes_geojson_data = empty_geojson
    district_label_text = "No district selected"

    if selected_id is not None:
        poly_file_path = DISTRICT_FILES.get(selected_id)
        edge_file_path = DISTRICT_EDGE_FILES.get(selected_id)
        node_file_path = DISTRICT_NODE_FILES.get(selected_id)

        if poly_file_path:
            poly_path_obj = Path(poly_file_path)
            edge_path_obj = Path(edge_file_path) if edge_file_path else None
            node_path_obj = Path(node_file_path) if node_file_path else None

            if poly_path_obj.exists():
                geojson_data = load_geojson(poly_file_path)
                district_label_text = f"Selected: {selected_id} ({poly_file_path})"
            else:
                district_label_text = (
                    f"Selected: {selected_id} (missing polygon file: {poly_file_path})"
                )

            if node_path_obj is not None and node_path_obj.exists():
                nodes_geojson_data = load_geojson(node_file_path)

            if edge_path_obj is not None and edge_path_obj.exists():
                edges_geojson_data = load_geojson(edge_file_path)
        else:
            district_label_text = f"Selected: {selected_id} (no GeoJSON mapping found)"

    return (
        build_scatter_figure(DATA, selected_id),
        build_ternary_figure(DATA, selected_id),
        row_for_id(DATA, selected_id),
        geojson_data,
        edges_geojson_data,
        nodes_geojson_data,
        district_label_text,
        selected_id,
    )


if __name__ == "__main__":
    app.run(debug=True)
