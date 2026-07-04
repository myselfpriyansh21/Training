"""
AI-Driven Crime Analytics Platform
-----------------------------------
Tab 1: Geospatial Crime Map (Synthetic/Real Data + DBSCAN Hotspot Clustering)
Tab 2: Criminal Network Graph (Mock Relationship Web via NetworkX + PyVis)

Step 7 update: Fix blank-graph rendering bug + simplify Tab 2 for officers
  - FIX: PyVis was defaulting to local/relative JS asset paths (a "lib/"
    folder) that don't exist once the HTML is embedded inside Streamlit's
    sandboxed iframe -> blank canvas. Fixed by (a) using cdn_resources=
    "in_line" so the JS library is embedded directly inside the HTML, and
    (b) saving to a known, explicit file via pathlib and reading it back
    with Path.read_text(), rather than relying on a throwaway temp path.
  - Dark slate background + white labels + bright, high-contrast node
    colors to match the platform's dark theme.
  - Tab 2 controls simplified to ONE dropdown: "🔍 Select Target Suspect
    to Trace" — no jargon, no extra knobs.
  - Plain-language alert summary above the graph when a suspect is selected
    (e.g. "Target is linked to 2 crime scenes and shares evidence with
    Suspect 4").
  - Clean 3-color legend: Red = Suspects/Criminals (incl. Gangs),
    Yellow = Crime Scenes, Blue = Stolen Vehicles / Recovered Evidence.

Expected CSV columns for the geospatial tab (must match exactly):
    latitude, longitude, crime_type, severity_score, crime_hour

Run with:
    pip install streamlit folium streamlit-folium pandas numpy scikit-learn networkx pyvis
    streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import folium
from folium import plugins
import streamlit as st
import streamlit.components.v1 as components
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
import networkx as nx
from pyvis.network import Network

# =========================================================
# 1. CONSTANTS
# =========================================================
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
NUM_INCIDENTS = 150
CRIME_TYPES = ["Chain Snatching", "Burglary", "Assault", "Vehicle Theft"]
REQUIRED_COLUMNS = ["latitude", "longitude", "crime_type", "severity_score", "crime_hour"]

# Default DBSCAN parameters (overridable via the sidebar)
DEFAULT_EPS = 0.007
DEFAULT_MIN_SAMPLES = 3

# Network graph node-type -> color mapping (used in Tab 2).
# "Gang" is visually folded into the same Red bucket as "Suspect" since,
# for officers, a gang IS a criminal entity — keeping the legend to a
# clean 3 colors as requested.
NODE_TYPE_COLORS = {
    "Suspect": "#FF3B3B",       # bright red  -> Suspects/Criminals
    "Gang": "#FF3B3B",          # bright red  -> Suspects/Criminals (gangs)
    "Crime Scene": "#FFD400",   # bright yellow -> Crime Scenes
    "Evidence": "#3B82F6",      # bright blue -> Stolen Vehicles / Evidence
}

# Where the rendered PyVis graph is saved before being read back into
# Streamlit. Using an explicit path next to the script (via pathlib)
# avoids the blank-graph bug caused by relative/temp-file lookups.
GRAPH_HTML_PATH = Path(__file__).resolve().parent / "network.html"

# =========================================================
# 2. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Crime Analytics Platform",
    layout="wide",
)

# =========================================================
# 3. SYNTHETIC GEOSPATIAL DATA GENERATION (Tab 1 fallback data)
# =========================================================
@st.cache_data
def generate_synthetic_crime_data(n: int = NUM_INCIDENTS, seed: int = 42) -> pd.DataFrame:
    """
    Creates a synthetic DataFrame of crime incidents, scattered around
    Bengaluru with a realistic center-weighted distribution.
    """
    rng = np.random.default_rng(seed)

    lat_spread = 0.06
    lon_spread = 0.06

    latitudes = rng.normal(loc=BENGALURU_LAT, scale=lat_spread, size=n)
    longitudes = rng.normal(loc=BENGALURU_LON, scale=lon_spread, size=n)

    crime_types = rng.choice(CRIME_TYPES, size=n)
    severity_scores = rng.integers(1, 11, size=n)
    crime_hours = rng.integers(0, 24, size=n)

    df = pd.DataFrame(
        {
            "latitude": latitudes,
            "longitude": longitudes,
            "crime_type": crime_types,
            "severity_score": severity_scores,
            "crime_hour": crime_hours,
        }
    )
    return df


# =========================================================
# 4. CSV LOADING + VALIDATION HELPER (Tab 1)
# =========================================================
def load_uploaded_csv(uploaded_file) -> pd.DataFrame | None:
    """Attempts to read and validate an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Couldn't read that CSV file: {e}")
        return None

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        st.sidebar.error(
            "Uploaded CSV is missing required column(s): "
            f"{', '.join(missing_cols)}.\n\n"
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}"
        )
        return None

    df = df[REQUIRED_COLUMNS].copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce")
    df["crime_hour"] = pd.to_numeric(df["crime_hour"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    dropped = before - len(df)
    if dropped > 0:
        st.sidebar.warning(f"Dropped {dropped} row(s) with missing/invalid values.")

    df["severity_score"] = df["severity_score"].astype(int)
    df["crime_hour"] = df["crime_hour"].astype(int)

    return df


# =========================================================
# 5. AI HOTSPOT CLUSTERING — DBSCAN (Tab 1)
# =========================================================
def find_dense_hotspots(
    data: pd.DataFrame,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
):
    """Runs DBSCAN to find dense crime zones, excluding noise (-1) points
    from the center-of-mass calculation. Returns (hotspot_centers_df, noise_df)."""
    if data is None or data.empty:
        return None, pd.DataFrame(columns=data.columns if data is not None else [])

    coords = data[["latitude", "longitude"]].to_numpy()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    working = data.copy()
    working["cluster_id"] = labels

    core_points = working[working["cluster_id"] != -1]
    noise_df = working[working["cluster_id"] == -1].drop(columns=["cluster_id"])

    if core_points.empty:
        return None, noise_df

    rows = []
    for cluster_id in sorted(core_points["cluster_id"].unique()):
        cluster_points = core_points[core_points["cluster_id"] == cluster_id]
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "center_lat": cluster_points["latitude"].mean(),
                "center_lon": cluster_points["longitude"].mean(),
                "incident_count": len(cluster_points),
                "avg_severity": cluster_points["severity_score"].mean(),
            }
        )

    hotspot_centers_df = (
        pd.DataFrame(rows).sort_values("incident_count", ascending=False).reset_index(drop=True)
    )
    return hotspot_centers_df, noise_df


# =========================================================
# 6. MOCK RELATIONSHIP-WEB DATA (Tab 2)
# =========================================================
@st.cache_data
def generate_network_data(seed: int = 7):
    """
    Builds a mock criminal-network dataset with four node types:
    Suspects, Gangs, Crime Scenes, Evidence (incl. stolen vehicles/weapons).
    Relationships follow a logical structure so the graph reads like a
    plausible case file.
    """
    rng = np.random.default_rng(seed)

    suspects = [f"Suspect {i}" for i in range(1, 9)]
    gangs = ["Kabir Gang", "Shadow Syndicate", "Iron Fist Crew", "Silver Cobra"]
    crime_scenes = [f"Crime Scene #{i}" for i in range(1, 7)]
    evidence = [
        "9mm Pistol",
        ".38 Caliber Revolver",
        "Switchblade Knife",
        "Crowbar",
        "Stolen Vehicle Plates",
    ]

    nodes = (
        [{"id": s, "type": "Suspect"} for s in suspects]
        + [{"id": g, "type": "Gang"} for g in gangs]
        + [{"id": c, "type": "Crime Scene"} for c in crime_scenes]
        + [{"id": e, "type": "Evidence"} for e in evidence]
    )
    nodes_df = pd.DataFrame(nodes)

    edges = []
    for s in suspects:
        gang = rng.choice(gangs)
        edges.append((s, gang, "affiliated_with"))

    for s in suspects:
        n_scenes = rng.integers(1, 3)
        scenes = rng.choice(crime_scenes, size=n_scenes, replace=False)
        for sc in scenes:
            edges.append((s, sc, "present_at"))

    for s in suspects:
        n_ev = rng.integers(0, 3)
        if n_ev > 0:
            ev_items = rng.choice(evidence, size=n_ev, replace=False)
            for ev in ev_items:
                edges.append((s, ev, "linked_to"))

    for g in gangs:
        n_scenes = rng.integers(1, 4)
        scenes = rng.choice(crime_scenes, size=n_scenes, replace=False)
        for sc in scenes:
            edges.append((g, sc, "operates_in"))

    edges_df = pd.DataFrame(edges, columns=["source", "target", "relationship"])
    return nodes_df, edges_df


def build_full_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> nx.Graph:
    """Converts the mock nodes/edges DataFrames into a NetworkX graph."""
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row["id"], type=row["type"])
    for _, row in edges_df.iterrows():
        G.add_edge(row["source"], row["target"], relationship=row["relationship"])
    return G


def render_graph_html(G: nx.Graph) -> str:
    """
    Renders a NetworkX graph as a self-contained, dark-themed PyVis HTML
    string, ready for st.components.v1.html().

    Bug fix #1 (blank graph): PyVis can default to referencing its JS
    library via a relative "lib/" folder on disk. That folder doesn't
    travel with the HTML once it's embedded inside Streamlit's sandboxed
    iframe, which is what was causing the completely blank graph. Fixed
    by cdn_resources="in_line" -> embeds the JS library directly inside
    the HTML string (no external/relative file lookups at all).

    Bug fix #2 (UnicodeEncodeError on Windows): net.save_graph() opens
    the output file internally using Python's platform-default encoding,
    which on Windows is typically cp1252 ("charmap") rather than UTF-8 —
    this throws a UnicodeEncodeError as soon as the graph contains any
    character cp1252 can't represent. PyVis's save_graph() doesn't expose
    an `encoding` parameter to fix this directly, so instead we use
    net.generate_html() to get the HTML as an in-memory string, then
    write it to disk ourselves with an explicit encoding="utf-8".
    """
    try:
        net = Network(
            height="650px",
            width="100%",
            bgcolor="#1E293B",     # dark slate background
            font_color="#FFFFFF",  # white text labels
            notebook=False,
            cdn_resources="in_line",
        )
    except TypeError:
        # Fallback for older PyVis versions without the cdn_resources kwarg.
        net = Network(
            height="650px",
            width="100%",
            bgcolor="#1E293B",
            font_color="#FFFFFF",
            notebook=False,
        )

    # Physics-based "spiderweb" layout — bright nodes spread out naturally
    # and stay fully draggable.
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.04,
        damping=0.09,
        overlap=0.1,
    )

    for node_id, attrs in G.nodes(data=True):
        node_type = attrs.get("type", "Unknown")
        color = NODE_TYPE_COLORS.get(node_type, "#CCCCCC")
        degree = G.degree(node_id)
        net.add_node(
            node_id,
            label=node_id,
            title=f"{node_type}: {node_id}  ({degree} connection{'s' if degree != 1 else ''})",
            color=color,
            size=16 + (degree * 2.5),
        )

    for source, target, attrs in G.edges(data=True):
        net.add_edge(source, target, title=attrs.get("relationship", ""), color="#8A94A6")

    # --- Generate the HTML in-memory, then write it out ourselves with
    # an explicit UTF-8 encoding (avoids Windows' cp1252/"charmap" default
    # that net.save_graph() would otherwise use internally). ---
    html_content = net.generate_html()
    with open(GRAPH_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    return html_content


def generate_suspect_summary(G: nx.Graph, suspect: str) -> str:
    """
    Builds a plain-language, officer-friendly summary of a suspect's
    connections — e.g. "Target is linked to 2 crime scenes and shares
    evidence with Suspect 4." Avoids any technical/graph-theory jargon.
    """
    if suspect not in G:
        return f"⚠️ No connection data found for **{suspect}**."

    neighbors = list(G.neighbors(suspect))
    crime_scenes = sorted(n for n in neighbors if G.nodes[n].get("type") == "Crime Scene")
    evidence_items = sorted(n for n in neighbors if G.nodes[n].get("type") == "Evidence")
    gangs = [n for n in neighbors if G.nodes[n].get("type") == "Gang"]

    # Find other suspects connected indirectly through a shared
    # gang / crime scene / piece of evidence (a 2-hop relationship).
    shared_links = []
    seen_pairs = set()
    for mid_node in neighbors:
        for second_hop in G.neighbors(mid_node):
            if second_hop != suspect and G.nodes[second_hop].get("type") == "Suspect":
                key = (second_hop, mid_node)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    shared_links.append((second_hop, mid_node))

    facts = []
    if crime_scenes:
        word = "crime scene" if len(crime_scenes) == 1 else "crime scenes"
        facts.append(f"linked to {len(crime_scenes)} {word} ({', '.join(crime_scenes)})")
    if evidence_items:
        word = "piece of evidence" if len(evidence_items) == 1 else "pieces of evidence"
        facts.append(f"connected to {len(evidence_items)} {word} ({', '.join(evidence_items)})")
    if gangs:
        facts.append(f"affiliated with **{gangs[0]}**")

    if facts:
        summary = f"⚠️ Alert: **{suspect}** is " + "; ".join(facts) + "."
    else:
        summary = f"⚠️ Alert: **{suspect}** currently has no recorded connections."

    if shared_links:
        link_phrases = [
            f"shares **{shared_node}** with **{other_suspect}**"
            for other_suspect, shared_node in shared_links[:2]
        ]
        summary += " Notably, this suspect " + "; ".join(link_phrases) + "."

    return summary


# =========================================================
# 7. STREAMLIT UI — TITLE & SIDEBAR (sidebar controls Tab 1's map)
# =========================================================
st.title("🚨 AI-Driven Crime Analytics Platform")

with st.sidebar:
    st.header("Data Source")
    st.caption("Controls below apply to the 🌐 Geospatial Hotspot Map tab.")

    uploaded_file = st.file_uploader(
        "Upload cleaned crime data (CSV)",
        type=["csv"],
        help=f"Expected columns: {', '.join(REQUIRED_COLUMNS)}",
    )

    using_real_data = False
    if uploaded_file is not None:
        real_df = load_uploaded_csv(uploaded_file)
        if real_df is not None and not real_df.empty:
            crime_df = real_df
            using_real_data = True
        else:
            st.sidebar.info("Falling back to synthetic mock data.")
            crime_df = generate_synthetic_crime_data()
    else:
        crime_df = generate_synthetic_crime_data()

    if using_real_data:
        st.success(f"✅ Using uploaded data ({len(crime_df)} rows)")
    else:
        st.info(f"ℹ️ Using synthetic mock data ({len(crime_df)} rows)")

    with st.expander("Preview active data"):
        st.dataframe(crime_df.head(20), use_container_width=True)

    st.divider()
    st.header("Filters")

    selected_types = st.multiselect(
        "Crime Type",
        options=sorted(crime_df["crime_type"].unique()),
        default=sorted(crime_df["crime_type"].unique()),
    )

    time_of_day_range = st.slider(
        "Time of Day (Hour Range)",
        min_value=0,
        max_value=23,
        value=(0, 23),
        step=1,
        help="Drag to narrow the map down to incidents that occurred within "
             "a specific hour range (24-hour format).",
    )

    min_severity = st.slider(
        "Minimum Severity Score",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )

    st.divider()
    with st.expander("🤖 Advanced ML Tuning"):
        st.caption(
            "Fine-tune the DBSCAN algorithm that powers the AI Dense "
            "Hotspot Centers below."
        )
        eps_value = st.slider(
            "Hotspot Sensitivity Radius (Eps)",
            min_value=0.001,
            max_value=0.02,
            value=DEFAULT_EPS,
            step=0.001,
            format="%.3f",
            help="Maximum distance (in coordinate degrees) between two "
                 "incidents for them to be considered part of the same "
                 "hotspot. Roughly: 0.001 ≈ 100m, 0.01 ≈ 1.1km.",
        )
        min_samples_value = st.slider(
            "Minimum Incidents per Hotspot (Min Samples)",
            min_value=2,
            max_value=10,
            value=DEFAULT_MIN_SAMPLES,
            step=1,
            help="Minimum number of nearby incidents required for DBSCAN "
                 "to call a region a dense hotspot.",
        )

# =========================================================
# 8. MAIN TAB LAYOUT
# =========================================================
tab1, tab2 = st.tabs(["🌐 Geospatial Hotspot Map", "🕸️ Criminal Network Graph"])

# ---------------------------------------------------------
# TAB 1 — GEOSPATIAL HOTSPOT MAP (unchanged from previous step)
# ---------------------------------------------------------
with tab1:
    st.caption(
        "⚠️ Using **synthetic mock data** as a fallback — upload the real "
        "cleaned CSV in the sidebar to switch over automatically."
        if not using_real_data
        else "✅ Map is rendering the **real uploaded dataset**."
    )

    filtered_df = crime_df[
        (crime_df["crime_type"].isin(selected_types))
        & (crime_df["severity_score"] >= min_severity)
        & (crime_df["crime_hour"].between(time_of_day_range[0], time_of_day_range[1]))
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("Filtered Incidents", f"{len(filtered_df)}")
    col2.metric("Total Incidents (Active Dataset)", f"{len(crime_df)}")
    col3.metric(
        "Avg. Severity (Filtered)",
        f"{filtered_df['severity_score'].mean():.1f}" if len(filtered_df) > 0 else "—",
    )

    hotspot_centers_df, noise_df = find_dense_hotspots(
        filtered_df, eps=eps_value, min_samples=min_samples_value
    )

    def build_crime_map(data: pd.DataFrame, hotspots: pd.DataFrame | None) -> folium.Map:
        fmap = folium.Map(
            location=[BENGALURU_LAT, BENGALURU_LON],
            zoom_start=12,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )

        if data.empty:
            folium.Marker(
                location=[BENGALURU_LAT, BENGALURU_LON],
                tooltip="No incidents match the current filters.",
                icon=folium.Icon(color="gray", icon="info-sign"),
            ).add_to(fmap)
            return fmap

        heat_data = data[["latitude", "longitude", "severity_score"]].values.tolist()
        plugins.HeatMap(
            heat_data,
            name="Severity Heatmap",
            radius=20,
            blur=25,
            max_zoom=13,
            min_opacity=0.4,
            gradient={
                "0.2": "#330000",
                "0.4": "#660000",
                "0.6": "#b30000",
                "0.8": "#ff0000",
                "1.0": "#ff5500",
            },
        ).add_to(fmap)

        marker_layer = folium.FeatureGroup(name="Incident Markers")
        for _, row in data.iterrows():
            popup_html = (
                f"<b>Crime Type:</b> {row['crime_type']}<br>"
                f"<b>Severity:</b> {row['severity_score']} / 10<br>"
                f"<b>Hour:</b> {row['crime_hour']:02d}:00"
            )
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color="#FFD700",
                weight=1,
                fill=True,
                fill_color="#FFFF00",
                fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"{row['crime_type']} (Severity {row['severity_score']}, "
                        f"{row['crime_hour']:02d}:00)",
            ).add_to(marker_layer)
        marker_layer.add_to(fmap)

        if hotspots is not None and not hotspots.empty:
            hotspot_layer = folium.FeatureGroup(name="AI Dense Hotspot Centers")
            for rank, row in hotspots.iterrows():
                center = [row["center_lat"], row["center_lon"]]
                glow_radius = 250 + (row["incident_count"] * 25)
                folium.Circle(
                    location=center,
                    radius=glow_radius,
                    color="#ff1a1a",
                    weight=2,
                    fill=True,
                    fill_color="#ff1a1a",
                    fill_opacity=0.25,
                ).add_to(hotspot_layer)

                popup_html = (
                    f"<b>AI Dense Hotspot Center #{rank + 1}</b><br>"
                    f"Lat: {row['center_lat']:.5f}<br>"
                    f"Lon: {row['center_lon']:.5f}<br>"
                    f"Incidents in cluster: {row['incident_count']}<br>"
                    f"Avg. severity: {row['avg_severity']:.1f}"
                )
                folium.Marker(
                    location=center,
                    icon=folium.Icon(color="red", icon="exclamation-sign"),
                    tooltip=f"AI Dense Hotspot Center #{rank + 1}",
                    popup=folium.Popup(popup_html, max_width=240),
                ).add_to(hotspot_layer)
            hotspot_layer.add_to(fmap)

        folium.LayerControl(collapsed=False).add_to(fmap)
        return fmap

    crime_map = build_crime_map(filtered_df, hotspot_centers_df)
    st_folium(crime_map, width=1300, height=620, key="crime_map")

    st.subheader("🎯 AI Dense Hotspot Centers (DBSCAN)")
    noise_count = len(noise_df) if noise_df is not None else 0

    if hotspot_centers_df is None:
        st.info(
            "No dense hotspots found under the current settings "
            f"(eps={eps_value:.3f}, min_samples={min_samples_value}). "
            "Try increasing Eps or lowering Min Samples in '🤖 Advanced ML "
            "Tuning', or widen your filters."
        )
    else:
        st.caption(
            f"DBSCAN (eps={eps_value:.3f}, min_samples={min_samples_value}) found "
            f"{len(hotspot_centers_df)} dense hotspot(s) from the {len(filtered_df)} "
            f"currently filtered incidents, with {noise_count} isolated/noise "
            f"incident(s) excluded from the center calculations."
        )
        for rank, row in hotspot_centers_df.iterrows():
            st.markdown(
                f"**#{rank + 1} — Lat: `{row['center_lat']:.5f}`, "
                f"Lon: `{row['center_lon']:.5f}`** &nbsp;|&nbsp; "
                f"{row['incident_count']} incidents &nbsp;|&nbsp; "
                f"avg. severity {row['avg_severity']:.1f}"
            )

    with st.expander("View filtered data currently shown on the map"):
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# ---------------------------------------------------------
# TAB 2 — CRIMINAL NETWORK GRAPH (simplified for officers)
# ---------------------------------------------------------
with tab2:
    st.subheader("🕸️ Criminal Network Graph")
    st.caption(
        "⚠️ Demo mode: this relationship web uses a **synthetically generated** "
        "mock dataset. Drag any node to explore connections — the graph is "
        "fully interactive."
    )

    nodes_df, edges_df = generate_network_data()
    G_full = build_full_graph(nodes_df, edges_df)

    suspect_list = sorted(
        n for n, attrs in G_full.nodes(data=True) if attrs.get("type") == "Suspect"
    )

    # --- THE ONE control for this tab ---
    selected_suspect = st.selectbox(
        "🔍 Select Target Suspect to Trace",
        options=["Show Full Network"] + suspect_list,
    )

    # --- Clean, color-coded legend ---
    legend_html = """
    <div style="display:flex; gap:24px; align-items:center; flex-wrap:wrap;
                font-size:14px; margin: 10px 0 16px 0;">
      <div><span style="display:inline-block;width:14px;height:14px;
           background:#FF3B3B;border-radius:50%;margin-right:6px;
           vertical-align:middle;"></span>Red = Suspects / Criminals (incl. Gangs)</div>
      <div><span style="display:inline-block;width:14px;height:14px;
           background:#FFD400;border-radius:50%;margin-right:6px;
           vertical-align:middle;"></span>Yellow = Crime Scenes</div>
      <div><span style="display:inline-block;width:14px;height:14px;
           background:#3B82F6;border-radius:50%;margin-right:6px;
           vertical-align:middle;"></span>Blue = Stolen Vehicles / Recovered Evidence</div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    # --- Plain-language alert summary + graph scope ---
    if selected_suspect == "Show Full Network":
        G_display = G_full
        st.info(
            "Showing the complete network. Select a specific suspect above "
            "to trace their direct and indirect connections."
        )
    else:
        G_display = nx.ego_graph(G_full, selected_suspect, radius=2)
        summary_text = generate_suspect_summary(G_full, selected_suspect)
        st.warning(summary_text)

    graph_html = render_graph_html(G_display)
    components.html(graph_html, height=680, scrolling=False)

    n_col1, n_col2, n_col3 = st.columns(3)
    n_col1.metric("Nodes Shown", f"{G_display.number_of_nodes()}")
    n_col2.metric("Connections Shown", f"{G_display.number_of_edges()}")
    n_col3.metric("Total Suspects in Database", f"{len(suspect_list)}")

    with st.expander("View raw node / edge data (full network)"):
        st.markdown("**Nodes**")
        st.dataframe(nodes_df, use_container_width=True)
        st.markdown("**Edges (Relationships)**")
        st.dataframe(edges_df, use_container_width=True)