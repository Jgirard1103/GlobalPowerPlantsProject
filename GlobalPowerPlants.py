"""
CS 230 Final Project - Global Power Plant Explorer

Name: Jacob Girard
Section: 7
Dataset: WRI Global Power Plant Database (v1.3.0)
         https://github.com/wri/global-power-plant-database
App URL: <https://globalpowerplantsproject-5q3iaflmny7ydazy6ptdpv.streamlit.app/>

Description:
    An interactive Streamlit app for students, researchers, and anyone curious
    about the global energy industry. It answers three questions:
      1. What fuel sources does each selected country generate power from?
      2. Where are the renewable power plants located?
      3. What are the largest power plants in the world (or in one country)?

LLM(s) used: Claude (Anthropic)
References: Streamlit docs, PyDeck docs, Altair docs, WRI GPPD documentation
"""

from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Relative path so the app runs on any computer / Streamlit Cloud.
DATA_FILE = "global_power_plant_database.csv"

# Each fuel gets a category, a hex color and a short description.
FUEL_INFO = {
    "Hydro":          {"group": "Renewable", "color": "#1f77b4", "note": "Dams and run-of-river"},
    "Solar":          {"group": "Renewable", "color": "#ffbf00", "note": "Photovoltaic and solar thermal"},
    "Wind":           {"group": "Renewable", "color": "#2ca02c", "note": "Onshore and offshore wind"},
    "Biomass":        {"group": "Renewable", "color": "#8c6d31", "note": "Plant and agricultural matter"},
    "Geothermal":     {"group": "Renewable", "color": "#e377c2", "note": "Heat from the earth"},
    "Wave and Tidal": {"group": "Renewable", "color": "#17becf", "note": "Ocean energy"},
    "Coal":           {"group": "Fossil",    "color": "#4d4d4d", "note": "Fossil fuel"},
    "Gas":            {"group": "Fossil",    "color": "#ff7f0e", "note": "Fossil fuel"},
    "Oil":            {"group": "Fossil",    "color": "#7f3c0a", "note": "Fossil fuel"},
    "Petcoke":        {"group": "Fossil",    "color": "#a0a0a0", "note": "Refinery by-product"},
    "Cogeneration":   {"group": "Fossil",    "color": "#bcbd22", "note": "Combined heat and power"},
    "Nuclear":        {"group": "Nuclear",   "color": "#9467bd", "note": "Low-carbon, not renewable"},
    "Waste":          {"group": "Other",     "color": "#c49c94", "note": "Waste-to-energy"},
    "Storage":        {"group": "Other",     "color": "#aec7e8", "note": "Batteries / pumped storage"},
    "Other":          {"group": "Other",     "color": "#dbdb8d", "note": "Unclassified"},
}
FUEL_COLORS = {fuel: info["color"] for fuel, info in FUEL_INFO.items()}
RENEWABLE_FUELS = [f for f, info in FUEL_INFO.items() if info["group"] == "Renewable"]

PAGES = [
    "Overview",
    "Q1: Fuel mix by country",
    "Q2: Renewable plant map",
    "Q3: Largest power plants",
]

DEFAULT_COUNTRIES = ["United States of America", "China", "India", "Germany"]


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------

def clean_text(series):
    """Strip edges and collapse repeated internal spaces in a text column."""
    return series.str.strip().str.replace(r"\s+", " ", regex=True)


@st.cache_data
def load_data(path=DATA_FILE):
    """Load the CSV, keep the columns the app uses, and clean them."""
    df = pd.read_csv(path, usecols=[
        "country_long", "name", "gppd_idnr", "capacity_mw", "latitude",
        "longitude", "primary_fuel", "commissioning_year", "owner",
    ])

    # Rename for readability.
    df = df.rename(columns={"country_long": "country"})

    # Messy text: stray spaces in plant names and owners.
    for col in ["name", "owner", "country", "primary_fuel"]:
        df[col] = clean_text(df[col])

    # Drop rows that cannot be used for any question.
    df = df.dropna(subset=["capacity_mw", "latitude", "longitude", "primary_fuel"])
    df = df[df["capacity_mw"] > 0]
    df = df.drop_duplicates(subset="gppd_idnr")

    # Missing owner is common; label it instead of leaving NaN.
    df["owner"] = df["owner"].fillna("Unknown")

    # Nullable integer so missing years stay blank, not 1997.0.
    df["commissioning_year"] = df["commissioning_year"].round().astype("Int64")

    # New column: energy group (Renewable / Fossil / Nuclear / Other).
    df["fuel_group"] = df["primary_fuel"].map(
        {fuel: info["group"] for fuel, info in FUEL_INFO.items()}
    ).fillna("Other")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def fuel_mix(df, countries, measure="capacity_mw"):
    """Total capacity (or plant count) per country and fuel for chosen countries."""
    subset = df[df["country"].isin(countries)]
    if measure == "capacity_mw":
        mix = subset.groupby(["country", "primary_fuel"], as_index=False)["capacity_mw"].sum()
        mix = mix.rename(columns={"capacity_mw": "value"})
    else:
        mix = subset.groupby(["country", "primary_fuel"], as_index=False).size()
        mix = mix.rename(columns={"size": "value"})
    # Percent of each country's total, so countries of different size compare fairly.
    mix["percent"] = mix["value"] / mix.groupby("country")["value"].transform("sum") * 100
    return mix


def filter_plants(df, countries, fuels, min_capacity=0):
    """Filter by country AND fuel AND minimum capacity."""
    mask = (
        df["country"].isin(countries)
        & df["primary_fuel"].isin(fuels)
        & (df["capacity_mw"] >= min_capacity)
    )
    return df[mask]


def top_plants(df, country, fuels, n=10):
    """Return the n largest plants, optionally for a single country."""
    subset = df[df["primary_fuel"].isin(fuels)]
    if country != "All countries":
        subset = subset[subset["country"] == country]
    top = subset.sort_values("capacity_mw", ascending=False).head(n).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def hex_to_rgb(hex_color):
    """'#1f77b4' -> [31, 119, 180] for PyDeck."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(df):
    st.title("Global Power Plant Explorer")
    st.write(
        "How do countries make electricity, and where are the plants? "
        "This app is for students, researchers, and anyone interested in the "
        "global energy industry. Use the sidebar to move through three "
        "connected questions."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Power plants", f"{len(df):,}")
    col2.metric("Countries", df["country"].nunique())
    col3.metric("Total capacity", f"{df['capacity_mw'].sum() / 1000:,.0f} GW")
    renewable_share = df.loc[df["fuel_group"] == "Renewable", "capacity_mw"].sum() / df["capacity_mw"].sum()
    col4.metric("Renewable share", f"{renewable_share:.0%}")

    st.subheader("Fuel categories used in this app")
    legend = pd.DataFrame(
        [{"Fuel": fuel, "Category": info["group"], "Description": info["note"]}
         for fuel, info in FUEL_INFO.items()]
    )
    st.dataframe(legend, hide_index=True, use_container_width=True)

    st.info(
        "**How to read the numbers:** capacity (MW) is the maximum output a plant "
        "can produce, not how much it actually generates. A solar plant and a "
        "nuclear plant with the same capacity will produce very different "
        "amounts of electricity over a year. The database is also incomplete for "
        "some countries."
    )


def page_fuel_mix(df):
    st.header("Q1: What sources does each country get its energy from?")

    all_countries = sorted(df["country"].unique())
    defaults = [c for c in DEFAULT_COUNTRIES if c in all_countries]
    countries = st.multiselect("Countries to compare", all_countries, default=defaults)

    measure_label = st.radio(
        "Measure",
        ["Capacity (MW)", "Number of plants"],
        horizontal=True,
    )
    measure = "capacity_mw" if measure_label.startswith("Capacity") else "count"
    as_percent = st.toggle("Show as % of each country's total", value=False)

    if not countries:
        st.warning("Select at least one country to see its fuel mix.")
        return

    mix = fuel_mix(df, countries, measure)
    y_field = "percent" if as_percent else "value"
    y_title = "Share of country total (%)" if as_percent else measure_label

    chart = (
        alt.Chart(mix)
        .mark_bar()
        .encode(
            x=alt.X("country:N", title="Country", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y(f"{y_field}:Q", title=y_title, stack="zero"),
            color=alt.Color(
                "primary_fuel:N",
                title="Fuel type",
                scale=alt.Scale(domain=list(FUEL_COLORS), range=list(FUEL_COLORS.values())),
            ),
            tooltip=[
                alt.Tooltip("country:N", title="Country"),
                alt.Tooltip("primary_fuel:N", title="Fuel"),
                alt.Tooltip("value:Q", title=measure_label, format=",.0f"),
                alt.Tooltip("percent:Q", title="% of country", format=".1f"),
            ],
        )
        .properties(
            title=f"Fuel mix by country ({measure_label})",
            height=420,
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # Summary table: one row per country, renewable vs other groups.
    group_table = (
        df[df["country"].isin(countries)]
        .pivot_table(index="country", columns="fuel_group",
                     values="capacity_mw", aggfunc="sum", fill_value=0)
    )
    group_table["Total (MW)"] = group_table.sum(axis=1)
    group_table["Renewable %"] = (
        group_table.get("Renewable", 0) / group_table["Total (MW)"] * 100
    ).round(1)
    st.subheader("Capacity by energy category (MW)")
    st.dataframe(group_table.round(0), use_container_width=True)

    st.caption(
        "Caution: countries differ in size, so use the percent toggle when "
        "comparing a large country with a small one. Capacity is not the same "
        "as actual electricity generated."
    )


def page_map(df):
    st.header("Q2: Where are the renewable energy plants?")

    all_countries = sorted(df["country"].unique())
    col1, col2 = st.columns(2)
    countries = col1.multiselect("Countries", all_countries, default=["Germany"])
    fuels = col2.multiselect("Renewable fuel types", RENEWABLE_FUELS, default=RENEWABLE_FUELS)
    min_capacity = st.slider("Minimum plant capacity (MW)", 0, 1000, 10, step=5)
    size_scale = st.select_slider("Marker size", options=["Small", "Medium", "Large"], value="Medium")

    if not countries or not fuels:
        st.warning("Choose at least one country and one fuel type.")
        return

    plants = filter_plants(df, countries, fuels, min_capacity)
    if plants.empty:
        st.warning("No plants match these filters. Try lowering the minimum capacity "
                   "or adding more countries or fuel types.")
        return

    st.write(f"Showing **{len(plants):,}** plants with **{plants['capacity_mw'].sum():,.0f} MW** "
             "of total capacity.")

    plants = plants.copy()
    plants["color"] = plants["primary_fuel"].map(lambda f: hex_to_rgb(FUEL_COLORS[f]))
    radius_factor = {"Small": 40, "Medium": 100, "Large": 250}[size_scale]
    plants["radius"] = plants["capacity_mw"].clip(upper=2000) ** 0.5 * radius_factor

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=plants,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius="radius",
        radius_min_pixels=3,
        radius_max_pixels=40,
        pickable=True,
        opacity=0.75,
    )
    view = pdk.ViewState(
        latitude=float(plants["latitude"].mean()),
        longitude=float(plants["longitude"].mean()),
        zoom=3 if len(countries) > 1 else 4.5,
    )
    tooltip = {
        "html": "<b>{name}</b><br/>{primary_fuel}<br/>{capacity_mw} MW<br/>{country}",
        "style": {"backgroundColor": "#222", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                             tooltip=tooltip, map_style="light"))

    # Chart 2: capacity by fuel type for the plants on the map.
    by_fuel = plants.groupby("primary_fuel", as_index=False)["capacity_mw"].sum()
    bars = (
        alt.Chart(by_fuel)
        .mark_bar()
        .encode(
            x=alt.X("capacity_mw:Q", title="Total capacity (MW)"),
            y=alt.Y("primary_fuel:N", title="Fuel type", sort="-x"),
            color=alt.Color("primary_fuel:N", legend=None,
                            scale=alt.Scale(domain=list(FUEL_COLORS),
                                            range=list(FUEL_COLORS.values()))),
            tooltip=["primary_fuel", alt.Tooltip("capacity_mw:Q", format=",.0f")],
        )
        .properties(title="Capacity of the plants shown on the map", height=220)
    )
    st.altair_chart(bars, use_container_width=True)

    st.caption("Marker size reflects capacity. Some locations in the database are "
               "approximate, so a marker may be a few kilometres off.")


def page_top_plants(df):
    st.header("Q3: What are the largest power plants?")

    col1, col2 = st.columns(2)
    country = col1.selectbox("Country", ["All countries"] + sorted(df["country"].unique()))
    group = col2.selectbox("Energy category", ["All", "Renewable", "Fossil", "Nuclear", "Other"])
    n = st.slider("Number of plants to show", 5, 50, 10)

    if group == "All":
        fuels = list(FUEL_INFO)
    else:
        fuels = [f for f, info in FUEL_INFO.items() if info["group"] == group]

    top = top_plants(df, country, fuels, n)
    if top.empty:
        st.warning(f"No {group.lower()} plants found for {country}. Try a different "
                   "country or category.")
        return
    if len(top) < n:
        st.info(f"Only {len(top)} plants match, fewer than the {n} requested.")

    table = top[["rank", "name", "country", "primary_fuel", "capacity_mw",
                 "commissioning_year", "owner"]].rename(columns={
        "rank": "Rank", "name": "Plant", "country": "Country", "primary_fuel": "Fuel",
        "capacity_mw": "Capacity (MW)", "commissioning_year": "Year", "owner": "Owner"})
    st.dataframe(table, hide_index=True, use_container_width=True)

    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("capacity_mw:Q", title="Capacity (MW)"),
            y=alt.Y("name:N", title=None, sort="-x"),
            color=alt.Color("primary_fuel:N", title="Fuel",
                            scale=alt.Scale(domain=list(FUEL_COLORS),
                                            range=list(FUEL_COLORS.values()))),
            tooltip=["name", "country", "primary_fuel",
                     alt.Tooltip("capacity_mw:Q", format=",.0f", title="MW")],
        )
        .properties(title=f"Top {len(top)} plants by capacity ({country})",
                    height=max(200, 24 * len(top)))
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption("Capacity is the maximum possible output. Plants with unknown owner "
               "or year show 'Unknown' / blank.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Global Power Plant Explorer", page_icon="⚡", layout="wide")

    try:
        df = load_data()
    except FileNotFoundError:
        st.error(f"Data file not found at {DATA_FILE}. Make sure the data folder is "
                 "included with the app.")
        st.stop()

    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.caption("Data: WRI Global Power Plant Database")

    if page == PAGES[0]:
        page_overview(df)
    elif page == PAGES[1]:
        page_fuel_mix(df)
    elif page == PAGES[2]:
        page_map(df)
    else:
        page_top_plants(df)


if __name__ == "__main__":
    main()
