import folium
from streamlit_folium import st_folium

def render_sharaan_map(geojson_data, spatial_data=None):
    """Renders a Folium map of the Sharaan area with optional spatial data overlay."""
    if not geojson_data or 'features' not in geojson_data:
        return None

    # Calculate the center of the Sharaan area
    first_polygon_coords = geojson_data['features'][0]['geometry']['coordinates'][0]
    center_lat = sum(coord[1] for coord in first_polygon_coords) / len(first_polygon_coords)
    center_lon = sum(coord[0] for coord in first_polygon_coords) / len(first_polygon_coords)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

    # Add GeoJSON layer for Sharaan boundary
    folium.GeoJson(geojson_data, style_function=lambda feature: {
        'fillColor': 'purple',
        'color': 'red',
        'weight': 2,
        'fillOpacity': 0.2
    }).add_to(m)

    # Add spatial data as circle markers if provided
    if spatial_data is not None and not spatial_data.empty:
        for index, row in spatial_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=row['value'] * 2,
                color='red',
                fill=True,
                fill_color='orangered',
                fill_opacity=0.6
            ).add_to(m)

    return m

def display_folium_map(folium_map, width=700, height=500):
    """Displays a Folium map in Streamlit."""
    if folium_map:
        st_folium(folium_map, width=width, height=height)
