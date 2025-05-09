import streamlit as st
import geopandas as gpd
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium

# Define the GeoJSON URL directly in the code
geojson_url = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

def display_geojson_area_folium(geojson_url):
    """Displays the area defined in a GeoJSON file from a URL as a filled polygon using Folium."""
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()
        gdf = gpd.read_file(geojson_url)

        if not gdf.empty and 'geometry' in gdf.columns:
            st.subheader("Sharaan Protected Area (Folium)")

            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    # Folium expects coordinates in [latitude, longitude] order
                    polygon_coords_folium = [[lat, lon] for lon, lat in list(geom.exterior.coords)]
                    m = folium.Map(location=[polygon_coords_folium[0][0], polygon_coords_folium[0][1]], zoom_start=10) # Initial view

                    folium.Polygon(locations=polygon_coords_folium, color="red", fill=True, fill_color="purple", fill_opacity=0.4).add_to(m)

                    st_folium(m, width=700, height=500)
                    return

                elif geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        polygon_coords_folium = [[lat, lon] for lon, lat in list(polygon.exterior.coords)]
                        m = folium.Map(location=[polygon_coords_folium[0][0], polygon_coords_folium[0][1]], zoom_start=10)

                        folium.Polygon(locations=polygon_coords_folium, color="red", fill=True, fill_color="purple", fill_opacity=0.4).add_to(m)

                        st_folium(m, width=700, height=500)
                        return

            st.warning("No Polygon or MultiPolygon geometry found in the GeoJSON data.")
        else:
            st.warning("No valid geometry data found in the GeoJSON URL.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GeoJSON from URL: {e}")
    except fiona.errors.DriverError:
        st.error("Error reading geospatial data from the URL. Ensure it's a valid GeoJSON format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.title("Sharaan Protected Area")

display_geojson_area_folium(geojson_url)
