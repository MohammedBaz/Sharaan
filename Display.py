import streamlit as st
import geopandas as gpd
import pandas as pd
import requests

def display_geojson_from_url(geojson_url):
    """Displays the area defined in a GeoJSON file from a URL."""
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()
        gdf = gpd.read_file(geojson_url)

        if not gdf.empty and 'geometry' in gdf.columns:
            st.subheader("Area from GeoJSON URL")

            # Calculate bounding box of the geometry
            bounds = gdf.total_bounds
            center_lat = (bounds[1] + bounds[3]) / 2
            center_lon = (bounds[0] + bounds[2]) / 2
            st.map(pd.DataFrame({'lat': [center_lat], 'lon': [center_lon]})) # Center map

            polygon_coords = []
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    polygon_coords.extend([list(coord) for coord in geom.exterior.coords])
                elif geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        polygon_coords.extend([list(coord) for coord in polygon.exterior.coords])

            if polygon_coords:
                df_polygon = pd.DataFrame(polygon_coords, columns=['lon', 'lat'])
                st.map(df_polygon)
            else:
                st.warning("No polygon coordinates found in the GeoJSON data.")
        else:
            st.warning("No valid geometry data found in the GeoJSON URL.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GeoJSON from URL: {e}")
    except fiona.errors.DriverError:
        st.error("Error reading geospatial data from the URL. Ensure it's a valid GeoJSON format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.title("Display Area from GeoJSON URL")

geojson_url = st.text_input("Enter the URL of your GeoJSON file on GitHub:",
                            value="https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson")

if geojson_url:
    display_geojson_from_url(geojson_url)
