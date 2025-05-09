import streamlit as st
import geopandas as gpd
import pandas as pd
import requests

# Define the GeoJSON URL directly in the code
geojson_url = "https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson"

def display_geojson_area(geojson_url):
    """Displays the area defined in a GeoJSON file from a URL, focusing the view."""
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()
        gdf = gpd.read_file(geojson_url)

        if not gdf.empty and 'geometry' in gdf.columns:
            st.subheader("Sharaan Area")

            polygon_coords = []
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    polygon_coords.extend([list(coord) for coord in geom.exterior.coords])
                elif geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        polygon_coords.extend([list(coord) for coord in polygon.exterior.coords])

            if polygon_coords:
                df_polygon = pd.DataFrame(polygon_coords, columns=['lon', 'lat'])

                # Calculate the bounding box of the polygon
                min_lat = df_polygon['lat'].min()
                max_lat = df_polygon['lat'].max()
                min_lon = df_polygon['lon'].min()
                max_lon = df_polygon['lon'].max()

                # Calculate the center of the bounding box
                center_lat = (min_lat + max_lat) / 2
                center_lon = (min_lon + max_lon) / 2

                # Display the map centered on the polygon
                st.map(df_polygon, latitude=center_lat, longitude=center_lon)
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

st.title("Sharaan Protected Area")

display_geojson_area(geojson_url)
