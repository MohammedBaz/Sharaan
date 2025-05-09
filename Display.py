import streamlit as st
import geopandas as gpd
import pandas as pd
import requests

def display_geojson_from_url(geojson_url):
    """Displays the area defined in a GeoJSON file from a URL."""
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        geojson_data = response.json()
        gdf = gpd.read_file(geojson_url)  # geopandas can directly read from a URL

        if not gdf.empty and 'geometry' in gdf.columns:
            st.subheader("Area from GeoJSON URL")
            # Extract centroid to center the map
            centroid = gdf.geometry.unary_union.centroid
            st.map(pd.DataFrame({'lat': [centroid.y], 'lon': [centroid.x]}))

            # Display the boundaries
            polygon_coords = []
            for geom in gdf.geometry:
                if geom.geom_type == 'Polygon':
                    polygon_coords.extend(list(geom.exterior.coords))
                elif geom.geom_type == 'MultiPolygon':
                    for polygon in geom.geoms:
                        polygon_coords.extend(list(polygon.exterior.coords))

            if polygon_coords:
                df_polygon = pd.DataFrame(polygon_coords, columns=['lon', 'lat'])
                st.map(df_polygon)
        else:
            st.warning("No valid geometry data found in the GeoJSON URL.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GeoJSON from URL: {e}")
    except ValueError:
        st.error("Invalid JSON format at the provided URL.")
    except fiona.errors.DriverError:
        st.error("Error reading geospatial data from the URL. Ensure it's a valid GeoJSON format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.title("Display Area from GeoJSON URL")

geojson_url = st.text_input("Enter the URL of your GeoJSON file on GitHub:")

if geojson_url:
    display_geojson_from_url(geojson_url)
