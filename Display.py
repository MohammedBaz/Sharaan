import streamlit as st
import geopandas as gpd
import pandas as pd
import requests

def display_geojson_from_url(geojson_url):
    """Displays the area defined in a GeoJSON file from a URL with debugging."""
    try:
        response = requests.get(geojson_url)
        response.raise_for_status()
        gdf = gpd.read_file(geojson_url)

        if not gdf.empty and 'geometry' in gdf.columns:
            st.subheader("Area from GeoJSON URL")

            for index, row in gdf.iterrows():
                geometry = row['geometry']
                if geometry.geom_type == 'Polygon':
                    exterior_coords = list(geometry.exterior.coords)
                    st.write("Polygon Coordinates (First 10):", exterior_coords[:10]) # Log first 10
                    df_polygon = pd.DataFrame(exterior_coords, columns=['lon', 'lat'])
                    st.map(df_polygon)
                    return # Exit after processing the first polygon
                elif geometry.geom_type == 'MultiPolygon':
                    for polygon in geometry.geoms:
                        exterior_coords = list(polygon.exterior.coords)
                        st.write("MultiPolygon Coordinates (First 10 of first):", exterior_coords[:10]) # Log first 10
                        df_polygon = pd.DataFrame(exterior_coords, columns=['lon', 'lat'])
                        st.map(df_polygon)
                        return # Exit after processing the first multi-polygon

            st.warning("No Polygon or MultiPolygon geometry found in the GeoJSON data.")
        else:
            st.warning("No valid geometry data found in the GeoJSON URL.")
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching GeoJSON from URL: {e}")
    except fiona.errors.DriverError:
        st.error("Error reading geospatial data from the URL. Ensure it's a valid GeoJSON format.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.title("Display Area from GeoJSON URL (Debugging)")

geojson_url = st.text_input("Enter the URL of your GeoJSON file on GitHub:",
                            value="https://raw.githubusercontent.com/MohammedBaz/Sharaan/main/Sharaan.geojson")

if geojson_url:
    display_geojson_from_url(geojson_url)
