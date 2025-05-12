# Add missing imports at the top
import matplotlib
import matplotlib.colors

# In generate_random_spatial_data_geojson:
@st.cache_data
def generate_random_spatial_data_geojson(geojson, variable, num_points_per_polygon=2000):
    """Generates GeoJSON-like data for heatmap visualization."""
    features = geojson['features']
    point_features = []
    gdf = gpd.GeoDataFrame.from_features(features)  # Use existing GeoJSON data

    for _, feature in gdf.iterrows():
        geom = feature.geometry
        if geom.geom_type == 'MultiPolygon':
            polygons = list(geom.geoms)
        else:
            polygons = [geom]

        for polygon in polygons:
            random_points = generate_random_points_in_polygon(polygon, num_points_per_polygon)
            for lat, lon in random_points:
                if variable == 'Temperature':
                    intensity = np.random.uniform(0.5, 1.0)
                elif variable == 'Precipitation':
                    intensity = np.random.uniform(0.2, 0.8)
                elif variable == 'Wind Speed':
                    intensity = np.random.uniform(0.4, 0.9)
                else:
                    intensity = np.random.rand()
                point_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {"intensity": intensity}
                })
    return {"type": "FeatureCollection", "features": point_features}

# Heatmap styling adjustment
def heatmap_style(feature):
    intensity = feature['properties']['intensity']
    # Use actual variable range or dynamic normalization
    normalized_intensity = (intensity - 0.2) / (1.0 - 0.2)  # Adjust based on variable
    color = plt.cm.viridis(normalized_intensity)
    hex_color = matplotlib.colors.rgb2hex(color)
    return {'radius': 8, 'fillColor': hex_color, 'color': hex_color, 'fillOpacity': 0.7}
