import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import KDTree
from geopy.distance import distance
from scipy.optimize import linear_sum_assignment


def proxy_finder(dataframe: pd.DataFrame, number_of_closest_matches: int, selected_index: list, visualisation: bool) -> pd.DataFrame:
    # --------------------------------------------------
    # 1. Clean data, select unit(s), visualise
    # --------------------------------------------------

    # --------------------------------------------------
    # 1.1 Sanity checks
    # --------------------------------------------------

    if isinstance(number_of_closest_matches, bool) or not isinstance(number_of_closest_matches, int):
        raise TypeError("`number_of_closest_matches` must be an int, a number.")

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("`dataframe` must be a pd.DataFrame.")

    if not isinstance(selected_index, list):
        raise TypeError("'selected_index' must be of type list")

    if not isinstance(visualisation, bool):
        raise TypeError("'visualisation' must be of type boolean")

    dataframe = dataframe[["lon", "lat"]].copy().apply(pd.to_numeric, errors="coerce")

    df_array = dataframe[["lat", "lon"]].values
    mask = ~np.isfinite(df_array)
    rows_with_bad_values = mask.any(axis=1)
    bad_rows = dataframe.loc[rows_with_bad_values, ["lat", "lon"]]

    if len(bad_rows) != 0:
        raise ValueError(
            "Invalid coordinates format. Inspect your coordinates to ensure that they meet\n 'lon': (-180, 180),\n'lat': (-90, 90)")

    dataframe_coords = dataframe[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")

    if not dataframe_coords["lon"].between(-180, 180).all() or not dataframe_coords["lat"].between(-90, 90).all():
        raise ValueError(
            f"Invalid coordinates format. Inspect your coordinates to ensure that they meet:\n"
            f"'lon': (-180, 180),\n"
            f"'lat': (-90, 90)"
        )

    # --------------------------------------------------
    # 1.2 Select unit(s)
    # --------------------------------------------------

    df_sample = dataframe[dataframe.index.isin(selected_index)]
    if df_sample.empty:
        raise ValueError("You've chosen coordinate(s) which do(es) not exist in the dataframe.")
    df_out = dataframe[~dataframe.index.isin(selected_index)]

    # --------------------------------------------------
    # 1.3 Initiate visualisation
    # --------------------------------------------------

    print(f"Selected coordinate(s):\n{df_sample}") # Selected coordinates
    if visualisation:
        # figure, interactive, spotlighting spatial position of both in-sample and out of sample coordinates
        fig = px.scatter_map(
            df_out,
            lat="lat",
            lon="lon",
            zoom=1
        )
        fig.update_traces(name="All catches",
                          hovertemplate="<b>%{hovertext}</b><br>Latitude: %{lat}<br>Longitude: %{lon}<extra></extra>",
                          showlegend=True)

        # Make df_out hover match df_sample
        fig.update_traces(
            selector=dict(type="scattermap"),
            customdata=df_out[["lat", "lon"]].to_numpy(),
            text=df_out.index.astype(str),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude: %{customdata[0]:.5f}<br>"
                "Longitude: %{customdata[1]:.5f}"
                "<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scattermap(
                lat=df_sample["lat"],
                lon=df_sample["lon"],
                mode="markers",
                marker=go.scattermap.Marker(
                    size=10,
                    color='rgb(255, 0, 0)'
                ),
                customdata=df_sample[["lat", "lon"]].to_numpy(),
                text=df_sample.index.astype(str),
                name=f"Selected coordinate(s)",
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Latitude: %{customdata[0]:.5f}<br>"
                    "Longitude: %{customdata[1]:.5f}"
                    "<extra></extra>"
                )
            )
        )
        fig.show()

    # --------------------------------------------------
    # 2. Find the nearest coordinates (greedy algorithm)
    # --------------------------------------------------

    # --------------------------------------------------
    # 2.1. Greedy (fast, globally suboptimal) algorithm
    # --------------------------------------------------

    # Setting optimal number for first (greedy) phase of matching
    number_of_closet_matches_greedy = 15 * number_of_closest_matches

    def get_distance(df):
        # Return distance in meters between (lat1, lon1) and (lat2, lon2).
        return df.apply(
            lambda row: distance(
                (row["lat"], row["lon"]),
                (row["nearest_match_lat"], row["nearest_match_lon"])
            ).meters,
            axis=1
        )

    def latlon_to_unit_xyz(lat_lon_array):
        """
        Convert an array of [lat, lon] in degrees to 3D unit-sphere Cartesian coordinates.
        This makes KDTree neighbour search wrap correctly on the sphere, including:
        - longitude wrapping at +/-180°
        - stable behaviour near the poles
        """
        lat_rad = np.deg2rad(lat_lon_array["lat"].to_numpy()) # captures degrees in radians dimensions, e.g. [0,0,90] -> [0,0,pi/2]
        lon_rad = np.deg2rad(lat_lon_array["lon"].to_numpy())

        cos_lat = np.cos(lat_rad) # cos(0) = 1 and cos(+-90)=0
        x = cos_lat * np.cos(lon_rad) # locate longitude
        y = cos_lat * np.sin(lon_rad) # locate longitude
        z = np.sin(lat_rad) # z = 1 is at poles

        return np.column_stack((x, y, z))

    # Extract List A coordinates
    sample_coords = df_sample[["lat", "lon"]]
    sample_xyz = latlon_to_unit_xyz(sample_coords)

    new_df = []
    used_indices = set()

    for i in range(number_of_closet_matches_greedy):
        if i == 0:
            df_candidates = df_out
        else:
            df_candidates = df_out
            df_candidates = df_candidates[~df_candidates.index.isin(used_indices)].copy()

        if df_candidates.empty:
            break

        df_coords = df_candidates[["lat", "lon"]]
        df_xyz = latlon_to_unit_xyz(df_coords)

        df_temp = df_sample.copy()

        tree = KDTree(df_xyz)

        # if the number of selected sample coordinates becomes smaller than the pool of remaining candidates, then the system saturates and terminates
        k = min(10, len(df_candidates))
        distances, pos = tree.query(sample_xyz, k=k)

        chosen_pos = []
        chosen_distances = []

        for p_row, d_row in zip(pos, distances):
            if np.isscalar(p_row):
                p_row = [p_row]
                d_row = [d_row]

            for candidate_pos, candidate_dist in zip(p_row, d_row):
                real_idx = df_candidates.index[candidate_pos]
                if real_idx not in used_indices:
                    chosen_pos.append(candidate_pos)
                    chosen_distances.append(candidate_dist)  # KD-space proxy distance in 3D chord space
                    used_indices.add(real_idx)
                    break
            else:
                chosen_pos.append(None)
                chosen_distances.append(None)

        valid_mask = [p is not None for p in chosen_pos]  # masks only successful matches
        df_temp = df_temp[valid_mask].copy()  # This keeps only the rows of df_temp that got a match.
        valid_chosen_pos = [p for p, valid in zip(chosen_pos, valid_mask) if valid]
        valid_chosen_distances = [d for d, valid in zip(chosen_distances, valid_mask) if valid]

        df_temp["nearest_match_lon"] = df_candidates.iloc[valid_chosen_pos]["lon"].values
        df_temp["nearest_match_lat"] = df_candidates.iloc[valid_chosen_pos]["lat"].values
        df_temp["og_index"] = df_candidates.iloc[valid_chosen_pos].index.values
        df_temp["nearest_KDTree_distance"] = valid_chosen_distances
        df_temp["rank"] = i + 1

        new_df.append(df_temp)

    new_df = pd.concat(new_df, ignore_index=True)
    new_df["advanced_distance_meters"] = get_distance(new_df)  # Exact geodesic distance in meters

    # --------------------------------------------------
    # 2.2. Visualise (greedy)
    # --------------------------------------------------

    if visualisation:
        df_out_vis_greedy = (
            df_out[~df_out.index.isin(used_indices)]
            .copy()
        )

        # figure, interactive, spotlighting spatial position of both in-sample and out of sample catches
        fig = px.scatter_map(df_out_vis_greedy,
                             lat=df_out_vis_greedy.lat,
                             lon=df_out_vis_greedy.lon,
                             hover_name=df_out_vis_greedy.index,
                             zoom=1
                             )
        fig.update_traces(name="All catches",
                          hovertemplate="<b>%{hovertext}</b><br>Latitude: %{lat}<br>Longitude: %{lon}<extra></extra>",
                          showlegend=True)

        fig.add_trace(go.Scattermap(
            lat=df_sample.lat,
            lon=df_sample.lon,
            mode='markers',
            marker=go.scattermap.Marker(
                size=10,
                color='rgb(255, 0, 0)'
            ),
            name=f'Selected coordinate(s)',
            text=df_sample.index.astype(str),
            hovertemplate="<b>%{text}</b><br>Latitude: %{lat}<br>Longitude: %{lon}<extra></extra>"
        ))

        fig.add_trace(go.Scattermap(
            lat=new_df.nearest_match_lat,
            lon=new_df.nearest_match_lon,
            mode='markers',
            marker=go.scattermap.Marker(
                size=9,
                color=new_df["rank"],
                colorscale=[
                    [0.0, 'rgb(255, 220, 0)'],  # darkest for best rank
                    [1.0, 'rgb(255, 251, 0)']  # brightest for worse rank
                ],
                showscale=False,
                cmin=new_df["rank"].min(),
                cmax=new_df["rank"].max(),
                # colorbar=dict(title='Rank')
            ),
            name='Nearest matches',
            text=new_df["og_index"],
            customdata=new_df[["rank", "advanced_distance_meters"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude: %{lat}<br>"
                "Longitude: %{lon}<br>"
                "Rank: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:,.0f} m"
                "<extra></extra>"
            )
        ))

        fig.add_trace(go.Scattermap(
            lat=new_df[new_df['rank'] <= number_of_closest_matches].nearest_match_lat,
            lon=new_df[new_df['rank'] <= number_of_closest_matches].nearest_match_lon,
            mode='markers',
            marker=go.scattermap.Marker(
                size=9,
                color=new_df["rank"],
                colorscale=[
                    [0.0, 'rgb(255, 147, 0)'],  # brightest for best rank
                    [1.0, 'rgb(255, 147, 0)']  # darker for worse rank
                ],
                showscale=False,
                cmin=new_df["rank"].min(),
                cmax=new_df["rank"].max(),
                # colorbar=dict(title='Rank')
            ),
            name=f'Nearest {number_of_closest_matches} matches',
            text=new_df["og_index"],
            customdata=new_df[["rank", "advanced_distance_meters"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude: %{lat}<br>"
                "Longitude: %{lon}<br>"
                "Rank: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:,.0f} m"
                "<extra></extra>"
            )
        ))

        modelcost_greedy = new_df[new_df['rank'] <= number_of_closest_matches]['advanced_distance_meters'].sum()

        fig.add_annotation(
            text=f"Data points: {len(dataframe)}. Model cost ({number_of_closest_matches} matches): {modelcost_greedy:,.2f}",
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            align="left"
        )

        fig.update_layout(
            legend_title_text="Model 1 (greedy, fast algorithm)",
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=80)
        )

        fig.update_traces(showlegend=True)

        fig.show()

    # --------------------------------------------------
    # 3. Find the nearest catches (Hungarian algorithm)
    # --------------------------------------------------

    # --------------------------------------------------
    # 3.1. Massage both df to create cost function and appropriate data structure for the Hungarian algorithm
    # --------------------------------------------------

    df_out.columns = ['lon', 'lat']
    df_out = (
        df_out[df_out.index.isin(new_df['og_index'])]
        .rename(columns={'lon': 'lon_nearest', 'lat': 'lat_nearest'})
        .reset_index(drop=False)
        .rename(columns={'index': 'og_index'})
    )
    # Get all combinations
    result = (
        df_sample
        .reset_index(drop=False)
        .rename(columns={'index': 'og_index_sample'})
        .merge(df_out, how="cross")
    )

    def get_distance_hungarian_cost_function(df_to_be_transformed):
        # Return distance in meters between (lat1, lon1) and (lat2, lon2).
        return df_to_be_transformed.apply(
            lambda row: distance(
                (row["lat"], row["lon"]),
                (row["lat_nearest"], row["lon_nearest"])
            ).meters,
            axis=1
        )

    result["cost"] = get_distance_hungarian_cost_function(result)  # Converting degrees to meters (exact: Vincenty’s ellipsoidal formula)
    df_cost_function = result.copy().loc[:, ['og_index_sample','og_index','cost']]

    # --------------------------------------------------
    # 3.2. Hungarian algorithm from scipy
    # --------------------------------------------------

    used_indices_hun = set()
    result_hun = []

    for i in range(number_of_closest_matches):
        if i == 0:
            df_pivot = df_cost_function.pivot_table(
                index='og_index_sample',
                columns='og_index',
                values='cost'
            )
            cost_matrix = df_pivot.to_numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            df_cost_function_plus = df_cost_function[~df_cost_function.og_index.isin(used_indices_hun)].copy()
            df_pivot = df_cost_function_plus.pivot_table(
                index='og_index_sample',
                columns='og_index',
                values='cost'
            )
            cost_matrix = df_pivot.to_numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

        if df_pivot.empty:
            break

        assigned_rows = []
        assigned_cols = []
        assigned_costs = []

        for r, c in zip(row_ind, col_ind):
            actual_col = df_pivot.columns[c]
            actual_row = df_pivot.index[r]

            if actual_col not in used_indices_hun:
                used_indices_hun.add(actual_col)
                assigned_rows.append(actual_row)
                assigned_cols.append(actual_col)
                assigned_costs.append(cost_matrix[r, c])

        if assigned_rows:
            df_to_be_appended = pd.DataFrame({
                'og_index_sample': assigned_rows,
                'og_index': assigned_cols,
                'advanced_distance_meters': assigned_costs,
                'rank': i + 1
            })
        else:
            break

        result_hun.append(df_to_be_appended)


    result_hun = pd.concat(result_hun, ignore_index=True)
    df_sample_ad = df_sample.reset_index(drop=False).rename(columns={'index': 'og_index_sample'})
    result_hun = result_hun.merge(df_sample_ad, on='og_index_sample', how='inner')
    result_hun = result_hun.merge(df_out, on='og_index', how='inner')

    # --------------------------------------------------
    # 3.3. Visualise (final, globally optimal)
    # --------------------------------------------------

    if visualisation:
        df_out_vis_hun = (
            df_out[~df_out.og_index.isin(used_indices_hun)]
            .copy()
        )

        # figure, interactive, spotlighting spatial position of both in-sample and out of sample catches
        fig = px.scatter_map(df_out_vis_hun,
                             lat=df_out_vis_hun.lat_nearest,
                             lon=df_out_vis_hun.lon_nearest,
                             hover_name=df_out_vis_hun.og_index,
                             zoom=1
                             )
        fig.update_traces(name="Fuzzy Model 1 matches",
                          marker=dict(color='rgb(255, 220, 0)', size=8),
                          hovertemplate="<b>%{hovertext}</b><br>Latitude: %{lat}<br>Longitude: %{lon}<extra></extra>",
                          showlegend=True)

        fig.add_trace(go.Scattermap(
            lat=df_sample.lat,
            lon=df_sample.lon,
            mode='markers',
            marker=go.scattermap.Marker(
                size=10,
                color='rgb(255, 0, 0)'
            ),
            name=f'Selected coordinate(s)',
            text=df_sample.index.astype(str),
            hovertemplate="<b>%{text}</b><br>Latitude: %{lat}<br>Longitude: %{lon}<extra></extra>"
        ))
        fig.add_trace(go.Scattermap(
            lat=result_hun.lat_nearest,
            lon=result_hun.lon_nearest,
            mode='markers',
            marker=go.scattermap.Marker(
                size=9,
                color=result_hun["rank"],
                colorscale=[
                    [0.0, 'rgb(255, 147, 0)'],
                    [1.0, 'rgb(255, 147, 0)']
                ],
                showscale=False,
                cmin=result_hun["rank"].min(),
                cmax=result_hun["rank"].max()
            ),
            name=f'Nearest {number_of_closest_matches} matches',
            text=result_hun["og_index"],
            customdata=result_hun[["rank", "advanced_distance_meters"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latitude: %{lat}<br>"
                "Longitude: %{lon}<br>"
                "Rank: %{customdata[0]}<br>"
                "Distance: %{customdata[1]:,.0f} m"
                "<extra></extra>"
            )
        ))

        modelcost_hun = result_hun['advanced_distance_meters'].sum()

        fig.add_annotation(
            text=f"Data points: {len(dataframe)}. Model cost ({number_of_closest_matches} matches): {modelcost_hun:,.2f}",
            x=0.01,
            y=0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            align="left"
        )

        fig.update_layout(
            legend_title_text="Model 2 (Hungarian, optimal algorithm)",
            showlegend=True,
            margin=dict(l=20, r=20, t=40, b=80)
        )
        fig.update_traces(showlegend=True)
        fig.show()

    # --------------------------------------------------
    # 4. Return selected closest matches
    # --------------------------------------------------
    final_df_to_be_returned = (
        result_hun[
        ['og_index_sample', 'lat', 'lon', 'rank', 'og_index', 'lat_nearest', 'lon_nearest','advanced_distance_meters']]
        .rename(columns={'advanced_distance_meters': 'distance_meters', 'og_index_sample': 'selected_id_og_index'})
        .sort_values(by=['selected_id_og_index', 'rank'], ascending=True)
        .reset_index(drop=True)
    )

    return final_df_to_be_returned
