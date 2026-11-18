import pandas as pd
from main_proxies_finder.proxies_finder_main import proxy_finder

"""
    The algorithm provides a globally optimal approach to identifying the nearest coordinates (i.e. proxies), 
    with significant time savings due to its two-stage, staggered approach:
        1. It identifies potential nearest coordinates, using greedy, suboptimal matching
        with scipy.spatial.KDTree in a Cartesian 3D unit-sphere  
        2. It uses scipy.optimize.linear_sum_assignment to select globally optimal nearest coordinates
    
    To parametrise the model, you will need:
        1. dataframe: data frame containing the geographic coordinates of your observations, with
            lon: longitude in (-180, 180)
            lat: latitude in (-90, 90)
        2. number_of_closest_matches: number (int) of closest matches to retrieve for each selected observation.
        3. selected_index: list of selected indices corresponding to the entries for 
                        which you want to find the nearest matches.
        4. visualisation: visualisation flag as boolean, with
            True = yes, show me results in my browser plotted on the Earth's map
            False = no

"""
"""
    Select coordinate(s) which you would like to find proxies for by identifying them by their index position, 
    for instance for 34th entry insert it as int to the list: [33].
    If you want to match multiple entries, then list them like this: [9056,1,1866].
"""
"""
    If you would like to test it first with an artificial dataset, you may use this:
    
        import numpy as np
        # Reproducibility
        np.random.seed(13)
        # Number of random geo coordinates
        n = 10000

        dataframe = pd.DataFrame({
            'lon': np.random.uniform(-180, 180, n),
            'lat': np.random.uniform(-90, 90, n)
        })
"""

dataframe = pd.DataFrame()
number_of_closest_matches = type[int]
selected_index = type[list]
visualisation = type[bool]

if __name__ == '__main__':
    df_with_proxies = proxy_finder(dataframe, number_of_closest_matches, selected_index, visualisation)
    print(f"Best matches for selected coordinates:\n{df_with_proxies}")
