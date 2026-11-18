# geo-coordinates-proxy-finder
It's a globally optimal algorithm which identifies the nearest geo coordinates (i.e. proxies), with significant time savings due to its two-stage, staggered approach.
1. It identifies potential nearest coordinates, using greedy, suboptimal matching
   with scipy.spatial.KDTree in a Cartesian 3D unit-sphere  
2. It uses scipy.optimize.linear_sum_assignment to select globally optimal nearest coordinates
