from point_cloud import PointCloud

file = "data/construction_site.las"
#file = "data/example_quarry2_group1_densified_point_cloud.las"
pc = PointCloud(filename=file, points_proportion=0.1)
#pc.apply_dbscan()
#pc.write_path_output("output.json", nb_points_of_interest=15)
pc.get_epsilon()
