from point_cloud import PointCloud

#file = "data/construction_site.las"
file = "data/example_quarry2_group1_densified_point_cloud.las"
pc = PointCloud(filename=file, points_proportion=0.1)
epsilon = pc.get_epsilon()
print(f"Recommended epsilon value : {epsilon}")
pc.apply_dbscan(epsilon)
pc.write_path_output("output.json", nb_points_of_interest=10)
#img = pc.generate_debug_image(2560, 1440, 5)
#img.show()
