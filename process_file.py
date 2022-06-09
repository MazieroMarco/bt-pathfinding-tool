from point_cloud import PointCloud

pc = PointCloud(filename="data/construction_site.las", points_proportion=0.1)
pc.apply_dbscan()
image = pc.generate_debug_image(width=2560, height=1440, zoom_level=5)
image.show()