import numpy as np
from matplotlib import pyplot as plt
from pathfinder import PointCloud
import time
import logging

POINT_CLOUD_FILE = 'data/Romont 3D_group1_densified_point_cloud.las'


def perf_time_density():
    logging.disable()
    x = np.arange(0.01, 0.2, 0.2)
    y = np.array([])

    for i in x:
        start_time = time.time()
        pc = PointCloud(filename=POINT_CLOUD_FILE, points_proportion=i)
        e = pc.get_epsilon()
        pc.apply_dbscan(e)
        pc.write_path_output(f'test/perf_time_density_{i*100}.json', 5)
        exec_time = (time.time() - start_time)
        y = np.append(y, exec_time)
        print(f'Computed path with {round(i * 100)}% of the points. Exec. time:\t{exec_time}\tseconds.')

    plt.plot(x, y, ".")
    plt.show()

perf_time_density()