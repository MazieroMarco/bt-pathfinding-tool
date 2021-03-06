import argparse
import json
import logging
import os
import random
import time
from datetime import datetime

import laspy
import numpy as np
import scipy.spatial
from PIL import Image, ImageDraw
from mathutils import Vector
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


# from matplotlib import pyplot as plt


class PointCloud:
    """
    This class represents a processed point cloud las file
    Clustering on the data geometry can be applied to extract relevant clusters
    """

    filename: str  # The name of the given LAS file
    nb_points: int  # The amount of points in the dataset
    header: laspy.LasHeader  # The metadata of the point cloud dataset
    points: np.ndarray  # The list of points shaped like [[x,y,z],...]
    clusters: np.ndarray  # The list of clusters found in the dataset
    epsilon: float  # Optimal epsilon value computed by the algorithm
    clusters_computed: bool  # Boolean that tells if the clusters where computed by the algorithm

    def __init__(self, filename: str, points_proportion: float = 0.5):
        """
        Builds a PointCloud instance with a given file as input
        :param filename: The name of the point cloud file to read
        :param points_proportion: The percentage (between 0 and 1) of points to read in the dataset
        """

        #  Saves the filename
        self.filename = filename

        # Validates the input
        if 0 > points_proportion > 1:
            raise ValueError("points_proportion must be between 0 and 1")

        # Initiates the clusters boolean
        self.clusters_computed = False

        # Reads the LAS file
        with laspy.open(filename) as file:
            self.nb_points = int(file.header.point_count * points_proportion)
            self.header = file.header
            self.points = self.__extract_points(file.read())
            file.close()

    def __extract_points(self, reader: laspy.LasData) -> np.ndarray:
        """
        Extract the (x,y,z) points contained inside the dataset
        :param reader: The point cloud reader opened from the file
        :return: A numpy array containing arrays of coordinates for each point like so -> [[x,y,z],...]
        """
        # Shuffles and selects only the first nb_points (faster)
        logging.info(f"Extracting {self.nb_points} points from file {self.filename}. This may take a while ...")
        rnd_indices = np.random.choice(len(reader.xyz), size=self.nb_points)
        extracted_points = reader.xyz[rnd_indices]

        logging.info(f"Successfully extracted {self.nb_points} points from file {self.filename} !")
        return extracted_points

    def __get_camera_targets(self, nb_clusters) -> np.ndarray:
        """
        Calculates the average position of all the points contained in the computed clusters
        :param nb_clusters: The number of clusters to get
        :return: An array containing the x,y,z values for each cluster center
        """

        # Checks if clusters where computed
        self.__verify_clusters_computed()

        # Retrieves the biggest clusters
        clusters_no_noise = self.clusters[self.clusters != -1]  # Removes noise
        unique, counts = np.unique(clusters_no_noise, return_counts=True)
        clusters_dict = dict(zip(unique, counts))
        sorted_clusters = [k for k, v in sorted(clusters_dict.items(), key=lambda item: item[1], reverse=True)]

        # Calculates clusters centers
        cluster_centers = np.empty((0, 3))
        for i in range(min(nb_clusters, len(sorted_clusters))):
            indices = np.where(self.clusters == sorted_clusters[i])
            cluster_average = np.mean(self.points[indices], axis=0)

            # Adds the center to the array
            cluster_centers = np.append(cluster_centers, np.array([cluster_average]), axis=0)

        # Sort targets by euclidian distance to avoid long movements with camera
        sorted_centers = np.empty((0, 3))
        sorted_centers = np.append(sorted_centers, np.array([cluster_centers[0]]), axis=0)
        cluster_centers = np.array(cluster_centers[cluster_centers != np.array(sorted_centers[0])]).reshape((-1, 3))
        while len(cluster_centers) != 0:
            tree = scipy.spatial.KDTree(cluster_centers)
            closest_center = cluster_centers[tree.query(sorted_centers[-1])[1]]
            sorted_centers = np.append(sorted_centers, np.array([closest_center]), axis=0)
            cluster_centers = np.array(cluster_centers[cluster_centers != np.array(closest_center)]).reshape((-1, 3))

        logging.info(f"{nb_clusters} points of interest have been computed.")
        return sorted_centers

    def get_epsilon(self) -> float:
        """
        Computes an approximation of the epsilon value for the DBSCAN algorithm
        Warning! This method is highly demanding and can take time to finish!
        :return: The approximated epsilon value
        """
        logging.info(f"Finding the best parameters for clustering. This may take a while ...")

        # Config values
        k = 20  # The number of neighbors to evaluate (the bigger, the slower)
        deriv_goal = 1  # The ideal value for the derivative
        corr_factor = -0.05  # The correction factor to apply to the final value

        # Calculates maximum distances average for KNN neighbors
        data = self.points
        dks = np.array([0])  # The first 0 value corresponds to the dk when k = 0 (no neighbors)
        neighbors = NearestNeighbors(n_neighbors=k).fit(data)
        distances, _ = neighbors.kneighbors(data)  # Find distances up to K neighbors
        d_maxis = np.sort(distances, axis=1)  # Sort distances from closest to farthest
        averages = np.average(d_maxis, axis=0)  # Calculates average distances for all columns
        dks = np.append(dks, averages)  # Appends the distances to the dks array

        # Set up the values
        x = np.array(range(k + 1))  # X-axis contains the K values (0 - K)
        y = dks  # Y-axis contains the dk averages for each K value

        # Calculates the fitting function of dks
        fit_coef = np.polyfit(x, y, 2)
        y_fit = np.polyval(fit_coef, x)

        # Get a function that evaluates the linear spline at any x
        f = InterpolatedUnivariateSpline(x, y_fit, k=1)

        # Get a function that evaluates the derivative of the linear spline at any x
        dfdx = f.derivative()

        # Evaluate the derivative dydx at each x location...
        y_der = np.array([])
        for v in x:
            y_der = np.append(y_der, dfdx(v))
        y_der = y_der * 100

        # Finds the value where the derivative is closest to the derivative goal
        difference_array = np.absolute(y_der - deriv_goal)  # Finds the closest value to ideal value for derivative
        id_closest_to_one = difference_array.argmin()

        # Retrieves the dk value where the derivative is close to the derivative goal
        dk_closest_to_der = y_fit[id_closest_to_one] + corr_factor

        # The lines below can be used as debug to output a graphic
        # representing the curve of Dmax averages over K values

        # plt.plot(x, y, ".", label='Dmax average')
        # plt.plot(x, y_fit, label='Fitting function')
        # plt.plot(y_der / 10, label='Derivative values')
        # plt.xticks(x)
        # plt.title("Average of maximum distances by number of neighbors")
        # plt.xlabel("Number of neighbors (K)")
        # plt.ylabel("Average of maximum distances (Dmax)")
        # plt.plot(id_closest_to_one, dk_closest_to_der - corr_factor, ".", label="Approx. of optimal epsilon")
        # plt.legend()
        # plt.show()

        return dk_closest_to_der

    def __get_camera_positions(self, camera_targets: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Computes the positions the camera will go when looking at the targets
        :param camera_targets: The targets the camera will have to look at
        :param epsilon: The epsilon best value (Distance max for two points to be in same cluster)
        :return: A numpy array containing one position per given target
        """
        random.seed(datetime.now().timestamp())

        positions = np.empty((0, 3))  # The array containing the final camera positions
        data = self.points  # The data to use for the KDTree
        tree = scipy.spatial.KDTree(data)  # The KD tree to compute KNN distances

        # Goes through each target
        target_index = 0
        while True:
            # Retrieves the current target
            t = camera_targets[target_index]

            # Generates random camera position
            pos = np.array([t[0], t[1], t[2]])
            pos[0] += (random.random() - 0.5) * 150  # X
            pos[1] += (random.random() - 0.5) * 150  # Y
            pos[2] += random.random() * 45  # Z

            # Lerps through the vector between position and target
            v1 = Vector((pos[0], pos[1], pos[2]))
            v2 = Vector((t[0], t[1], t[2]))

            # Checks for occlusion
            is_occlusion = False
            for i in np.arange(0, 0.8, 0.1):
                interp_pos = v1.lerp(v2, i)

                # Uses KNN to check if there are points around
                distance, _ = tree.query(interp_pos)
                if distance < epsilon * 3:
                    is_occlusion = True
                    break

            # Decides to break or continue the loop
            if is_occlusion:
                continue

            target_index += 1
            positions = np.append(positions, np.array([pos]), axis=0)

            if target_index >= len(camera_targets):
                break
            else:
                continue

        logging.info(f"{len(camera_targets)} camera positions have been computed.")

        return positions

    def __verify_clusters_computed(self):
        """
        Checks if clusters where computed
        :return: False if the clusters where not computed
        """
        if not self.clusters_computed:
            raise RuntimeError("The clusters are not computed, please apply the clustering algorithm.")

    def apply_dbscan(self, epsilon: float) -> None:
        """
        Applies the DBSCAN data clustering algorithm to identify clusters in the dataset
        :param epsilon: The best value for the epsilon parameter (The max distance for two points to be in same cluster)
        :return: Numpy array containing the cluster labels for each given input point
        """

        # Applies DBSCAN on the points
        logging.info(f"Starting DBSCAN clustering algorithm on {self.filename} with epsilon of {epsilon} ...")
        self.clusters = DBSCAN(eps=epsilon, algorithm='kd_tree', n_jobs=-1).fit_predict(np.array(self.points))
        self.clusters_computed = True
        logging.info("Successfully computed DBSCAN algorithm. The clusters are saved in memory.")

    def write_path_output(self, json_output_file: str, epsilon: float, nb_points_of_interest: int = 5) -> Image:
        """
        Writes an JSON output file with the camera targets and positions that can be used in a visualization tool
        :param epsilon: The best value possible for the epsilon parameter
                        (The distance between points for them to be in the same cluster)
        :param json_output_file: The name of the output file
        :param nb_points_of_interest: The number of wanted targets
        """
        # Checks if clusters where computed
        self.__verify_clusters_computed()

        targets = self.__get_camera_targets(nb_points_of_interest)
        positions = self.__get_camera_positions(targets, epsilon)

        # Defines the dictionary object with positions and targets
        data = {
            "positions": positions.tolist(),
            "targets": targets.tolist()
        }

        with open(json_output_file, 'w') as outfile:
            json.dump(data, outfile)

        logging.info(f"Camera targets and positions were saved in file {json_output_file}")

    def generate_debug_image(self, width: int, height: int, zoom_level: float) -> Image:
        """
        Generates an output image representing a top view of the scene clustering
        :param width: The output image width
        :param height: The output image height
        :param zoom_level: The zoom level for debug image generation
        :return: The created image
        """

        # Checks if clusters where computed
        self.__verify_clusters_computed()

        def generate_color(seed: int) -> (int, int, int):
            """
            Generates a random color based on the given seed
            :param seed: The seed to generate the color
            :return:
            """
            random.seed(seed)
            red = random.randint(0, 255)
            random.seed(seed + 1)
            grn = random.randint(0, 255)
            random.seed(seed + 2)
            blu = random.randint(0, 255)
            return red, grn, blu

        # Creates a 2D picture
        logging.info(f"Creating a new picture of size {width}x{height} px ...")
        output_img = Image.new(mode="RGB", size=(width, height))
        draw = ImageDraw.Draw(output_img)

        # Goes through each point and draws it on the picture
        for i, p in enumerate(self.points):
            # Pass if noise
            if self.clusters[i] == -1:
                continue

            x = int(p[0] * zoom_level) + width / 2 - int(self.header.x_offset)
            y = int(p[1] * zoom_level) + height / 2 - int(self.header.y_offset)

            # Gets the cluster color
            r, g, b = generate_color(self.clusters[i])

            # Draws the point
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), (r, g, b, 255))

        logging.info(f"The picture was successfully created !")
        return output_img


if __name__ == "__main__":
    # Execution time calculation
    start_time = time.time()

    # Setup logging config
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


    def dir_path(s):
        """
        Ensures the given string is a valid file path
        :param s: The string to check
        :return: The string if valid. Otherwise, an error
        """
        if os.path.exists(s) or os.access(os.path.dirname(s), os.W_OK):
            return s
        else:
            raise NotADirectoryError(s)


    # Sets up the arguments parser
    parser = argparse.ArgumentParser(description="Finds interesting locations and a camera path "
                                                 "inside a given LAS point cloud data file.")
    parser.add_argument("input",
                        type=dir_path,
                        metavar="INPUT",
                        help="The path of the input LAS data file")
    parser.add_argument("--output", "-o",
                        metavar="DIR",
                        default="./pathfinder_output.json",
                        help="The output directory for the generated JSON file")
    parser.add_argument("--poi", "-p",
                        type=int, metavar="N",
                        default=5,
                        help="The amount of points of interest to output")
    parser.add_argument("--quantity", "-q",
                        type=float,
                        metavar="N",
                        default=0.1,
                        help="The proportion of points to keep in the working data sample [0 < q < 1]. "
                             "Warning, a big number slows down the algorithm.")
    parser.add_argument("--epsilon", "-e",
                        type=float,
                        metavar="N",
                        default=-1,
                        help="The epsilon parameter used for the data clustering. "
                             "This parameter is approximated if no value is given.")

    arguments = parser.parse_args()

    if not 0 < arguments.quantity <= 1:
        logging.warning("The given quantity of points should be between 0 and 1. Taking default value 0.1 instead.")
        arguments.quantity = 0.1

    if not 1 < arguments.poi <= 50:
        logging.warning(
            "The amount of desired points of interest should be between 1 and 50. Computing only 5 points of interest.")
        arguments.poi = 5

    # Executes the pathfinding algorithm
    pc = PointCloud(filename=arguments.input, points_proportion=arguments.quantity)

    if arguments.epsilon <= 0.0:
        arguments.epsilon = pc.get_epsilon()

    pc.apply_dbscan(arguments.epsilon)

    pc.write_path_output(json_output_file=arguments.output,
                         epsilon=arguments.epsilon,
                         nb_points_of_interest=arguments.poi)

    # Logging exec time
    logging.info(f"Algorithm total execution time: {(time.time() - start_time)} seconds")
