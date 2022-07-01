import random
import laspy
import logging
import numpy as np
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import json


class PointCloud:
    filename: str  # The name of the given LAS file
    nb_points: int  # The amount of points in the dataset
    scale_x: int  # The point cloud scale on the X axis
    scale_y: int  # The point cloud scale on the Y axis
    scale_z: int  # The point cloud scale on the Z axis
    offset_x: int  # The position offset of the point cloud on the X axis
    offset_y: int  # The position offset of the point cloud on the Y axis
    offset_z: int  # The position offset of the point cloud on the Z axis
    points: np.ndarray  # The list of points shaped like [[x,y,z],...]
    clusters: np.ndarray  # The list of clusters found in the dataset

    def __init__(self, filename: str, points_proportion: float = 0.5):
        """
        Builds a PointCloud instance with a given file as input
        :param filename: The name of the point cloud file to read
        :param points_proportion: The percentage (between 0 and 1) of points to read in the dataset
        """
        # Setup logging config
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        #  Saves the filename
        self.filename = filename

        # Validates the input
        if 0 > points_proportion > 1:
            raise ValueError("points_proportion must be between 0 and 1")

        # Reads the LAS file
        with laspy.open(filename) as file:
            self.nb_points = int(file.header.point_count * points_proportion)
            self.scale_x = file.header.x_scale
            self.scale_y = file.header.y_scale
            self.scale_z = file.header.z_scale
            self.offset_x = file.header.x_offset * self.scale_x
            self.offset_y = file.header.x_offset * self.scale_y
            self.offset_z = file.header.x_offset * self.scale_z
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

        logging.info(f"Successfully extracted {self.nb_points} from file {self.filename} !")
        return extracted_points

    def __get_camera_targets(self, nb_clusters) -> np.ndarray:
        """
        Calculates the average position of all the points contained in the computed clusters
        :param nb_clusters: The number of clusters to get
        :return: An array containing the x,y,z values for each cluster center
        """
        # TODO Checks if clusters where calculated

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

        return cluster_centers

    def get_epsilon(self):
        # TODO
        logging.info(f"Finding the best parameters for clustering ...")

        #data = np.array([[1, 2], [1, 7], [2, 4], [3, 3],[5, 1], [6, 5]])
        data = self.points
        dks = np.empty(1)
        dks = np.append(dks, PointCloud.get_dks(data, 20))
        print(f"Calculated Dk for k = {20}")

        x = np.array(range(21))
        y = dks
        #fitting = np.polyfit(x=x, y=y, deg=6)

        coef1 = np.polyfit(x, y, 0)
        y1 = np.polyval(coef1, x)
        plt.plot(x, y, '.')
        coef3 = np.polyfit(x, y, 2)
        y3 = np.polyval(coef3, x)
        #derivative = np.polyder(y3)
        #print(derivative)
        plt.plot(x, y3)
        x_der, y_der = PointCloud.first_derivative(x, y3)
        id_closest_to_one = PointCloud.find_closest_element_index(y_der, 0.01)
        dk_closest_to_der = y3[id_closest_to_one]
        plt.show()
        return dk_closest_to_der - 0.5

    @staticmethod
    def get_dks(data, k):
        neigh = NearestNeighbors(n_neighbors=k)
        nbrs = neigh.fit(data)
        distances, _ = nbrs.kneighbors(data)
        d_maxis = np.sort(distances, axis=1)
        averages = np.average(d_maxis, axis=0)
        return averages

    @staticmethod
    def first_derivative(x_data, y_data):
        y_prime = np.diff(y_data) / np.diff(x_data)
        x_prime = np.array([])
        for i in range(len(y_prime)):
            x_temp = (x_data[i+1] + x_data[i]) / 2
            x_prime = np.append(x_prime, x_temp)

        print(x_prime)
        print(y_prime)
        plt.plot(x_prime, y_prime)

        return x_prime, y_prime

    @staticmethod
    def find_closest_element_index(arr, element):
        difference_array = np.absolute(arr - element)
        index = difference_array.argmin()
        return index

    @staticmethod
    def __get_camera_positions(camera_targets: np.ndarray) -> np.ndarray:
        random.seed(datetime.now().timestamp())

        def randomize_position(pos):
            pos[0] += (random.random() - 0.5) * 150  # X
            pos[1] += (random.random() - 0.5) * 150  # Y
            pos[2] += random.random() * 60           # Z
            return pos

        return np.array([randomize_position([t[0], t[1], t[2]]) for t in camera_targets])

    def apply_dbscan(self, epsilon) -> None:
        """
        Applies the DBSCAN data clustering algorithm to identify clusters in the dataset
        :return: Numpy array containing the cluster labels for each given input point
        """
        # Finds the average distance between points
        # logging.info(f"Calculating epsilon value for DBSCAN algorithm.")
        # avg_dist = 0
        # for i in range(len(self.points) - 1):
        #     avg_dist += np.linalg.norm(self.points[i] - self.points[i + 1])
        # avg_dist /= len(self.points)
        # logging.info(f"Computed epsilon value : {avg_dist}")

        # Applies DBSCAN on the points
        logging.info(f"Starting DBSCAN clustering algorithm on {self.filename} ...")
        self.clusters = DBSCAN(eps=epsilon, algorithm='kd_tree', n_jobs=-1).fit_predict(np.array(self.points))
        logging.info("Successfully computed DBSCAN algorithm. The clusters are saved in memory.")

    def write_path_output(self, json_output_file: str, nb_points_of_interest=5):
        """
        Writes an JSON output file with the camera targets and positions that can be used in a visualization tool
        :param json_output_file: The name of the output file
        :param nb_points_of_interest: The number of wanted targets
        """
        # TODO Checks if clusters where calculated
        targets = self.__get_camera_targets(nb_points_of_interest)
        positions = PointCloud.__get_camera_positions(targets)

        # Defines the dictionary object with positions and targets
        data = {
            "positions": positions.tolist(),
            "targets": targets.tolist()
        }

        with open(json_output_file, 'w') as outfile:
            json.dump(data, outfile)

        logging.info(f"Camera targets and positions were saved in file {json_output_file}")

    def generate_debug_image(self, width: int, height: int, zoom_level: int) -> Image:
        """
        Generates an output image
        :param width: The output image width
        :param height: The output image height
        :param zoom_level: The zoom level for debug image generation
        :return: None
        """

        # TODO Verify if clusters where calculated

        def generate_color(seed: int) -> (int, int, int):
            """
            Generates a random color based on the given seed
            :param seed: The seed to generate the color
            :return:
            """
            random.seed(seed)
            r = random.randint(0, 255)
            random.seed(seed + 1)
            g = random.randint(0, 255)
            random.seed(seed + 2)
            b = random.randint(0, 255)
            return r, g, b

        # Creates a 2D picture
        logging.info(f"Creating a new picture of size {width}x{height} px ...")
        output_img = Image.new(mode="RGB", size=(width, height))
        draw = ImageDraw.Draw(output_img)

        # Goes through each point and draws it on the picture
        for i, p in enumerate(self.points):
            # Pass if noise
            if self.clusters[i] == -1:
                continue

            x = int(p[0] * zoom_level) + width / 2 - int(self.offset_x)
            y = int(p[1] * zoom_level) + height / 2 - int(self.offset_y)

            # Gets the cluster color
            r, g, b = generate_color(self.clusters[i])

            # Draws the point
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), (r, g, b, 255))

        logging.info(f"The picture was successfully created !")
        return output_img
