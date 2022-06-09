import random
import laspy
import logging
import numpy as np
import pandas as pd
import sklearn.cluster as sklearn
from PIL import Image, ImageDraw


class PointCloud:
    filename: str  # The name of the given LAS file
    nb_points: int  # The amount of points in the dataset
    scale_x: int  # The point cloud scale on the X axis
    scale_y: int  # The point cloud scale on the Y axis
    scale_z: int  # The point cloud scale on the Z axis
    offset_x: int  # The position offset of the point cloud on the X axis
    offset_y: int  # The position offset of the point cloud on the Y axis
    offset_z: int  # The position offset of the point cloud on the Z axis
    points: np.ndarray  # The list of points shaped like [[x,y,z,r,g,b],...]
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
            reader = file.read()
            self.nb_points = int(file.header.point_count * points_proportion)
            self.scale_x = file.header.x_scale
            self.scale_y = file.header.y_scale
            self.scale_z = file.header.z_scale
            self.offset_x = file.header.x_offset * self.scale_x
            self.offset_y = file.header.x_offset * self.scale_y
            self.offset_z = file.header.x_offset * self.scale_z
            self.points = self.__extract_points(reader)
            file.close()

    def __extract_points(self, reader: laspy.LasData) -> np.ndarray:
        """
        Extract the (x,y,z) points contained inside the dataset
        :param reader: The point cloud reader opened from the file
        :return: A numpy array containing arrays of coordinates for each point like so -> [[x,y,z,r,g,b],...]
        """
        # Shuffles and selects only columns X,Y,Z and only the first nb_points (faster)
        logging.info(f"Extracting {self.nb_points} points from file {self.filename}. This may take a while ...")
        extracted_points = np.random.choice(reader.points.array, size=self.nb_points)[['X', 'Y', 'Z']]

        # Converts [(),(),()] into [[],[],[]] to have a shape like (n,m) n=record, m=features (x,y,z)
        # TODO This is very inefficient, need to find another way to do it

        # def f(coords):
        #     return np.array([coords[0] * self.scale_x, coords[1] * self.scale_y, coords[2] * self.scale_z])
        #
        # print(extracted_points)
        # print("--------------------")
        # extracted_points = extracted_points.astype([('X', '<i4'), ('Y', '<i4'), ('Z', '<i4')]).view('<i4')
        # extracted_points = np.reshape(extracted_points, (-1, 3))
        # extracted_points = f(extracted_points)
        # print(extracted_points)

        extracted_points = np.array([list((x[0] * self.scale_x, x[1] * self.scale_y, x[2] * self.scale_z)) for x in extracted_points])
        logging.info(f"Successfully extracted {self.nb_points} from file {self.filename} !")

        return extracted_points

    def apply_dbscan(self) -> None:
        """
        Applies the DBSCAN data clustering algorithm to identify clusters in the dataset
        :return: Numpy array containing the cluster labels for each given input point
        """
        # Finds the mean distance between points


        # Applies DBSCAN on the points
        logging.info(f"Starting DBSCAN clustering algorithm on {self.filename} ...")
        self.clusters = sklearn.DBSCAN(eps=0.3, algorithm='kd_tree', n_jobs=-1).fit_predict(np.array(self.points))
        logging.info("Successfully computed DBSCAN algorithm. The clusters are saved in memory.")

    def generate_debug_image(self, width: int, height: int, zoom_level: int) -> Image:
        """
        Generates an output image
        :param width: The output image width
        :param height: The output image height
        :param zoom_level: The zoom level for debug image generation
        :return: None
        """

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
            x = int(p[0] * zoom_level) + width / 2 - int(self.offset_x)
            y = int(p[1] * zoom_level) + height / 2 - int(self.offset_y)

            # Gets the cluster color
            r, g, b = generate_color(self.clusters[i])

            # Draws the point
            if 0 <= x < width and 0 <= y < height:
                draw.point((x, y), (r, g, b, 255))

        logging.info(f"The picture was successfully created !")
        return output_img
