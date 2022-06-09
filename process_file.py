import laspy
import sklearn.cluster as sklearn
import numpy as np
from PIL import Image, ImageDraw
import random

file = laspy.open("../data/construction_site.las")
#file = laspy.open("../data/example_quarry2_group1_densified_point_cloud.las")
print("Nb points: " + str(file.header.point_count))

reader = file.read()

# Variables
nb_points = 2_000_000
zoom = 5
scaleX = file.header.x_scale
scaleY = file.header.y_scale
scaleZ = file.header.z_scale
offsetX = file.header.x_offset * scaleX
offsetY = file.header.y_offset * scaleY
offsetZ = file.header.z_offset * scaleZ


# Select only columns X,Y,Z and only the first nb_points (faster)
print("Shuffling points ...")
all_points = reader.points.array.copy()
np.random.shuffle(all_points)

points_tuples = all_points[['X', 'Y', 'Z']][1:nb_points]
print(points_tuples)

# Converts [(),(),()] into [[],[],[]] to have a shape like (n,m) n=record, m=features (x,y,z)
print("Converting tuples into arrays ...")
points_coords = np.array([list((x[0] * scaleX, x[1] * scaleY, x[2] * scaleZ)) for x in points_tuples])
print(points_coords.shape)
print(points_coords)

# Applies DBSCAN on [[],[],[]]
print("Starting DBSCAN ...")
clusters = sklearn.DBSCAN(eps=0.3, algorithm='kd_tree', n_jobs=-1).fit_predict(points_coords)
print(clusters.shape)
print(np.unique(clusters))
print(clusters)

# Creates a KD tree fr nearest neighbors
# print(points_coords.shape)
# print(points_coords)
# scipy.KDTree(points_coords)
print("All points have been read")

# Generates random color with given seed
def generate_color(seed):
    random.seed(seed)
    r = random.randint(0, 255)
    random.seed(seed + 1)
    g = random.randint(0, 255)
    random.seed(seed + 2)
    b = random.randint(0, 255)
    return r, g, b


# Creates a 2D picture with clusters results
width = 5000
height = 5000

print("Creating file of size " +str(width) + "x" + str(height))
img = Image.new(mode="RGB", size=(width, height))
draw = ImageDraw.Draw(img)

for i in range(0, len(clusters)):
    p = points_coords[i]

    x = int(p[0] * zoom) + width / 2 - int(offsetX)
    y = int(p[1] * zoom) + height / 2 - int(offsetY)
    r, g, b = generate_color(clusters[i])
    if 0 <= x < width and 0 <= y < height:
        draw.point((x, y), (r, g, b, 255))

img.show()