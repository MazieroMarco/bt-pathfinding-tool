import laspy
import vtkmodules.all as vtk
from PIL import Image, ImageDraw
import numpy as np
import sklearn
import scipy

#file = laspy.open("C:\\Users\\mazie\\Downloads\\construction_site.las")
file = laspy.open("../data/construction_site.las")
#file = laspy.open("/Users/marcomaziero/Desktop/example_quarry2_group1_densified_point_cloud.las")
print("Nb points: " + str(file.header.point_count))

reader = file.read()

i = 0
maxIter = 500_000 #file.header.point_count
zoom = 5
scaleX = file.header.x_scale
scaleY = file.header.y_scale
scaleZ = file.header.z_scale
offsetX = file.header.x_offset * scaleX
offsetY = file.header.y_offset * scaleY

width = 15000
height = 15000

print("Creating file of size " +str(width) + "x" + str(height))
img = Image.new(mode="RGB", size=(width, height))
draw = ImageDraw.Draw(img)

for i in range(i, maxIter):
    #print(width * p.X / 2**32 * width)
    #print("Retrieveing point nb.: " + str(int(i * file.header.point_count / maxIter)) + " / " + str(file.header.point_count))
    p = reader.points[int(i * file.header.point_count / maxIter)]

    x = int(p.X * scaleX * zoom) + width / 2 - int(offsetX)
    y = int(p.Y * scaleY * zoom) + height / 2 - int(offsetY)
    r = int(255 * p.red / 2**16)
    g = int(255 * p.green / 2**16)
    b = int(255 * p.blue / 2**16)
    #print("(" + str(x) + ", " + str(y) + ") | " + str(r) + " " + str(g) + " " + str(b))
    if 0 <= x < width and 0 <= y < height:
        #print("(" + str(x) + ", " + str(y) + ") | " + str(r) + " " + str(g) + " " + str(b))
        if i % 1000 == 0:
            print("Computed : " + str(round(i / maxIter * 100, 2)) + "% of points")

        draw.point((x, y), (r, g, b, 255))

for i in range(int(file.header.x_min * scaleX * zoom), int(file.header.x_max * scaleX * zoom)):
    draw.point((i, int(file.header.y_min * scaleY * zoom)), (255, 255, 255, 255))

img.show()
