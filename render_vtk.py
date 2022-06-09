import laspy
import vtkmodules.all as vtk
from PIL import Image, ImageDraw
import numpy as np

#file = laspy.open("C:\\Users\\mazie\\Downloads\\construction_site.las")
file = laspy.open("/Users/marcomaziero/Desktop/construction_site.las")
print("Nb points: " + str(file.header.point_count))

reader = file.read()

width = 1920
height = 1080

img = Image.new(mode="RGB", size=(width, height))
draw = ImageDraw.Draw(img)

# Renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.12, 0.11, 0.13)

i = 0
maxIter = 10000
zoom = 3

for p in reader.points:

    #print(width * p.X / 2**32 * width)
    x = int(p.X * file.header.x_scale * zoom)# + width / 2
    y = int(p.Y * file.header.y_scale * zoom)# + height / 2
    z = int(p.Z * file.header.z_scale * zoom)
    r = int(255 * p.red / 2**16)
    g = int(255 * p.green / 2**16)
    b = int(255 * p.blue / 2**16)
    I = int(z * 100 / file.header.z_max)

    if 0 <= x < width and 0 <= y < height:
        #print("(" + str(x) + ", " + str(y) + ") | " + str(r) + " " + str(g) + " " + str(b))
        if i % 1000 == 0:
            print("Computed : " + str(round(i / maxIter * 100, 2)) + "% of points")

        point_source = vtk.vtkPointSource()

        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputConnection(point_source.GetOutputPort())

        point_actor = vtk.vtkActor()
        point_actor.SetMapper(point_mapper)
        point_actor.SetPosition(x, y, z)
        renderer.AddActor(point_actor)

    i += 1

    if i >= maxIter:
        break



# Window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(1280, 720)

# interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

style = vtk.vtkInteractorStyleTrackballCamera()
interactor.SetInteractorStyle(style)

interactor.Initialize()
interactor.Start()
