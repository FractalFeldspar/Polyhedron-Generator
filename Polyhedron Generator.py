import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def tetrahedron_volume(the_points, A, B, C, D):
    AB = np.array(the_points[B]) - np.array(the_points[A])
    AC = np.array(the_points[C]) - np.array(the_points[A])
    AD = np.array(the_points[D]) - np.array(the_points[A])
    volume = abs(np.dot(AB, np.cross(AC, AD))) / 6
    return volume

# Generate random 3D points
num_points = 30
min_coordinate = 0
max_coordinate = 100
points = [[random.uniform(min_coordinate, max_coordinate) for _ in range(3)] for _ in range(num_points)]
# points = np.random.uniform(min_coordinate, max_coordinate, size=(num_points, 3))
print("points: ", points)

# Generate tetrahedron. Points shouldn't repeat
points_no_extremes = list(enumerate(points))
print(points_no_extremes)
min_x_point = min(points_no_extremes, key=lambda p: p[1][0])[0]
print("min_x_point: ", min_x_point)
print("min_x_point actual value: ", points[min_x_point])
points_no_extremes = [item for item in points_no_extremes if item[0] != min_x_point]
# print("points_no_extreme: ", points_no_extremes)
max_x_point = max(points_no_extremes, key=lambda p: p[1][0])[0]
points_no_extremes = [item for item in points_no_extremes if item[0] != max_x_point]
min_y_point = min(points_no_extremes, key=lambda p: p[1][1])[0]
points_no_extremes = [item for item in points_no_extremes if item[0] != min_y_point]
max_y_point = max(points_no_extremes, key=lambda p: p[1][1])[0]
points_no_extremes = [item for item in points_no_extremes if item[0] != max_y_point]
min_z_point = min(points_no_extremes, key=lambda p: p[1][2])[0]
points_no_extremes = [item for item in points_no_extremes if item[0] != min_z_point]
max_z_point = max(points_no_extremes, key=lambda p: p[1][2])[0]
points_no_extremes = [item for item in points_no_extremes if item[0] != max_z_point]
print("points_no_extreme: ", points_no_extremes)
extreme_points = [min_x_point, max_x_point, min_y_point, max_y_point, min_z_point, max_z_point]
print("extreme points: ", extreme_points)
print("points: ", points)
max_vol = 0
best_combo = None
for combo in itertools.combinations(extreme_points, 4):
    vol = tetrahedron_volume(points, *combo)
    if vol > max_vol:
        max_vol = vol
        best_combo = combo

print("max_vol: ", max_vol)
print("best_combo: ", best_combo)
# print(type(best_combo))
faces = [list(face) for face in itertools.combinations(best_combo, 3)]
print("faces: ", faces)


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot interior points (not part of the convex hull)
interior_indices = [i for i in range(len(points))]
x_vals = [points[i][0] for i in interior_indices]
y_vals = [points[i][1] for i in interior_indices]
z_vals = [points[i][2] for i in interior_indices]
ax.scatter(x_vals, y_vals, z_vals,
           color='gray', alpha=0.4, s=20, label='Interior Points')

# # Plot hull vertices
# # Make sure you rewrite this code to work with Python lists, not numpy arrays
# ax.scatter(points[hull_vertex_indices, 0],
#            points[hull_vertex_indices, 1],
#            points[hull_vertex_indices, 2],
#            color='red', s=30, label='Hull Vertices')

# Plot convex hull faces with transparency and edge lines
faces = [[points[point] for point in face] for face in faces]
poly3d = Poly3DCollection(faces, facecolors='skyblue', linewidths=0.8, edgecolors='navy', alpha=0.4)
ax.add_collection3d(poly3d)

# Better lighting and axes
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=20, azim=30)  # Starting camera angle

plt.legend(loc='upper right')
plt.title("Convex Polyhedron Hull")
plt.tight_layout()
plt.show()






# import numpy as np
# from scipy.spatial import ConvexHull
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # Generate random 3D points
# num_points = 30
# points = np.random.rand(num_points, 3)

# # Compute convex hull
# hull = ConvexHull(points)

# # Identify which points are on the convex hull
# hull_vertex_indices = np.unique(hull.simplices.flatten())
# interior_indices = [i for i in range(len(points)) if i not in hull_vertex_indices]

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# # Plot interior points (not part of the convex hull)
# ax.scatter(points[interior_indices, 0],
#            points[interior_indices, 1],
#            points[interior_indices, 2],
#            color='gray', alpha=0.4, s=20, label='Interior Points')

# # Plot hull vertices
# ax.scatter(points[hull_vertex_indices, 0],
#            points[hull_vertex_indices, 1],
#            points[hull_vertex_indices, 2],
#            color='red', s=30, label='Hull Vertices')

# # Plot convex hull faces with transparency and edge lines
# faces = [points[simplex] for simplex in hull.simplices]
# poly3d = Poly3DCollection(faces, facecolors='skyblue', linewidths=0.8, edgecolors='navy', alpha=0.4)
# ax.add_collection3d(poly3d)

# # Better lighting and axes
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_box_aspect([1, 1, 1])
# ax.view_init(elev=20, azim=30)  # Starting camera angle

# plt.legend(loc='upper right')
# plt.title("Convex Hull with Enhanced 3D Clarity")
# plt.tight_layout()
# plt.show()

