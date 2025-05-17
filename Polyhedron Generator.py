import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Generate random 3D points
num_points = 30
min_coordinate = 0
max_coordinate = 100
points = np.random.uniform(min_coordinate, max_coordinate, size=(num_points, 3))

# Generate tetrahedron. Points shouldn't repeat
points_no_extremes = points
min_x_point = min(points_no_extremes, key=lambda p: p[0])
points_no_extremes.remove(min_x_point)
max_x_point = max(points_no_extremes, key=lambda p: p[0])
points_no_extremes.remove(max_x_point)
min_y_point = min(points_no_extremes, key=lambda p: p[1])
points_no_extremes.remove(min_y_point)
max_y_point = max(points_no_extremes, key=lambda p: p[1])
points_no_extremes.remove(max_y_point)
min_z_point = min(points_no_extremes, key=lambda p: p[2])
points_no_extremes.remove(min_z_point)
max_z_point = max(points_no_extremes, key=lambda p: p[2])
points_no_extremes.remove(max_z_point)
extreme_points = [min_x_point, max_x_point, min_y_point, max_y_point, min_z_point, max_z_point]


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot interior points (not part of the convex hull)
interior_indices = [i for i in range(len(points))]
ax.scatter(points[interior_indices, 0],
           points[interior_indices, 1],
           points[interior_indices, 2],
           color='gray', alpha=0.4, s=20, label='Interior Points')

# # Plot hull vertices
# ax.scatter(points[hull_vertex_indices, 0],
#            points[hull_vertex_indices, 1],
#            points[hull_vertex_indices, 2],
#            color='red', s=30, label='Hull Vertices')

# # Plot convex hull faces with transparency and edge lines
# faces = [points[simplex] for simplex in hull.simplices]
# poly3d = Poly3DCollection(faces, facecolors='skyblue', linewidths=0.8, edgecolors='navy', alpha=0.4)
# ax.add_collection3d(poly3d)

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

