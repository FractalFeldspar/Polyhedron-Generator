import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh
from stl import Mode  # for specifying ASCII
from datetime import datetime

class Face:
    face_id = 0

    def __init__(self, vertices):
        """
        vertices: list of 3 point indices (integers)
        """
        self.vertices = vertices
        self.face_id = Face.face_id
        self.neighbors = []
        self.visible_points = []
        self.normal = None
        self.offset = None
        Face.face_id += 1

    def is_point_above(self, point, points):
        """
        point: list containing coordinates of the point (not the index of the point)
        Returns True if point is on the normal side of the face
        """
        a, b, c = (points[i] for i in self.vertices)
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n = n / np.linalg.norm(n)
        self.normal = n
        self.offset = np.dot(n, a) # this is the distance from the coordinate system origin to the face
        return np.dot(self.normal, point) > self.offset + 1e-8

    def norm_dist_from_face(self, point_idx, points):
        point = points[point_idx]
        return np.dot(self.normal, point) - self.offset

    def add_neighbor_face(self, neighbor_face_id):
        self.neighbors.append(neighbor_face_id)

    @staticmethod
    def are_vertices_counterclockwise(vertex_indices, origin, points):
        a, b, c = [points[vertex_index] for vertex_index in vertex_indices]
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n = n / np.linalg.norm(n)
        normal = n
        offset = np.dot(n, a)
        if np.dot(normal, np.array(origin)) > offset:
            return False
        else:
            return True

    @classmethod
    def how_many(cls):
        return cls.count

    @staticmethod
    def reverse_vertex_order(vertex_indices):
        a, b, c = vertex_indices
        return [c, b, a]

class Point:
    def __init__(self, index):
        self.index = index
        self.visible_faces = []

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
points = np.array([[random.uniform(min_coordinate, max_coordinate) for _ in range(3)] for _ in range(num_points)])
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
# print("points_no_extreme: ", points_no_extremes)
extreme_points = [min_x_point, max_x_point, min_y_point, max_y_point, min_z_point, max_z_point]
print("extreme points: ", extreme_points)
# print("points: ", points)
max_vol = 0
best_combo = None
for combo in itertools.combinations(extreme_points, 4):
    vol = tetrahedron_volume(points, *combo)
    if vol > max_vol:
        max_vol = vol
        best_combo = combo
print("best_combo: ", best_combo)
best_combo_values = [points[idx] for idx in best_combo]
# print("best_combo_values: ", best_combo_values)
best_combo_x, best_combo_y, best_combo_z = zip(*best_combo_values)
# print("best_combo_x: ", best_combo_x)
# print("best_combo_y: ", best_combo_y)
tetrahedron_center = [sum(best_combo_x)/len(best_combo_x), sum(best_combo_y)/len(best_combo_y), sum(best_combo_z)/len(best_combo_z)]
print("tetrahedron_center: ", tetrahedron_center)


print("max_vol: ", max_vol)
# print(type(best_combo))
faces_list = [list(face) for face in itertools.combinations(best_combo, 3)]
print("faces_list: ", faces_list)
# Make sure the points are arranged in counterclockwise order
for idx, face in enumerate(faces_list):
    if not Face.are_vertices_counterclockwise(face, tetrahedron_center, points):
        new_face = Face.reverse_vertex_order(face)
        faces_list[idx] = new_face
print("faces_list, all counterclockwise: ", faces_list)
face_objects = {}
point_objects = {}
for idx, face in enumerate(faces_list):
    new_face = Face(face)
    for face_idx in range(len(faces_list)):
        if face_idx != idx:
            new_face.add_neighbor_face(face_idx)
    visible_points = [point_idx for point_idx in range(len(points)) if new_face.is_point_above(points[point_idx], points)]
    visible_points.sort(key=lambda p: new_face.norm_dist_from_face(p, points), reverse=True)
    new_face.visible_points = visible_points
    face_objects[new_face.face_id] = new_face
    # turn the visible point into an object if it isn't an object already. Also update its visible_faces if necessary
    for visible_point in visible_points:
        if visible_point in point_objects:
            point_objects[visible_point].visible_faces.append(new_face.face_id)
        else:
            new_point_object = Point(visible_point)
            new_point_object.visible_faces.append(new_face.face_id)
            point_objects[visible_point] = new_point_object            
            
print("face objects: ", [[face_object.face_id, face_object.vertices, face_object.neighbors, face_object.visible_points] for key, face_object in face_objects.items()])
print("point objects: ", [[point_object.index, point_object.visible_faces] for key, point_object in point_objects.items()])

face_object_keys = list(face_objects.keys())
print("face_object_keys: ", face_object_keys)

while face_object_keys:
    face_key = face_object_keys.pop(0)
    face = face_objects[face_key]
    if len(face.visible_points) > 0:
        new_vertex = face.visible_points[0]
        seen_faces = set(point_objects[new_vertex].visible_faces)
        neighbor_faces = set()
        seen_face_boundary = []
        for seen_face in seen_faces:           
            seen_face_neighbors = face_objects[seen_face].neighbors
            neighbor_faces.update(seen_face_neighbors)
            a, b, c = face_objects[seen_face].vertices
            if [b, a] in seen_face_boundary:
                seen_face_boundary.remove([b, a])
            else:
                seen_face_boundary.append([a, b])
            if [c, b] in seen_face_boundary:
                seen_face_boundary.remove([c, b])
            else:
                seen_face_boundary.append([b, c])
            if [a, c] in seen_face_boundary:
                seen_face_boundary.remove([a, c])
            else:
                seen_face_boundary.append([c, a])
        # Make sure seen_face_boundary edges are arranged in continuous order
        seen_face_boundary_copy = seen_face_boundary
        seen_face_boundary = [seen_face_boundary_copy[0]]
        seen_face_boundary_copy.pop(0)
        # while seen_face_boundary_copy:
        #     current_point = seen_face_boundary[-1][1]
        #     for edge in seen_face_boundary_copy:
        #         if edge[0] == current_point:
        #             seen_face_boundary.append(edge)
        #             seen_face_boundary_copy.remove(edge)
        while seen_face_boundary_copy:
            current_point = seen_face_boundary[-1][1]
            matched = False
            for edge in seen_face_boundary_copy:
                if edge[0] == current_point:
                    seen_face_boundary.append(edge)
                    seen_face_boundary_copy.remove(edge)
                    matched = True
                    break
            if not matched:
                print("Could not find a matching edge to continue boundary stitching")
                break
        neighbor_faces = neighbor_faces - seen_faces # make sure neighbor_faces and seen_faces are mutually exclusive
        print("new_vertex: ", new_vertex)
        print("seen faces: ", seen_faces)
        print("neighbor faces: ", neighbor_faces)
        print("face boundary edges: ", seen_face_boundary)

        candidate_points = set() # points that may be on the normal side of the new faces

        for face in seen_faces | neighbor_faces:
            face_object = face_objects[face]
            candidate_points.update(face_object.visible_points)
        print("candidate points: ", candidate_points)

        for face in seen_faces:
            face_objects.pop(face)
            if face in face_object_keys: # need to check this because I already pop the current face out of the keys list at the beginning of the loop
                face_object_keys.remove(face)
            for neighbor_face in neighbor_faces:
                if face in face_objects[neighbor_face].neighbors:
                    face_objects[neighbor_face].neighbors.remove(face)
            for key, point in point_objects.items(): # you can probably limit the search to a subset of the points
                point.visible_faces = [visible_face for visible_face in point.visible_faces if visible_face != face]

        current_face_id = Face.face_id
        lowest_face_id = current_face_id
        highest_face_id = lowest_face_id + len(seen_face_boundary) - 1
        for edge in seen_face_boundary:
            new_face = Face([edge[0], edge[1], new_vertex])
            if current_face_id == lowest_face_id:
                neighbor_face_a = current_face_id + 1
                neighbor_face_b = highest_face_id
            elif current_face_id == highest_face_id:
                neighbor_face_a = lowest_face_id
                neighbor_face_b = current_face_id - 1
            else:
                neighbor_face_a = current_face_id + 1
                neighbor_face_b = current_face_id - 1
            current_face_id += 1
            neighbor_face_c = None
            for neighbor_face in neighbor_faces:
                a, b, c = face_objects[neighbor_face].vertices
                if set([a, b])==set(edge) or set([b, c])==set(edge) or set([c, a])==set(edge):
                    neighbor_face_c = neighbor_face
                    face_objects[neighbor_face].neighbors.append(new_face.face_id)
            new_face.neighbors.extend([neighbor_face_a, neighbor_face_b, neighbor_face_c])

            visible_points = [candidate_point for candidate_point in candidate_points if new_face.is_point_above(points[candidate_point], points)]
            visible_points.sort(key=lambda p: new_face.norm_dist_from_face(p, points), reverse=True)
            new_face.visible_points = visible_points

            for visible_point in visible_points:
                if visible_point in point_objects:
                    point_objects[visible_point].visible_faces.append(new_face.face_id)
                else:
                    new_point_object = Point(visible_point)
                    new_point_object.visible_faces.append(new_face.face_id)
                    point_objects[visible_point] = new_point_object    

            face_objects[new_face.face_id] = new_face
            face_object_keys.append(new_face.face_id)

print("face objects: ", [[face_object.face_id, face_object.vertices, face_object.neighbors, face_object.visible_points] for key, face_object in face_objects.items()])
print("face object keys: ", list(face_objects.keys()))



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
for i in range(len(points)):
    ax.text(x_vals[i] + 0.2, y_vals[i] + 0.2, z_vals[i] + 0.2, str(i), fontsize=8)

# # Plot hull vertices
# # Make sure you rewrite this code to work with Python lists if points is a list
# ax.scatter(points[hull_vertex_indices, 0],
#            points[hull_vertex_indices, 1],
#            points[hull_vertex_indices, 2],
#            color='red', s=30, label='Hull Vertices')

# Plot convex hull faces with transparency and edge lines
faces = [[points[vertex] for vertex in face.vertices] for key, face in face_objects.items()]
poly3d = Poly3DCollection(faces, facecolors='skyblue', linewidths=0.8, edgecolors='navy', alpha=0.4)
ax.add_collection3d(poly3d)
for key, face_object in face_objects.items():
    face_points = [points[vertex] for vertex in face_object.vertices]
    centroid = np.mean(face_points, axis=0)
    ax.text(centroid[0], centroid[1], centroid[2], str(key), color='red', fontsize=8)

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




# Export polyhedron as STL file
triangles = []

for face in face_objects.values():
    a, b, c = face.vertices
    triangle = [points[a], points[b], points[c]]
    triangles.append(triangle)

scale_factor = 0.1
triangles_np = scale_factor * np.array(triangles)

hull_mesh = mesh.Mesh(np.zeros(triangles_np.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(triangles_np):
    hull_mesh.vectors[i] = f

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"convex_hull_polyhedron_{timestamp}.stl"
hull_mesh.save(filename, mode=Mode.ASCII)
print("STL file", filename, "saved")



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

