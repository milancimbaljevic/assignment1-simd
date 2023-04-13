import trimesh
import numpy as np
import math
from assignment import transform_vertices as transform_vertices_c_simd


def compare_c_python(vertex_buffer, transformation_matrix):
    vertex_buffer_c = vertex_buffer.copy()
    vertex_buffer_py = vertex_buffer.copy()

    vertex_buffer_py = transform_vertices_python(
        vertex_buffer_py, transformation_matrix)
    transform_vertices_c_simd(vertex_buffer_c, transformation_matrix)

    if len(vertex_buffer_c) != len(vertex_buffer_py):
        print("Error, buffers size don't match!")
        return

    for i in range(len(vertex_buffer_c)):
        if abs(vertex_buffer_c[i] - vertex_buffer_py[i]) > 0.000001:
            print("Error, buffers don't match (", i ,")!")
            print("C:", vertex_buffer_c[i])
            print("Python:", vertex_buffer_py[i])
            return          

    print("Success, buffers match!")


def transform_vertices_python(vertex_buffer, transformation_matrix):
    num_of_cordinates = len(vertex_buffer)
    vertex_buffer = np.append(vertex_buffer, [1.0])

    i = 0
    while i != num_of_cordinates:
        temp = vertex_buffer[i+3]
        vertex_buffer[i+3] = 1.0

        for r in range(3):
            temp1 = 0
            for c in range(4):
                temp1 += transformation_matrix[r*4+c] * vertex_buffer[i+c]
            vertex_buffer[i+r] = temp1

        vertex_buffer[i+3] = temp
        i = i + 3

    vertex_buffer = np.delete(vertex_buffer, len(vertex_buffer) - 1)
    return vertex_buffer


model = trimesh.load("teapot.obj")
# model.show()

s_x = 5
s_y = 1.0
s_z = 1.0

dx = 2
dy = 2
dz = 2

theta_x = math.pi/2

transformation_matrix1 = np.array([
    s_x, 0.0, 0.0, 0.0,
    0.0, s_y, 0.0, 0.0,
    0.0, 0.0, s_z, 0.0,
    0.0, 0.0, 0.0, 1.0
], dtype=np.float32)

transformation_matrix2 = np.array([
    1.0, 0.0, 0.0, dx,
    0.0, 1.0, 0.0, dy,
    0.0, 0.0, 1.0, dz,
    0.0, 0.0, 0.0, 1.0
], dtype=np.float32)

transformation_matrix3 = np.array([
    1.0, 0.0, 0.0, 0.0,
    0.0, math.cos(theta_x), -math.sin(theta_x), 0,
    0.0, math.sin(theta_x), math.cos(theta_x), 0,
    0.0, 0.0, 0.0, 1.0
], dtype=np.float32)


# Extract vertices to a float buffer:
vertex_buffer = np.ravel(model.vertices.view(np.ndarray)).astype(np.float32)
# Transform the vertex buffer with your C processing function:

transform_vertices_c_simd(vertex_buffer, transformation_matrix1)
transform_vertices_c_simd(vertex_buffer, transformation_matrix2)
transform_vertices_c_simd(vertex_buffer, transformation_matrix3)

# compare_c_python(vertex_buffer, transformation_matrix1)
# compare_c_python(vertex_buffer, transformation_matrix2)
# compare_c_python(vertex_buffer, transformation_matrix3)

# vertex_buffer = transform_vertices_python(vertex_buffer, transformation_matrix1)
# vertex_buffer = transform_vertices_python(vertex_buffer, transformation_matrix2)
# vertex_buffer = transform_vertices_python(vertex_buffer, transformation_matrix3)

# Reassign the transformed vertices to the model object and visualize it:
model.vertices = vertex_buffer.reshape((-1, 3))
model.show()
