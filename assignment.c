#include <xmmintrin.h>
#include <pmmintrin.h>
#include <stdio.h>
#include <Python.h>
#include <intrin.h>

static void transform_vertices_simd(float *vertex_buffer, size_t number_of_cordinates, const float *transformation_matrix)
{
    __m128 *tm_r1 = (__m128 *)(transformation_matrix);
    __m128 *tm_r2 = (__m128 *)(transformation_matrix + 4);
    __m128 *tm_r3 = (__m128 *)(transformation_matrix + 8);

    __m128 x_new;
    __m128 y_new;
    __m128 z_new;

    float *xp = (float *)&x_new;
    float *yp = (float *)&y_new;
    float *zp = (float *)&z_new;

    int i = 0;
    while (i != number_of_cordinates)
    {
        float temp = vertex_buffer[i + 3];
        vertex_buffer[i + 3] = 1.0;
                                    
        __m128 *vertex = (__m128 *)(vertex_buffer + i);

        x_new = _mm_mul_ps(*vertex, *tm_r1);
        y_new = _mm_mul_ps(*vertex, *tm_r2);
        z_new = _mm_mul_ps(*vertex, *tm_r3);

        x_new = _mm_hadd_ps(x_new, y_new);
        z_new = _mm_hadd_ps(z_new, z_new);

        vertex_buffer[i + 0] = xp[0] + xp[1];
        vertex_buffer[i + 1] = xp[2] + xp[3];
        vertex_buffer[i + 2] = zp[0] + zp[1];

        vertex_buffer[i + 3] = temp;
        i += 3;
    }
}

//__m128 vertex = _mm_setr_ps(vertex_buffer[i+0], vertex_buffer[i+1], vertex_buffer[i+2], 1.0);

static PyObject *transform_vertices(PyObject *self, PyObject *args)
{
    PyObject *vtxarray_obj;
    PyObject *mat4_obj;
    Py_buffer vtxarray_buf;
    Py_buffer mat4_buf;
    // Try to read two arguments: the vertex array and the transformation matrix
    if (!PyArg_ParseTuple(args, "OO", &vtxarray_obj, &mat4_obj))
        return NULL;
    // Try to read the objects as contiguous buffers
    if (PyObject_GetBuffer(vtxarray_obj, &vtxarray_buf,
                           PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1 ||
        PyObject_GetBuffer(mat4_obj, &mat4_buf,
                           PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    // Check that the vertex buffer has one dimension and contains floats ("f")
    if (vtxarray_buf.ndim != 1 || strcmp(vtxarray_buf.format, "f") != 0)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Vertex buffer must be a flat array of floats");
        PyBuffer_Release(&vtxarray_buf);
        return NULL;
    }
    // Check that the transformation matrix buffer has 16 float elements
    if (mat4_buf.ndim != 1 || mat4_buf.shape[0] != 16 ||
        strcmp(mat4_buf.format, "f") != 0)
    {
        PyErr_SetString(PyExc_TypeError,
                        "Transformation matrix must be an array of 16 floats");
        PyBuffer_Release(&mat4_buf);
        return NULL;
    }
    // Call the actual processing function
    transform_vertices_simd(vtxarray_buf.buf, vtxarray_buf.shape[0], mat4_buf.buf);
    // Clean up resources and return to Python code
    PyBuffer_Release(&vtxarray_buf);
    PyBuffer_Release(&mat4_buf);
    Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit_assignment(void)
{
    static PyMethodDef module_methods[] = {{"transform_vertices",
                                            (PyCFunction)transform_vertices,
                                            METH_VARARGS,
                                            "Transform vertex array"},
                                           {NULL}};
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "assignment",
        "This module contains my assignment code",
        -1,
        module_methods};
    return PyModule_Create(&module_def);
}
