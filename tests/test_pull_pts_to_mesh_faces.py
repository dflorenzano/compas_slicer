import numpy as np
from compas.datastructures import Mesh
from compas.geometry import distance_point_point

from compas_slicer.utilities.utils import pull_pts_to_mesh_faces


def grid(n=20):
    mesh = Mesh.from_meshgrid(dx=100.0, nx=n)
    mesh.quads_to_triangles()  # leaves gaps in the face keys
    return mesh


def test_closest_faces_match_brute_force():
    mesh = grid()
    rng = np.random.default_rng(0)
    points = (rng.random((200, 3)) * [100, 100, 5]).tolist()
    closest_fks, projected = pull_pts_to_mesh_faces(mesh, points)
    for point, fk in zip(points, closest_fks):
        nearest = min(distance_point_point(point, mesh.face_centroid(f)) for f in mesh.faces())
        assert np.isclose(distance_point_point(point, mesh.face_centroid(fk)), nearest)
    assert all(abs(p[2]) < 1e-9 for p in projected)  # flat grid: projections lie in z = 0


def test_noncontiguous_face_keys():
    mesh = grid(4)
    assert sorted(mesh.faces()) != list(range(mesh.number_of_faces()))
    closest_fks, _ = pull_pts_to_mesh_faces(mesh, [[90.0, 90.0, 1.0]])
    assert closest_fks[0] in set(mesh.faces())
