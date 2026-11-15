import math
from pathlib import Path

import numpy as np
import pytest

from nyc_drone_sim.dress import pose_matrix, triangulate
from nyc_drone_sim.populate import rects

FLEET = Path(__file__).resolve().parent.parent / "data" / "assets" / "valdrada" / "Content" / "models" / "fleet24"


def test_rects_area_and_heading():
    r = np.atleast_1d(rects(10.0, -5.0, 30.0, 4.0, 2.0))[0]
    assert r.area == pytest.approx(8.0)
    assert (r.centroid.x, r.centroid.y) == pytest.approx((10.0, -5.0))
    # the long side points along the heading
    c = np.asarray(r.exterior.coords)
    long_edge = max((c[i + 1] - c[i] for i in range(4)), key=np.linalg.norm)
    ang = math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180
    assert ang == pytest.approx(30.0)


def test_pose_matrix_is_row_vector_scale_rotate_translate():
    m = np.array(pose_matrix(3.0, 4.0, 90.0, 2.0))
    p = np.array([1.0, 0.0, 0.5, 1.0]) @ m          # one metre ahead of the model, half a metre up
    assert p[:3] == pytest.approx([3.0, 6.0, 1.0])


def test_triangulate_keeps_area_and_faces_up():
    from shapely.geometry import Polygon
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (2, 4), (4, 4), (4, 2)]])
    pts, tris = triangulate([poly], 0.0)
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    signed = 0.5 * ((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
    assert (signed > 0).all()
    assert signed.sum() == pytest.approx(96.0)


@pytest.mark.skipif(not (FLEET / "taxi.glb").exists(), reason="model set not downloaded")
def test_decoded_taxi_wheelbase():
    from nyc_drone_sim.gltf import Glb
    prims = Glb(FLEET / "taxi.glb").primitives(0, extra=("_WHEEL",))
    w = np.concatenate([p["_WHEEL"].ravel() for p in prims if "_WHEEL" in p])
    pts = np.vstack([p["points"] for p in prims if "_WHEEL" in p])
    centre = {int(k): (pts[w == k].max(0) + pts[w == k].min(0)) / 2 for k in np.unique(w) if k > 0}
    z = sorted(c[2] for c in centre.values())
    assert z[-1] - z[0] == pytest.approx(2.913, abs=0.01)      # Ford Crown Victoria wheelbase, metres
    assert all(p["triangles"].max() == len(p["points"]) - 1 for p in prims)
