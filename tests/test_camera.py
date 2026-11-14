import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from nyc_drone_sim.common import load_config
from nyc_drone_sim.paths import Pinhole, camera_to_world, frustum_ground, project


def test_field_of_view():
    h, v = Pinhole(load_config()["camera"]).fov_deg
    assert abs(h - 73.74) < 0.005
    assert abs(v - 45.75) < 0.005


def test_image_centre_hits_ground_at_altitude_ahead():
    cam = Pinhole(load_config()["camera"])
    m = camera_to_world([0, 0, 45.72], 0.0, cam.pitch_deg)
    u, v, front = project(m, cam, (45.72, 0.0, 0.0))
    assert front and abs(u) < 1e-9 and abs(v) < 1e-9
    assert abs(np.linalg.det(m[:3, :3]) - 1) < 1e-12


def test_farthest_ground_matches_brief():
    cam = Pinhole(load_config()["camera"])
    for alt, far in ((45.72, 140), (152.4, 468)):
        g = frustum_ground(camera_to_world([0, 0, alt], 0.0, cam.pitch_deg), cam)
        assert abs(np.hypot(g[:, 0], g[:, 1]).max() - far) < 1.0
        assert abs(alt / math.sin(math.radians(45)) - {45.72: 64.7, 152.4: 215.5}[alt]) < 0.05
