import math

import numpy as np


def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
    return view


def perspective(fov_y, aspect, near, far):
    f = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def rot_x_matrix(theta):
    return np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [
                0.0,
                math.cos(theta),
                -math.sin(theta),
                0.0,
            ],
            [
                0.0,
                math.sin(theta),
                math.cos(theta),
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_y_matrix(theta):
    return np.matrix(
        [
            [
                math.cos(theta),
                0.0,
                -math.sin(theta),
                0.0,
            ],
            [0.0, 1.0, 0.0, 0.0],
            [
                math.sin(theta),
                0.0,
                math.cos(theta),
                0.0,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_z_matrix(theta):
    return np.matrix(
        [
            [
                math.cos(theta),
                -math.sin(theta),
                0.0,
                0.0,
            ],
            [
                math.sin(theta),
                math.cos(theta),
                0.0,
                0.0,
            ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def setup_transforms(width, height, pitch, yaw, roll):
    projection = perspective(np.radians(45.0), width / height, 0.1, 100.0)
    inv_projection = np.linalg.inv(projection)

    eye = np.array(
        [
            0.0,
            0.0,
            4.0,
        ],
        dtype=np.float32,
    )
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    view = look_at(eye, target, up)
    inv_view = np.linalg.inv(view)

    model = (
        rot_z_matrix(roll)
        @ rot_x_matrix(pitch)
        @ rot_y_matrix(yaw)
        @ np.matrix(
            [
                [2.0, 0.0, 0.0, -1.0],
                [0.0, 2.0, 0.0, -0.8],
                [0.0, 0.0, 1.8, -1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )
    inv_model = np.linalg.inv(model)

    return model, inv_model, view, inv_view, projection, inv_projection
