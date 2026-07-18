"""Test / demo surface meshes (icosphere)."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def icosphere(subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of a unit icosphere for tests and demos."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    for _ in range(subdivisions):
        cache, new_faces, verts = {}, [], list(verts)

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            m = (verts[a] + verts[b]) / 2.0
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts, faces = np.array(verts), np.array(new_faces)
    return verts, faces
