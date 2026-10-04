import math, numpy as np
from pusht_keyboard_sim import rotation_matrix
def circle_vs_T_fast(circle_pos, radius, body_pos, body_angle, vertices):
    R = rotation_matrix(body_angle)
    c = R.T @ (circle_pos - body_pos)
    A = vertices; AB = np.roll(vertices, -1, axis=0) - A
    t = np.clip(((c - A) * AB).sum(1) / (AB * AB).sum(1), 0.0, 1.0)
    Q = A + t[:, None] * AB
    d2 = ((Q - c) ** 2).sum(1)
    i = int(np.argmin(d2)); d = math.sqrt(d2[i])
    yi, yj = A[:, 1], np.roll(A[:, 1], 1)
    xi, xj = A[:, 0], np.roll(A[:, 0], 1)
    cross = (yi > c[1]) != (yj > c[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        xs = (xj - xi) * (c[1] - yi) / (yj - yi) + xi
    inside = bool(np.count_nonzero(cross & (c[0] < xs)) % 2)
    if not inside and d >= radius:
        return None
    direction = (Q[i] - c) / d if d > 1e-12 else np.array([1.0, 0.0])
    n_local, pen = (-direction, radius + d) if inside else (direction, radius - d)
    n = R @ n_local; n /= np.linalg.norm(n) + 1e-12
    return body_pos + R @ Q[i], n, pen
