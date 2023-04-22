import taichi as ti
import taichi.math as tm
from pyquaternion import Quaternion

inf = float('inf')


def omega_to_quat(omega):
    return Quaternion(0, omega.x, omega.y, omega.z)


@ti.kernel
def velocity(v: tm.vec3, omega: tm.vec3, r: tm.vec3) -> tm.vec3:
    # r = collision_point - CoM
    return v + omega.cross(r)


@ti.kernel
def k_matrix(m: float, r: tm.vec3, I_inv: tm.mat3) -> tm.mat3:
    inverse_mass = (1 / m) * tm.eye(3)
    r_mat = cross_prod_matrix(r)
    r_mat_T = r_mat.transpose()
    return inverse_mass + (r_mat_T @ I_inv @ r_mat)


@ti.kernel
def impulse(restitution: float, n: tm.vec3, v_rel: tm.vec3, K_T: tm.mat3) -> float:
    v_norm = n.dot(v_rel)
    numerator = -(1 + restitution) * v_norm
    denom = n.dot(K_T @ n)
    return numerator / denom


def ang_vel_update(I_inv: tm.mat3, r: tm.vec3, j: tm.vec3):
    return I_inv @ (r.cross(j))


@ti.func
def cross_prod_matrix(r: tm.vec3) -> tm.mat3:
    return tm.mat3(
        [0, -r.z, r.y],
        [r.z, 0, -r.x],
        [-r.y, r.x, 0]
    )


@ti.kernel
def compute_transform(p: tm.vec3, scale: tm.vec3, axis: tm.vec3, angle: float) -> tm.mat4:
    S = tm.scale(scale.x, scale.y, scale.z)
    T = tm.translate(p.x, p.y, p.z)
    R = tm.eye(4)
    if axis.norm() > 0:
        R = tm.rot_by_axis(axis, angle)
    return T @ R @ S


@ti.kernel
def inverse_transform(m: tm.mat4) -> tm.mat4:
    return m.inverse()


@ti.func
def lookup_distance_field(df: ti.template(), mins: tm.vec3, dx: float, v: tm.vec3) -> float:
    # trilinear interpolate on the distance field
    # thanks wikipedia!

    x = ((v.x - mins.x) / dx) + 0.5
    i = int(x)
    xd_1 = tm.mod(x, 1)
    xd_0 = 1 - xd_1

    y = ((v.y - mins.y) / dx) + 0.5
    j = int(y)
    yd_1 = tm.mod(y, 1)
    yd_0 = 1 - yd_1

    z = ((v.z - mins.z) / dx) + 0.5
    k = int(z)
    zd_1 = tm.mod(z, 1)
    zd_0 = 1 - zd_1

    f00 = (xd_0 * df[i, j, k]) + (xd_1 * df[i + 1, j, k])
    f01 = (xd_0 * df[i, j, k + 1]) + (xd_1 * df[i + 1, j, k + 1])
    f10 = (xd_0 * df[i, j + 1, k]) + (xd_1 * df[i + 1, j, k])
    f11 = (xd_0 * df[i, j + 1, k + 1]) + (xd_1 * df[i + 1, j + 1, k + 1])

    f0 = (yd_0 * f00) + (yd_1 * f10)
    f1 = (yd_0 * f01) + (yd_1 * f11)

    f = (zd_0 * f0) + (zd_1 * f1)

    return f


@ti.func
def normal_from_df(df: ti.template(), mins: tm.vec3, dx: float, v: tm.vec3) -> tm.vec4:
    # Uses forward differences to approximate the (homogenous) normal from a distance field
    # maybe central differences better? but would have to check bounds :(
    # also doesn't have any interpolation, but with decent sized dfs should be fine...?
    x = ((v.x - mins.x) / dx) + 0.5
    i = int(x)
    y = ((v.y - mins.y) / dx) + 0.5
    j = int(y)
    z = ((v.z - mins.z) / dx) + 0.5
    k = int(z)

    f = df[i, j, k]
    fx = df[i + 1, j, k]
    fy = df[i, j + 1, k]
    fz = df[i, j, k + 1]

    n = tm.vec4(fx - f, fy - f, fz - f, 0)
    return n.normalized()


# hyper simplified .obj mesh loading; only cares about (3D) vertices and (triangular) faces
# returns (name: string, vertices: ti.Vector.field, indices: ti.field)
def parse_obj(filename):
    name = None
    verts = []
    faces = []
    with open(filename) as f:
        for line in f:
            line = line.split()

            if line[0] == 'o':
                name = line[1]

            elif line[0] == 'v':
                verts.append(ti.math.vec3(float(line[1]), float(line[2]), float(line[3])))

            elif line[0] == 'f':
                for i in range(1, len(line)):
                    # face values are given in v, v/vt, v/vt/vn, or v//vn format
                    # only care about the v part
                    face_i = line[i].split('/')[0]
                    faces.append(int(face_i) - 1)       # -1 since .obj is 1-indexed but Taichi expects 0-indexed

    # convert to Taichi types
    vertices = ti.Vector.field(3, dtype=ti.float32, shape=len(verts))
    for i in range(len(verts)):
        vertices[i] = verts[i]

    indices = ti.field(dtype=ti.i32, shape=len(faces))
    for i in range(len(faces)):
        indices[i] = faces[i]

    return name, vertices, indices
