# A rigid body, with geometry (mesh) and signed distance function

import taichi as ti
import taichi.math as tm
import util
from pyquaternion import Quaternion

K_0 = tm.mat3(0)

@ti.data_oriented
class RigidBodyInstance:
    def __init__(self, geometry, pinned=False,
                 p0=tm.vec3(0, 0, 0),
                 v0=tm.vec3(0, 0, 0),
                 scale=tm.vec3(1, 1, 1),
                 restitution=0.5,
                 friction=0.5,
                 density=1,
                 q0=Quaternion(),
                 omega0=tm.vec3(0, 0, 0)
                 ):

        self.geometry = geometry
        geometry.add_instance(self)
        self.pinned = pinned  # object is stationary

        self.scale = scale  # nonuniform scale
        scale_multiplier = scale[0] * scale[1] * scale[2]
        self.m = scale_multiplier * density * geometry.mass  # mass
        self.restitution = restitution  # coefficient of restitution
        self.friction = friction  # coefficient of friction

        self.p = p0  # position (3D vector)
        self.v = v0  # velocity (3D vector)
        self.p_dt = tm.vec3(0, 0, 0)  # predicted position at next time step

        self.q = q0  # orientation (unit quaternion)
        self.omega = omega0  # angular velocity
        self.q_dt = Quaternion()  # predicted orientation at next time step
        self.I = self.compute_inertia()
        self.I_inv = tm.mat3([1 / self.I[0, 0], 0, 0], [0, 1 / self.I[1, 1], 0], [0, 0, 1 / self.I[2, 2]])

        self.transform = None
        self.transform_dt = None
        self.inv_transform_dt = None

        self.compute_transform()

        # I don't know how tf taichi is supposed to work but structs are wack so now we get this
        self.collision_result = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        # n, r1, r2


    def compute_inertia(self):
        dims = self.geometry.AABB_max - self.geometry.AABB_min
        x_sqr = dims[0] * dims[0]
        y_sqr = dims[1] * dims[1]
        z_sqr = dims[2] * dims[2]
        Ixx = (1 / 12) * self.m * (y_sqr + z_sqr)
        Iyy = (1 / 12) * self.m * (x_sqr + z_sqr)
        Izz = (1 / 12) * self.m * (x_sqr + y_sqr)
        return tm.mat3([Ixx, 0, 0], [0, Iyy, 0], [0, 0, Izz])

    def predict_transform(self, g, dt):
        if self.pinned:
            dt = 0
        self.p_dt = self.p + dt * (self.v + dt * g)
        self.q_dt = self.q + 0.5 * dt * (util.omega_to_quat(self.omega) * self.q)

        axis = self.q_dt.axis
        angle = self.q_dt.angle
        self.transform_dt = util.compute_transform(self.p_dt, self.scale, axis, angle)
        self.inv_transform_dt = util.inverse_transform(self.transform_dt)

    def compute_transform(self):
        axis = self.q.axis
        angle = self.q.angle
        self.transform = util.compute_transform(self.p, self.scale, axis, angle)

    def plane_collision_test(self, plane, g, dt, inelastic=False):
        self.predict_transform(g, dt)
        n = plane.n
        p = plane.p
        temp = plane_collision(self.geometry.vertices, self.transform_dt, n, p)            # 4th value is indicator of collision (<0 = collision)
        if temp.w < 0.0:
            if inelastic:
                restitution = 0
            else:
                restitution = min(self.restitution, plane.restitution)

            r = temp.xyz
            if self.pinned:
                v_at_r = tm.vec3(0)
                K = K_0
            else:
                v_at_r = util.velocity(self.v, self.omega, r)
                K = util.k_matrix(self.m, r, self.I_inv)

            j_n = util.impulse(restitution, n, v_at_r, K)
            j_n = abs(j_n)
            j = n * j_n

            if not self.pinned:
                self.v += j / self.m
                self.omega -= util.ang_vel_update(self.I_inv, r, j)

            self.predict_transform(g, dt)


    def collision_test(self, other_body, g, dt, inelastic=False):
        self.predict_transform(g, dt)
        other_body.predict_transform(g, dt)
        # do coarse (bounding box) test
        possible_collision = coarse_collision(
            self.geometry.AABB,
            self.transform_dt,
            other_body.geometry.AABB_min,
            other_body.geometry.AABB_max,
            other_body.inv_transform_dt)

        # do finer test
        if possible_collision:
            # friction = max(self.friction, other_body.friction)        # if only
            if inelastic:
                restitution = 0
            else:
                restitution = min(self.restitution, other_body.restitution)

            for i in range(5):
                mins = other_body.geometry.AABB_min
                maxs = other_body.geometry.AABB_max
                df = other_body.geometry.distance_field
                dx = other_body.geometry.cell_size

                # this is hideous
                collision_v = fine_collision(self.collision_result,
                                             self.geometry.vertices,
                                             self.transform_dt,
                                             mins, maxs,
                                             df, dx,
                                             other_body.inv_transform_dt)

                if collision_v >= 0:
                    n = self.collision_result[0]
                    r1 = self.collision_result[1]
                    r2 = self.collision_result[2]

                    if self.pinned:
                        v_at_r1 = tm.vec3(0)
                        K_1 = K_0
                    else:
                        K_1 = util.k_matrix(self.m, r1, self.I_inv)
                        v_at_r1 = util.velocity(self.v, self.omega, r1)

                    if other_body.pinned:
                        v_at_r2 = tm.vec3(0)
                        K_2 = K_0
                    else:
                        K_2 = util.k_matrix(other_body.m, r2, other_body.I_inv)
                        v_at_r2 = util.velocity(other_body.v, other_body.omega, r2)

                    v_rel = v_at_r1 - v_at_r2
                    K_T = K_1 + K_2

                    j_n = util.impulse(restitution, n, v_rel, K_T)

                    # I don't know why this works but I think it DOOOOES?
                    # might be due to not culling collision that are already separating?
                    j_n = abs(j_n)

                    j = n * j_n

                    # update velocity and angular velocity
                    if not self.pinned:
                        self.v += j / self.m
                        self.omega -= util.ang_vel_update(self.I_inv, r1, j)
                    if not other_body.pinned:
                        other_body.v -= j / other_body.m
                        other_body.omega += util.ang_vel_update(other_body.I_inv, r2, j)

                    self.predict_transform(g, dt)
                    other_body.predict_transform(g, dt)

                else:
                    # no more collisions found; don't check for more
                    return False

        return possible_collision


@ti.kernel
def coarse_collision(obj1_aabb: ti.template(), W_from_obj1: tm.mat4, obj2_mins: tm.vec3, obj2_maxs: tm.vec3,
                     obj2_from_W: tm.mat4) -> bool:
    # converts object1's AABB in its local frame to another AABB in object2's frame, and checks for possible overlap
    inf = util.inf
    mins = tm.vec4(inf, inf, inf, 1)
    maxs = tm.vec4(-inf, -inf, -inf, 1)
    M = obj2_from_W @ W_from_obj1
    ti.loop_config(serialize=True)  # atomic_min seems to be broken and doesn't work atomically :(((
    for i in range(8):
        x = M @ obj1_aabb[i]
        ti.atomic_min(mins, x)
        ti.atomic_max(maxs, x)

    mins_3d = mins.xyz
    maxs_3d = maxs.xyz
    temp1 = mins_3d < obj2_maxs
    temp2 = maxs_3d > obj2_mins

    return temp1.all() and temp2.all()


# causes Taichi to lag the first time 2 objects get near each other;
# Maybe compiling a new function for each pair? Probably my fault, but I blame Taichi.
@ti.kernel
def fine_collision(result: ti.template(),
                   obj1_verts: ti.template(), W_from_obj1: tm.mat4,
                   obj2_mins: tm.vec3, obj2_maxs: tm.vec3,
                   obj2_df: ti.template(), dx: float, obj2_from_W: tm.mat4) -> int:

    # find the deepest intersection of obj1's vertices with obj2's distance field
    deepest_intersection = 0.0
    deepest_v = -1
    M = obj2_from_W @ W_from_obj1

    for i in obj1_verts:
        v = obj1_verts[i]
        v_obj2 = M @ tm.vec4(v, 1)  # convert to homogenous and move to obj2 frame
        v_3d = v_obj2.xyz
        temp1 = v_3d < obj2_maxs
        temp2 = v_3d > obj2_mins

        if temp1.all() and temp2.all():
            # trilinear interpolation on the distance field
            dist = util.lookup_distance_field(obj2_df, obj2_mins, dx, v_3d)

            ti.atomic_min(deepest_intersection, dist)
            if deepest_intersection == dist:
                deepest_v = i

    if deepest_v >= 0:
        obj1_p = obj1_verts[deepest_v]
        obj2_p = (M @ tm.vec4(obj1_p, 1)).xyz

        p = W_from_obj1 @ tm.vec4(obj1_verts[deepest_v], 1)     # point of collision in world coords
        r1 = p - (W_from_obj1 @ tm.vec4(0, 0, 0, 1))
        r2 = p - (obj2_from_W.inverse() @ tm.vec4(0, 0, 0, 1))
        n_4d = obj2_from_W.inverse() @ util.normal_from_df(obj2_df, obj2_mins, dx, obj2_p)
        n = n_4d.xyz

        result[0].xyz = n.x, n.y, n.z         # gross
        result[1].xyz = r1.x, r1.y, r1.z
        result[2].xyz = r2.x, r2.y, r2.z

    return deepest_v


@ti.kernel
def plane_collision(verts: ti.template(), W_from_obj: tm.mat4, n: tm.vec3, p: tm.vec3) -> tm.vec4:
    deepest_intersection = 0.0
    deepest_v = -1
    for i in verts:
        v = verts[i]
        v_world = W_from_obj @ tm.vec4(v, 1)
        v = v_world.xyz - p
        dist = v.dot(n)

        ti.atomic_min(deepest_intersection, dist)
        if deepest_intersection == dist:
            deepest_v = i

    point = W_from_obj @ tm.vec4(verts[deepest_v], 1)
    obj_origin = W_from_obj @ tm.vec4(0, 0, 0, 1)
    r = (point - obj_origin).xyz
    output = tm.vec4(r.x, r.y, r.z, deepest_intersection)

    return output

