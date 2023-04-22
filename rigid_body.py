# A template for a rigid body, with geometry (mesh) and signed distance function
# Instances of rigid_body must be created using rigid_body_instance

import taichi as ti
import taichi.math as tm
import math
import util
from rigid_body_instance import RigidBodyInstance


@ti.data_oriented
class RigidBodyGeometry:
    def __init__(self, obj_file, distance_field_res=32):
        # geometry
        self.name, self.vertices, self.indices = util.parse_obj(obj_file)
        self.num_verts = self.vertices.shape[0]
        self.AABB, self.AABB_min, self.AABB_max = self._bounding_box()
        self.center_of_mass = self._compute_center_of_mass()
        self._recenter()

        dims = self.AABB_max - self.AABB_min
        self.mass = dims[0] * dims[1] * dims[2]

        # print('Center of mass:', self.center_of_mass)
        # print('Bounding box:', self.AABB_min, self.AABB_max)
        # print('Number of vertices:', self.num_verts)
        # object space signed distance function
        df_res = self._level_set_shape(distance_field_res)
        self.distance_field = ti.field(dtype=ti.f32, shape=df_res)
        self._compute_level_set()

        # for testing
        x_n, y_n, z_n = self.distance_field.shape
        # print('DISTANCE FIELD SHAPE: ', df_res)
        size = x_n * y_n * z_n
        self._df_verts = ti.Vector.field(3, dtype=ti.f32, shape=size)
        self._df_colors = ti.Vector.field(3, dtype=ti.f32, shape=size)
        self._distance_field_verts()

        self.instances = []
        self.num_instances = 0
        self.transforms = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=1)

    # computes an object frame axis-aligned bounding box
    def _bounding_box(self):
        aabb = ti.Vector.field(n=4, dtype=ti.f32, shape=8)
        inf = float('inf')
        mins = ti.math.vec3(inf, inf, inf)
        maxs = ti.math.vec3(-inf, -inf, -inf)
        for v in range(self.num_verts):
            vert = self.vertices[v]
            for i in range(3):
                if vert[i] < mins[i]:
                    mins[i] = vert[i]
                if vert[i] > maxs[i]:
                    maxs[i] = vert[i]
        # eight corners of the bounding box
        aabb[0] = tm.vec4(mins[0], mins[1], mins[2], 1)
        aabb[1] = tm.vec4(mins[0], mins[1], maxs[2], 1)
        aabb[2] = tm.vec4(mins[0], maxs[1], mins[2], 1)
        aabb[3] = tm.vec4(mins[0], maxs[1], maxs[2], 1)
        aabb[4] = tm.vec4(maxs[0], mins[1], mins[2], 1)
        aabb[5] = tm.vec4(maxs[0], mins[1], maxs[2], 1)
        aabb[6] = tm.vec4(maxs[0], maxs[1], mins[2], 1)
        aabb[7] = tm.vec4(maxs[0], maxs[1], maxs[2], 1)

        return aabb, mins, maxs

    # very simple center of mass approximation; computes center of axis-aligned bounding box
    def _compute_center_of_mass(self):
        x = (self.AABB_min.x + self.AABB_max.x) / 2
        y = (self.AABB_min.y + self.AABB_max.y) / 2
        z = (self.AABB_min.z + self.AABB_max.z) / 2
        return ti.math.vec3(x, y, z)

    # ensures the origin (in object frame) is at the center of mass
    def _recenter(self):
        for v in range(self.num_verts):
            self.vertices[v] -= self.center_of_mass
        for i in range(8):
            self.AABB[i] -= tm.vec4(self.center_of_mass, 0)
        self.AABB_min -= self.center_of_mass
        self.AABB_max -= self.center_of_mass
        self.center_of_mass -= self.center_of_mass

    # compute the size of the uniform grid holding distance function information
    # the largest dimension has size <resolution + 2> -- other dimensions may be smaller if mesh is oblong
    # distance grid is padded by 1 cell on every side to simplify interpolating the distance gradient
    def _level_set_shape(self, resolution: int):
        len_x = self.AABB_max.x - self.AABB_min.x
        len_y = self.AABB_max.y - self.AABB_min.y
        len_z = self.AABB_max.z - self.AABB_min.z
        self.cell_size = max(len_x, len_y, len_z) / resolution

        res_x = math.ceil(len_x / self.cell_size)
        res_y = math.ceil(len_y / self.cell_size)
        res_z = math.ceil(len_z / self.cell_size)

        return res_x + 2, res_y + 2, res_z + 2

    # calculate the signed distance function of the mesh on a uniform grid.
    @ti.kernel
    def _compute_level_set(self):
        dx = self.cell_size
        min_x, min_y, min_z = self.AABB_min

        for i, j, k in self.distance_field:
            self.distance_field[i, j, k] = util.inf

            # -0.5 accounts for padding
            p = tm.vec3(
                min_x + dx * (i - 0.5),
                min_y + dx * (j - 0.5),
                min_z + dx * (k - 0.5)
            )

            # compute distance to every triangle
            # not as elegant as something like fast marching, but way easier to implement
            # and not terribly inefficient for reasonable meshes
            for f_ind in range(self.indices.shape[0] // 3):
                f = f_ind * 3
                v1 = self.vertices[self.indices[f]]
                v2 = self.vertices[self.indices[f + 1]]
                v3 = self.vertices[self.indices[f + 2]]

                d = dist_to_tri(p, v1, v2, v3)

                # only update distance if this triangle is the closest
                if abs(d) < abs(self.distance_field[i, j, k]):
                    self.distance_field[i, j, k] = d

    # converts the Rigid Body's distance field to a set of 3D points
    # to be drawn in the scene (for debug purposes)
    # output is in OBJECT FRAME, needs to be transformed if object isn't at origin
    @ti.kernel
    def _distance_field_verts(self):
        # find min and max distance for more a e s t h e t i c rendering
        min_dist = float('inf')
        max_dist = -min_dist
        for i, j, k in self.distance_field:
            if self.distance_field[i, j, k] < min_dist:
                min_dist = self.distance_field[i, j, k]
            if self.distance_field[i, j, k] > max_dist:
                max_dist = self.distance_field[i, j, k]
        min_color = 1.1 / min_dist
        max_color = 1.1 / max_dist

        x_n, y_n, z_n = self.distance_field.shape
        size = x_n * y_n * z_n
        dx = self.cell_size
        min_x, min_y, min_z = self.AABB_min
        for i in range(x_n):
            x = min_x + dx * (i - 0.5)
            for j in range(y_n):
                y = min_y + dx * (j - 0.5)
                for k in range(z_n):
                    z = min_z + dx * (k - 0.5)
                    idx = (i * y_n * z_n) + (j * z_n) + k
                    dist = self.distance_field[i, j, k]

                    color = tm.vec3(max_color * dist, 0.0, 0.0)
                    if dist < 0:
                        color = tm.vec3(0.0, 0.0, min_color * dist)

                    self._df_colors[idx] = color
                    self._df_verts[idx] = tm.vec3(x, y, z)

    def get_drawable_distance_field(self):
        return self._df_verts, self._df_colors

    def add_instance(self, new_instance):
        self.instances.append(new_instance)
        self.num_instances += 1

    @ti.kernel
    def update_transforms_helper(self, i: int, m: tm.mat4):
        self.transforms[i] = m

    def update_transforms(self):
        if self.num_instances > 0:
            if self.num_instances != self.transforms.shape[0]:
                self.transforms = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=self.num_instances)
            for i in range(self.num_instances):
                self.update_transforms_helper(i, self.instances[i].transform)


# Computes the signed distance of a point p to the triangle formed by points a, b, c
# You wouldn't think it would be such a pain, but alas
# referenced code at: https://github.com/embree/embree/blob/master/tutorials/common/math/closest_point.h
@ti.func
def dist_to_tri(p: tm.vec3, a: tm.vec3, b: tm.vec3, c: tm.vec3) -> ti.f32:
    done = False

    ab = b - a
    ac = c - a
    ap = p - a
    n = (ab.cross(ac)).normalized()

    dist = n.dot(ap)                # signed distance of p to the plane of the triangle
    sign = 1.0
    if dist < 0:
        sign = -1.0

    d1 = ab.dot(ap)
    d2 = ac.dot(ap)
    if d1 <= 0 and d2 <= 0:
        dist = sign * tm.length(ap)
        done = True

    # Taichi doesn't support multiple return statements in a function so we get this mess
    d3, d4, d5, d6, va, vb, vc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if not done:
        bp = p - b
        d3 = ab.dot(bp)
        d4 = ac.dot(bp)
        if d3 >= 0 and d4 <= d3:
            dist = sign * tm.length(bp)
            done = True

    if not done:
        cp = p - c
        d5 = ab.dot(cp)
        d6 = ac.dot(cp)
        if d6 >= 0 and d5 <= d6:
            dist = sign * tm.length(cp)
            done = True

    if not done:
        vc = d1 * d4 - d3 * d2
        if vc <= 0 and d1 >= 0 and d3 <= 0:
            v = d1 / (d1 - d3)
            dist = sign * tm.length(p - (a + (v * ab)))
            done = True

    if not done:
        vb = d5 * d2 - d1 * d6
        if vb <= 0 and d2 >= 0 and d6 <= 0:
            v = d2 / (d2 - d6)
            dist = sign * tm.length(p - (a + (v * ac)))
            done = True

    if not done:
        va = d3 * d6 - d5 * d4
        if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
            v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            dist = sign * tm.length(p - (b + v * (c - b)))
            done = True

    if not done:
        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        dist = sign * tm.length(p - (a + (v * ab) + (w * ac)))

    return dist


if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    RigidBodyGeometry('res/cube.obj')

