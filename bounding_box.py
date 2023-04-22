import util
import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Bounding_Boxes:
    def __init__(self, num_bb):
        _, self.verts, self.indices = util.parse_obj("res/cube.obj")
        self.transforms = ti.Matrix.field(n=4, m=4, dtype=ti.f32, shape=num_bb)

    @ti.kernel
    def update_transform(self, i: int, M: tm.mat4, scale: tm.vec3):
        S = tm.scale(scale.x, scale.y, scale.z)
        self.transforms[i] = M @ S
