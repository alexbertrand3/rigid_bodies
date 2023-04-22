import taichi.math as tm
import util


class Plane:
    def __init__(self, n=tm.vec3(0, 1, 0), p=tm.vec3(0, 0, 0), restitution=0):
        self.n = n.normalized()
        self.p = p
        self.restitution = restitution
        self.name, self.vertices, self.indices = util.parse_obj("res/plane.obj")
