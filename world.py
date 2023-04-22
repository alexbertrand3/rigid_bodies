# defines some interesting scenes
from rigid_body import RigidBodyGeometry
from rigid_body_instance import RigidBodyInstance
import taichi.math as tm
from plane import Plane
import bounding_box

loaded_geometries = {}
meshes = []
objects = []


def clear():
    for g in loaded_geometries.keys():
        loaded_geometries[g].instances = []
        loaded_geometries[g].num_instances = 0

    objects.clear()
    meshes.clear()


def update():
    for m in meshes:
        m.update_transforms()


def teapot_through_torus():
    clear()

    cam_pos = (0, 7, 10)
    cam_lookat = (0, 4, 0)
    gravity = True

    plane = Plane()
    torus = load('res/torus.obj')
    teapot = load('res/teapot.obj')

    meshes.append(torus)
    meshes.append(teapot)

    objects.append(RigidBodyInstance(torus, pinned=True, p0=tm.vec3(0, 3, 0), scale=tm.vec3(2, 1, 2)))
    objects.append(RigidBodyInstance(teapot, p0=tm.vec3(4, 6, 0), v0=tm.vec3(-3, 0, 0)))

    torus.update_transforms()
    teapot.update_transforms()

    bbs = bounding_box.Bounding_Boxes(len(objects))

    return plane, objects, gravity, bbs, cam_pos, cam_lookat


def colliding_toruses():
    clear()

    cam_pos = (0, 7, 10)
    cam_lookat = (0, 5, 0)
    gravity = False

    plane = Plane()
    torus = load('res/torus.obj')
    meshes.append(torus)

    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(-2, 3, 0), v0=tm.vec3(3, 1, 0), omega0=tm.vec3(0, 0, -1)))
    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(4, 4, 0), v0=tm.vec3(-2, 0, 0), omega0=tm.vec3(0, 0, 1)))
    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(0, 0, 0), v0=tm.vec3(0, 0, 0), omega0=tm.vec3(-2, 0, 0)))
    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(0, 6, 0), v0=tm.vec3(1, -2, 0), omega0=tm.vec3(0, 3, 0)))

    torus.update_transforms()

    bbs = bounding_box.Bounding_Boxes(len(objects))

    return plane, objects, gravity, bbs, cam_pos, cam_lookat


def stack():
    clear()

    cam_pos = (0, 5, 10)
    cam_lookat = (0, 2, 0)
    gravity = True

    plane = Plane()
    torus = load('res/torus.obj')
    meshes.append(torus)

    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(0, 1, 0)))
    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(0, 3, 0)))
    objects.append(RigidBodyInstance(torus, pinned=False, p0=tm.vec3(0, 5, 0)))

    torus.update_transforms()

    bbs = bounding_box.Bounding_Boxes(len(objects))

    return plane, objects, gravity, bbs, cam_pos, cam_lookat


def load(obj_file_str):
    # for some reason trying to reuse the meshing so they don't have to get reloaded every times
    # completely breaks rendering? No idea why but that's the way it is
    # hence resetting completely reloads the meshes and rebuilds the distance function
    # whatever

    # if obj_file_str not in loaded_geometries:
    #     loaded_geometries[obj_file_str] = RigidBodyGeometry(obj_file_str)
    loaded_geometries[obj_file_str] = RigidBodyGeometry(obj_file_str)
    return loaded_geometries[obj_file_str]
