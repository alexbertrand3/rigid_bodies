# By [REDACTED]
# COMP 559 Final Project - Rigid Bodies
# Based on "Nonconvex Rigid Bodies with Stacking" by Guendelman, Bridson, and Fedkiw (2003)

import taichi as ti
import taichi.math as tm

# import bounding_box
# import rigid_body
# from rigid_body_instance import RigidBodyInstance
# from plane import Plane

import world
import util

# for some reason cpu is faster for me, but you can try arch=ti.gpu and see if it's better
ti.init(arch=ti.cpu)


def main():
    # simulation parameters
    paused = False
    dt = 1e-2
    collision_substeps = 5
    contact_substeps = 2

    g = tm.vec3(0, -9.8, 0)
    do_gravity = True
    wireframe = False
    draw_bounding_boxes = False

    plane, objects, bbs, do_gravity, cam_pos, cam_lookat = world.teapot_through_torus()

    # Window Setup
    width = 1280
    height = 720
    window = ti.ui.Window('Rigid Bodies', res=(width, height), pos=(150, 150), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.3, 0.3, 0.3))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.projection_mode = 0  # 0 = perspective, 1 = orthographic

    # Render loop
    while window.running:
        # keyboard input

        while window.get_event(ti.ui.PRESS):
            e = window.event
            if e.key == ' ':
                paused = not paused
            if e.key == 'b':
                draw_bounding_boxes = not draw_bounding_boxes
            if e.key == 'g':
                do_gravity = not do_gravity
            if e.key == 'w':
                wireframe = not wireframe

            # SCENE SELECT
            if e.key == 'u':
                plane, objects, bbs, do_gravity, cam_pos, cam_lookat = world.teapot_through_torus()
            if e.key == 'i':
                plane, objects, bbs, do_gravity, cam_pos, cam_lookat = world.colliding_toruses()
            if e.key == 'o':
                plane, objects, bbs, do_gravity, cam_pos, cam_lookat = world.stack()

            if e.key == 'r':
                recording = not recording

            elif e.key == ti.ui.ESCAPE:
                window.running = False

        if paused:
            continue

        if do_gravity:
            g_t = g
        else:
            g_t = tm.vec3(0, 0, 0)

        # Time step
        collision_detection(objects, g_t, dt, collision_substeps, plane)
        velocity_step(objects, g_t, dt)
        contact_resolution(objects, dt, contact_substeps, plane)
        position_step(objects, dt)

        world.update()

        # Camera stuff
        camera.position(*cam_pos)
        camera.lookat(*cam_lookat)
        scene.set_camera(camera)

        scene.point_light(pos=(1, 2, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        for mesh in world.meshes:
            scene.mesh_instance(
                mesh.vertices,
                mesh.indices,
                transforms=mesh.transforms,
                color=(1.0, 1.0, 1.0),
                show_wireframe=wireframe
            )

        scene.mesh_instance(
            plane.vertices,
            plane.indices,
            color=(0.6, 0.5, 0.3)
        )

        if draw_bounding_boxes:
            for i in range(len(objects)):
                obj = objects[i]
                s = 0.5 * (obj.geometry.AABB_max - obj.geometry.AABB_min)
                bbs.update_transform(i, obj.transform, s)

            scene.mesh_instance(
                bbs.verts,
                bbs.indices,
                transforms=bbs.transforms,
                color=(1.0, 1.0, 1.0),
                show_wireframe=True
            )

        # df_points, df_colors = mesh.get_drawable_distance_field()
        # scene.particles(
        #     df_points,
        #     radius=mesh.cell_size / 6,
        #     per_vertex_color=df_colors
        # )

        canvas.scene(scene)
        window.show()


def collision_detection(objects, g, dt, substeps, plane):
    for _ in range(substeps):
        collisions_found = False

        for obj in objects:
            if obj.pinned:
                obj.v = tm.vec3(0)
                obj.omega = tm.vec3(0)
            cf = obj.plane_collision_test(plane, g, dt)
            collisions_found = collisions_found or cf

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                cf = objects[i].collision_test(objects[j], g, dt)
                collisions_found = collisions_found or cf

        if not collisions_found:
            break


def velocity_step(objects, g, dt):
    for obj in objects:
        if not obj.pinned:
            obj.v += g * dt


def contact_resolution(objects, dt, contact_substeps, plane):
    for _ in range(contact_substeps):
        for obj in objects:
            if obj.pinned:
                obj.v = tm.vec3(0)
                obj.omega = tm.vec3(0)
            obj.plane_collision_test(plane, tm.vec3(0), dt, inelastic=True)

        collisions_found = False

        for i in range(len(objects) - 1):
            for j in range(i + 1, len(objects)):
                cf = objects[i].collision_test(objects[j], tm.vec3(0), dt, inelastic=True)
                collisions_found = collisions_found or cf

        if not collisions_found:
            break

def position_step(objects, dt):
    for obj in objects:
        if not obj.pinned:
            obj.p += obj.v * dt
            obj.q += 0.5 * dt * (util.omega_to_quat(obj.omega) * obj.q)
            obj.q = obj.q.normalised
        obj.compute_transform()


if __name__ == '__main__':
    main()
