import bpy
import math
import sys


tx = 0.0
ty = -0.0
tz = 5.0

rx = 0.0
ry = 0.0
rz = 180.0



bpy.ops.import_scene.fbx(filepath = 'michael0_128_11_c0001.fbx')
scene = bpy.data.scenes[0]

scene.camera.rotation_mode = 'XYZ'
scene.camera.rotation_euler[0] = rx*(math.pi/180.0)
scene.camera.rotation_euler[1] = ry*(math.pi/180.0)
scene.camera.rotation_euler[2] = rz*(math.pi/180.0)

scene.camera.location.x = tx
scene.camera.location.y = ty
scene.camera.location.z = tz


bpy.data.objects['character'].location.x = 0.0
bpy.data.objects['character'].location.y = 0.0
bpy.data.objects['character'].location.z = 0.0

bpy.data.objects['character'].scale = (1.0, 1.0, 1.0)


scene.frame_end = 50


scene.render.fps = 24
scene.render.image_settings.file_format='AVI_JPEG'
scene.render.filepath = 'michael0_128_11_c0001'

bpy.ops.render.render(animation = True)
