"""
Checker file

Gonna read and explore 4DM's calibrations files
"""
import os
import json
import numpy as np
import quaternion as quat

json_file = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/4DM/easy/378d43cea3872b0551900a15c92e0ec4/0001-partial.bin'

#json_file = os.path.join(dataset_root_folder, mobile, '%04d.bin' % fr)

# mobile pose
with open(json_file, 'r') as jf:
    json_text = jf.read()
    meta = json.loads(json_text)

for key,value in meta.items():
    print(key)
    print(value)

pp = np.asarray([meta['principalPoint']['x'], meta['principalPoint']['y']])
fl = np.asarray([meta['focalLength']['x'], meta['focalLength']['y']])

rotation = quat.quaternion(meta['rotation']['w'],
                           -meta['rotation']['x'],
                           meta['rotation']['y'],
                           -meta['rotation']['z'])

centre = np.asarray([meta['position']['x'],
                     -meta['position']['y'],
                     meta['position']['z']])

#http://ksimek.github.io/2012/08/22/extrinsic/
K = np.asarray([[fl[0], 0, pp[0]],
                [0, fl[1], pp[1]],
                [0, 0, 1]])

R = quat.as_rotation_matrix(rotation)

print("\n\n Start of new checker\n\n")
