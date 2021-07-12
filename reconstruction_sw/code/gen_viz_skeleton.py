"""
Progress:

I have to change the code to read our own calibration file. From lines 282+.

I think the calibration files from Mohamed are in OpenCV format.
"""

import open3d as o3d
import cv2 as cv
import numpy as np
import quaternion as quat
import glob
import os
import json
import matplotlib.pyplot as plt
from ctypes import *
import time
import copy
from utils import LineMesh
import argparse

parser = argparse.ArgumentParser(description='Process input.')
parser.add_argument('--vis_open3d', action='store_true')
parser.add_argument('--vis_opencv', action='store_true')
parser.add_argument('--vis_debug', action='store_true')

args = parser.parse_args()


def my_load_calib(path):   
    
    def load_intrinsic(path):
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        camera_matrix = cv_file.getNode("K").mat()
        dist_matrix = cv_file.getNode("D").mat()

        cv_file.release()
        return [camera_matrix, dist_matrix]

    def load_extrinsic(path):
        cv_file = cv.FileStorage(path, cv.FILE_STORAGE_READ)
        R_matrix = cv_file.getNode("R").mat()
        t_matrix = cv_file.getNode("t").mat()

        cv_file.release()
        return [R_matrix, t_matrix]

    intrinsic = path + "/intrinsic/cam.cal"
    extrinsic = path + "/extrinsic/extr.cal"

    camera_matrix, dist_matrix = load_intrinsic(intrinsic)
    R, t = load_extrinsic(extrinsic)
    #R = np.linalg.inv(R)

    pp = camera_matrix[0,2]/3, camera_matrix[1,2]/3
    fl = camera_matrix[0,0]/3, camera_matrix[1,1]/3
    centre = np.squeeze(np.asarray(t))

    rotation = quat.from_rotation_matrix(R)

    return camera_matrix, pp, fl, R, centre, rotation

def get_pts_2d_and_cams(mobile_captures):

    n_cams = len(mobile_captures.keys())
    camera_params = np.zeros((n_cams, 11))

    pts_2d = np.empty((2, 0), dtype=float)
    pts_2d_3d_indices = np.empty((0, ), dtype=int)
    pts_2d_cam_indices = np.empty((0, ), dtype=int)

    i = 0
    for cam_key, cam_par in mobile_captures.items():

        # organise 2d points in an array
        _pts_2d = np.asarray([cam_par['person_pose'][::3], cam_par['person_pose'][1::3]])
        _pts_2d_3d_indices = np.where(_pts_2d[0, :] > 0)
        pts_2d = np.hstack((pts_2d, _pts_2d[:, _pts_2d_3d_indices[0]]))
        pts_2d_3d_indices = np.hstack((pts_2d_3d_indices, _pts_2d_3d_indices[0]))
        pts_2d_cam_indices = np.hstack((pts_2d_cam_indices, i * np.ones(_pts_2d_3d_indices[0].shape, dtype=int)))

        # organise camera parameters in a matrix
        camera_params[i, :2] = cam_par['focal_length']
        camera_params[i, 2:4] = cam_par['principal_point']
        camera_params[i, 4:8] = quat.as_float_array(cam_par['quaternion'])
        camera_params[i, 8:] = cam_par['position']

        i += 1

    return pts_2d, pts_2d_3d_indices, pts_2d_cam_indices, camera_params


def compute_reprojection_error(pts_3d, camera_params, mobile_captures):

    reproj_error = []
    n_cams = len(mobile_captures.keys())

    feat_points = np.zeros((2, 25))

    for c in range(n_cams):
        cam_key = c + 1

        fx, fy = camera_params[c, :2]
        px, py = camera_params[c, 2:4]
        R = quat.as_rotation_matrix(quat.quaternion(*camera_params[c, 4:8]))
        C = camera_params[c, 8:]

        K = np.asarray([[fx, 0, px],
                        [0, fy, py],
                        [0, 0, 1]])

        #P = np.dot(K, np.c_[R.T, - np.dot(R.T, C)])
        P = np.dot(K, np.c_[R, C])

        pts_2d = np.dot(P, np.vstack((pts_3d, np.ones((1, pts_3d.shape[1])))))

        pts_2d = pts_2d[:2, :] / pts_2d[2, :]

        feat_points[0, :] = mobile_captures[cam_key]['person_pose'][::3]
        feat_points[1, :] = mobile_captures[cam_key]['person_pose'][1::3]

        idx_to_compute = np.where(feat_points[0, :] > 0)[0]

        re = np.linalg.norm(pts_2d[:, idx_to_compute] - feat_points[:, idx_to_compute]) / idx_to_compute.size

        reproj_error.append(re)

    reproj_error = np.asarray(reproj_error)

    return np.sum(reproj_error) / n_cams


def project(pts_3d, camera_params, mobile_captures, pts_2d_3d_indices, pts_2d_cam_indices, title, viz=0):

    PTS_2D = np.empty((0, 2))
    n_cams = len(mobile_captures.keys())

    for c in range(n_cams):
        print("C", c)
        fx, fy = camera_params[c, :2]
        px, py = camera_params[c, 2:4]
        R = quat.as_rotation_matrix(quat.quaternion(*camera_params[c, 4:8]))
        C = camera_params[c, 8:]

        # select the indices of only the 3d points triangulated from the current camera
        pts_3d_idx = pts_2d_3d_indices[pts_2d_cam_indices == c]

        _pts_3d = pts_3d[:, pts_3d_idx]

        K = np.asarray([[fx, 0, px],
                        [0, fy, py],
                        [0, 0, 1]])

        #P = np.dot(K, np.c_[R.T, - np.dot(R.T, C)])
        P = np.dot(K, np.c_[R, C])

        pts_2d = np.dot(P, np.vstack((_pts_3d, np.ones((1, _pts_3d.shape[1])))))

        pts_2d = pts_2d[:2, :] / pts_2d[2, :]

        PTS_2D = np.vstack((PTS_2D, pts_2d.T))

        cam_key = c + 1  # todo: substitute this by taking the key directly from the key list
        
        #if viz == cam_key or viz == -1:
        plt.figure()
        _img = copy.deepcopy(mobile_captures[cam_key]['im'])
        _img[:, :, 0] = mobile_captures[cam_key]['im'][:, :, 2]
        _img[:, :, 2] = mobile_captures[cam_key]['im'][:, :, 0]
        plt.imshow(_img)
        plt.plot(mobile_captures[cam_key]['person_pose'][::3], mobile_captures[cam_key]['person_pose'][1::3],
                    'o', color='green', markersize=4)
        plt.plot(pts_2d[0, :], pts_2d[1, :],
                    'o', color='red', markersize=4)
        plt.draw()
        plt.title(title)
        plt.pause(1)
        plt.close()


    return PTS_2D.T


###################################################
############### MAIN
###################################################

do_cv_viz = args.vis_opencv
do_o3d_viz = args.vis_open3d
do_debug_reprojection = args.vis_debug

do_white_background = False
do_save_results = True

assert (not (do_cv_viz is True and do_o3d_viz is True)), 'Choose either OpenCV or Open3D visualisation, not both!'


dataset_root_folder = '/media/weber/Ubuntu2/ubuntu2/Human_Pose/QMUL-data/four_viewpoints_ballet/scenario-3' # substitute this with the correct path to the dataset
open_pose_data = dataset_root_folder + "/openpose-results"
frame_res_det_folder = '../results/skeleton'
camera_calib_folder = '/media/weber/Ubuntu2/ubuntu2/Human_Pose/QMUL-data/four_viewpoints_ballet/calib/'
if not os.path.isdir(frame_res_det_folder):
    os.makedirs(frame_res_det_folder)

#dataset_name = 'easy' # choose between easy, medium, hard
cameras = {1: 'point1',
           2: 'point2',
           3: 'point3',
           4: 'point4'}

mobile_host = 1 # what is this?

line_mesh_radius = 0.02

# optimisation modalities sparse bundle adjustment
# 0 -> BA_MOTSTRUCT
# 1 -> BA_MOT
# 2 -> BA_STRUCT
BA_MODALITY = 2

# define function from sparse bundle adjustment
libmysba = cdll.LoadLibrary('libmysba.so')
libmysba.testlib()
libmysba.sba.argtypes = [c_char_p, c_char_p, POINTER(c_double), c_int, c_int,
                         POINTER(c_float), POINTER(c_float), POINTER(c_float)]


# get the sorted frames TODO (sort better)
frame_list = glob.glob(os.path.join(dataset_root_folder, 'images', cameras[mobile_host], '*.png'))
frame_list.sort()
print("\n", frame_list, "\n")

number_of_frames = len(frame_list)
print("Number of frames:", number_of_frames)
#number_of_frames = int(os.path.splitext(os.path.basename(frame_list[-1]))[0]) + 1

start_frame = 1
end_frame = number_of_frames + 1

# CAMERA VIZ MODEL
mobile_viz_points = 0.4 * np.array([[-0.5, -0.5, 1], [0.5, -0.5, 1], [-0.5, 0.5, 1], [0.5, 0.5, 1], [0, 0, 0]])
mobile_viz_lines = [[0, 1], [2, 3], [2, 0], [1, 3], [4, 0], [4, 1], [4, 2], [4, 3]]
mobile_viz_colors = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]
if do_white_background:
    mobile_viz_colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

limbs = np.asarray([[0, 1],
                    [0, 16],
                    [0, 18],
                    [0, 15],
                    [15, 17],
                    [1, 2],
                    [1, 5],
                    [5, 6],
                    [6, 7],
                    [2, 3],
                    [3, 4],
                    [1, 8],
                    [8, 12],
                    [8, 9],
                    [12, 13],
                    [13, 14],
                    [14, 21],
                    [14, 19],
                    [19, 20],
                    [9, 10],
                    [10, 11],
                    [11, 24],
                    [11, 22],
                    [22, 23]])

if do_cv_viz:
    imshow_tag = 'viz'
    cv.namedWindow(imshow_tag, cv.WINDOW_NORMAL)

if do_o3d_viz:
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=1920, height=1080-240)
    vis.get_render_option().background_color = [0, 0, 0]
    if do_white_background:
        vis.get_render_option().background_color = [1, 1, 1]

    local_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    vis.add_geometry(local_frame)

    vis.update_renderer()

    flag_viz_camera_rotation = 1

# init mobile captures
mobile_captures = dict()
for mobile_number, mobile in cameras.items():
    print("Mobile number:", mobile_number, "| Mobile:", mobile)
    mobile_captures[mobile_number] = {'im': [],
                                      'position': [],
                                      'rotation': [],
                                      'principal_point': [],
                                      'focal_length': [],
                                      'K': [],
                                      'frustum': [],
                                      'person_pose': []}

skeleton = []

# evaluation
reprojection_error_end_sequence = []
elapsed_ba_sequence = []

go_to_frame = 0

fr_counter = 0
for fr in range(start_frame, end_frame):
    
    print("frame:", fr)
    
    if fr < go_to_frame:
        continue

    flag_skip_frame = 0  # frame skipping occurs when partial frames are detected

    t_all = time.time()

    ####################
    #################### load camera data and estimated poses for each mobile
    ####################
    t_load_data = time.time()

    for mobile_number, mobile in cameras.items():

        png_file = os.path.join(dataset_root_folder, 'images', mobile, 'out{}.png'.format(fr))
        #json_file = os.path.join(dataset_root_folder, mobile, '%04d.bin' % fr) # xavier: camera data
        
        ### read our calibration file
        K, pp, fl, R, centre, rotation = my_load_calib(os.path.join(camera_calib_folder, mobile))
        ###

        open_pose_file = os.path.join(open_pose_data, mobile, 'keypoints', 'out{}_keypoints.json'.format(fr))

        if not os.path.isfile(png_file):
            flag_skip_frame = 1
            continue

        im = cv.imread(png_file)

        # mobile pose
        # with open(json_file, 'r') as jf:
        #     json_text = jf.read()
        #     meta = json.loads(json_text)

        #pp = np.asarray([meta['principalPoint']['x'], meta['principalPoint']['y']])
        #fl = np.asarray([meta['focalLength']['x'], meta['focalLength']['y']])

        # rotation = quat.quaternion(meta['rotation']['w'],
        #                            -meta['rotation']['x'],
        #                            meta['rotation']['y'],
        #                            -meta['rotation']['z'])

        # centre = np.asarray([meta['position']['x'],
        #                      -meta['position']['y'],
        #                      meta['position']['z']])

        #http://ksimek.github.io/2012/08/22/extrinsic/
        K = np.asarray([[fl[0], 0, pp[0]],
                        [0, fl[1], pp[1]],
                        [0, 0, 1]])

        #R = quat.as_rotation_matrix(rotation)

        # quat_float = quat.as_float_array(rotation)
        # print(quat_float)
        # quat_float = [quat_float[0], -quat_float[1], quat_float[2], -quat_float[3]]
        # rotation = quat.as_quat_array(quat_float)
        # print(quat_float)

        print("\nPrinting Camera Calibration Results")
        print("K:", K)
        print("pp:", pp)
        print("fl:", fl)
        print("R:", R)
        print("centre:", centre)
        print("rotation:", rotation)
        print("\ns")

        mobile_captures[mobile_number]['im'] = im
        mobile_captures[mobile_number]['principal_point'] = pp
        mobile_captures[mobile_number]['focal_length'] = fl
        mobile_captures[mobile_number]['quaternion'] = rotation # rotation matrix as quat
        mobile_captures[mobile_number]['rotation'] = R # rotation matrix
        mobile_captures[mobile_number]['position'] = centre
        mobile_captures[mobile_number]['K'] = K

        # open pose
        with open(open_pose_file, 'r') as jf:
            json_text = jf.read()
            open_pose_meta = json.loads(json_text)

        torso_lens = []
        area_occupied = []
        inv_dist_from_centre = []
        for person in open_pose_meta['people']:
            x, y, c = np.asarray(person['pose_keypoints_2d'][::3]), \
                      np.asarray(person['pose_keypoints_2d'][1::3]), \
                      np.asarray(person['pose_keypoints_2d'][2::3])

            if c[1] == 0 or c[8] == 0:
                torso_lens.append(0)
                area_occupied.append(0)
                inv_dist_from_centre.append(0)
                continue

            torso_lens.append(np.linalg.norm([x[1] - x[8], y[1] - y[8]]))

            bbox_centre = [(np.min(x[x > 0]) + np.max(x[x > 0])) / 2,
                           (np.min(y[y > 0]) + np.max(y[y > 0])) / 2]
            
            # NOTE - be careful: I believe these are hard-coded image dimensions
            inv_dist_from_centre.append(1 / np.linalg.norm([bbox_centre[0] - 640/2, bbox_centre[1] - 480/2]))

            area_occupied.append((np.max(x[x > 0]) - np.min(x[x > 0])) * (np.max(y[y > 0]) - np.min(y[y > 0])))

        idx = np.argmax(np.multiply(area_occupied, inv_dist_from_centre))
        mobile_captures[mobile_number]['person_pose'] = open_pose_meta['people'][idx]['pose_keypoints_2d']

        if do_cv_viz:
            _im = copy.deepcopy(im)

            for l, limb in enumerate(limbs):

                kpts = np.asarray([mobile_captures[mobile_number]['person_pose'][::3],
                                   mobile_captures[mobile_number]['person_pose'][1::3]]).T

                if kpts[limb[0]][0] == 0 or kpts[limb[1]][0] == 0:
                    continue

                _im = cv.line(_im, tuple(kpts[limb[0]].astype(int)), tuple(kpts[limb[1]].astype(int)), [0, 255*0.906, 255], 3, lineType=cv.LINE_AA)

            if do_save_results:
                if not os.path.exists('../results/img_poses'):
                    os.mkdir('../results/img_poses')
                cv.imwrite(os.path.join('../results/img_poses', '{:04d}_{}.png'.format(fr_counter, mobile)), _im)

            cv.imshow(imshow_tag, _im)
            cv.waitKey(1)

    elapsed_load_data = (time.time() - t_load_data) * 1000

    if flag_skip_frame:
        continue

    ####################
    #################### triangulate from multiple views
    ####################
    t_ba = time.time()

    #mobile_combs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    #mobile_combs = [(1, 2), (2, 3), (3, 4), (4,2), (4,1), (3,1)] 
    mobile_combs = [(1, 2), (1,3), (1,4), (2, 3), (2, 4), (3,4)] 

    pts_3d_multiple_views = np.zeros((5, 25, len(mobile_combs)))

    c = 0

    for comb in mobile_combs:

        m1 = comb[0]
        m2 = comb[1]

        # K1 = mobile_captures[m1]['K'] # intrinsics
        # R1 = mobile_captures[m1]['rotation'].T # rotation matrix (tranposed)
        # T1 = - np.dot(R1, mobile_captures[m1]['position']) # ?

        # K2 = mobile_captures[m2]['K']
        # R2 = mobile_captures[m2]['rotation'].T
        # T2 = - np.dot(R2, mobile_captures[m2]['position'])

        # P1 = np.dot(K1, np.c_[R1, T1]) # projection matrix of camera 1
        # P2 = np.dot(K2, np.c_[R2, T2]) # projection matrix of camera 2

        K1 = mobile_captures[m1]['K']
        R1 = mobile_captures[m1]['rotation']
        T1 = np.expand_dims(mobile_captures[m1]['position'], axis=1)

        K2 = mobile_captures[m2]['K']
        R2 = mobile_captures[m2]['rotation']
        T2 = np.expand_dims(mobile_captures[m2]['position'], axis=1)

        P1 = np.dot(K1, np.hstack((R1, T1)))
        P2 = np.dot(K2, np.hstack((R2, T2)))

        # 2d points from 2 different views
        pts_2d_1 = np.asarray( [mobile_captures[m1]['person_pose'][::3], mobile_captures[m1]['person_pose'][1::3]] )
        pts_2d_2 = np.asarray( [mobile_captures[m2]['person_pose'][::3], mobile_captures[m2]['person_pose'][1::3]] )

        # triangulate points from different views
        pts_3d = cv.triangulatePoints(P1, P2, pts_2d_1, pts_2d_2)

        pts_3d = pts_3d / pts_3d[3, :]

        # here we store the 3d points together with their confidence values
        # 3d points
        pts_3d_multiple_views[0:3, :, c] = pts_3d[0:3, :]
        # confidence scores
        pts_3d_multiple_views[3, :, c] = mobile_captures[m1]['person_pose'][2::3]
        pts_3d_multiple_views[4, :, c] = mobile_captures[m2]['person_pose'][2::3]

        c += 1

    ####################
    #################### bundle adjustment
    ####################

    th = 0

    pts_3d_x = np.sum(pts_3d_multiple_views[0, :, :] * np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0),
                      axis=1) / (np.sum(np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0), axis=1) + 1e-12)

    pts_3d_y = np.sum(pts_3d_multiple_views[1, :, :] * np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0),
                      axis=1) / (np.sum(np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0), axis=1) + 1e-12)

    pts_3d_z = np.sum(pts_3d_multiple_views[2, :, :] * np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0),
                      axis=1) / (np.sum(np.prod(pts_3d_multiple_views[3:, :, :] > th, axis=0), axis=1) + 1e-12)

    # final 3d points of the pose
    pts_3d = np.asarray([pts_3d_x, pts_3d_y, pts_3d_z])

    pts_2d, pts_2d_3d_indices, pts_2d_cam_indices, camera_params = get_pts_2d_and_cams(mobile_captures)

    # debug reprojection error
    if do_debug_reprojection:
        print('before ba')
        pts_2d_proj = project(pts_3d, camera_params, mobile_captures, pts_2d_3d_indices, pts_2d_cam_indices, title='before ba',
                              viz=do_debug_reprojection)

    idx_to_bundle = np.where(pts_3d[0, :] > 0)
    
    # prepare point file
    pts_file = open('pts.txt', 'w')
    for p in idx_to_bundle[0]:
        pt_3d = pts_3d[:, p]
        pt_3d_idx = np.where(pts_2d_3d_indices == p)

        pts_file.write(str(pt_3d[0]) + ' ' + str(pt_3d[1]) + ' ' + str(pt_3d[2]) + ' ' +
                       str(pt_3d_idx[0].size) + ' ')

        for pp in pt_3d_idx[0]:
            pts_file.write(str(pts_2d_cam_indices[pp]) + ' ' + str(pts_2d[0, pp]) + ' ' + str(pts_2d[1, pp]) + ' ')

        pts_file.write('\n')
    pts_file.close()

    # prepare camera file
    # todo: this can be substituted by preparing the vector to pass to the function directly
    cam_file = open('cams.txt', 'w')
    for cam in camera_params:
        #q = quat.quaternion(*cam[4:8]).inverse()
        q = quat.quaternion(*cam[4:8])
        R = quat.as_rotation_matrix(q)
        C = cam[8:]
        #T = - np.dot(R, C)
        T = C
        cam_file.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n' %
                       (cam[0], cam[2], cam[3], cam[1]/cam[0], 0, q.w, q.x, q.y, q.z, T[0], T[1], T[2]))
    cam_file.close()

    # len(mobile_captures.keys() -> this is because the quaternion is returned with the 3 independent components
    P = np.zeros((12 * len(mobile_captures.keys()) + 3 * idx_to_bundle[0].size - len(mobile_captures.keys()),),
                 dtype=c_double)

    error_init = np.zeros((1,), dtype=c_float)
    error_end = np.zeros((1,), dtype=c_float)
    elapsed_ba_internal = np.zeros((1,), dtype=c_float)

    libmysba.sba(c_char_p(b'cams.txt'), c_char_p(b'pts.txt'), P.ctypes.data_as(POINTER(c_double)), BA_MODALITY, 0,
                 error_init.ctypes.data_as(POINTER(c_float)),
                 error_end.ctypes.data_as(POINTER(c_float)),
                 elapsed_ba_internal.ctypes.data_as(POINTER(c_float)))

    elapsed_ba = (time.time() - t_ba) * 1000

    # evaluation
    elapsed_ba_sequence.append(elapsed_ba)

    newcams = P[:11 * len(mobile_captures.keys())].reshape((len(mobile_captures.keys()), 11))

    newpts = P[11 * len(mobile_captures.keys()):].reshape((idx_to_bundle[0].size, 3)).T

    new_camera_params = np.zeros((len(mobile_captures.keys()), 11))

    # this is necessary because sba uses only the independent components of the quaternion
    new_camera_params[:, 0] = newcams[:, 0]
    new_camera_params[:, 1] = newcams[:, 0] * newcams[:, 3]
    new_camera_params[:, 2] = newcams[:, 1]
    new_camera_params[:, 3] = newcams[:, 2]
    new_camera_params[:, 4] = np.sqrt(1 - newcams[:, 5] ** 2. - newcams[:, 6] ** 2. - newcams[:, 7] ** 2)
    new_camera_params[:, 5:8] = newcams[:, 5:8]
    new_camera_params[:, 8:] = newcams[:, -3:]

    # tranform R and T in the original format
    for c in range(len(mobile_captures.keys())):
        #q = quat.quaternion(*new_camera_params[c, 4:8]).inverse()
        q = quat.quaternion(*new_camera_params[c, 4:8])
        R = quat.as_rotation_matrix(q)
        #C = - np.dot(R, new_camera_params[c, 8:])
        C = new_camera_params[c, 8:] # NOTE
        new_camera_params[c, 4:8] = quat.as_float_array(q)
        new_camera_params[c, 8:] = C

    pts_3d[:, idx_to_bundle[0]] = newpts

    elapsed_all = time.time() - t_all

    reprojection_error_end = compute_reprojection_error(pts_3d, new_camera_params, mobile_captures)
    print('reprojection error: %.4f' % reprojection_error_end)
    reprojection_error_end_sequence.append(reprojection_error_end)

    if do_debug_reprojection:
        print('after ba')
        pts_2d_proj = project(pts_3d, new_camera_params, mobile_captures, pts_2d_3d_indices, pts_2d_cam_indices, title='after ba',
                              viz=do_debug_reprojection)

    if do_o3d_viz:

        for mobile_number, mobile in cameras.items():

            # visualise cameras
            camidx = mobile_number - 1
            R = quat.as_rotation_matrix(quat.quaternion(*new_camera_params[camidx, 4:8]))
            C = new_camera_params[camidx, 8:]

            mobile_pose_world = np.dot(R, mobile_viz_points.T).T + C

            if fr_counter > 0:
                [vis.remove_geometry(l) for l in mobile_captures[mobile_number]['frustum']]
            line_mesh = LineMesh(mobile_pose_world, mobile_viz_lines, mobile_viz_colors, radius=line_mesh_radius)
            line_mesh_geoms = line_mesh.cylinder_segments
            mobile_captures[mobile_number]['frustum'] = line_mesh_geoms
            [vis.add_geometry(l) for l in mobile_captures[mobile_number]['frustum']]

        # filter zero pts_3d
        idx_to_filter = np.where(pts_3d[0, :] == 0)[0]
        _limbs = limbs
        for i in idx_to_filter:
            t = _limbs[:, 0] != i
            _limbs = _limbs[t, :]
            t = _limbs[:, 1] != i
            _limbs = _limbs[t, :]

        # visualise skeleton
        if fr_counter > 0:
            [vis.remove_geometry(l) for l in skeleton]
        line_mesh = LineMesh(pts_3d.T, _limbs, [1, 1, 1], radius=line_mesh_radius)
        if do_white_background:
            line_mesh = LineMesh(pts_3d.T, _limbs, [0, 0, 0], radius=line_mesh_radius)
        line_mesh_geoms = line_mesh.cylinder_segments
        skeleton = line_mesh_geoms
        [vis.add_geometry(l) for l in skeleton]

        if flag_viz_camera_rotation:
            vis.run()
            camera_parameter = vis.get_view_control().convert_to_pinhole_camera_parameters()
            flag_viz_camera_rotation = 0
            vis.run()

        vis.get_view_control().convert_from_pinhole_camera_parameters(camera_parameter)
        vis.poll_events()
        vis.update_renderer()

        if do_save_results:
            vis.capture_screen_image(os.path.join(frame_res_det_folder, '{:04d}.png'.format(fr_counter, fr)))
    fr_counter += 1

    print('All: %.2fms - load_data: %.2fms - ba: %.2fms' %
          (elapsed_all, elapsed_load_data, elapsed_ba))

if do_o3d_viz:
    vis.destroy_window()
