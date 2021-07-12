"""
Checker file

Gonna read and explore Mohamed's calibrations files
"""
import cv2

def load_intrinsic(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

def load_extrinsic(path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    R_matrix = cv_file.getNode("R").mat()
    t_matrix = cv_file.getNode("t").mat()

    cv_file.release()
    return [R_matrix, t_matrix]

for point in ['point1', 'point2', 'point3', 'point4']:

    print(point)

    # paths
    calib = '/media/weber/Ubuntu2/ubuntu2/Human_Pose/QMUL-data/four_viewpoints_ballet/calib/' + point
    calib_intrinsic = calib + "/intrinsic/cam.cal"
    calib_extrinsic = calib + "/extrinsic/extr.cal"

    # load intrinsics
    camera_matrix, dist_matrix = load_intrinsic(calib_intrinsic)
    print("INTRINSICS")
    print("Camera matrix:\n{}\n".format(camera_matrix))
    print("Dist matrix:\n{}\n".format(dist_matrix))

    print("fl:", camera_matrix[0,0], camera_matrix[1,1])
    print("pp:", camera_matrix[0,2], camera_matrix[1,2])

    # load extrinsics
    R_matrix, t_matrix = load_extrinsic(calib_extrinsic)
    print("EXTRINSICS")
    print("Rotation matrix:\n{}\n".format(R_matrix))
    print("Translation vector:\n{}".format(t_matrix))
