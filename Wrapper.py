#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 1: AutoCalib

Author(s): 
Uday Girish Maradana - RBE MS Student
"""

import numpy as np
import cv2
from scipy import optimize
from helpers.logger import setup_logger
from helpers.utils import FileHandler
import os
import copy
from helpers.cv2_calib import CamcalibCV2
import shutil
import argparse


class CameraCalibration:
    def __init__(
        self, logger, file_handler, path, checker_size=(9, 6), square_size=21.5
    ):
        """
        Initializes the Wrapper object.

        Args:
            logger: The logger object for logging messages.
            file_handler: The file handler object for handling file operations.
            path: The path to the directory containing the images for calibration.
            checker_size: The size of the checkerboard pattern (rows, columns).
            square_size: The size of each square in the checkerboard pattern.

        Returns:
            None
        """
        self.description = "Camera Calibration for Pinhole Camera Model."
        self.logger = logger
        self.file_handler = file_handler
        self.path = path
        self.checker_size = checker_size
        self.square_size = square_size
        self._get_world_coordinates()

    def _read_images(self):
        """
        Reads and returns a list of images and their corresponding filenames from the specified path.

        Returns:
            images (list): A list of images read from the path.
            filenames (list): A list of filenames corresponding to the images.
        """
        images = []
        filenames = []
        for file in os.listdir(self.path):
            if file.endswith(".jpg"):
                images.append(cv2.imread(os.path.join(self.path, file)))
                filenames.append(file)
        return images, filenames

    def _get_world_coordinates(self):
        """
        Calculate the world coordinates of the checkerboard corners.

        Returns:
            numpy.ndarray: Array of world coordinates of the checkerboard corners.
        """
        world_points = []
        for i in range(1, self.checker_size[1] + 1):
            for j in range(1, self.checker_size[0] + 1):
                world_points.append([i * self.square_size, j * self.square_size])
        self.world_points = np.array(world_points)

    def _get_image_points(self, images):
        """
        Get the image points (corners) of chessboard pattern in the given images.

        Args:
            images (List[np.ndarray]): List of input images.

        Returns:
            np.ndarray: Array of image points (corners) for each image.
        """
        image_points = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                image_points.append(corners)
        return np.array(image_points)

    def plot_original_corners(
        self, images, corners, names, corner_path="original_corners/"
    ):
        """
        Plot the original corners on the images and save the output images.

        Args:
            images (list): List of input images.
            corners (list): List of corner coordinates for each image.
            names (list): List of image names.
            corner_path (str, optional): Path to save the output images. Defaults to "original_corners/".
        """
        for i in range(len(images)):
            img = copy.deepcopy(images[i])
            img = cv2.drawChessboardCorners(img, (9, 6), corners[i], True)
            output_path = str(names[i].split(".")[0]) + ".jpg"
            self.file_handler.write_output(img, corner_path, output_path)

    def plot_corners(
        self, images, corners, projected, names, corner_path="comparison/"
    ):
        """
        Plot corners on the input images and save the output images.

        Args:
            images (list): List of input images.
            corners (list): List of corner coordinates for each image.
            projected (list): List of projected corner coordinates for each image.
            names (list): List of image names.
            corner_path (str, optional): Path to save the output images. Defaults to "comparison/".
        """
        for i in range(len(images)):
            img = copy.deepcopy(images[i])
            for j in range(len(corners[i])):
                # Draw outer circles without filling
                img = cv2.circle(
                    img,
                    (int(corners[i][j][0][0]), int(corners[i][j][0][1])),
                    10,
                    (0, 0, 255),
                    1,  # Set thickness to a positive value for outer circle without filling
                )
                img = cv2.circle(
                    img,
                    (int(projected[i][j][0][0]), int(projected[i][j][0][1])),
                    10,
                    (0, 255, 0),
                    1,  # Set thickness to a positive value for outer circle without filling
                )
            output_path = str(names[i].split(".")[0]) + ".jpg"
            self.file_handler.write_output(img, corner_path, output_path)

    def _find_homography(self, image_point):
        """
        Computes the homography matrix given a set of image points and corresponding world points.

        Args:
            image_point (ndarray): Array of image points with shape (N, 1, 2), where N is the number of points.

        Returns:
            ndarray: Homography matrix with shape (3, 3).
        """
        H = []
        for i in range(len(image_point)):
            x_i = image_point[i][0, 0]
            y_i = image_point[i][0, 1]
            X_i = self.world_points[i, 0]
            Y_i = self.world_points[i, 1]
            aX_T = np.array([-X_i, -Y_i, -1, 0, 0, 0, x_i * X_i, x_i * Y_i, x_i])
            aY_T = np.array([0, 0, 0, -X_i, -Y_i, -1, y_i * X_i, y_i * Y_i, y_i])
            H.append(aX_T)
            H.append(aY_T)
        H = np.array(H)
        U, S, V_T = np.linalg.svd(H, full_matrices=True)
        H = V_T.T[:, -1]
        H = H / H[8]
        H = H.reshape(3, 3)
        return H

    def _get_all_homographies(self, image_points):
        """
        Calculates the homography for each image point in the given list.

        Args:
            image_points (list): A list of image points.

        Returns:
            list: A list of homographies corresponding to each image point.
        """
        homographies = []
        for image_point in image_points:
            homographies.append(self._find_homography(image_point))
        return homographies

    def get_Vij(self, H, i, j):
        """
        Calculate the V matrix for given H, i, and j.

        Parameters:
        H (numpy.ndarray): Homography matrix.
        i (int): Index i.
        j (int): Index j.

        Returns:
        numpy.ndarray: V matrix.
        """
        H = H.T
        V = np.array(
            [
                [H[i][0] * H[j][0]],
                [H[i][0] * H[j][1] + H[i][1] * H[j][0]],
                [H[i][1] * H[j][1]],
                [H[i][2] * H[j][0] + H[i][0] * H[j][2]],
                [H[i][2] * H[j][1] + H[i][1] * H[j][2]],
                [H[i][2] * H[j][2]],
            ]
        )
        return V.T

    def _compute_B(self, H_all):
        """
        Compute the matrix B using the given list of homography matrices.

        Parameters:
        - H_all: List of homography matrices

        Returns:
        - B: Matrix B computed from the homography matrices
        """
        v = []
        for i in range(len(H_all)):
            h = H_all[i]
            v.append(self.get_Vij(h, 0, 1))
            v.append(self.get_Vij(h, 0, 0) - self.get_Vij(h, 1, 1))
        v = np.array(v)
        v = v.reshape(-1, 6)
        _, _, V_T = np.linalg.svd(v, full_matrices=True)
        b = V_T[-1]

        B = np.array(
            [
                [b[0], b[1], b[3]],
                [b[1], b[2], b[4]],
                [b[3], b[4], b[5]],
            ]
        )
        return B

    def _compute_intrinsic(self, B):
        """
        Compute the intrinsic camera matrix K using the given matrix B.

        Parameters:
        - B: numpy.ndarray, shape (3, 3), the matrix B

        Returns:
        - K: numpy.ndarray, shape (3, 3), the intrinsic camera matrix K
        """
        v0 = (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2]) / (
            B[0, 0] * B[1, 1] - B[0, 1] ** 2
        )
        lambd = (
            B[2, 2]
            - (B[0, 2] ** 2 + v0 * (B[0, 1] * B[0, 2] - B[0, 0] * B[1, 2])) / B[0, 0]
        )
        alpha = np.sqrt(lambd / B[0, 0])
        beta = np.sqrt((lambd * B[0, 0]) / (B[0, 0] * B[1, 1] - B[0, 1] ** 2))
        gamma = -1 * (B[0, 1] * alpha**2 * beta) / lambd
        u0 = (gamma * v0 / beta) - (B[0, 2] * alpha**2 / lambd)
        K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        return K

    def get_extrinsics(self, H_all, K):
        """
        Calculate the extrinsics (rotation and translation) given a set of homography matrices and the camera intrinsic matrix.

        Parameters:
        - H_all (list of numpy arrays): List of homography matrices.
        - K (numpy array): Camera intrinsic matrix.

        Returns:
        - R (list of numpy arrays): List of rotation matrices.
        - T (list of numpy arrays): List of translation vectors.
        """
        R = []
        T = []
        for H in H_all:
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]
            lam = 1 / np.linalg.norm(np.dot(np.linalg.inv(K), h1), ord=2)
            r1 = lam * np.dot(np.linalg.inv(K), h1)
            r2 = lam * np.dot(np.linalg.inv(K), h2)
            # r3 = np.cross(r1, r2)  # Making r3
            t = lam * np.dot(np.linalg.inv(K), h3)
            R.append(np.array([r1, r2]))
            T.append(t)
        return R, T

    def _get_intrinsics(self, image_points):
        """
        Compute the intrinsic camera matrix and homographies.

        Args:
            image_points (list): List of image points.

        Returns:
            tuple: A tuple containing the intrinsic camera matrix (K) and homographies (H_all).
        """
        H_all = self._get_all_homographies(image_points)
        B = self._compute_B(H_all)
        K = self._compute_intrinsic(B)

        return K, H_all

    def calculate_reprojection_error(self, params, R, T, image_points, world_points):
        """
        Calculates the reprojection error for a given set of camera parameters, rotation matrices, translation vectors,
        image points, and corresponding world points.

        Args:
            params (tuple): Camera parameters (alpha, gamma, beta, u0, v0, k1, k2).
            R (list): List of rotation matrices.
            T (list): List of translation vectors.
            image_points (list): List of image points.
            world_points (list): List of corresponding world points.

        Returns:
            error (ndarray): Array of reprojection errors for each image.
            reprojected_corners (ndarray): Array of reprojected corner points for each image.
        """
        alpha, gamma, beta, u0, v0, k1, k2 = params
        K = np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1]])
        error = []
        reprojected_corners = []
        for i in range(len(image_points)):
            img_corners = image_points[i]
            RT = np.vstack((R[i], T[i]))
            RT = RT.T
            H = np.dot(K, RT)
            temp_error = 0
            temp_reprojected_corners = []
            for j in range(image_points[i].shape[0]):
                world_point = world_points[j]
                world_point = np.append(world_point, 1)
                world_point = world_point.reshape(-1, 1)
                world_point = world_point.T
                reprojected_point = np.matmul(RT, world_point.T)
                reprojected_point = reprojected_point / reprojected_point[2]
                corner_point_orig = img_corners[j]
                corner_point_orig = np.array(
                    [corner_point_orig[0, 0], corner_point_orig[0, 1], 1]
                )
                corner_point = np.matmul(H, world_point.T)
                corner_point = corner_point / corner_point[2]

                x = reprojected_point[0]
                y = reprojected_point[1]
                u = corner_point[0]
                v = corner_point[1]

                r = np.square(x) + np.square(y)
                u_hat = u + (u - u0) * (k1 * r + k2 * np.square(r))
                v_hat = v + (v - v0) * (k1 * r + k2 * np.square(r))

                corner_hat = np.array([u_hat, v_hat, 1], dtype=np.float32)
                temp_reprojected_corners.append(
                    np.array((corner_hat[0], corner_hat[1]))
                )

                temp_error += np.linalg.norm((corner_point_orig - corner_hat), ord=2)
                # corner_hat.astype(np.float32)
            error.append(temp_error / image_points[i].shape[0])
            reprojected_corners.append(temp_reprojected_corners)

        return np.array(error), np.array(reprojected_corners)

    def optimization_function(self, params, R, T, image_points, world_points):
        """
        Calculates the error of the optimization function.

        Args:
            params (numpy.ndarray): The parameters of the optimization function.
            R (numpy.ndarray): The rotation matrix.
            T (numpy.ndarray): The translation vector.
            image_points (numpy.ndarray): The image points.
            world_points (numpy.ndarray): The world points.

        Returns:
            numpy.ndarray: The flattened error array.
        """
        error, _ = self.calculate_reprojection_error(
            params, R, T, image_points, world_points
        )
        return error.flatten()


def main():
    """
    Main function for camera calibration.

    This function performs camera calibration using a set of calibration images.
    It computes the camera intrinsics, extrinsics, and distortion parameters,
    and then optimizes them using the Levenberg-Marquardt algorithm.
    Finally, it compares the results with the baseline calibration from OpenCV.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Camera Calibration")
    parser.add_argument(
        "--path",
        type=str,
        default="Calibration_Imgs/",
        help="Path to the directory containing the images for calibration.",
        required=False,
    )

    args = parser.parse_args()
    # Initialize logger and file handler
    logger = setup_logger()
    file_handler = FileHandler()
    if os.path.exists("Results"):
        logger.info("Cleaning Old Results Folder")
        shutil.rmtree("Results")
    logger.info("ORIGINAL IMPLEMENTATION OF CAMERA CALIBRATION")
    logger.info("Initial Values")

    # Define the path to the images
    path = str(args.path)

    # Create an instance of CameraCalibration
    cam_calib = CameraCalibration(logger, file_handler, path)

    images, names = cam_calib._read_images()
    # Get image points from the images
    image_points = cam_calib._get_image_points(images)

    # Compute intrinsics
    K, H_all = cam_calib._get_intrinsics(image_points)

    # Get extrinsics (rotation and translation)
    R, T = cam_calib.get_extrinsics(H_all, K)

    # Initial guess for distortion parameters
    K_distortion = np.array([0, 0])

    logger.info(K)
    logger.info(K_distortion)

    # Convert to Param vector
    param = np.array(
        [K[0, 0], K[0, 1], K[1, 1], K[0, 2], K[1, 2], K_distortion[0], K_distortion[1]]
    )
    logger.info(param)

    reprojection_error, _ = cam_calib.calculate_reprojection_error(
        param, R, T, image_points, cam_calib.world_points
    )
    logger.info("Reprojection Error for all images:{}".format(reprojection_error))

    reprojection_error = np.mean(reprojection_error)

    logger.info("Optimization without Distortion")
    # Optimize the intrinsics
    optimized_params = optimize.least_squares(
        cam_calib.optimization_function,
        param,
        args=(R, T, image_points, cam_calib.world_points),
        method="lm",
    )

    res_params = optimized_params.x

    reprojection_error_new, reprojected_corners_new = (
        cam_calib.calculate_reprojection_error(
            res_params, R, T, image_points, cam_calib.world_points
        )
    )
    logger.info(
        "Reprojection  New error for all images:{}".format(reprojection_error_new)
    )
    reprojection_error_new = np.mean(reprojection_error_new)

    new_K = np.array(
        [
            [res_params[0], res_params[1], res_params[3]],
            [0, res_params[2], res_params[4]],
            [0, 0, 1],
        ]
    )
    new_distortion = np.array([res_params[5], res_params[6]])
    logger.info(new_K)
    logger.info(new_distortion)
    logger.info("Reprojection Error Before Optimization: %f", reprojection_error)
    logger.info("Reprojection Error After Optimization: %f", reprojection_error_new)

    logger.info("Optimization with Distortion")
    # Optimize the intrinsics and distortion parameters

    # Comparison with baseline
    logger.info("BASELINE COMPARISON WITH CV2 - OPENCV Function")
    cam_calib_cv2 = CamcalibCV2()
    mtx, dst = cam_calib_cv2.calib()

    logger.info("CV2 Baseline Intrinsics:{}".format(mtx))
    logger.info("CV2 Baseline Distortion:{}".format(dst))

    cv2_projection_error = cam_calib_cv2.calculate_projection_err()
    logger.info("CV2 Baseline Reprojection Error:{}".format(cv2_projection_error))
    # Reproject images using the optimized parameters
    cam_calib.plot_original_corners(images, image_points, names)
    distortion_error = dict()
    for i in range(len(images)):
        img = images[i]
        mtx = new_K
        dist = np.array([K_distortion[0], K_distortion[1], 0, 0, 0])
        # dist = np.array([0.1684, 0.7002, 0, 0, 0])
        img = cv2.undistort(img, mtx, dst)
        output_path = str(names[i].split(".")[0]) + ".jpg"
        file_handler.write_output(img, "undistorted/", output_path)

    # Plot new reprojected corners
    # Convert reprojected corners to int

    # convert the format of reprojected corners new to same as image_points
    reprojected_corners_mod = []
    for i in range(len(reprojected_corners_new)):
        temp = []
        for j in range(len(reprojected_corners_new[i])):
            temp.append(
                [
                    [
                        reprojected_corners_new[i][j][0],
                        reprojected_corners_new[i][j][1],
                    ]
                ]
            )
        reprojected_corners_mod.append(temp)
    cam_calib.plot_original_corners(
        images, np.array(reprojected_corners_mod), names, "reprojected/"
    )

    cam_calib.plot_corners(
        images, image_points, np.array(reprojected_corners_mod), names, "comparison/"
    )


if __name__ == "__main__":
    main()
