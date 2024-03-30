########################################################################
# Rahil Shah
# 25|1|2024
########################################################################

import pyzed.sl as sl
import cv2
import numpy as np
import math
from camera_geometry import CameraGeometry
# zed = sl.Camera()

def extract_camera_parameters(K):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    image_width = int(2 * cx)
    image_height = int(2 * cy)

    focal_length = (fx + fy) / 2.0
    sensor_size = np.sqrt(image_width**2 + image_height**2)
    field_of_view = 2 * np.arctan(sensor_size / (2 * focal_length))

    field_of_view_deg = np.degrees(field_of_view)

    return field_of_view_deg, image_width, image_height
def find_3d_coordinates(x_2d, y_2d):
    # calibration_params = zed.get_camera_information().calibration_parameters
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    h_fov = calibration_params.left_cam.h_fov
    image_size = zed.get_camera_information().camera_configuration.resolution
    image_width = image_size.width
    image_height = image_size.height
    cam_geom = CameraGeometry(height=0.08, yaw_deg=0, pitch_deg=0, roll_deg=0, image_width=image_width, image_height=image_height, field_of_view_deg=h_fov)


    # Define pixel coordinates (u, v)
    u_pixel = x_2d
    v_pixel = y_2d

    # Convert pixel coordinates to 3D coordinates (X, Y, Z) in the road frame
    X, Y, Z = cam_geom.uv_to_roadXYZ_roadframe(u_pixel, v_pixel)
    return X,Y,Z
    
def mouse_callback(event, x, y, flags, param):
    global zed
    if event == cv2.EVENT_LBUTTONDOWN:
        # Find 3D coordinates for the clicked point
        point_3d = find_3d_coordinates(x, y)
        if point_3d is not None:
            # Print the first three values of the 3D coordinates
            print("Clicked pixel coordinates:", x, y)
            # for x in point_3d:
            #     if x != "NaN":
            #         x = int(x)
            print(f"{point_3d[0]*100:.3f}   {point_3d[1]*100:.3f}   {point_3d[2]*100:.3f}\n")  # Extract first three values

            # Save the pixel coordinates and the first three values of 3D coordinates to a CSV file
            with open('pixel_and_3d_coordinates.csv', mode='a') as file:
                file.write(f"{x},{y},{point_3d[2]*100:.3f}\n")

def main():
    global zed
    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.camera_fps = 30
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    # init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(-1)

    cv2.namedWindow("ZED Image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ZED Image", mouse_callback)

    while True:
        image = sl.Mat()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            image_data = image.get_data()
            opencv_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

            # Display the image
            cv2.imshow("ZED Image", image_data)

        # Check for ESC key press to exit the loop
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Close the camera
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()