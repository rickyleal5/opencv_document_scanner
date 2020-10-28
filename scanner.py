# Document scanner using OpenCV

# Libraries
import cv2
import numpy as np
from skimage.filters import threshold_local
import time

# Scanner class
class Scanner:
    # Constructor
    def __init__(self, output_width=480, output_height=640):
        self.__image = None
        self.__kernel = np.ones((3, 3))
        self.__output_width = output_width
        self.__output_height = output_height
        self.__output = None
        self.__image_copy = None

    # Checks if the image is not empty
    def __check_image_size(self, file_path):
        isValid = False
        image = cv2.imread(file_path)
        if image is not None:
            isValid = True
        return isValid

    # Complete process of scanning the image
    def scanImage(self, file_path):
        #If the image is not empty, scan
        if (self.__check_image_size(file_path) == True):
            self.__set_image(cv2.imread(file_path))
            self.__set_image_copy(cv2.imread(file_path))
            self.__preprocess_image()
            biggest = self.__getContours()
            #Check if the scanner did not find any contours
            if biggest.size == 0:
                print("Scanner could not find any contours on the image")
            else:
                warped_image = self.__getWarpedImage(biggest)
                output = self.__convert_to_black_and_white(warped_image)
                self.__set_output(output)
                self.__save_output_file()
        else:
            print("Invalid image")

    #Transform the image from gray-scale to black and white pixels
    def __convert_to_black_and_white(self, warped_image):
        #Convert to gray scale
        warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        #Create a threshold to convert the gray values to either black or white
        threshold = threshold_local(warped_image, 7, offset = 10, method = "gaussian")
        #Convert values to 1s and 0s, then convert the 1s to 255s
        warped_image = (warped_image > threshold).astype(np.uint8) * 255
        return warped_image


    def __set_image(self, image):
        self.__image = np.copy(image)

    def __set_image_copy(self, image):
        self.__image_copy = np.copy(image)

    def __preprocess_image(self):
        #Preprocess the copy of the image
        frame = cv2.cvtColor(self.__image_copy, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.Canny(frame, 100, 200)
        frame = cv2.morphologyEx(frame, op=cv2.MORPH_CLOSE, kernel=self.__kernel,
                                 iterations = 2)
        #Update the copy of the image
        self.__set_image_copy(frame)

    def __findContours(self):
        return cv2.findContours(self.__image_copy, mode=cv2.RETR_EXTERNAL,
                                    method=cv2.CHAIN_APPROX_NONE)

    def __getContours(self):
        #Variables
        biggest_contour = np.array([])
        biggest_area = np.array([])
        max_area = 0
        #Find contours
        contours, _ = self.__findContours()
        #Check each contour
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area > 10000:
                perimeter = cv2.arcLength(contour, closed=True)
                contour_approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, closed=True)
                #If it found a squared contour and the area of this contour is more than the max_area, save this contour
                if contour_area > max_area and len(contour_approximation) == 4:
                    biggest_contour = contour_approximation
                    max_area = contour_area

        return biggest_contour

    def __getWarpedImage(self, biggest):
        #Reorganize coordinates to make sure a rectangular output is made
        biggest = self.reorder_coordinates(biggest)
        coordinates = np.float32(biggest)
        ordered_coordinates = np.float32([[0, 0], [self.__output_width, 0], [0, self.__output_height],
                                          [self.__output_width, self.__output_height]])

        #Warp perspective
        array = cv2.getPerspectiveTransform(coordinates, ordered_coordinates)
        warped_image = cv2.warpPerspective(self.__image, array, (self.__output_width, self.__output_height))

        return warped_image


    def reorder_coordinates(self, coordinates):
        #Reorganize the 4 coordinate points
        coordinates = coordinates.reshape(4, 2)
        new_coordinates = np.zeros((4, 1, 2), dtype=np.int32)
        add = np.sum(coordinates, axis=1)
        diff = np.diff(coordinates, axis=1)
        new_coordinates[0] = coordinates[np.argmin(add)]
        new_coordinates[1] = coordinates[np.argmin(diff)]
        new_coordinates[2] = coordinates[np.argmax(diff)]
        new_coordinates[3] = coordinates[np.argmax(add)]

        return new_coordinates

    def __set_output(self, image):
        self.__output = image

    def __save_output_file(self):
        cv2.imwrite('output_{}.png'.format(str(time.time()).replace('.','_')), self.__output)


def main():
    #Image filepath
    file_path = 'george.jpg'

    #Output dimensions
    output_width, output_height = 480, 640

    #Create a Scanner instance
    scanner = Scanner(output_width, output_height)
    #Scan image
    scanner.scanImage(file_path)

main()
