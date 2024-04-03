# ----------------------------- Image processing ------------------------------------

"""
    Task:
        Perform preprocessing of images to facilitate the subsequent identification of certain objects in the image.
        The images provided are screenshots from a video of a dog moving quickly on the dogwalk.
        It is necessary to determine whether the hind legs of the dog have entered the exit zone of the dogwalk.
        The dogwalk is yellow, and the exit zone is pink/red.
    Stages of the Process:
    1. Formalization of the Image Processing Task:
        After research, 4 main approaches to object edge detection were selected:
            1. Laplacian filter
            2. Image clustering using the k-means method
            3. Canny filter
            4. Creation of a boolean mask
        To ensure the correct operation of these methods, various combinations of image preprocessing were used:
            1. Histogram equalization
            2. Image enhancement by adjusting its hue, saturation, and value
            3. Image smoothing (noise reduction)
            4. Image segmentation based on color features

    2. Data Formation:
        The input data consists of pre-selected screenshots from the video,
        compiled in a project folder named 'images/Joy'.

    3. Presentation of Results:
        The result of each image processing pipeline is saved in a separate project folder with the respective name
        'images/Joy_out_pipeline_name'.
        Based on the results of all pipelines, a final compilation of processed images is formed
        in the folder 'images/Joy_out_comparison'.

"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


# ------------------ Filters block ------------------------

def hist_equalization(image):
    """
        Apply histogram equalization to the input image to enhance its contrast.

        This function takes an input image and applies histogram equalization to the Y channel
        of its YUV color space representation. It then converts the equalized YUV image back to
        the BGR color space and displays the original and equalized images side by side.

        Parameters:
        - image (numpy.ndarray): The input image to be equalized.

        Returns:
        - numpy.ndarray: The histogram equalized image.
    """
    img = image

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    cv2.imshow('Color input image', img)
    cv2.imshow('Histogram equalized', img_output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_output


def color_segmentation(image):
    """
        Segment an image based on color using masks derived from HSV color space.

        This function converts the input image to RGB and then to HSV color spaces.
        It defines masks for specific color ranges (pink, grey, green, yellow) in the HSV color space.
        The final segmented image is obtained by bitwise ANDing the original image with the combined mask.

        Parameters:
        - image (numpy.ndarray): The input image to be segmented.

        Returns:
        - tuple: A tuple containing:
            - numpy.ndarray: The segmented image.
            - numpy.ndarray: The combined mask used for segmentation.
    """
    image = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    ## mask of pink (145,100,20) ~ (175, 255,255)
    mask_pink = cv2.inRange(hsv_img, (140, 50, 10), (175, 255, 255))

    ## mask of grey (110,50,50) ~ (130,255,255)
    mask_grey = cv2.inRange(hsv_img, (0, 0, 0), (255, 10, 255))
    not_mask_grey = cv2.bitwise_not(mask_grey)

    ## mask of green (36,0,0) ~ (70, 255,255)
    mask_green = cv2.inRange(hsv_img, (36, 0, 0), (70, 255, 255))
    not_mask_green = cv2.bitwise_not(mask_green)

    ## mask of yellow (15,0,0) ~ (36, 255, 255)
    mask_yellow = cv2.inRange(hsv_img, (15, 0, 0), (36, 255, 255))
    not_mask_yellow = cv2.bitwise_not(mask_yellow)

    mask_all = (not_mask_green & not_mask_yellow & not_mask_grey) | mask_pink
    result_image = cv2.bitwise_and(image, image, mask=mask_all)

    plt.subplot(1, 2, 1)
    plt.imshow(mask_all, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.show()

    return result_image, mask_all


def color_enhancement(image, hue_coeff, saturation_coeff, val_coeff):
    """
        Enhance the color of an image by adjusting its hue, saturation, and value.

        This function converts an input image from the BGR color space to the HSV color space.
        Then, it adjusts the hue, saturation, and value of the image based on the provided coefficients.
        After the adjustments, the image is converted back to the BGR color space.

        Parameters:
        - image (numpy.ndarray): The input image to be color-enhanced.
        - hue_coeff (float): Coefficient to adjust the hue of the image.
        - saturation_coeff (float): Coefficient to adjust the saturation of the image.
        - val_coeff (float): Coefficient to adjust the value of the image.

        Returns:
        - numpy.ndarray: The color-enhanced image.
    """
    # Convert the image from BGR to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Adjust the hue, saturation, and value of the image
    # Adjusts the hue by multiplying it by hue_coeff
    image[:, :, 0] = image[:, :, 0] * hue_coeff
    # Adjusts the saturation by multiplying it by saturation_coeff
    image[:, :, 1] = image[:, :, 1] * saturation_coeff
    # Adjusts the value by multiplying it by val_coeff
    image[:, :, 2] = image[:, :, 2] * val_coeff

    # Convert the image back to BGR color space
    color_enhancement_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    cv2.imshow('color enhancement', color_enhancement_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return color_enhancement_image


def remove_noise(image):
    """
        Remove noise from an image using a median filter.

        This function applies a median filter to the input image to remove noise.
        A median filter is effective in reducing salt-and-pepper noise without blurring edges.

        Parameters:
        - image (numpy.ndarray): The input image with noise.

        Returns:
        - numpy.ndarray: The image after noise removal using median filtering.
    """
    # Remove noise using a median filter
    filtered_image = cv2.medianBlur(image, 5)

    cv2.imshow('Median Blur', filtered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_image


# ------------------ Edge detectors block ------------------------

def laplacian(image):
    """
        Sharpen an image using the Laplacian operator.

        The Laplacian operator is applied to enhance the edges in an image, which results in sharpening.
        This function takes an input image and applies the Laplacian operator to sharpen it.

        Parameters:
        - image (numpy.ndarray): The input image to be sharpened.

        Returns:
        - numpy.ndarray: The sharpened image using the Laplacian operator.
    """
    # Sharpen the image using the Laplacian operator
    sharpened_image = cv2.Laplacian(image, cv2.CV_64F, ksize=9)

    cv2.imshow('Laplacian Sharpening', sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return sharpened_image


def kmeans(image):
    """
        Apply the K-means clustering algorithm to segment an image.

        This function reshapes the input image into a format suitable for K-means clustering.
        It then applies the K-means algorithm to segment the image into 'K' clusters.
        The resulting clustered image is displayed using OpenCV.

        Parameters:
        - image (numpy.ndarray): The input image to be segmented.

        Returns:
        - numpy.ndarray: The image after applying K-means clustering.
    """
    img = image
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    kmeans_image = res.reshape((img.shape))
    cv2.imshow('kmeans clustered image', kmeans_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return kmeans_image


def canny(image):
    """
        Apply the Canny edge detection algorithm to the input image.

        This function applies the Canny edge detection algorithm to the input image using
        predefined parameters for lower and upper thresholds, aperture size, and gradient
        calculation method (L2Gradient). It then displays the resulting edge-detected image.

        Parameters:
        - image (numpy.ndarray): The input image to be processed.

        Returns:
        - numpy.ndarray: The image after applying the Canny edge detection.
    """
    img = image

    # Defining all the parameters
    t_lower = 100  # Lower Threshold
    t_upper = 300  # Upper threshold
    aperture_size = 3  # Aperture size
    L2Gradient = True  # Boolean

    # Applying the Canny Edge filter
    # with Aperture Size and L2Gradient
    canny_img = cv2.Canny(img, t_lower, t_upper,
                          apertureSize=aperture_size,
                          L2gradient=L2Gradient)

    cv2.imshow('Canny', canny_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return canny_img


# ------------------ Pipelines block ------------------------


def image_laplacian_pipeline(image_path):
    """
        Process an image using a sequence of operations:
        histogram equalization, color segmentation, and Laplacian sharpening.

        Parameters:
        - image_path (str): Path to the input image file.

        Returns:
        - sharpened_image: Image after applying the Laplacian sharpening.
    """
    image = cv2.imread(image_path)
    image = hist_equalization(image)
    image, mask = color_segmentation(image)
    image = laplacian(image)

    return image


def image_kmeans_pipeline(image_path):
    """
        Process an image using a sequence of operations:
        color segmentation, color enhancement, and K-means clustering.

        Parameters:
        - image_path (str): Path to the input image file.

        Returns:
        - kmeans_image: Image after applying K-means clustering.
    """
    image = cv2.imread(image_path)
    image, mask = color_segmentation(image)
    image = color_enhancement(image, 1, 0.4, 0.7)
    image = kmeans(image)

    return image


def image_mask_pipeline(image_path):
    """
        Process an image using a sequence of operations:
        histogram equalization, color enhancement, noise removal,
        and color segmentation to generate a mask.

        Parameters:
        - image_path (str): Path to the input image file.

        Returns:
        - mask: Generated mask after segmentation.
    """
    image = cv2.imread(image_path)
    image = hist_equalization(image)
    image = color_enhancement(image, 0.7, 1.5, 0.5)
    image = remove_noise(image)
    image, mask = color_segmentation(image)
    return mask


def image_canny_pipeline(image_path):
    """
        Process an image using a sequence of operations:
        histogram equalization, color segmentation, noise removal,
        and edge detection using Canny.

        Parameters:
        - image_path (str): Path to the input image file.

        Returns:
        - image: Image after applying the Canny edge detection.
    """
    image = cv2.imread(image_path)
    image = hist_equalization(image)
    image, mask = color_segmentation(image)
    image = remove_noise(image)
    image = canny(image)
    return image


# ------------------ Image processing block ------------------------


def concat_images(*images_pathes):
    """
        Concatenate multiple images vertically with white borders in between.

        This function takes a variable number of image paths as input.
        It loads each image, adds a white border to the bottom of each image,
        and concatenates them vertically. The resulting image is returned.

        Parameters:
        - *images_pathes (str): Variable number of paths to the images to be concatenated.

        Returns:
        - numpy.ndarray: The concatenated image.
    """
    pathes = images_pathes
    white = [255, 255, 255]
    image1 = cv2.imread(pathes[0], 0)
    image1 = cv2.copyMakeBorder(image1, 0, 20, 0, 0, cv2.BORDER_CONSTANT, value=white)
    res = image1
    for path in pathes[1:]:
        image = cv2.imread(path, 0)
        image = cv2.copyMakeBorder(image, 0, 20, 0, 0, cv2.BORDER_CONSTANT, value=white)
        res = cv2.vconcat([res, image])

    return res


def compare_directories(*directories_path_in, directory_path_out):
    """
        Compare images from multiple input directories and concatenate them vertically.
        Save the resulting concatenated images to an output directory.

        This function takes multiple input directories and an output directory path.
        For each filename found in the first input directory, it looks for the same
        filename in all input directories, concatenates the corresponding images
        vertically with white borders in between, and saves the resulting image to the
        output directory.

        Parameters:
        - *directories_path_in (str): Variable number of paths to input directories.
        - directory_path_out (str): Path to the output directory where concatenated images will be saved.

        Returns:
        - None
    """
    directories = directories_path_in
    for filename in os.listdir(directories[0]):
        filenames = []
        for directory in directories:
            filepath = os.path.join(directory, filename)
            filenames.append(filepath)

        result_image = concat_images(*filenames)
        f_out = os.path.join(directory_path_out, filename)
        cv2.imwrite(f_out, result_image)
    return


def process_directory(directory_path_in, process, directory_path_out):
    """
        Process images from a directory using a specified image processing function
        and save the results to an output directory.

        Parameters:
        - directory_path_in (str): Input directory path containing images.
        - process (function): Image processing function to apply to each image.
        - directory_path_out (str): Output directory path to save processed images.

        Returns:
        None
    """
    for filename in os.listdir(directory_path_in):
        f_in = os.path.join(directory_path_in, filename)
        result_image = process(f_in)
        f_out = os.path.join(directory_path_out, filename)
        cv2.imwrite(f_out, result_image)
    return


# ------------------ Main function block ------------------------

def main():
    """
        Main function to process images from the 'Joy' directory using various image
        processing pipelines and save the results to respective output directories.
        Additionally, it compiles the processed images from different directories
        into a single user-friendly format.

        Parameters:
        None

        Returns:
        None
    """
    # assign directory
    directory_input = 'images/Joy'
    directory_laplacian_output = 'images/Joy_out_laplacian'
    directory_kmeans_output = 'images/Joy_out_kmeans'
    directory_mask_output = 'images/Joy_out_mask'
    directory_canny_output = 'images/Joy_out_canny'

    # running pipelines for image processing
    process_directory(directory_input, image_laplacian_pipeline, directory_laplacian_output)
    process_directory(directory_input, image_kmeans_pipeline, directory_kmeans_output)
    process_directory(directory_input, image_mask_pipeline, directory_mask_output)
    process_directory(directory_input, image_canny_pipeline, directory_canny_output)

    # compile results of image processing in user-friendly format
    directory_output = 'images/Joy_out_comparison'

    directories = [directory_input,
                   directory_laplacian_output,
                   directory_kmeans_output,
                   directory_mask_output,
                   directory_canny_output]
    compare_directories(*directories, directory_path_out=directory_output)

    print('Image processing complete.\n You can find results in folder "Joy_out_comparison"')


if __name__ == '__main__':
    main()
