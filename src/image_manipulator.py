# Image Manipulator File
# date: 01/04/2021
# name: Ibrahim Sheriff Sururu
# description: This file holds the ImageM class which can be used to load image
# files into memory for manipulation, feature detection, display or altering. The
# class can read an image from a file or work with loaded image data. There are a
# range of functions to get object related data such as the dimensions of an image
# as well as functions to set the values of object related data such as the pixel
# values of the image.
# One can obtain a greyscale version of an image, a scaled version of an image,
# the features of an image, do feature matching between two images, compute matching
# accuracy and repeatability, apply a filter to an image and do unsharp masking.
# At each stage of image manipulation one can view the results in a UI window or
# save the results to an image file.
import cv2 as cv
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import math
import statistics as stats


class ImageM:
    """
    This is the class that can be used to collect, modify and manipulate image
    data. The class can be used to obtain a greyscale version of an image, a
    scaled version of an image, the features of an image, do feature matching
    between two images, compute matching accuracy and repeatability, apply a
    filter to an image and do unsharp masking.
    """

    def __init__(self, *args):
        """
        Constructor that is used to contrust an ImageM object. *args can hold
        an address to an image to load from a file or hold an image data array.
        """
        if isinstance(args[0], str):
            self.image_addr = args[0]
            self.image = cv.imread(self.image_addr)
        else:
            self.image_addr = None
            self.image = args[0]
        self.average_pixel_value = None
        self.rows = self.image.shape[0]
        self.cols = self.image.shape[1]
        self.nbh_avgs = None
        if len(self.image.shape) == 2:
            self.channels = 1
        else:
            self.channels = self.image.shape[2]

    def copy(self):
        temp = ImageM(self.image_addr)
        return temp

    def get_image_data(self):
        """
        Get the image data array of the image loaded into the object.

        :return the image data array of the loaded image
        """
        return self.image

    def get_row_num(self):
        """
        Get the number of rows of the image data array.

        :return the number of rows of the image data array
        """
        return self.rows

    def get_col_num(self):
        """
        Get the number of columns of the image data array.

        :return the number of columns of the image data array
        """
        return self.cols

    def get_channel_num(self):
        """
        Get the number of channels of the image loaded into the object i.e. 3
        if the image can be represented using the RGB system and 1 if it is a
        greyscale image.

        :return the number of channels of the image
        """
        return self.channels

    def get_pixel_num(self):
        """
        Get the number of pixels in the image data.

        :return the number of pixels of the loaded image
        """
        return self.image.size

    def get_pixel_value(self, row, col):
        """
        Get the pixel value at the given row and column position in the image data.

        :param row: the row position of the pixel
        :param col: the column position of the pixel
        :return the value of the pixel at a given row and column position
        """
        return self.image[row, col]

    def resize_image(self, width, height):
        """
        Resize the image associated with this object to the given width and height.

        :param width: desired width of the image
        :param height: desired height of the image
        """
        self.cols = width
        self.rows = height
        self.image = cv.resize(self.image, (width, height))

    def compute_average_pixel_value(self):
        """
        Compute the average pixel value for each core colour of the image i.e.
        Red, Blue and Red, for the entire image. The result is stored as a
        (B, G, R) tuple.
        """
        total_blue = 0
        total_green = 0
        total_red = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                total_blue = total_blue + self.image[i, j][0]
                total_green = total_green + self.image[i, j][1]
                total_red = total_red + self.image[i, j][2]

        pixel_total = self.rows * self.cols
        avg_blue = int(total_blue / pixel_total)
        avg_green = int(total_green / pixel_total)
        avg_red = int(total_red / pixel_total)
        self.average_pixel_value = (avg_blue, avg_green, avg_red)

    def get_average_pixel_value(self):
        """
        Return the  (B, G, R) tuple instance variable holding the average 
        pixel value for each main colour of the image.
        """
        return self.average_pixel_value

    def compute_neighbourhood_pixel_values(self, nbh_rows, nbh_cols, w_rows, w_cols):
        """
        Compute the average pixel value for each core colour of each neighbourhood of
        the image.

        :param nbh_rows: the number of rows in the neighbourhood
        :param nbh_cols: the number of columns in the neighbourhood
        :param w_rows: width of the neighbourhood
        :param w_cols: width of the neighbourhood
        """
        self.nbh_avgs = [[0.0] * nbh_cols for n in range(0, nbh_rows)]
        row_start = 0
        for i in range(0, nbh_rows):
            col_start = 0
            for j in range(0, nbh_cols):
                temp_nbh_avg = self.get_average_neighbourhood_pixel_value(
                    row_start, col_start, w_rows, w_cols)
                self.nbh_avgs[i][j] = temp_nbh_avg
                col_start = col_start + w_cols
            row_start = row_start + w_rows

    def get_average_neighbourhood_pixel_value_for_tile(self, r, c):
        """
        Returns the average pixel value for each core colour of the specified 
        neighbourhood of the image.

        :param r: the starting row position of the neighbourhood
        :param c: the starting column position of the neighbourhood
        :return: the average pixel value of the neighbourhood
        """

        return self.nbh_avgs[r][c]

    def get_average_neighbourhood_pixel_value(self, i, j, w_rows, w_cols):
        """
        Returns the pixels in the neighbourhood of size wxw collected from the
        starting row position i and starting column position j.

        :param i: the starting row position
        :param j: the starting column position
        :param w_rows: number of rows in the neighbourhood
        :param w_cols: number of columns in the neighbourhood
        :return: an image data array of size wxw with the pixels in the given neighbourhood
        """
        nbh = np.zeros((w_rows, w_cols, self.channels), np.uint8)
        r = math.ceil(w_rows/2)
        c = math.ceil(w_cols/2)
        start_row = i - r
        start_col = j - c
        nbh_row = 0
        nbh_col = 0
        for k in range(start_row, start_row + w_rows):
            nbh_col = 0
            for l in range(start_col, start_col + w_cols):
                # ensure we are still within the image
                if k >= 0 and k < self.rows and l >= 0 and l < self.cols:
                    nbh[nbh_row, nbh_col] = self.image[k, l]
                else:
                    # for boundary cases
                    if self.channels == 1:
                        nbh[nbh_row, nbh_col] = 0
                    else:
                        nbh[nbh_row, nbh_col] = [0, 0, 0]
                nbh_col = nbh_col + 1
            nbh_row = nbh_row + 1

        total_blue = 0
        total_green = 0
        total_red = 0
        for i in range(0, w_rows):
            for j in range(0, w_cols):
                total_blue = total_blue + nbh[i, j][0]
                total_green = total_green + nbh[i, j][1]
                total_red = total_red + nbh[i, j][2]

        pixel_total = w_rows * w_cols
        avg_blue = int(total_blue / pixel_total)
        avg_green = int(total_green / pixel_total)
        avg_red = int(total_red / pixel_total)
        t = (avg_blue, avg_green, avg_red)
        return t

    def fill_neighbourhood(self, i, j, w_rows, w_cols, tile_image_m):
        """
        Returns the pixels in the neighbourhood of size wxw collected from the
        starting row position i and starting column position j.

        :param i: the starting row position
        :param j: the starting column position
        :param w_rows: number of rows in the neighbourhood
        :param w_cols: number of columns in the neighbourhood
        :param tile_image_m: the tile ImageM to get the pixel to fill the neighbourhood
        :return: an image data array of size wxw with the pixels in the given neighbourhood
        """
        r = math.ceil(w_rows/2)
        c = math.ceil(w_cols/2)
        start_row = i - r
        start_col = j - c
        nbh_row = 0
        for k in range(start_row, start_row + w_rows):
            nbh_col = 0
            for l in range(start_col, start_col + w_cols):
                self.image[k, l] = tile_image_m.image[nbh_row, nbh_col]
                nbh_col = nbh_col + 1
            nbh_row = nbh_row + 1

    def get_colour_difference(self, colour1, colour2):
        part = (colour1[0] - colour2[0]) ** 2 + (colour1[1] -
                                                 colour2[1]) ** 2 + (colour1[2] - colour2[2]) ** 2
        return math.sqrt(part)

    def get_colour_perception_measure(self, colour1, colour2):
        """
        Computes and returns the human colour perception value between the two given colours.

        :param colour1: the first colour
        :param colour2: the second colour
        :return: he human colour perception value between the two given colours
        """
        underline_r = (colour1[2] + colour2[2]) / 2
        delta_r = colour1[2] - colour2[2]
        delta_g = colour1[1] - colour2[1]
        delta_b = colour1[0] - colour2[0]
        part1 = (2 + (underline_r/256)) * (delta_r ** 2)
        part2 = 4 * (delta_g ** 2)
        part3 = (2 + ((255 - underline_r)/256)) * (delta_b ** 2)
        return math.sqrt(part1 + part2 + part3)

    def get_greyscale_image1(self):
        """
        Get the greyscale version of the loaded image using the cv library.

        :return the greyscale version of the loaded image
        """
        return ImageM(cv.cvtColor(self.image, cv.COLOR_BGR2GRAY))

    def get_greyscale_image2(self):
        """
        Get the greyscale version of the loaded image using the Weighted(Luminosity)
        method to obtain a value of each colour pixel.

        :return the greyscale version of the loaded image
        """
        new_image = np.zeros((self.rows, self.cols, 1), np.uint8)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                # output_value = (0.2989 x R) + (0.5870 x G) + (0.1140 * B)
                new_image[i, j] = (0.2989 * self.image[i, j][2]) + (0.5870 *
                                                                    self.image[i, j][1]) + (0.1140 * self.image[i, j][0])

        return ImageM(new_image)

    def get_greyscale_image3(self):
        """
        Get the greyscale version of the loaded image using the Weighted(Luminosity)
        method to obtain a value of each colour pixel except the output image has
        3 channels meaning that it can support colour pixels too.

        This image can be useful for displaying experimental results as the greyscale
        image does not bright colours meaning the output image can clearly display results
        as coloured points.

        :return 3 channel greyscale version of the loaded image
        """
        new_image = np.zeros((self.rows, self.cols, 3), np.uint8)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                new_image[i, j] = (0.2989 * self.image[i, j][2]) + (0.5870 *
                                                                    self.image[i, j][1]) + (0.1140 * self.image[i, j][0])

        return ImageM(new_image)

    def set_pixel_value(self, row, col, value):
        """
        Set the pixel at a particular row and col point to the given value.

        :param row: the row position of the pixel to set
        :param col: the column position of the pixel to set
        :param value: the pixel value to use to set the position with
        """
        self.image[row, col] = value

    def replace_green_screen(self, new_background):
        """
        Replace the green pixels in a green screen background in the loaded image
        with the corresponding pixels in the image in the given background ImageM
        object.

        :param new_background: ImageM object holding the image data to use to replace
                                                   the green screen background with.
        """
        green_temp = self.image[0, 0]
        # use floats for computation purposes to avoid unnecessary clipping
        blue_val = float(green_temp[0])
        green_val = float(green_temp[1])
        red_val = float(green_temp[2])
        for r in range(0, self.rows):
            for c in range(0, self.cols):
                temp_value = self.image[r, c]
                temp_diff = (float(temp_value[0]) - blue_val) ** 2 + (
                    float(temp_value[1]) - green_val) ** 2 + (float(temp_value[2]) - red_val) ** 2
                temp_diff = temp_diff ** 0.5
                if temp_diff <= 100.0:
                    self.image[r, c] = new_background.get_pixel_value(r, c)

    def get_green_screen_silhouette(self):
        """
        Replace the green pixels in a green screen background in the loaded image
        using thresholding meaning that the green pixels are set to black and the
        desired remaining pixels are set to white.

        :return: a silhouette looking image as the result of thresholding
        """
        new_image = np.zeros((self.rows, self.cols, self.channels), np.uint8)
        green_temp = self.image[0, 0]
        # use floats for computation purposes to avoid unnecessary clipping
        blue_val = float(green_temp[0])
        green_val = float(green_temp[1])
        red_val = float(green_temp[2])
        for r in range(0, self.rows):
            for c in range(0, self.cols):
                temp_value = self.image[r, c]
                temp_diff = (float(temp_value[0]) - blue_val) ** 2 + (
                    float(temp_value[1]) - green_val) ** 2 + (float(temp_value[2]) - red_val) ** 2
                temp_diff = temp_diff ** 0.5
                if temp_diff <= 100.0:
                    new_image[r, c] = [0, 0, 0]
                else:
                    new_image[r, c] = [255, 255, 255]

        return ImageM(new_image)

    def get_median_filtered_image(self, w):
        """
        Returns the output image after the Median Filter is applied using a neighbourhood
        of size wxw.

        :param w: the value used to determine the size of the neighbourhood to use
                          when filtering
        :return: Median Filtered output image
        """
        new_image = np.zeros((self.rows, self.cols, self.channels), np.uint8)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                temp_nbh = self.get_neighbourhood(i, j, w)
                # prepare the neighbourhood to calculate the median pixel value
                if self.channels == 1:
                    # for greyscale images
                    temp_list = []
                    for k in range(0, w):
                        for l in range(0, w):
                            temp_list.append(temp_nbh[k, l])
                    new_image[i, j] = stats.median(temp_list)
                else:
                    # for colour images
                    temp_b_list = []
                    temp_g_list = []
                    temp_r_list = []
                    for k in range(0, w):
                        for l in range(0, w):
                            temp_b_list.append(temp_nbh[k, l][0])
                            temp_g_list.append(temp_nbh[k, l][1])
                            temp_r_list.append(temp_nbh[k, l][2])
                    new_image[i, j] = [stats.median(temp_b_list), stats.median(
                        temp_g_list), stats.median(temp_r_list)]

        return ImageM(new_image)

    def get_sharpened_image(self, blurred, k):
        """
        Returns the output image after applying unsharp masking which can be used
        to remove the blurr from images.

        :param blurred: the ImageM object holding the blurred image
        :param k: the sharpening constant(k = 1 for unsharp masking)
        """
        # sharpened = orginal + (original - blurred) * amount
        return ImageM(np.add(self.image, np.subtract(self.image, blurred.get_image_data()) * k))

    def get_histogram_for_channel_i(self, channel_i):
        """
        Returns a list with the distribution of pixel values for a given channel
        of the RGB points in the image associated with this object i.e. R or G
        or B. =

        :param channel_i: the integer presentation of the channel to obtain a
        distribution for. If channel_i is 1 we will compute for the B channel, if
        it is 2, the G channel and if it is 3 the R channel.
        :return: a list with the distribution of pixel values for a given channel
        """
        if self.channels == 1 and channel_i != 1:
            return None
        histogram = [0] * 256
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                if self.channels == 1:
                    histogram[self.image[i][j]
                              ] = histogram[self.image[i][j]] + 1
                else:
                    index = self.image[i][j][channel_i - 1]
                    histogram[index] = histogram[index] + 1

        return histogram

    def save_colour_histogram(self, figure_file_path, channel_i):
        """
        Save the image colour histogram with the given channel figure to a file.

        :param figure_file_path: the file path to store the figure to.
        :param channel_i: the channel of the histogram to store.
        :return: the image colour histogram with the given channel
        """
        histogram = self.get_histogram_for_channel_i(channel_i)
        # fig = plt.figure()
        # ax = fig.add_axes(histogram)
        # ax = fig.add_axes([0, 0, 1, 1])
        # ax.bar([i for i in range(0, 256)], histogram)
        # plt.style.use('ggplot')
        # plt.savefig(figure_file_path)
        # plt.clf()
        return histogram

    def compare_histograms(self, figure_file_path, histogram1, histogram2):
        """
        Compare the image colour histograms and store the resulting figure to a
        file to the specified file path.

        :param figure_file_path: the file path to store the figure to.
        :param histogram1: the first histogram for comparison
        :param histogram2: the second histogram for comparison
        :return: the difference values between the two given colour histograms
        """
        diff_values = []
        for i in range(0, len(histogram1)):
            diff_values.append(abs(histogram1[i] - histogram2[i]))
        # plt.style.use('ggplot')
        # plt.hist(diff_values, bins=256)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar([i for i in range(0, 256)], diff_values)
        # plt.style.use('ggplot')
        ax.xlabel("Intensity i")
        ax.ylabel("Average Absolute Difference")
        ax.title("Input image-Picture Mosaic Image Histogram Differences")
        plt.savefig(figure_file_path)
        return diff_values

    def compare_all_histograms(self, figure_file_path, histogram_b1, histogram_b2, histogram_g1, histogram_g2,
                               histogram_r1, histogram_r2):
        """
        Compare the image colour histograms and store the resulting figure to a
        file to the specified file path. The histograms are provided split according
        to the image R, G, B values.

        :param figure_file_path: the file path to store the figure to.
        :param histogram_b1: the first histogram Blue stream
        :param histogram_b2: the second histogram Blue stream
        :param histogram_g1: the first histogram Green stream
        :param histogram_g2: the second histogram Green stream
        :param histogram_r1: the first histogram Red stream
        :param histogram_r2: the second histogram Red stream
        :return: the difference values between the two given colour histograms streams
        """
        diff_values_b = []
        diff_values_g = []
        diff_values_r = []
        for i in range(0, len(histogram_b1)):
            diff_values_b.append(abs(histogram_b1[i] - histogram_b2[i]))
            diff_values_g.append(abs(histogram_g1[i] - histogram_g2[i]))
            diff_values_r.append(abs(histogram_r1[i] - histogram_r2[i]))
        # plt.style.use('ggplot')
        # plt.hist(diff_values, bins=256)
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar([i for i in range(0, 256)], diff_values_b)
        # plt.style.use('ggplot')
        plt.xlabel("Intensity i")
        plt.ylabel("Average Absolute Difference")
        plt.title("Input image-Picture Mosaic Image Histogram Differences")
        plt.savefig(figure_file_path)
        return diff_values_b

    def get_normalised_histogram(self, histogram):
        """
        Returns a normalised colour distribution histogram which is computed using
        the given histogram.

        :param histogram: a list with the distribution of pixel values for a
        given channel
        :return: a list with the normalised colour distribution histogram
        """
        divisor = self.rows * self.cols
        return [h_i/divisor for h_i in histogram]

    def get_T_mapped_constrast_pixel_value(normalised_histogram, s):
        """
        Uses the given normalised colour distribution histogram to map a given
        pixel value to a constrasted value.

        :param normalised_histogram: a list with the normalised colour distribution
        histogram
        :param s: the integer representation of the pixel value to map
        :return: the histogram equalisation mapped value of the given pixel
        """
        total_sum = 0
        for i in range(0, s + 1):
            total_sum = total_sum + normalised_histogram[i]
        return 256 * total_sum

    def adjust_constrast_with_base_image_for_colour_image(self, base_image):
        """
        Runs a histogram matching operation of the image associated this object using
        the colour histogram distribution of the given base_image so that they have
        the same contrast level.

        :param base_image: the image to use to obtain the colour histogram
        distribution to map the pixels of the image associated this object
        """
        if self.channels:
            return None
        histogram1 = base_image.get_histogram_for_channel_i(1)
        histogram2 = base_image.get_histogram_for_channel_i(2)
        histogram3 = base_image.get_histogram_for_channel_i(3)
        normalised1 = base_image.get_normalised_histogram(histogram1)
        normalised2 = base_image.get_normalised_histogram(histogram2)
        normalised3 = base_image.get_normalised_histogram(histogram3)
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                temp_B = self.get_T_mapped_constrast_pixel_value(
                    normalised1, self.image[i][j][0])
                temp_G = self.get_T_mapped_constrast_pixel_value(
                    normalised2, self.image[i][j][1])
                temp_R = self.get_T_mapped_constrast_pixel_value(
                    normalised3, self.image[i][j][2])
                self.image[i][j] = [temp_B, temp_G, temp_R]

    def get_image_from_histogram_matching(self, base_image):
        """
        Runs a histogram matching operation of the image associated this object using
        the colour histogram distribution of the given base_image so that they have
        the same contrast level.

        :param base_image: the image to use to obtain the colour histogram
        distribution to map the pixels of the image associated this object
        :return: histogram matched ImageM object
        """
        reference = base_image.get_image_data()
        image = self.image
        matched = match_histograms(image, reference, multichannel=True)
        return ImageM(matched)

    def show_histogram_matching(self, base_image):
        """
        Runs a histogram matching operation of the image associated this object using
        the colour histogram distribution of the given base_image so that they have
        the same contrast level. The results of the histogram matching processing
        are shown on the screen.

        :param base_image: the image to use to obtain the colour histogram
        distribution to map the pixels of the image associated this object
        """
        reference = base_image.get_image_data()
        image = self.image
        matched = match_histograms(image, reference, multichannel=True)
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                            figsize=(8, 3), sharex=True, sharey=True)
        for aa in (ax1, ax2, ax3):
            aa.set_axis_off()
        ax1.imshow(image)
        ax1.set_title('Source')
        ax2.imshow(reference)
        ax2.set_title('Reference')
        ax3.imshow(matched)
        ax3.set_title('Matched')
        plt.tight_layout()
        plt.show()

    def get_resized_image_using_nearest_neighbour(self, s):
        """
        Scale the loaded image using Nearest Neighbour Interpolation.

        :param s: the scale factor to use when scaling the image
        :return: the s scaled version of the loaded image
        """
        new_rows = int(self.rows * s)
        new_cols = int(self.cols * s)
        new_image = np.zeros((new_rows, new_cols, self.channels), np.uint8)
        for i in range(0, new_rows):
            for j in range(0, new_cols):
                temp_row = math.ceil(i / s)
                temp_col = math.ceil(j / s)
                if temp_row == self.rows:
                    temp_row = self.rows - 1
                if temp_col == self.cols:
                    temp_col = self.cols - 1
                new_image[i, j] = self.image[temp_row, temp_col]

        return ImageM(new_image)

    def get_resized_image_using_bilinear_interpolation(self, s):
        """
        Scale the loaded image using Bilinear Interpolation.

        :param s: the scale factor to use when scaling the image
        :return: the s scaled version of the loaded image
        """
        new_rows = int(self.rows * s)
        new_cols = int(self.cols * s)
        new_image = np.zeros((new_rows, new_cols, self.channels), np.uint8)
        for i in range(0, new_rows):
            for j in range(0, new_cols):
                temp_row_f = math.floor(i / s)
                temp_col_f = math.floor(j / s)
                temp_row_c = temp_row_f + 1
                temp_col_c = temp_col_f + 1
                if temp_row_c == self.rows:
                    temp_row_c = self.rows - 1
                if temp_col_c == self.cols:
                    temp_col_c = self.cols - 1
                new_image[i, j] = (temp_row_c - (i / s)) * ((temp_col_c - (j / s)) * self.image[temp_row_f, temp_col_f] + ((j / s) - temp_col_f) * self.image[temp_row_f, temp_col_c]) \
                    + ((i / s) - temp_row_f) * ((temp_col_c - (j / s)) * self.image[temp_row_c, temp_col_f] + (
                        (j / s) - temp_col_f) * self.image[temp_row_c, temp_col_c])

        return ImageM(new_image)

    def get_neighbourhood(self, i, j, w):
        """
        Returns the pixels in the neighbourhood of size wxw collected from the
        starting row position i and starting column position j.

        :param i: the starting row position
        :param j: the starting column position
        :param w: used to compute the size of the neighbourhood i.e. wxw
        :return: an image data array of size wxw with the pixels in the given neighbourhood
        """
        nbh = np.zeros((w, w, self.channels), np.uint8)
        r = c = end = round(w/2)
        start_row = i - r
        start_col = j - c
        nbh_row = 0
        nbh_col = 0
        for k in range(start_row, start_row + w):
            nbh_col = 0
            for l in range(start_col, start_col + w):
                # ensure we are still within the image
                if k >= 0 and k < self.rows and l >= 0 and l < self.cols:
                    nbh[nbh_row, nbh_col] = self.image[k, l]
                else:
                    # for boundary cases
                    if self.channels == 1:
                        nbh[nbh_row, nbh_col] = 0
                    else:
                        nbh[nbh_row, nbh_col] = [0, 0, 0]
                nbh_col = nbh_col + 1
            nbh_row = nbh_row + 1

        return nbh

    def get_r_c_neighbourhood(self, i, j, w_rows, w_cols):
        """
        Returns the pixels in the neighbourhood of size wxw collected from the
        starting row position i and starting column position j.

        :param i: the starting row position
        :param j: the starting column position
        :param w_rows: number of rows in the neighbourhood
        :param w_cols: number of columns in the neighbourhood
        :return: an image data array of size wxw with the pixels in the given neighbourhood
        """
        nbh = np.zeros((w_rows, w_cols, self.channels), np.uint8)
        r = w_rows/2
        c = w_cols/2
        start_row = i - r
        start_col = j - c
        nbh_row = 0
        nbh_col = 0
        for k in range(start_row, start_row + w_rows):
            nbh_col = 0
            for l in range(start_col, start_col + w_cols):
                # ensure we are still within the image
                if k >= 0 and k < self.rows and l >= 0 and l < self.cols:
                    nbh[nbh_row, nbh_col] = self.image[k, l]
                else:
                    # for boundary cases
                    if self.channels == 1:
                        nbh[nbh_row, nbh_col] = 0
                    else:
                        nbh[nbh_row, nbh_col] = [0, 0, 0]
                nbh_col = nbh_col + 1
            nbh_row = nbh_row + 1

        return nbh

    def get_sift_keypoints_and_descriptors(self):
        """
        Returns the keypoints and descriptor vectors of the loaded image using the
        SIFT Feature Detection Algorithm.

        :return: the image feature keypoints and descriptor vectors
        """
        sift = cv.SIFT_create()
        return sift.detectAndCompute(self.image, None)

    def get_coordinates_of_keypoints1(self, keypoints):
        """
        Returns a list of unrounded (x, y) coordinate tuples holding the row and
        column values of the location of the features in the image.

        :param keypoints: the image feature keypoints
        :return: (x, y) coordinate tuples holding location of image features
        """
        return [point.pt for point in keypoints]

    def get_coordinates_of_keypoints2(self, keypoints):
        """
        Returns a list of rounded (x, y) coordinate tuples holding the row and
        column values of the location of the features in the image.

        :param keypoints: the image feature keypoints
        :return: (x, y) coordinate tuples holding location of image features
        """
        return [point.pt for point in keypoints]

    def show_sift_feature_matching1(self, compare_image):
        """
        Displays the results from feature matching between two images using a UI
        window. Feature Detection is done using SIFT and Feature Matching is done
        using the Brute-Force matcher. The results are displayed by showing the
        greyscale version of the images next to each and drawing lines to features
        that were matched.

        :param compare_image: the image to match image features with
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # feature matching
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all
        # other features in second set using some distance calculation. And the closest one is returned.
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        grey_image1 = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        grey_image2 = cv.cvtColor(
            compare_image.get_image_data(), cv.COLOR_BGR2GRAY)
        display_image = cv.drawMatches(
            grey_image1, keypoints1, grey_image2, keypoints2, matches[:50], grey_image2, flags=2)
        plt.imshow(display_image)
        plt.show()

    def show_sift_feature_matching2(self, compare_image):
        """
        Displays the results from feature matching between two images using a UI
        window. Feature Detection is done using SIFT and Feature Matching is done
        using the Flann matcher. The results are displayed by showing the
        greyscale version of the images next to each and drawing lines to features
        that were matched.

        :param compare_image: the image to match image features with
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # feature matching
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # FLANN stands for Fast Library for Approximate Nearest Neighbors. It contains a collection of algorithms optimized
        # for fast nearest neighbor search in large datasets and for high dimensional features. It works faster than BFMatcher for large datasets.
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        grey_image1 = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        grey_image2 = cv.cvtColor(
            compare_image.get_image_data(), cv.COLOR_BGR2GRAY)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < (0.7 * n.distance):
                matchesMask[i] = [1, 0]
        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(
            255, 0, 0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
        display_image = cv.drawMatchesKnn(
            grey_image1, keypoints1, grey_image2, keypoints2, matches, None, **draw_params)
        plt.imshow(display_image)
        plt.show()

    def show_sift_feature_matching_dot_line1(self, compare_image):
        """
        Displays the results from feature matching between two images using a UI
        window. Feature Detection is done using SIFT and Feature Matching is done
        using the Brute-Force matcher. The results are displayed on a greyscale
        version of the loaded image. Features of the loaded image are shown using
        a dark yellow dot and bright yellow lines are drawn to the feature in the
        comparison image that was matched to the feature represented by the dark
        yellow dot.

        :param compare_image: the image to match image features with
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        # Brute-Force matcher is simple. It takes the descriptor of one feature in first set and is matched with all
        # other features in second set using some distance calculation. And the closest one is returned.
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        grey_image1 = self.get_greyscale_image3().image
        for match in matches:
            temp_image1_kp_index = match.queryIdx
            temp_image2_kp_index = match.trainIdx
            temp_coord1 = keypoints1[temp_image1_kp_index].pt
            temp_coord2 = keypoints2[temp_image2_kp_index].pt
            # show keypoint in the first image with a circle
            cv.circle(grey_image1, (int(temp_coord1[0]), int(
                temp_coord1[1])), 2, (10, 245, 245), -1)
            # line joining match points in first and second points
            cv.line(grey_image1, (int(temp_coord1[0]), int(temp_coord1[1])), (int(
                temp_coord2[0]), int(temp_coord2[1])), (10, 194, 245), 1)

        cv.imshow("ImageM Feature Matching", grey_image1)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite("sempermatched.jpg", grey_image1)

    def get_repeatability1(self, compare_image):
        """
        Returns the repeatability of the feature detection and matching process
        of SIFT. The Brute-Force Matcher is used.

        Repeatability is the percentage of A’s features that are detected in B,
        regardless of how they are matched

        :param compare_image: the image to match image features with
        :return: loaded image repeatability percentage when matched to the given image
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        return (float(len(matches)) / float(len(keypoints1))) * 100

    def get_repeatability2(self, compare_image):
        """
        Returns the repeatability of the feature detection and matching process
        of SIFT. The Flann Matcher is used.

        Repeatability is the percentage of A’s features that are detected in B,
        regardless of how they are matched

        :param compare_image: the image to match image features with
        :return: loaded image repeatability precentage when matched to the given image
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        return (float(len(matches)) / float(len(keypoints1))) * 100

    def get_match_accuracy1(self, compare_image, scale_factor):
        """
        Returns the matching accuracy of the feature detection and matching process
        of SIFT. The Brute-Force Matcher is used.

        Matching accuracy is the percentage of matches between A and B that are
        in fact correct.

        :param compare_image: the image to match image features with
        :return: loaded image match accuracy when matched to the given image
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        correct = 0
        for match in matches:
            temp_image1_kp_index = match.queryIdx
            temp_image2_kp_index = match.trainIdx
            temp_coord1 = keypoints1[temp_image1_kp_index].pt
            temp_coord2 = keypoints2[temp_image2_kp_index].pt
            if int(temp_coord1[0] * scale_factor) == int(temp_coord2[0]) and \
                    int(temp_coord1[1] * scale_factor) == int(temp_coord2[0]):
                correct = correct + 1

        return (float(correct) / float(len(matches))) * 100

    def get_repeatability_and_match_accuracy1(self, compare_image, scale_factor):
        """
        Returns in a list the repeatability matching accuracy of the feature
        detection and matching process of SIFT. The Brute-Force Matcher is used.

        Note: This function is to be used to match different scaled versions of
        the same image for experimental purposes.

        Repeatability is the percentage of A’s features that are detected in B,
        regardless of how they are matched.
        Matching accuracy is the percentage of matches between A and B that are
        in fact correct.

        :param compare_image: the image to match image features with
        :param scale_factor: the scale factor used to scale the comparison image
        :return: loaded image repeatability and match accuracy when matched to
                         the given image
        """
        keypoints1, descriptors1 = self.get_sift_keypoints_and_descriptors()
        keypoints2, descriptors2 = compare_image.get_sift_keypoints_and_descriptors()
        # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
        bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)
        correct = 0
        for match in matches:
            temp_image1_kp_index = match.queryIdx
            temp_image2_kp_index = match.trainIdx
            temp_coord1 = keypoints1[temp_image1_kp_index].pt
            temp_coord2 = keypoints2[temp_image2_kp_index].pt
            if int(temp_coord1[0] * scale_factor) == int(temp_coord2[0]) and int(temp_coord1[1] * scale_factor) == int(temp_coord2[0]):
                correct = correct + 1

        repeatability = (float(len(matches)) / float(len(keypoints1))) * 100.0
        accuracy = (float(correct) / float(len(matches))) * 100
        return [round(repeatability, 2), round(accuracy, 2)]

    def get_camera_matrix_P(self, corresponding_points):
        """
        Returns the P matrix for the image camera computed using the given corresponding points.

        :param corresponding_points: the list of corresponding points to used in the computation.
        :return: the P matrix for the image camera
        """
        if len(corresponding_points) < 6:
            return None

        # corresponding_points format corresponding_points[i] = [(x_1, y_1), (x'_1, y'_1)]
        # use the corresponding_points to obtain the A matrix to then calculate the SVD for
        A = []
        for point_pair in corresponding_points:
            # P^T 0^T -uP^T
            temp_row_1 = [0, 0, 0, 0, -point_pair[1][0], -point_pair[1][1], -point_pair[1][2], -1, point_pair[0][1] *
                          point_pair[1][0], point_pair[0][1] * point_pair[1][1], point_pair[0][1] * point_pair[1][2], point_pair[0][1]]
            # 0^T P^T -vP^T
            temp_row_2 = [point_pair[1][0], point_pair[1][1], point_pair[1][2], 1, 0, 0, 0, 0, -point_pair[0][0] *
                          point_pair[1][0], -point_pair[0][0] * point_pair[1][1], -point_pair[0][0] * point_pair[1][2], -point_pair[0][0]]
            A.append(temp_row_1)
            A.append(temp_row_2)

        U, S, V = np.linalg.svd(np.array(A), full_matrices=True)
        # use the last column of V to obtain the h_i values for the H matrix
        index = len(V) - 1
        print(V[index])
        # store the h_i values for i = 1,2,...,9 in the 3x3 matrix H
        H = [
            [V[index][0], V[index][1], V[index][2], V[index][3]],
            [V[index][4], V[index][5], V[index][6], V[index][7]],
            [V[index][8], V[index][9], V[index][10], V[index][11]]
        ]

        return np.array(H)

    def decompose_camera_matrix_P(self, P):
        """
        Returns the decomposed P matrix for the image camera.

        :param P: the P matrix for the image camera
        :return: decomposed P matrix for the image camera
        """
        W = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # calculate K and R up to sign
        Qt, Rt = np.linalg.qr((W.dot(P[:, 0:3])).T)
        K = W.dot(Rt.T.dot(W))
        R = W.dot(Qt.T)

        D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if K[0, 0] < 0:
            D[0, 0] = -1
        if K[1, 1] < 0:
            D[1, 1] = -1
        if K[2, 2] < 0:
            D[2, 2] = -1
        K = K.dot(D)
        R = D.dot(R)

        # calculate c
        c = -R.T.dot(np.linalg.inv(K).dot(P[:, 3]))

        return K, R, c

    def difference_between_3D_points(self, c1, c2):
        """
        Returns the Euclidean distance between the given 3D points.
        :param c1: the first 3D point to use in the computation
        :param c2: the second 3D point to use in the computation
        :return: the Euclidean distance between the given 3D points
        """
        return ((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2 + (c2[2] - c1[2]) ** 2) ** 0.5

    def angle_between_vectors(self, v1, v2):
        """
        Returns the angle(in degrees) between the given 2D points.

        :param v1: the first point to use in the computation
        :param v2: the second point to use in the computation
        :return: the angle(in degrees) between the given 2D points
        """
        numerator = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        part1 = v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2
        part2 = v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2
        denominator = ((part1) ** 0.5) * ((part2) ** 0.5)
        return math.degrees(math.acos(numerator/denominator))

    def show_given_matches_between_images(self, compare_image, match_coordinates):
        """
        Displays the given feature matching coordinates between the given image
        and the image associated with this object on the screen.

        :param compare_image: the image the image associated with this object is
         being feature matched to
        :param match_coordinates: the list of feature matching coordinates
        """
        # We assume the images have the same dimensions
        display_image = np.concatenate(
            (self.image, compare_image.get_image_data()), axis=1)
        for coord in match_coordinates:
            # show keypoint in the first image with a circle
            x_prime = coord[1][0] + self.cols
            y_prime = coord[1][1]  # + self.rows
            cv.circle(display_image, (int(coord[0][0]), int(
                coord[0][1])), 2, (10, 245, 245), -1)
            cv.circle(display_image, (int(x_prime), int(y_prime)),
                      2, (10, 245, 245), -1)
            # line joining match points in first and second points
            cv.line(display_image, (int(coord[0][0]), int(coord[0][1])), (int(
                x_prime), int(y_prime)), (10, 194, 245), 1)
        cv.imshow('ImageM Show Image Matches', display_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_given_matches_between_images_single_image_display(self, match_coordinates):
        """
        Displays the given feature matching coordinates between the given image
        and a compare image on the screen on the image associated with this
        object. Note: the compare image is not needed only the match coordinates
        for the two images and the images are assumed to have the same
        dimensions.

        :param match_coordinates: the list of feature matching coordinates
        """
        display_image = np.copy(self.image)
        p = 0
        for coord in match_coordinates:
            # show keypoint in the first image with a circle
            x_prime = coord[1][0]
            y_prime = coord[1][1]
            cv.circle(display_image, (int(coord[0][0]), int(
                coord[0][1])), 2, (10, 245, 245), -1)
            p = p + 1
            # line joining match points in first and second points
            cv.line(display_image, (int(coord[0][0]), int(coord[0][1])), (int(
                x_prime), int(y_prime)), (10, 194, 245), 1)
        print("p drawn: " + str(p))
        cv.imshow('ImageM Show Image Matches', display_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_given_matches_between_images_single_image_display(self, match_coordinates, file_path):
        """
        Displays the given feature matching coordinates between the given image
        and a compare image on the screen on the image associated with this
        object. Note: the compare image is not needed only the match coordinates
        for the two images and the images are assumed to have the same
        dimensions.

        :param match_coordinates: the list of feature matching coordinates
        """
        display_image = np.copy(self.get_greyscale_image3().image)
        for coord in match_coordinates:
            # show keypoint in the first image with a circle
            x_prime = coord[1][0]
            y_prime = coord[1][1]
            cv.circle(display_image, (int(coord[0][0]), int(
                coord[0][1])), 2, (10, 245, 245), -1)
            # cv.circle(display_image, (int(x_prime), int(y_prime)),
            #          2, (10, 245, 245), -1)
            # line joining match points in first and second points
            cv.line(display_image, (int(coord[0][0]), int(coord[0][1])), (int(
                x_prime), int(y_prime)), (10, 194, 245), 1)
        # save to file
        cv.imwrite(file_path, display_image)

    def show_camera_details(self, title, world_origin, P, R, C_tilde, s):
        """
        Display the details of the image camera on the screen.

        :param title: the title of the display window
        :param world_origin: the coordinates origin in the real world of the image
        :param P: the P matrix to use in computation of the display points
        :param R: the R matrix to use in computation of the display points
        :param C_tilde: the C tilde vector to use in computation of the display points
        "param s: the s value to use in computation of the display points
        """
        C_tilde_x = np.matmul(np.transpose(
            R), np.array([100, 0, 0])) + (C_tilde)
        C_tilde_y = np.matmul(np.transpose(
            R), np.array([0, 100, 0])) + (C_tilde)
        C_tilde_z = np.matmul(np.transpose(
            R), np.array([0, 0, 100])) + (C_tilde)
        C_tilde_2D = np.matmul(P, np.append(C_tilde, [1]))
        C_x_2D = np.matmul(P, np.append(C_tilde_x, [1]))
        C_y_2D = np.matmul(P, np.append(C_tilde_y, [1]))
        C_z_2D = np.matmul(P, np.append(C_tilde_z, [1]))
        plt.title(title)
        # format of world origin [(x_1, y_1), (x_2, y_2)]
        x1, y1 = [world_origin[0][0][0], world_origin[0][1][0]], [
            world_origin[0][0][1], world_origin[0][1][1]]  # z
        x2, y2 = [world_origin[1][0][0], world_origin[1][1][0]], [
            world_origin[1][0][1], world_origin[1][1][1]]  # y
        x3, y3 = [world_origin[2][0][0], world_origin[2][1][0]], [
            world_origin[2][0][1], world_origin[2][1][1]]  # y
        plt.plot(x1, y1, x2, y2, x3, y3, marker='o', color='red')
        plt.text(world_origin[0][1][0] + s, world_origin[0]
                 [1][1], "z", color='red')  # z label
        plt.text(world_origin[1][1][0] + s, world_origin[1]
                 [1][1], "y", color='red')  # y label
        plt.text(796 + 2 - (s + 40), 1892, "x", color='red')  # x label
        plt.imshow(self.get_greyscale_image1().get_image_data(), cmap='gray')
        plt.show()

    def show_camera_details_in_3d(self, P, R, C_tilde):
        """
        Display the details of the image camera on the screen in 3D for a better
        perspective.

        :param title: the title of the display window
        :param world_origin: the coordinates origin in the real world of the image
        :param P: the P matrix to use in computation of the display points
        :param R: the R matrix to use in computation of the display points
        :param C_tilde: the C tilde vector to use in computation of the display points
        "param s: the s value to use in computation of the display points
        """
        C_tilde_x = np.matmul(np.transpose(
            R), np.array([100, 0, 0])) + (C_tilde)
        C_tilde_y = np.matmul(np.transpose(
            R), np.array([0, 100, 0])) + (C_tilde)
        C_tilde_z = np.matmul(np.transpose(
            R), np.array([0, 0, 100])) + (C_tilde)
        C_tilde_2D = np.matmul(P, np.append(C_tilde, [1]))
        C_x_2D = np.matmul(P, np.append(C_tilde_x, [1]))
        C_y_2D = np.matmul(P, np.append(C_tilde_y, [1]))
        C_z_2D = np.matmul(P, np.append(C_tilde_z, [1]))
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        zline = [C_tilde[2]]
        yline = [C_tilde[1]]
        xline = [C_tilde[0]]
        ax.scatter3D(xline, yline, zline, c=zline)
        plt.show()

    def get_epipole(self, P, c):
        """
        Compute the epipole of the image camera using the P matric and c vector.

        :return: the epipole of the image camera
        """
        return np.matmul(P, np.append(c, [1]))

    def get_fundamental_matrix(self, epipole, P_1, P_2):
        """
        Compute the fundamental matrix, F, of the image camera.

        :param epipole: epipole of the image camera
        :param P_1: the P1 matrix of the image camera computation
        :param P_2: the P2 matrix of the image camera computation
        :return: the fundamental matrix, F, of the image camera
        """
        e_x = np.array([
            [0, -epipole[2], epipole[1]],
            [epipole[2], 0, -epipole[0]],
            [-epipole[1], epipole[0], 0]
        ])
        PP_plus = np.matmul(P_2, la.pinv(P_1))
        return np.matmul(e_x, PP_plus)

    def show_epipolar_lines(self, F, points, P_1, P_2):
        """
        Show the epipolar lines of the image camera on the screen.

        :param F: the fundamental matrix for the image camera
        :param points: the points of the epipolar lines
        :param P_1: the P1 matrix of the image camera computation
        :param P_2: the P2 matrix of the image camera computation
        """
        color_options = [
            "#DFFF00", "#FFBF00", "#FF7F50", "#DE3163", "#9FE2BF",
            "#40E0D0", "#6495ED", "#CCCCFF", "#154360", "#D35400",
            "#1E8449", "#884EA0", "#B7950B", "#34495E"
        ]
        j = 0
        for point in points:
            vec = np.array([point[0][0], point[0][1], 1])
            temp = np.matmul(F, vec)
            x_t, y_t = [0, 2560], [point[0][1], point[0][1]]
            plt.plot(x_t, y_t, color=color_options[j])
            j = j + 1
        plt.imshow(self.get_greyscale_image1().get_image_data(), cmap='gray')
        plt.show()

    def show_x_y_points(self, title, points):
        """
        plt.title(title)
        figure, axes = plt.subplots(1)
        axes.set_aspect('equal')
        axes.imshow(self.get_greyscale_image1().get_image_data(), cmap='gray')
        for point in points:
            temp_circle = Circle((point[0], point[1]), 0.2, color='r')
            axes.add_patch(temp_circle)
        plt.show()
        """
        display_image = self.get_greyscale_image3().get_image_data()
        for point in points:
            # 237, 35, 17
            cv.circle(display_image, (point[0],
                      point[1]), 5, (17, 35, 237), -1)

        # cv.imshow("ImageM image", display_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        cv.imwrite("lego2_points.jpg", display_image)

    def save_given_matches_between_images(self, compare_image, match_coordinates, file_path):
        """
        Save an image displaying the given feature matching coordinates between
        the given image and the image associated with this object on the screen.

        :param compare_image: the image the image associated with this object is
         being feature matched to
        :param match_coordinates: the list of feature matching coordinates
        """
        # We assume the images have the same dimensions
        display_image = np.concatenate(
            (self.image, compare_image.get_image_data()), axis=1)
        for coord in match_coordinates:
            # show keypoint in the first image with a circle
            x_prime = coord[1][0] + self.cols
            y_prime = coord[1][1]  # + self.rows
            cv.circle(display_image, (int(coord[0][0]), int(
                coord[0][1])), 2, (10, 245, 245), -1)
            cv.circle(display_image, (int(x_prime), int(y_prime)),
                      2, (10, 245, 245), -1)
            # line joining match points in first and second points
            cv.line(display_image, (int(coord[0][0]), int(coord[0][1])), (int(
                x_prime), int(y_prime)), (10, 194, 245), 1)

        cv.imwrite(file_path, display_image)

    def show_image_edges(self):
        """
        Displays on the screen the visual showing the edges detected after running
        a edge detection algorithm.
        """
        # run the edge detection
        edges = cv.Canny(self.image, 100, 200)
        # set up the plot to display
        plt.subplot(121), plt.imshow(self.image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def show_image_sift_keypoints(self):
        """
        Displays the keypoints generated using the SIFT Feature Detection Algorithm.
        """
        sift = cv.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(self.image, None)
        plt.imshow(cv.drawKeypoints(cv.cvtColor(
            self.image, cv.COLOR_BGR2GRAY), keypoints, self.image))
        plt.show()

    def show_image(self):
        """
        Displays the loaded image data in a UI window.
        """
        cv.imshow("ImageM image", self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_image_with_title(self, title):
        """
        Displays the loaded image data in a UI window.

        :param title: the title to use for the display window
        """
        cv.imshow(title, self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def save_image_to_file(self, file_path):
        """
        Saves the loaded image data to the file with the given file path.

        :param file_path: the file path to use to save the image to
        """
        cv.imwrite(file_path, self.image)

    def save_image_data_to_file(self, image_data, file_path):
        """
        Saves the loaded image data to the file with the given file path.

        :param image_data: the array holding the image data
        :param file_path: the file path to use to save the image to
        """
        cv.imwrite(file_path, image_data)

    def __str__(self):
        return "rows: " + str(self.rows) + " columns: " + str(self.cols)


if __name__ == "__main__":
    # for testing purposes
    test_img = ImageM("test_files/family_image2.JPG")
    grey_img = test_img.get_greyscale_image2()
