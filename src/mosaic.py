# Mosaic File
# date: 26/05/2021
# name: Ibrahim Sheriff Sururu
# description: This file can be used to run an Evolutionary Algorithm(EA) to produce picture mosaics
# based on a database of images and using a evolutionary algorithm to evolve the picture mosaic to avoid 
# the computational inefficiencies of a brute-force method.
# The results can be great if there is a large number of images in the image database with diverse colour
# profiles.
import cv2 as cv
import numpy as np
import math
import os
import sys
import uuid
import json
import requests
import ast
from imutils import build_montages
import time
import random
import matplotlib.pyplot as plt
from image_manipulator import ImageM

# For Terminal Colour Display
HEADER = '\033[95m'
OK_BLUE = '\033[94m'
OK_CYAN = '\033[96m'
OK_GREEN = '\033[92m'
WARNING_COLOUR = '\033[93m'
FAIL_COLOUR = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

SRC_PATH = os.path.dirname(__file__)[:len(os.path.dirname(__file__))-3]
# Image directory path
IMAGES_FILE_PATH = SRC_PATH + "images/"
# Pexels API Access Key
API_KEY = "563492ad6f91700001000001e59d3ba0d6a9471e80b924ed9c4db018"
REQUESTS_HEADERS = {
    'Authorization': API_KEY
}


def get_file_lines(file_addr):
    """
    Returns the stripped lines of the file with the given file address.

    :param file_addr: the full path to the file
    :return: the stripped lines of the file with the given file address
    """
    with open(file_addr, "r") as f:
        return [line.rstrip() for line in f]


# a list of images that have been downloaded
PHOTO_LIST = get_file_lines(IMAGES_FILE_PATH + "download_list.txt")
# A list of available Pexels search terms to use when searching for images to download
AVAILABLE_SEARCH_TERMS = ["plants", "ocean", "outdoors", "happy", "nature", "dogs", "plants", "food", "people", "sport",
                          "school", "cats", "summer", "waves", "winter", "sleeping", "planet", "safari", "fire", "light"
                          , "technology", "business", "meeting"]


def add_to_download_list(image_url):
    """
    Adds the given image URL to the list of downloaded images to avoid duplication when
    downloading.

    :param image_url: the image URL to add to the list of downloaded images.
    """
    PHOTO_LIST.append(image_url)
    f = open(IMAGES_FILE_PATH + "download_list.txt", "a")
    f.write(image_url + "\n")
    f.close()


def add_histogram_values_to_file(histogram_list, colour):
    """
    Adds the given colour histogram values from an experiment to a file.

    :param histogram_list: the list containing the values of the colour histogram
    :param colour: the colour histogram stream value i.e. Blue, Green or Red.
    """
    f = open(IMAGES_FILE_PATH + "histogram_differences_" + colour + ".txt", "a")
    f.write(str(histogram_list) + "\n")
    f.close()


def compute_histogram_avg(colour):
    """
    Compute the average of the stored colour histogram values for a given colour stream. The
    histogram values are obtained from text files. The result is then displayed on the screen
    for viewing.

    :param colour: the colour stream(i.e. Blue, Green or Red) of the histograms to use in
    the computation.
    """
    file_data = get_file_lines(IMAGES_FILE_PATH + "histogram_differences_" + colour + ".txt")
    histogram_diff_list = []
    for data in file_data:
        histogram_diff_list.append(ast.literal_eval(data))

    histogram_sums = []
    for hist in histogram_diff_list:
        for i in range(0, len(hist)):
            histogram_sums[i] = histogram_sums[i] + hist[i]

    n = len(file_data)
    for j in range(0, len(histogram_sums)):
        histogram_sums[j] = histogram_sums[j] / n

    plt.style.use('ggplot')
    plt.hist(histogram_sums, bins=256)
    plt.savefig("average_histogram_colour_differences_" + colour + ".txt")


def download_image(search_term, image_data):
    """
    Download the image found using the given search term with the given Pexels image
    data.

    :param search_term: the search term used to obtain the image.
    :param image_data: the image data JSON object returned by the Pexels API which contains
    the metadata of the image.
    :return: True if all went well and the download was successfully completed, and False
    if there was an error and the download could not be completed.
    """
    if image_data["src"]["original"] in PHOTO_LIST:
        return False
    response = requests.get(image_data["src"]["original"], "wb")
    file_name = IMAGES_FILE_PATH + "downloaded/" + search_term + "_" + str(uuid.uuid4()) + ".jpeg"
    file = open(file_name, "wb")
    file.write(response.content)
    file.close()
    add_to_download_list(image_data["src"]["original"])
    return True


def get_images(search_term, number_of_tiles, database_multiplier):
    """
    Download images using the Pexels API that can found using the given search terms.

    :param search_term: the search term to use in the downloading process.
    :param number_of_tiles: the number of tiles the picture mosaic is going to split into.
    :param database_multiplier: the number of times to grow the image database by.
    """
    search_runs = math.ceil(number_of_tiles / 80 * database_multiplier)
    search_url = "https://api.pexels.com/v1/search?query={search_term}&page={page_num}&per_page=80"
    page_i = 0
    i = 0
    while i < search_runs:
        search_url_temp = search_url.format(search_term=search_term, page_num=str(page_i))
        response = requests.get(search_url_temp, headers=REQUESTS_HEADERS)
        temp_photos_list = json.loads(response.text)
        for photo in temp_photos_list["photos"]:
            if download_image(search_term, photo):
                i = i + 1
        page_i = page_i + 1


def download_images():
    """
    Download images using a search list with search terms and add them to the image database.
    """
    search_list = ["sleeping", "planet", "safari", "fire", "light", "technology", "business", "meeting", "autumn",
                   "spring"]
    for term in search_list:
        # number of tiles of the picture mosaic = 50
        # database multiplier = 1
        get_images(term, 50, 1)


def get_list_of_available_images(search_term):
    """
    Collect the images in the image database using the given search term.

    :param search_term: the term that relates to the desired images to be collected
    :return: a list of image file names 
    """
    image_files = os.listdir(IMAGES_FILE_PATH + "downloaded/")
    # ignore the .gitignore file as it is not an image file
    image_files.remove(".gitignore")
    if search_term is not None:
        for image in image_files:
            if image.split("_")[0] != search_term:
                image_files.remove(image)
    # add a sense of random by shuffling, important for the diversity of the algorithm
    random.shuffle(image_files)
    return image_files


def create_image_montage(source_image_m, file_path, images, tile_width, tile_height, tile_rows, tile_cols):
    """
    Create a image montage using the tile images of an output picture mosaic. This function puts together
    the output of the Evolutionary Algorithm(EA) to create the picture mosaic file for exporting purposes.

    :param source_image_m: the ImageM object of the picture mosaic image
    :param file_path: the file path to stgore the montage to
    :param images: the images that make up the montage i.e. tiles of the picture mosaic
    :param tile_width: the width, in pixels, of the picture mosaic tile
    :param tile_height: the height, in pixels, of the picture mosaic tile
    :param tile_rows: the number of rows in a tile of the picture mosaic
    :param tile_cols: the number of columns in a tile of the picture mosaic
    """
    montages = build_montages(images, (tile_width, tile_height), (tile_rows, tile_cols))
    source_image_m.save_image_data_to_file(montages, file_path)


def display_montage_on_screen(montages):
    """
    Display the given list of montages parts on the screen as a montage.
    """
    for montage in montages:
        cv.imshow("Montage", montage)
        cv.waitKey(0)


# number of trials to use in generating a list of unique numbers
LARGE_TRIAL_NUMBER = 1000000


def generate_unique_random_numbers(higher_bound, trials):
    """
    Generates a list of unique random numbers that are uniformly distributed.

    :param higher_bound: the highest possible number for the list of random numbers
    :param trials: the required number of random numbers
    :return: a list of unique random numbers
    """
    if higher_bound + 1 == trials:
        rand_nums = [i for i in range(0, trials)]
        random.shuffle(rand_nums)
        return rand_nums
    rand_nums = []
    i = 0
    counter = -1
    # keep going until we have enough unique numbers or our trails have run out
    while i < trials:
        counter = counter + 1
        temp_nums = np.random.randint(0, higher_bound, trials)
        for num in temp_nums:
            if num not in rand_nums:
                # to avoid repeating numbers
                rand_nums.append(num)
                i = i + 1
        if counter >= LARGE_TRIAL_NUMBER:
            # if we get stuck in iterations due to the random nature of things
            print("counter went past 1million")
            for j in range(0, higher_bound + 1):
                if j not in rand_nums:
                    rand_nums.append(j)

    return rand_nums


def get_num_vertical_tiles(image_rows, image_cols, tiles_num):
    """
    Computes the number of tiles that make the vertical side of the image.

    :param image_rows: the number of rows in the image
    :param image_cols: the number of columns in the image
    :param tiles_num: the number of tiles that entire image is going to be split into
    :return: the number of tiles that make the vertical side of the imag
    """
    return int(math.sqrt(tiles_num))


def get_num_horizontal_tiles(image_rows, image_cols, tiles_num):
    """
    Computes the number of tiles that make the horizontal side of the image.

    :param image_rows: the number of rows in the image
    :param image_cols: the number of columns in the image
    :param tiles_num: the number of tiles that entire image is going to be split into
    :return: the number of tiles that make the horizontal side of the imag
    """
    return int(math.sqrt(tiles_num))


class Individual:
    """
    The Individual class represents a specific Individual in the population of Individuals
    of possible picture mosaics to select from in a given generation. It holds a fitness,
    average fitness of tiles and the tiles of the picture mosaic.

    :param tiles: the tiles that make up the picture mosaic of this Individual
    """
    def __init__(self, tiles):
        self.fitness = [0] * len(tiles)
        self.average_fitness = None
        self.tiles = tiles

    def compute_average_fitness(self):
        """
        Computes the fitness for this Individual in the population.
        """
        self.average_fitness = sum(self.fitness)/len(self.fitness)

    def __str__(self):
        return "fitness:\n" + str(self.fitness) + "\ntiles:\n" + str(self.tiles) + "\naverage fitness: "\
               + str(self.average_fitness)


# used to growing the image database on downloading
DATABASE_MULTIPLIER = 2
# the chosen stop condition of the Evolutionary Algorithm(EA) for experimentaion
STOPPING_CONDITION = "iterations"
# if the stop condition is 'iterations' then this is the number of iterationns the EA will be run for
MAX_ITERATIONS = 50


class MosaicMaker:
    """
    The MosaicMaker class encapsulates the components of the Evolutionary Algorithm(EA) that is used
    to improve a starting population through mutation and crossover to generate the best possible
    picture mosaic using the given set of images in the image database.

    :param source_iamge_addr: the file path to the input image
    :param num_of_tiles: the number of tiles to split the input image into
    :param population_size: the number of Individuals in the population
    :param search_term: the search term that best relates to the input image
    """
    def __init__(self, source_image_addr, num_of_tiles, population_size, search_term):
        self.source_image_m = ImageM(source_image_addr)
        self.num_of_tiles = num_of_tiles
        self.num_tile_rows = 0
        self.num_tile_cols = 0
        self.population_size = population_size
        self.search_term = search_term
        self.tiles_images = []
        self.t = 0  # generation counter
        self.individual_list = []

    def create_tile_images(self):
        """
        Creates the images of the tiles of the picture mosaics of the Individuals of the population.
        """
        photos = get_list_of_available_images(self.search_term)
        if len(photos) < self.num_of_tiles * DATABASE_MULTIPLIER:
            # too few photos for this search term so download more
            get_images(self.search_term, len(photos) - self.num_of_tiles * DATABASE_MULTIPLIER, DATABASE_MULTIPLIER)
            photos = get_list_of_available_images(self.search_term)
            if len(photos) < self.num_of_tiles * DATABASE_MULTIPLIER:
                print(FAIL_COLOUR + "Failed to get enough photos to create the mosaic" + ENDC)
                sys.exit()
        # resize source image if necessary, measure out tiles
        self.num_tile_rows = get_num_vertical_tiles(self.source_image_m.rows, self.source_image_m.cols,
                                                    self.num_of_tiles)
        self.num_tile_cols = get_num_horizontal_tiles(self.source_image_m.rows, self.source_image_m.cols,
                                                      self.num_of_tiles)
        new_rows = 0
        new_cols = 0
        if self.source_image_m.rows % self.num_tile_rows != 0:
            new_rows = self.source_image_m.rows + (self.source_image_m.rows % self.num_tile_rows)
            if new_rows / self.num_tile_rows != self.num_tile_rows:
                new_rows = self.source_image_m.rows - (self.source_image_m.rows % self.num_tile_rows)
        if self.source_image_m.cols % self.num_tile_cols != 0:
            new_cols = self.source_image_m.cols + (self.source_image_m.cols % self.num_tile_cols)
            if new_cols / self.num_tile_cols != self.num_tile_cols:
                new_cols = self.source_image_m.cols - (self.source_image_m.cols % self.num_tile_cols)
        # make a good fit
        new_rows = int(new_rows)
        new_cols = int(new_cols)
        if new_rows != 0 or new_cols != 0:
            if new_rows == 0:
                new_rows = self.source_image_m.rows
            if new_cols == 0:
                new_cols = self.source_image_m.cols
            # print("rows are now: " + str(new_rows))
            # print("cols are now: " + str(new_cols))
            self.source_image_m.resize_image(new_rows, new_cols)
        tile_row_dim = int(self.source_image_m.rows / self.num_tile_rows)
        tile_col_dim = int(self.source_image_m.cols / self.num_tile_cols)
        # print("title rows is: " + str(self.num_tile_rows))
        # print("title cols is: " + str(self.num_tile_cols))
        # print("source rows: " + str(self.source_image_m.rows))
        # print("source cols: " + str(self.source_image_m.cols))
        # save on gas
        self.source_image_m.compute_neighbourhood_pixel_values(self.num_tile_rows, self.num_tile_cols, tile_row_dim,
                                                               tile_col_dim)
        i = 0
        for photo in photos:
            if i > self.num_of_tiles * DATABASE_MULTIPLIER:
                break
            # avoid the gitignore file in the folder
            temp_image_m = ImageM(IMAGES_FILE_PATH + "downloaded/" + photo)
            temp_image_m.resize_image(tile_row_dim, tile_col_dim)
            # print("compute pixels avg " + str(i) + "photo: " + photo)
            temp_image_m.compute_average_pixel_value()
            self.tiles_images.append(temp_image_m)
            i = i + 1

    def create_and_initialise_population(self):
        """
        Represents the creation and initialisation step of the EA.
        """
        self.create_tile_images()
        # Create and initialize an n_x-dimensional population of n_s individuals
        total_avail_tiles = len(self.tiles_images)
        for p in range(0, self.population_size):
            temp_tiles = generate_unique_random_numbers(total_avail_tiles - 1, self.num_of_tiles)
            temp_individual = Individual(temp_tiles)
            self.individual_list.append(temp_individual)

    def stop_condition(self):
        """
        Represents the stop condition check of the EA.
        """
        if STOPPING_CONDITION == "iterations":
            if MAX_ITERATIONS <= self.t:
                return True
            else:
                return False

    def evaluate_fitness_of_individual(self, individual_i):
        """
        Evaluates the fitness of a given individua in the population.

        :param individual_i: the position of the individual in the list of Individuals of the population.
        """
        tile_num = 0
        for i in range(0, self.num_tile_rows):
            for j in range(0, self.num_tile_cols):
                nbh_avg = self.source_image_m.get_average_neighbourhood_pixel_value_for_tile(i, j)
                temp_image_m = self.tiles_images[individual_i.tiles[tile_num]]
                tile_avg = temp_image_m.get_average_pixel_value()
                individual_i.fitness[tile_num] = self.source_image_m.get_colour_difference(nbh_avg, tile_avg)
                tile_num = tile_num + 1

        individual_i.compute_average_fitness()

    def evaluate_fitness_of_individuals(self):
        """
        Evaluates the fitness of all the Individuals in the population.
        """
        for p in range(0, self.population_size):
            self.evaluate_fitness_of_individual(self.individual_list[p])

    def crossover(self, parent_i, parent_j):
        """
        Represents the crossover step of the EA. Random tiles are selected from the parents and the tiles
        with the best fitness are put into the offspring.

        :param parent_i: the position of the first parent in the list of Individuals of the population.
        :param parent_i: the position of the second parent in the list of Individuals of the population.
        :return: a new Individual, the offspring, of the given parents
        """
        offspring = Individual([0] * self.num_of_tiles)
        for i in range(0, self.num_of_tiles):
            if self.individual_list[parent_i].fitness[i] < self.individual_list[parent_j].fitness[i]:
                offspring.tiles[i] = self.individual_list[parent_i].tiles[i]
                offspring.fitness[i] = self.individual_list[parent_i].fitness[i]
            elif self.individual_list[parent_i].fitness[i] > self.individual_list[parent_j].fitness[i]:
                offspring.tiles[i] = self.individual_list[parent_j].tiles[i]
                offspring.fitness[i] = self.individual_list[parent_j].fitness[i]
            else:
                # case where the fitness for the tile is the same
                if self.individual_list[parent_i].average_fitness > self.individual_list[parent_j].average_fitness:
                    offspring.tiles[i] = self.individual_list[parent_j].tiles[i]
                    offspring.fitness[i] = self.individual_list[parent_j].fitness[i]
                else:
                    # if the fitness of parent_i is smaller or if they are both equal then just go with the one of
                    # either parent
                    offspring.tiles[i] = self.individual_list[parent_i].tiles[i]
                    offspring.fitness[i] = self.individual_list[parent_i].fitness[i]

        offspring.compute_average_fitness()
        return offspring

    def crossover_parents(self):
        """
        Selects parents for crossover and sends to the crossover function for production of offsring.

        :return: a list of new offspring obtained by crossover
        """
        num_to_crossover = np.random.randint(2, self.population_size, 1)[0]
        if num_to_crossover % 2 != 0:
            num_to_crossover = num_to_crossover + 1
        parents_to_crossover = generate_unique_random_numbers(self.population_size - 1, num_to_crossover)
        offspring = []
        i = 0
        while i < num_to_crossover - 1:
            parent_i = parents_to_crossover[i]
            i = i + 1
            parent_j = parents_to_crossover[i]
            offspring.append(self.crossover(parent_i, parent_j))
            i = i + 1

        return offspring

    def mutate(self, individual_i):
        """
        Mutates the given Individual using uniformly distributed(random) mutation.

        :param individual_i: the position in the list of Individuals of the population
        """
        tiles_to_mutate = np.random.randint(1, self.num_of_tiles - 1, 1)[0]
        positions_to_mutate = generate_unique_random_numbers(self.num_of_tiles - 1, tiles_to_mutate)
        tile_numbers = [i for i in range(0, len(self.tiles_images))]
        random.shuffle(tile_numbers)
        for position in positions_to_mutate:
            for t in tile_numbers:
                if t not in individual_i.tiles:
                    individual_i.tiles[position] = t
        self.evaluate_fitness_of_individual(individual_i)

    def mutate_population(self):
        """
        Represents the mutation step of the EA.
        """
        num_to_mutate = np.random.randint(1, self.population_size, 1)[0]
        individuals_to_mutate = generate_unique_random_numbers(self.population_size - 1, num_to_mutate)
        for individual_i in individuals_to_mutate:
            self.mutate(self.individual_list[individual_i])

    def random_selection(self, selection_pool):
        """
        Represents the Random Selection operator of the EA.

        :param selection_pool: the selection pool to use in the Random Selection operator
        """
        return generate_unique_random_numbers(selection_pool - 1, self.population_size)

    def elitist_selection(self, new_offspring):
        """
        Represents the Elitist Selection operator of the EA.

        :param new_offspring: the new offspring to use in the Elitist Selection operator
        """
        full_list = self.individual_list
        if new_offspring is not None and len(new_offspring) > 0:
            full_list = full_list + new_offspring
        avg_fitness_list = [full_list[f].average_fitness for f in range(0, len(full_list))]
        # to get the biggest values
        avg_fitness_list.sort(reverse=True)
        new_individuals_i_s = []
        count = 0
        for j in range(0, len(full_list)):
            for k in range(0, len(full_list)):
                if avg_fitness_list[j] == full_list[k].average_fitness and k not in new_individuals_i_s:
                    new_individuals_i_s.append(k)
                    count = count + 1
                    if count == self.population_size:
                        return new_individuals_i_s

    def proportional_selection(self, new_offspring):
        """
        Represents the Proportional Selection operator of the EA.

        :param new_offspring: the new offspring to use in the Proportional Selection operator
        """
        full_list = self.individual_list
        if new_offspring is not None and len(new_offspring):
            full_list = full_list + new_offspring
        phis = [0.0] * len(full_list)
        fitnesses = [full_list[f].average_fitness for f in range(0, len(full_list))]
        fit_sum = sum(fitnesses)
        for i in range(0, len(full_list)):
            phis[i] = full_list[i].average_fitness / fit_sum
        # to get the biggest values
        phis.sort(reverse=True)
        new_individuals_i_s = []
        for j in range(0, self.population_size):
            for k in range(0, len(full_list)):
                if phis[j] == full_list[k].average_fitness and k not in new_individuals_i_s:
                    new_individuals_i_s[j] = k

        return new_individuals_i_s

    def get_mutation_probability(self):
        """
        Returns the probability of mutation of the EA.
        """
        return (1 / 240) + (0.11375 / 2 ** self.t)

    def get_crossover_probability(self):
        """
        Returns the probability of crossover of the EA.
        """
        return 0.50

    def should_mutate(self):
        """
        Determines if mutation should take place in this iteration of the EA.

        :return: True if mutation should take place, and False if not.
        """
        if np.random.uniform(0, 1, 1)[0] < self.get_mutation_probability():
            print("mutation")
            return True
        return False

    def should_cross_over(self):
        """
        Determines if crossover should take place in this iteration of the EA.

        :return: True if crossover should take place, and False if not.
        """
        if np.random.uniform(0, 1, 1)[0] < self.get_crossover_probability():
            return True
        return False

    def run(self):
        """
        Executes all the steps of the EA to output a file population of Individuals to select a 
        picture mosaic.
        """
        self.create_and_initialise_population()
        while not self.stop_condition():
            # Evaluate the fitness
            self.evaluate_fitness_of_individuals()
            # Perform reproduction to create offspring
            offspring = []
            change = False
            if self.should_mutate():
                self.mutate_population()
                change = True
            if self.should_cross_over():
                offspring = self.crossover_parents()
                change = True
            if not change:
                self.t = self.t + 1
                continue
            # Select the new population, C(t + 1)
            total_population = self.population_size + len(offspring)
            # new_population = self.random_selection(total_population)
            new_population = self.elitist_selection(offspring)
            temp_individuals = []
            for p in new_population:
                if p < self.population_size:
                    temp_individuals.append(self.individual_list[p])
                else:
                    temp_individuals.append(offspring[p - self.population_size])
            for i in range(0, self.population_size):
                self.individual_list[i] = temp_individuals[i]
            # Advance to the new generation, i.e. t = t + 1
            self.t = self.t + 1

    def get_best_individual(self):
        """
        Returns the Individual with the best fitness value in the population.

        :return: the Individual with the best fitness value in the population
        """
        min_i = 0
        min_value = self.individual_list[0].average_fitness
        for i in range(1, self.population_size):
            if self.individual_list[i].average_fitness < min_value:
                min_value = self.individual_list[i].average_fitness
                min_i = i

        return self.individual_list[min_i]

    def use_best_individual_to_generate_mosaic(self):
        """
        Uses the Individual with the best fitness value in the population to generate a picture
        mosaic that best matches the given input image. The tiles of the Individuals are stiched
        together using a montage building function and the result stored in a image file.
        """
        best_individual = self.get_best_individual()
        tile_cols = self.tiles_images[best_individual.tiles[0]].cols
        tile_rows = self.tiles_images[best_individual.tiles[0]].rows
        pic_name = self.source_image_m.image_addr.split("/")
        name_parts = pic_name[len(pic_name) - 1].split(".")
        id = str(uuid.uuid4())
        file_name = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_" + id + "." + name_parts[1]
        source_image_copy = self.source_image_m.copy()
        fig_name1 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_OG_histogramB_" + id + "." + name_parts[1]
        fig_name2 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_OG_histogramG_" + id + "." + name_parts[1]
        fig_name3 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_OG_histogramR_" + id + "." + name_parts[1]
        og_histogram_b = self.source_image_m.save_colour_histogram(fig_name1, 1)
        og_histogram_g = self.source_image_m.save_colour_histogram(fig_name2, 2)
        og_histogram_r = self.source_image_m.save_colour_histogram(fig_name3, 3)
        tile_num = 0
        row_start = 0
        for i in range(0, self.num_tile_rows):
            col_start = 0
            for j in range(0, self.num_tile_cols):
                self.source_image_m.fill_neighbourhood(row_start, col_start, tile_rows, tile_cols,
                                                       self.tiles_images[best_individual.tiles[tile_num]])
                tile_num = tile_num + 1
                col_start = col_start + tile_cols
            row_start = row_start + tile_rows
        self.source_image_m.save_image_to_file(file_name)
        matched = self.source_image_m.get_image_from_histogram_matching(source_image_copy)
        file_name2 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                      + str(self.num_tile_cols) + "_matched_" + id + "." + name_parts[1]
        fig_name21 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                      + str(self.num_tile_cols) + "_mosaic_histogramB_" + id + "." + name_parts[1]
        fig_name22 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                      + str(self.num_tile_cols) + "_mosaic_histogramG_" + id + "." + name_parts[1]
        fig_name23 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                      + str(self.num_tile_cols) + "_mosaic_histogramR_" + id + "." + name_parts[1]
        mosaic_histogram_b = self.source_image_m.save_colour_histogram(fig_name21, 1)
        mosaic_histogram_g = self.source_image_m.save_colour_histogram(fig_name22, 2)
        mosaic_histogram_r = self.source_image_m.save_colour_histogram(fig_name23, 3)
        diff_fig_name1 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_histogram_diff_B_" + id + "." + name_parts[1]
        diff_fig_name2 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_histogram_diff_G_" + id + "." + name_parts[1]
        diff_fig_name3 = IMAGES_FILE_PATH + "output/" + name_parts[0] + "_" + str(self.num_tile_rows) + "x"\
                                     + str(self.num_tile_cols) + "_histogram_diff_R_" + id + "." + name_parts[1]
        diff_values_b = self.source_image_m.compare_histograms(diff_fig_name1, og_histogram_b, mosaic_histogram_b)
        diff_values_g = self.source_image_m.compare_histograms(diff_fig_name2, og_histogram_g, mosaic_histogram_g)
        diff_values_r = self.source_image_m.compare_histograms(diff_fig_name3, og_histogram_r, mosaic_histogram_r)
        matched.save_image_to_file(file_name2)


def average_for_difference(image_name):
    """
    Prints the average difference values from experiments of the image with the given name on the screen.

    Values computated for display include:
    - Average pixel difference
    - Minimum pixel difference
    - Standard deviation of the pixel difference
    - Average runtime to generate a picture mosaic
    """
    start_time = time.time()
    t_num_of_tiles = 144
    t_population_size = 30
    t_search_term = None
    total_sum = 0
    total_run_time = 0
    trials = 3
    min_cost = 1000000
    avgs = np.array([0.0] * trials)
    for i in range(0, trials):
        mosaic_maker = MosaicMaker(IMAGES_FILE_PATH + image_name, t_num_of_tiles, t_population_size, t_search_term)
        mosaic_maker.run()
        temp_best = mosaic_maker.get_best_individual()
        total_sum = total_sum + temp_best.average_fitness
        total_run_time = total_run_time + time.time() - start_time
        avgs[i] = temp_best.average_fitness
        if min_cost > temp_best.average_fitness:
            min_cost = temp_best.average_fitness

    print("for: " + image_name)
    print("avg diff is: " + str(total_sum/trials))
    print("min diff is: " + str(min_cost))
    print("std dev is: " + str(np.std(avgs)))
    print("avg runtime: " + str(total_run_time/trials))


def make_mosaic(image_name):
    """
    Creates a picture mosaic using the input image with the given image name. An Evolutionary Algorithm(EA)
    is used in the generation process.

    :param image name: the name of the input image used as the source image of the picture mosaic generation
    process
    """
    start_time = time.time()
    t_num_of_tiles = 144
    t_population_size = 30
    t_search_term = None
    mosaic_maker = MosaicMaker(IMAGES_FILE_PATH + image_name, t_num_of_tiles, t_population_size, t_search_term)
    mosaic_maker.run()
    mosaic_maker.use_best_individual_to_generate_mosaic()
    end_time = time.time() - start_time
    print("Execution time was: " + str(end_time) + " seconds")


if __name__ == "__main__":
    # place the input images in the images directory so that they can be used by the picture mosaic
    # Evolutionary Algorithm(EA)
    make_mosaic("/sample_1/black_dice.jpg")
    make_mosaic("/sample_2/ibra.jpeg")
    make_mosaic("/sample_3/green_waterfall.jpg")
    make_mosaic("/sample_4/red_ballons.jpg")

