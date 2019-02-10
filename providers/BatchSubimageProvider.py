import os
from functools import reduce

import numpy as np
from imageio import imread
from tqdm import tqdm as progress


training_set = 'training'
evaluation_set = 'evaluation'

S2_satellite = 'S2'
S3_satellite = 'S3'


class BatchSubimageProvider:
    __slots__ = (
        '_root_path', '_input_bands', '_target_bands',
        'training_input_subimages', 'training_target_subimages',
        'evaluation_input_subimages', 'evaluation_target_subimages'
    )

    def __init__(self, subimages_folder_path, input_bands=[i for i in range(21)],
                 target_bands=[i for i in range(13)]):

        self._root_path = subimages_folder_path
        self._input_bands = input_bands
        self._target_bands = target_bands

        self._load_training_input_subimages()
        self._load_training_target_subimages()
        self._load_evaluation_input_subimages()
        self._load_evaluation_target_subimages()

    #
    # Public methods
    #

    def training_subimage_count(self):
        return len(self.training_input_subimages)

    def evaluation_subimage_count(self):
        return len(self.evaluation_input_subimages)

    def subimage_shape(self):
        return self.training_input_subimages[0, :, :, 0].shape

    def random_training_subimage_batch(self, batch_size=None):
        # If batch size is not specified then all images are picked
        if batch_size is None:
            batch_size = self.training_subimage_count()

        rand_indexes = np.random.choice(len(self.training_input_subimages), size=batch_size)
        rand_input = self.training_input_subimages[rand_indexes]
        rand_target = self.training_target_subimages[rand_indexes]

        return rand_input, rand_target

    def evaluation_subimages(self):
        return self.evaluation_input_subimages, self.evaluation_target_subimages

    #
    # Private methods
    #

    def _load_training_input_subimages(self):
        self.training_input_subimages = self._load_subimages(training_set, S3_satellite, self._input_bands,
                                                             progress_desc='Loading training input subimages...')

    def _load_training_target_subimages(self):
        self.training_target_subimages = self._load_subimages(training_set, S2_satellite, self._target_bands,
                                                              progress_desc='Loading training target subimages...')

    def _load_evaluation_input_subimages(self):
        self.evaluation_input_subimages = self._load_subimages(evaluation_set, S3_satellite, self._input_bands,
                                                               progress_desc='Loading evaluation input subimages...')

    def _load_evaluation_target_subimages(self):
        self.evaluation_target_subimages = self._load_subimages(evaluation_set, S3_satellite, self._target_bands,
                                                                progress_desc='Loading evaluation target subimages...')

    def _load_subimages(self, set_type, satellite, bands, progress_desc='Loading subimages'):
        # List of places folders for the given satellite
        satellite_places = [os.path.join(place, satellite) for place in self._list_places_folders(set_type)]
        subimage_count = None
        progress_bar = None
        subimages = None

        # One band at a time
        for b, band in enumerate(bands):
            # List of places folders for the given satellite and band
            band_places_folders = [os.path.join(satellite_place, str(band + 1)) for satellite_place in satellite_places]

            for i, band_place_folder in enumerate(band_places_folders):
                band_files = os.listdir(band_place_folder)
                band_files.sort(key=lambda s: reduce(lambda x, y: y + x * 10e5,
                                                     [int(elem) for elem in s[:-4].split('_')]))

                band_place_files = [os.path.join(band_place_folder, file) for file in band_files
                                    if os.path.isfile(os.path.join(band_place_folder, file))]

                # Complete images amount if uncertain, follows "each folder has the same amount of files" convention
                if subimage_count is None:
                    subimage_count = len(band_place_files) * len(satellite_places)

                # Initialize progress bar if uninitialized
                if progress_bar is None:
                    progress_bar = progress(total=subimage_count * len(bands), desc=progress_desc, unit='imgs')

                for j, band_place_file in enumerate(band_place_files):
                    band_image = imread(band_place_file)
                    # Initialize 4-dimensional array if uninitialized
                    if subimages is None:
                        image_width, image_height = band_image.shape
                        subimages = np.empty([subimage_count, image_height, image_width, len(bands)],
                                             dtype=np.uint16)

                    # Indexes as follows: n images from place 0; n images from place 1; ...; n images from place p.
                    # Where: n is the amount of images per folder and p the amount of places
                    index = j + (i * len(band_place_files))
                    subimages[index, :, :, b] = band_image
                    progress_bar.update()

        progress_bar.close()
        return subimages

    def _list_places_folders(self, set_type):
        folder = os.path.join(self._root_path, set_type)
        return [os.path.join(folder, elem) for elem in os.listdir(folder) if
                os.path.isdir(os.path.join(folder, elem))]
