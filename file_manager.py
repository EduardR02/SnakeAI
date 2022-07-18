# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from pathlib import Path


class FileManager:

    def __init__(self, file_dir):
        self.file_dir = file_dir


class ModelFileManager(FileManager):

    def __init__(self, file_dir):
        super().__init__(file_dir)

    def save_models(self, snakes, best, best_gen):
        print("saving...")
        # don't print warnings
        tf.get_logger().setLevel('ERROR')
        for i in range(len(snakes)):
            snakes[i].brain.model.save(f"{self.file_dir}model_nr_{i}.h5")
        for i in range(len(best)):
            best[i].brain.model.save(f"{self.file_dir}model_nr_best_{i}.h5")
        for i in range(len(best_gen)):
            best_gen[i].brain.model.save(f"{self.file_dir}model_nr_best_gen_{i}.h5")
        print("models saved!")

    def load_models(self, population_size, best_size, create_snake_function, create_snake_function_with_brain):
        loaded_snakes = []
        best_all = []
        best_gen = []
        print("loading...")
        for i in range(population_size):
            try:
                loaded_snakes.append(create_snake_function_with_brain(
                    load_model(f"{self.file_dir}model_nr_{i}.h5", compile=False)))
            except OSError:
                loaded_snakes.append(create_snake_function())
        for i in range(best_size):
            try:
                best_all.append(create_snake_function_with_brain(
                    load_model(f"{self.file_dir}model_nr_best_{i}.h5", compile=False)))
            except OSError:
                best_all.append(create_snake_function())
        for i in range(best_size):
            try:
                best_gen.append(create_snake_function_with_brain(
                    load_model(f"{self.file_dir}model_nr_best_gen_{i}.h5", compile=False)))
            except OSError:
                best_gen.append(create_snake_function())
        print("models loaded!")
        return loaded_snakes, best_all, best_gen


class ReplayFileManager(FileManager):

    def __init__(self, file_dir):
        super().__init__(file_dir)

    def save_replay(self, data, filename):
        Path(self.file_dir).mkdir(parents=True, exist_ok=True)
        np.save(self.file_dir + filename + ".npy", data)

    def load_replay(self, filename):
        return np.load(self.file_dir + filename + ".npy", allow_pickle=False)
