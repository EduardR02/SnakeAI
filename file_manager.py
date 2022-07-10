# noinspection PyUnresolvedReferences
from tensorflow.keras.models import load_model


class FileManager:

    def __init__(self, models_file_path):
        self.models_file_path = models_file_path

    def save_models(self, snakes, best, best_gen):
        print("saving...")
        for i in range(len(snakes)):
            snakes[i].brain.model.save(f"{self.models_file_path}model_nr_{i}.h5")
        for i in range(len(best)):
            best[i].brain.model.save(f"{self.models_file_path}model_nr_best_{i}.h5")
        for i in range(len(best_gen)):
            best_gen[i].brain.model.save(f"{self.models_file_path}model_nr_best_gen_{i}.h5")
        print("models saved!")

    def load_models(self, population_size, best_size, create_snake_function, create_snake_function_with_brain):
        loaded_snakes = []
        best_all = []
        best_gen = []
        print("loading...")
        for i in range(population_size):
            try:
                loaded_snakes.append(create_snake_function_with_brain(
                    load_model(f"{self.models_file_path}model_nr_{i}.h5", compile=False)))
            except OSError:
                loaded_snakes.append(create_snake_function())
        for i in range(best_size):
            try:
                best_all.append(create_snake_function_with_brain(
                    load_model(f"{self.models_file_path}model_nr_best_{i}.h5", compile=False)))
            except OSError:
                best_all.append(create_snake_function())
        for i in range(best_size):
            try:
                best_gen.append(create_snake_function_with_brain(
                    load_model(f"{self.models_file_path}model_nr_best_gen_{i}.h5", compile=False)))
            except OSError:
                best_gen.append(create_snake_function())
        print("models loaded!")
        return loaded_snakes, best_all, best_gen
