import numpy as np
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, population_size, start_gen, filename):
        self.population_size = population_size
        self.generation = start_gen
        self.filename = filename
        self.score_color = "#61AFEF"
        self.fitness_color = "#5cdb95"
        self.background_color = "#2b2b2b"
        self.best_of_gens_list = []

    def update_graph(self, current_gen_best):
        self.best_of_gens_list += [current_gen_best]
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Score", color=self.score_color)
        ax1.plot(np.asarray(self.best_of_gens_list)[:, 0], color=self.score_color)
        ax1.tick_params(axis="y", labelcolor=self.score_color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel("Average Fitness", color=self.fitness_color)  # we already handled the x-label with ax1
        ax2.plot(np.asarray(self.best_of_gens_list)[:, 1] / self.population_size, color=self.fitness_color)
        ax2.tick_params(axis="y", labelcolor=self.fitness_color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        plt.title("Snake Neural Network")
        ax1.set_facecolor(self.background_color)
        plt.savefig(self.filename, bbox_inches="tight")
        plt.close("all")

    def reset(self):
        self.best_of_gens_list = []
