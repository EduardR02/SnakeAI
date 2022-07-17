import gui
import genetic_alg
import config
import controller
import time


def main():
    ga = genetic_alg.GeneticAlgorithm(config.population_size, config.mutation_rate,
                                      mutation_decay=config.mutation_rate_decay, random_crossover_bias=True,
                                      mutation_skip_rate=0.1, food_gain_times=config.food_gain_times,
                                      min_mutation_rate=config.minimum_mutation_rate,
                                      crossover_parents_percent=0.05)
    vis = gui.GUI(ga)
    listener = controller.KeyController()
    listener.start()
    update_counter = 0
    while True:
        ga.update_cycle()
        if config.no_graphics:
            continue
        update_counter += 1
        if update_counter >= config.updates_per_draw:
            vis.update()
            update_counter = 0
        if config.pause_between_frames_ms > 0:
            time.sleep(1e-3 * config.pause_between_frames_ms)


if __name__ == "__main__":
    main()
