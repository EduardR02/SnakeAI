import file_manager
import graph
import config
import tensorflow.keras.backend as K
import brain
import snake
import random
import utils
from functools import cmp_to_key


class GeneticAlgorithm:

    def __init__(self, population_size, mutation_rate, food_gain_times=1, mutation_decay=1.0,
                 min_mutation_rate=0.0, mutation_skip_rate=0.0, random_mutation_rate_in_interval=False,
                 crossover_bias=0.5, random_crossover_bias=False, crossover_parents_percent=0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.food_gain_times = food_gain_times
        self.mutation_decay = mutation_decay
        self.min_mutation_rate = min_mutation_rate
        self.mutation_skip_rate = mutation_skip_rate
        self.crossover_bias = crossover_bias
        self.random_crossover_bias = random_crossover_bias
        self.crossover_parents_percent = crossover_parents_percent
        # this will NOT randomize the mutation rate, it will randomize in the interval between 0.0 and mutation_rate
        self.random_mutation_rate_in_interval = random_mutation_rate_in_interval
        self.current_population = []
        self.current_snake = None
        self.current_snakes_food_pos = None
        self.generation_has_finished = False
        self.current_snake_idx = 0
        self.generation = 1
        self.cumulative_fitness = 0
        self.adjusted_fitness_sum = 0   # used in selection of next generation
        self.worst_fitness_in_generation = 0
        self.fitness_per_move = 1
        self.base_food_gain = config.grid_count.sum_xy() * 4
        self.initial_moves = config.grid_count.sum_xy() * 2
        self.moves_added_on_food = config.grid_count.sum_xy() * 2
        self.max_moves = self.initial_moves * 2
        # at the beginning of the generation these should be ordered from big -> small
        self.best_all_time = []
        self.best_of_gen = []
        self.best_snakes_list_length = 2    # minimum 1
        self.graph = graph.Graph(self.population_size, self.generation, config.graph_file_name)
        self.file_manager = file_manager.FileManager(config.models_file_path)
        # call these before any snakes are created
        self.init_mutation_params_in_brain()
        self.init_snake_params()

        self.initialize_population()
        self.reset_gen_alg_params()

    def initialize_population(self):
        assert self.best_snakes_list_length >= 1
        self.current_population = [self.create_new_snake() for _ in range(self.population_size)]
        self.best_of_gen = self.current_population[0:self.best_snakes_list_length]
        self.best_all_time = self.current_population[self.best_snakes_list_length:self.best_snakes_list_length*2]

    def reset_gen_alg_params(self):
        # only call when you have a population
        self.generation = 1
        self.cumulative_fitness = 0
        self.adjusted_fitness_sum = 0
        self.worst_fitness_in_generation = 0
        self.graph.reset()
        self.current_snake_idx = 0
        self.generation_has_finished = False
        self.current_snake = self.current_population[self.current_snake_idx]
        self.create_new_food()

    def save_models(self):
        if not config.save_m: return
        self.file_manager.save_models(self.current_population, self.best_all_time, self.best_of_gen)
        config.save_m = False

    def load_models(self):
        if not config.load_m: return
        # create new snaked passed as function, so it can be called
        self.current_population, self.best_all_time, self.best_of_gen = self.file_manager.load_models(
            self.population_size, self.best_snakes_list_length,
            self.create_new_snake, self.create_new_snake_model_already_generated)
        self.reset_gen_alg_params()
        config.load_m = False

    def init_next_agent(self):
        self.current_snake_idx += 1
        if self.current_snake_idx >= self.population_size:
            self.generation_has_finished = True
            return
        self.generation_has_finished = False
        self.current_snake = self.current_population[self.current_snake_idx]
        self.create_new_food()

    def update_cycle(self):
        if config.load_m:
            self.load_models()
        elif self.generation_has_finished:
            self.create_next_generation()
        elif self.current_snake.is_dead:
            self.init_next_agent()
        else:
            self.update_current_snake()

    def update_current_snake(self):
        self.current_snake.update(self.current_snakes_food_pos)
        self.current_snake_apply_fitness()

    def current_snake_apply_fitness(self):
        self.update_snake_fitness()
        if self.current_snake.is_dead:
            self.update_snake_died()
        else:
            if self.snake_head_on_food():
                self.update_snake_ate_food()

    def update_snake_ate_food(self):
        self.create_new_food()
        self.current_snake.score += self.food_gain_times
        self.current_snake.fitness += self.base_food_gain * self.food_gain_times
        self.current_snake.moves_left = max(self.max_moves, self.current_snake.moves_left + self.moves_added_on_food)
        self.current_snake.add_length_to_snake(self.food_gain_times)

    def update_snake_died(self):
        if not self.current_snake.changed_direction_at_least_once:
            # fitness per move is negative, this is punishment
            self.current_snake.fitness = 0
            self.current_snake.score = 0

    def snake_head_on_food(self):
        return self.current_snake.get_head_position() == self.current_snakes_food_pos

    def init_mutation_params_in_brain(self):
        # call this before any snakes are created
        brain.Brain.mutation_skip_rate = self.mutation_skip_rate
        brain.Brain.random_mutation_rate_in_interval = self.random_mutation_rate_in_interval
        brain.Brain.crossover_bias = self.crossover_bias
        brain.Brain.random_crossover_bias = self.random_crossover_bias

    # these methods are separated to avoid confusion
    def create_new_snake(self):
        new_brain = brain.Brain()
        new_snake = snake.Snake(new_brain)
        return new_snake

    def create_new_snake_model_already_generated(self, new_model):
        new_brain = brain.Brain(new_model)
        new_snake = snake.Snake(new_brain)
        return new_snake

    def init_snake_params(self):
        snake.Snake.starting_moves = self.initial_moves

    def update_snake_fitness(self):
        # basically moves cost fitness, and ofc scaled to grid count
        self.current_snake.fitness += self.fitness_per_move

    def calculate_generations_fitness(self):
        # assumes sorted list
        self.worst_fitness_in_generation = self.current_population[0].fitness
        self.cumulative_fitness = sum(snake.fitness for snake in self.current_population)
        self.calculate_adjusted_fitness_sum()

    def calculate_adjusted_fitness_sum(self):
        if self.worst_fitness_in_generation < 0:
            # self.worst_fitness_in_generation is negative, therefore minus
            self.adjusted_fitness_sum = self.cumulative_fitness - self.population_size * self.worst_fitness_in_generation
        else:
            self.adjusted_fitness_sum = self.cumulative_fitness

    def update_best_snakes_list(self):
        # replace best of all time with best of generation if appropriate
        self.update_best_of_generation()
        temp_best = []
        i = j = 0
        # combine best of gen and best of all time into single sorted array
        while i + j < self.best_snakes_list_length:
            if self.best_of_gen[i].compare_performance(self.best_all_time[j]) >= 0:
                temp_best.append(self.best_of_gen[i])
                i += 1
            else:
                temp_best.append(self.best_all_time[j])
                j += 1
        # assign best snakes to best all time
        self.best_all_time = temp_best
        assert len(self.best_all_time) == self.best_snakes_list_length

    def update_best_of_generation(self):
        # order from best -> worst
        self.best_of_gen = self.current_population[-self.best_snakes_list_length:][::-1]
        for best in self.best_of_gen:
            print("SCORE:", best.score, "FITNESS:", best.fitness)
        print("--------------------")
        assert len(self.best_of_gen) == self.best_snakes_list_length

    def create_next_generation(self):
        K.clear_session()
        self.sort_population()
        self.calculate_generations_fitness()
        self.update_best_snakes_list()
        if config.save_m:
            self.save_models()
        self.update_graph()
        self.pick_next_generation()
        self.generation += 1
        # this has to be at the end
        self.update_mutation_rate()
        self.new_gen_reset_values()

    def new_gen_reset_values(self):
        self.generation_has_finished = False
        self.current_snake_idx = 0
        self.current_snake = self.current_population[self.current_snake_idx]
        self.cumulative_fitness = 0
        self.adjusted_fitness_sum = 0
        self.worst_fitness_in_generation = 0
        self.create_new_food()

    def pick_next_generation(self):
        new_population = self.add_best_to_new_pop()
        amount_to_pick_crossover = self.population_size // 2 - len(self.best_all_time)
        # +1 in case the population size is odd
        amount_to_pick_pick_and_mutate = (self.population_size + 1) // 2 - len(self.best_of_gen)
        new_population += self.crossover(amount_to_pick_crossover)
        new_population += self.pick_and_mutate(amount_to_pick_pick_and_mutate)
        self.current_population = new_population

    def crossover(self, amount_to_pick):
        assert amount_to_pick <= self.population_size
        new_snakes = []
        parents = self.pick_snakes_by_fitness_no_dupes(round(self.population_size * self.crossover_parents_percent))
        # parents = self.current_population[-round(self.population_size * self.crossover_parents_percent):]
        # add the best ones for guarantee that these are in there, possibly duplicate but doesn't matter
        parents += self.best_of_gen + self.best_all_time
        # if odd, will create one child too many. in that case, remove that child
        for r in range((amount_to_pick + 1) // 2):
            # go from best to worst, second parent goes through all one by one
            p1_idx = p2_idx = 0
            while p2_idx == p1_idx and len(parents) > 1:
                # inclusive
                p2_idx = random.randint(0, len(parents) - 1)
                p1_idx = random.randint(0, len(parents) - 1)
            # this is fine, copy stuff dealt with in brain, this returns two completely new brains, also mutates
            child_model1, child_model2 = parents[p1_idx].crossover(parents[p2_idx], self.mutation_rate)
            child_snake1 = self.create_new_snake_model_already_generated(child_model1)
            child_snake2 = self.create_new_snake_model_already_generated(child_model2)
            new_snakes.append(child_snake1)
            new_snakes.append(child_snake2)
        if amount_to_pick % 2 == 1:
            new_snakes.pop()
        return new_snakes

    def add_best_to_new_pop(self):
        new_population = []
        for good_snake in self.best_all_time + self.best_of_gen:
            new_population.append(self.create_new_snake_model_already_generated(good_snake.brain.get_model_deep()))
        return new_population

    def pick_and_mutate(self, amount_to_pick):
        assert amount_to_pick <= self.population_size
        new_snakes = []
        picked_snakes = self.pick_dont_remove_dupes(amount_to_pick)
        for picked_snake in picked_snakes:
            # shallow / deep copy stuff handled in brain. Mutation returns new brain with new model
            child_model = picked_snake.mutate(self.mutation_rate)
            child_snake = self.create_new_snake_model_already_generated(child_model)
            new_snakes.append(child_snake)
        return new_snakes

    def pick_snakes_by_fitness_no_dupes(self, amount_to_pick):
        picked_snakes = []
        already_picked = set()
        for i in range(amount_to_pick):
            picked_index = self.get_random_snake_by_fitness()
            if picked_index in already_picked:
                while picked_index in already_picked:
                    picked_index -= 1
                    # in case it goes out of bounds
                    if picked_index < -self.population_size:
                        picked_index = -1
            already_picked.add(picked_index)
            picked_snakes.append(self.current_population[picked_index])
        return picked_snakes

    def pick_dont_remove_dupes(self, amount_to_pick):
        picked_snakes = []
        for i in range(amount_to_pick):
            picked_index = self.get_random_snake_by_fitness()
            picked_snakes.append(self.current_population[picked_index])
        return picked_snakes

    def get_random_snake_by_fitness(self):
        # assumes sorted worst -> best
        d = random.random()
        i = 1
        value_to_add = 0
        if self.worst_fitness_in_generation < 0:
            value_to_add = -self.worst_fitness_in_generation
        else:
            assert self.adjusted_fitness_sum == self.cumulative_fitness
        while d > 0:
            if self.adjusted_fitness_sum <= 0:
                i = random.randint(1, self.population_size)     # inclusive
                return -i
            d -= (self.current_population[-i].fitness + value_to_add) / self.adjusted_fitness_sum
            i += 1
            # in case of precision errors
            if i > self.population_size:
                i = 1
        i -= 1
        i = min(-i, -1)  # in case d is initialized to 0.0
        return i

    def create_new_food(self):
        # inefficient when snake covers most of the grids
        assert self.current_snake is not None
        random_point = utils.generate_random_point()
        while self.current_snake.is_point_with_snake_collision(random_point):
            random_point = utils.generate_random_point()
        self.current_snakes_food_pos = random_point

    def update_mutation_rate(self):
        self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * self.mutation_decay)

    def sort_population(self):
        # supposed to be smaller -> larger
        self.current_population.sort(key=cmp_to_key(snake.Snake.compare_performance))
        # possible that all values are the same, therefore >=
        assert self.current_population[-1].compare_performance(self.current_population[0]) >= 0

    def update_graph(self):
        self.graph.update_graph([self.best_of_gen[0].score, self.cumulative_fitness])

