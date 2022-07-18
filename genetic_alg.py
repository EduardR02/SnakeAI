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
        # self.fitness_per_move = -1
        self.initial_moves = int(config.grid_count.sum_xy() * 3.5)
        self.moves_added_on_food = int(config.grid_count.sum_xy() * 2)
        self.max_moves = int(self.initial_moves * 1.5)
        self.food_reward = int(config.grid_count.sum_xy() * 5)
        # at the beginning of the generation these should be ordered from big -> small
        self.best_all_time = []
        self.best_of_gen = []
        self.best_snakes_list_length = 2    # minimum 1
        self.graph = graph.Graph(self.population_size, self.generation, config.graph_file_name)
        self.file_manager = file_manager.ModelFileManager(config.models_file_path)
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
        self.graph.reset()
        self.current_snake_idx = 0
        self.generation_has_finished = False
        self.current_snake = self.current_population[self.current_snake_idx]
        self.create_new_food()
        self.calculate_snake_moves_to_food_and_avg()

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
        self.calculate_snake_moves_to_food_and_avg()

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
        if self.snake_head_on_food():
            self.update_snake_ate_food()
        if self.current_snake.is_dead:
            self.calculate_snake_fitness()

    def update_snake_ate_food(self):
        self.create_new_food()
        self.current_snake.score += self.food_gain_times
        # important that this step is AFTER score calculation
        self.calculate_snake_moves_to_food_and_avg()
        self.fitness_update_after_food()
        self.current_snake.moves_left = min(self.max_moves, self.current_snake.moves_left + self.moves_added_on_food)
        self.current_snake.add_length_to_snake(self.food_gain_times)

    def fitness_update_after_food(self):
        fitness_update = (2 ** min(self.current_snake.score, 10)) * self.current_snake.curr_gamma * self.food_reward
        self.current_snake.fitness += fitness_update * self.current_snake.score
        self.current_snake.curr_gamma = 1.0

    def calculate_snake_fitness(self):
        # self.fitness_get_food_fast()
        # self.fitness_move_much_get_food()
        self.gamma_fitness()
        # exponential / polynomial works nicely to enforce progress, linear is much, much worse

    def gamma_fitness(self):
        self.current_snake.fitness = int(self.current_snake.fitness)

    def fitness_get_food_fast(self):
        # reward less moves
        curr_score = self.current_snake.score
        total_moves = self.current_snake.total_moves
        moves_to_food_avg = self.current_snake.moves_to_food_avg
        if curr_score == 0:
            self.current_snake.fitness = total_moves
        else:
            # val between 1.0 and 2.0
            moves_to_food_avg = (moves_to_food_avg + self.max_moves) / self.max_moves
            moves_to_food_avg = ((2.0 - moves_to_food_avg) ** 3) * 8
            moves_to_food_avg = int(2 ** moves_to_food_avg)
            self.current_snake.fitness = (2 ** min(curr_score, 10)) * moves_to_food_avg * ((curr_score + 1) ** 2)
            self.current_snake.fitness += int(total_moves ** 1.5)

    def fitness_move_much_get_food(self):
        curr_score = self.current_snake.score
        total_moves = self.current_snake.total_moves
        self.current_snake.fitness = (2 ** min(curr_score, 10)) * total_moves * ((curr_score + 1) ** 2)

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

    def calculate_generations_fitness(self):
        # assumes sorted list
        self.cumulative_fitness = sum(snake.fitness for snake in self.current_population)

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
        print("GENERATION:", self.generation, "MUTATION RATE:", self.mutation_rate)
        for best in self.best_of_gen:
            print("SCORE:", best.score, "FITNESS:", best.fitness)
        print("--------------------")
        assert len(self.best_of_gen) == self.best_snakes_list_length

    def create_next_generation(self):
        K.clear_session()
        self.sort_population()
        self.calculate_generations_fitness()
        self.update_best_snakes_list()
        if config.save_m or (config.auto_save and self.was_new_best_set()):
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
        self.create_new_food()
        self.calculate_snake_moves_to_food_and_avg()

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
        parents = self.pick_dont_remove_dupes(round(self.population_size * self.crossover_parents_percent))
        # parents = self.current_population[-round(self.population_size * self.crossover_parents_percent):]
        parents += self.best_all_time
        # if odd, will create one child too many. in that case, remove that child
        for r in range((amount_to_pick + 1) // 2):
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
            if picked_snake.score == 0 and random.random() <= 0.1:
                child_snake = self.create_new_snake()
            else:
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
        while d > 0:
            if self.cumulative_fitness == 0:
                i = random.randint(1, self.population_size)     # inclusive
                return -i
            d -= self.current_population[-i].fitness / self.cumulative_fitness
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

    def calculate_snake_moves_to_food_and_avg(self):
        assert self.current_snakes_food_pos is not None
        # negative value of minimum moves to reach food, because each moves +1 value will track "unnecessary" moves
        moves_to_food = self.current_snakes_food_pos - self.current_snake.get_head_position()
        moves_to_food = -abs(moves_to_food.x) - abs(moves_to_food.y)
        if self.current_snake.score > 0:
            # calc running mean
            prev_avg = self.current_snake.moves_to_food_avg
            prev_avg = prev_avg + (1.0 / self.current_snake.score) * (self.current_snake.moves_to_food - prev_avg)
            self.current_snake.moves_to_food_avg = prev_avg
        # set moves_to_food to min number of steps to reach food (negative)
        self.current_snake.moves_to_food = int(moves_to_food)

    def update_mutation_rate(self):
        self.mutation_rate = max(self.min_mutation_rate, self.mutation_rate * self.mutation_decay)

    def was_new_best_set(self):
        # if they refer to the same object, means that a new best was set
        return self.best_of_gen[0] == self.best_all_time[0]

    def sort_population(self):
        # supposed to be smaller -> larger
        self.current_population.sort(key=cmp_to_key(snake.Snake.compare_performance))
        # possible that all values are the same, therefore >=
        assert self.current_population[-1].compare_performance(self.current_population[0]) >= 0

    def update_graph(self):
        cumulative_score = sum(s.score for s in self.current_population)
        self.graph.update_graph([self.best_of_gen[0].score, self.cumulative_fitness, cumulative_score])

