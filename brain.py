import random
import numpy as np

import config
from point import Point
import utils
import model
from contextlib import nullcontext
from torch.amp import autocast
from torch import float16, from_numpy, no_grad
import copy


device_type = 'cpu'
ctx = nullcontext() if device_type == 'cpu' else autocast(device_type=device_type, dtype=float16)
direction_dict = {"left": 0, "up": 1, "right": 2, "down": 3}


class Brain:
    use_bias = False
    directions_to_look = 8
    inputs_per_direction = 3
    input_size = directions_to_look * inputs_per_direction
    # next 4 values will be overwritten by the Genetic Algorithm class
    mutation_skip_rate = 0.05
    random_mutation_rate_in_interval = False
    crossover_bias = 0.5
    distrib_mul = 2.0
    max_size = max(config.grid_size.x, config.grid_size.y)
    random_crossover_bias = True
    object_dict = {"food": 0, "wall": 2, "body": 1}

    def __init__(self, ready_model=None):
        # ONLY PASS DEEP COPIES OR NEW MODELS HERE, DO NOT PASS SHALLOW COPY
        if ready_model is not None:
            self.model = ready_model
        else:
            self.model = self.create_model()
        self.model.to(device_type).eval()
        self.inputs = []
        # the snake shall pass itself to the brain when the snake is created, so the brain will be created with no snake
        # the snake is just needed for input params and collision checking etc.
        self.snake = None

    @no_grad()
    def think(self, food_position, current_direction):
        inputs = self.generate_inputs(food_position, current_direction)
        reshaped_inputs = inputs.reshape(-1, self.input_size).astype(np.float32)
        reshaped_inputs = from_numpy(reshaped_inputs).to(device_type)
        with ctx:
            predicted_move = self.model(reshaped_inputs)    # left, stay, right
        return (current_direction + (int(predicted_move) - 1)) % 4

    def mutate(self, rate):
        """
        BE EXTREMELY CAREFUL WITH SHALLOW COPYING
        """
        if random.random() <= self.mutation_skip_rate:
            # DO NOT RETURN SHALLOW COPY HERE
            return self.get_model_deep()
        # ONLY ASSIGN ANYTHING TO A NEW MODEL; NEVER ASSIGN A SHALLOW COPY HERE
        return self.model.mutate(rate, deepcopy=True)

    def crossover(self, other, mutation_rate):
        """
        BE EXTREMELY CAREFUL WITH SHALLOW COPYING
        """
        if self.random_crossover_bias:
            bias = random.random()  # how much is taken from one parent over the other
        else:
            bias = self.crossover_bias
        child1 = self.model.crossover(other.model, bias, deepcopy=True)
        child2 = self.model.crossover(other.model, 1.0 - bias, deepcopy=True)
        return child1.mutate(mutation_rate, deepcopy=False), child2.mutate(mutation_rate, deepcopy=False)

    def create_model(self):
        net = model.SimpleForward(self.input_size, 3, self.use_bias)
        return net

    def generate_inputs(self, food_position, current_direction):
        self.inputs = []
        self.surroundings_to_inputs(food_position)
        self.inputs_for_draw = self.inputs
        # self.inputs = self.inputs[:4*self.inputs_per_direction] + self.inputs[5*self.inputs_per_direction:] + self.inputs[4*self.inputs_per_direction:5*self.inputs_per_direction]
        self.align_to_direction(current_direction)
        return np.array(self.inputs)

    def align_to_direction(self, current_direction):
        # * 2 because we also are looking inbetween directions
        if current_direction == 1:
            return
        if current_direction == 0:
            current_direction = 4
        rotate_by = ((current_direction - 1) * 2) * self.inputs_per_direction
        self.inputs = self.inputs[rotate_by:] + self.inputs[:rotate_by]

    def look_in_direction(self, delta_point, food_position):
        moving_point = self.snake.body[0] + delta_point
        distance = 1
        body_found = False
        food_found = False
        res = [0, 0, 0]
        # now it is "see through", meaning all existing inputs in this direction are recorded, even if line of sight is
        # blocked
        while not utils.is_wall_collision(moving_point):
            if not food_found and moving_point == food_position:
                res[self.object_dict["food"]] = self.normalize_distance(distance)
                food_found = True
            if not body_found and self.snake.is_point_with_body_collision(moving_point):
                res[self.object_dict["body"]] = 1 if distance == 1 else self.normalize_distance(distance) * (- 1)
                body_found = True
            distance += 1
            moving_point += delta_point

        res[self.object_dict["wall"]] = 1 if distance == 1 else self.normalize_distance(distance) * (- 1)
        self.inputs += res

    def surroundings_to_inputs(self, food_position):
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    # because delta is 0
                    continue
                self.look_in_direction(Point(i - 1, j - 1), food_position)

    def get_inputs_around_head(self, food_position, index):
        temp = 0
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    # because this is head position
                    temp = 1
                    continue
                else:
                    observed_point = self.snake.get_head_position() + Point(i - 1, j - 1)
                    if food_position == observed_point:
                        self.inputs[i * 3 + j - temp + index] = 1
                    elif utils.is_wall_collision(observed_point) or self.snake.is_point_with_body_collision(
                            observed_point):
                        self.inputs[i * 3 + j - temp + index] = -1
                    else:
                        self.inputs[i * 3 + j - temp + index] = 0

    def get_direction_input(self, index):
        # as of python 3.7 dictionaries are guaranteed to be ordered, therefore this works
        for i, value in enumerate(self.snake.direction_dict.values()):
            if self.snake.current_direction == value:
                self.inputs[index + i] = 1
            else:
                self.inputs[index + i] = -1

    def get_model_deep(self):
        return copy.deepcopy(self.model)

    def set_snake(self, snake):
        self.snake = snake

    def normalize_distance(self, distance):
        return (self.max_size - distance) / (self.max_size - 1)
    
    def inverse_normalize_distance(self, normalized_distance):
        return self.max_size - normalized_distance * (self.max_size - 1)

    def print_inputs_sub(self, val, arr):
        print(val, "Food:", arr[0], "Distance:", arr[1])

    def print_inputs(self):
        print("Direction:", self.snake.direction_dict.get(int(np.argmax(self.inputs[:4], axis=-1))))
        vals = {0: "left-top", 1: "left", 2: "left-bottom", 3: "top", 4: "bottom",
                5: "right-top", 6: "right", 7: "right-bottom"}
        for i in range(8):
            self.print_inputs_sub("Vision, " + vals[i] + ":", self.inputs[4 + i * 2: 7 + i * 2])
        print("------------------------------------------------------------")