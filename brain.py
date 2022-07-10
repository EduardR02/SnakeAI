import random
import tensorflow.keras.initializers as initializers
from tensorflow.keras.layers import Dense
import tensorflow.keras.models as models
import numpy as np
from point import Point
import utils


class Brain:

    use_bias = False
    directions_to_look = 8
    inputs_per_direction = 3
    input_size = directions_to_look * inputs_per_direction
    # next 4 values will be overwritten by the Genetic Algorithm class
    mutation_skip_rate = 0.05
    random_mutation_rate_in_interval = False
    crossover_bias = 0.5
    random_crossover_bias = True
    object_dict = {"food": 0, "wall": 2, "body": 1}

    def __init__(self, ready_model=None):
        # ONLY PASS DEEP COPIES OR NEW MODELS HERE, DO NOT PASS SHALLOW COPY
        if ready_model is not None:
            self.model = ready_model
        else:
            self.model = self.create_model()
        self.inputs = np.zeros(self.input_size)
        # the snake shall pass itself to the brain when the snake is created, so the brain will be created with no snake
        # the snake is just needed for input params and collision checking etc.
        self.snake = None

    def think(self, food_position):
        self.generate_inputs(food_position)
        reshaped_inputs = self.inputs.reshape(-1, self.input_size)
        predicted_direction = np.argmax(self.model(reshaped_inputs), axis=-1)[0]
        return predicted_direction

    def mutate(self, rate):
        """
        BE EXTREMELY CAREFUL WITH SHALLOW COPYING
        """
        if random.random() <= self.mutation_skip_rate:
            # DO NOT RETURN SHALLOW COPY HERE
            return self.get_model_deep()
        # ONLY ASSIGN ANYTHING TO A NEW MODEL; NEVER ASSIGN A SHALLOW COPY HERE
        new_model = self.create_model()
        if self.random_mutation_rate_in_interval: rate *= random.random()  # rate becomes 0 - rate
        for i, layer in enumerate(self.model.layers):
            new_weights_for_layer = []
            # using the original weights here is necessary obviously, but they are only used for calculation
            for weight_array in layer.get_weights():
                # rand expects non tuple, ones expects tuple
                distribution_dev = (1.0 / weight_array.shape[0]) ** 0.5
                mutation_update = np.random.normal(0.0, distribution_dev, size=weight_array.shape)
                weight_mask = np.random.rand(*weight_array.shape) < rate
                weight_mask_opposite = np.ones(weight_array.shape) - weight_mask
                new_weights = weight_array * weight_mask_opposite + mutation_update * weight_mask
                new_weights_for_layer.append(new_weights)
            new_model.layers[i].set_weights(new_weights_for_layer)
        return new_model

    def _shallow_mutate(self, rate):
        """
        USE WITH CARE, only when you already created a new model
        """
        if random.random() <= self.mutation_skip_rate:
            return self.get_model_shallow()
        if self.random_mutation_rate_in_interval: rate *= random.random()  # rate becomes 0 - rate
        for i, layer in enumerate(self.model.layers):
            new_weights_for_layer = []
            # using the original weights here is necessary obviously, but they are only used for calculation
            for weight_array in layer.get_weights():
                # rand expects non tuple, ones expects tuple
                distribution_dev = (1.0 / weight_array.shape[0]) ** 0.5
                mutation_update = np.random.normal(0.0, distribution_dev, size=weight_array.shape)
                weight_mask = np.random.rand(*weight_array.shape) < rate
                weight_mask_opposite = np.ones(weight_array.shape) - weight_mask
                new_weights = weight_array * weight_mask_opposite + mutation_update * weight_mask
                new_weights_for_layer.append(new_weights)
            self.model.layers[i].set_weights(new_weights_for_layer)
        return self.model

    def crossover(self, other, mutation_rate):
        """
        BE EXTREMELY CAREFUL WITH SHALLOW COPYING
        """
        if self.random_crossover_bias:
            bias = random.random()  # how much is taken from one parent over the other
        else:
            bias = self.crossover_bias
        # create completely new net, to which new values will be assigned, DO NOT ASSIGN A SHALLOW COPY OF PARENTS HERE
        child1 = self.create_model()
        child2 = self.create_model()
        # parent net values SHALL ONLY BE USED FOR CALCULATING THE CHILD WEIGHTS; NEVER ASSIGN TO THEM
        p1_net = self.get_model_shallow()
        p2_net = other.get_model_shallow()
        for i, layer in enumerate(p1_net.layers):
            child_weights1 = []
            child_weights2 = []
            for p, weight_array in enumerate(layer.get_weights()):
                weight_array2 = p2_net.layers[i].get_weights()[p]
                # np rand expects non tuple as shape, ones expects tuple as shape
                weight_mask = np.random.rand(*weight_array.shape) < bias  # array of true and false
                weight_mask_2 = np.ones(weight_array2.shape) - weight_mask  # this works with true false vals
                # the children will never have weights from the same parent on the same spot, always opposite
                new_weights1 = weight_array * weight_mask + weight_array2 * weight_mask_2
                new_weights2 = weight_array * weight_mask_2 + weight_array2 * weight_mask
                child_weights1.append(new_weights1)
                child_weights2.append(new_weights2)
            child1.layers[i].set_weights(child_weights1)
            child2.layers[i].set_weights(child_weights2)
        # mutate children, because a new net already created, we can shallow mutate
        child1 = Brain(child1)._shallow_mutate(mutation_rate)
        child2 = Brain(child2)._shallow_mutate(mutation_rate)
        return child1, child2

    def create_model(self):
        model = models.Sequential()
        # initializer = initializers.RandomNormal(mean=0., stddev=(1.0 / 24.0) ** 0.5)
        # initializer2 = initializers.RandomNormal(mean=0., stddev=(1.0 / 8.0) ** 0.5)
        # initializer3 = initializers.RandomNormal(mean=0., stddev=(1.0 / 4.0) ** 0.5)
        model.add(Dense(8, activation="relu", input_dim=self.input_size, use_bias=self.use_bias,
                        # kernel_initializer=initializer, bias_initializer="zeros"
                        ))
        """model.add(Dense(8, activation="relu", use_bias=self.use_bias,
                        kernel_initializer=initializer2, bias_initializer="zeros"))"""
        model.add(Dense(8, activation="relu", use_bias=self.use_bias,
                        # kernel_initializer=initializer2, bias_initializer="zeros"
                        ))
        model.add(Dense(4, activation="softmax", use_bias=self.use_bias,
                        # kernel_initializer=initializer2
                        ))
        model.build(input_shape=(1, self.input_size))
        return model

    def generate_inputs(self, food_position):
        # self.get_direction_input(0) # unnecessary, as surrounding_inputs are enough to figure out direction
        self.surroundings_to_inputs(0, food_position, draw=False)  # 16 inputs

    def add_to_inputs_consistent(self, thing, distance, index):
        normalized_distance = self.normalize_distance(distance)
        self.inputs[index] = 1. if thing == self.object_dict["food"] else 0.
        self.inputs[index + 1] = 1. if thing == self.object_dict["body"] else 0.
        self.inputs[index + 2] = normalized_distance

    def look_in_direction(self, delta_point, food_position, index):
        moving_point = self.snake.body[0] + delta_point
        distance = 1
        # distinguish between body == -1, food == 1 and wall == 0
        while not utils.is_wall_collision(moving_point):
            if moving_point == food_position:
                # cannot be one, don't think it matters
                self.add_to_inputs_consistent(self.object_dict["food"], distance, index)
                return distance, self.object_dict["food"]
            if self.snake.is_point_with_body_collision(moving_point):
                self.add_to_inputs_consistent(self.object_dict["body"], distance, index)
                return distance, self.object_dict["body"]
            distance += 1
            moving_point += delta_point

        self.add_to_inputs_consistent(self.object_dict["wall"], distance, index)
        return distance, self.object_dict["wall"]

    def surroundings_to_inputs(self, index, food_position, draw=False):
        temp = 0
        res = []
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    # because delta is 0
                    temp = 1
                    continue
                distance, thing_found = self.look_in_direction(Point(i - 1, j - 1), food_position,
                                                               (i * 3 + j - temp) * self.inputs_per_direction + index)
                if draw:
                    res.append((i, j, distance, thing_found))
        return res

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
                    elif utils.is_wall_collision(observed_point) or self.snake.is_point_with_body_collision(observed_point):
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

    def get_model_shallow(self):
        return self.model

    def get_model_deep(self):
        new_model = self.create_model()
        for i, layer in enumerate(self.model.layers):
            new_weights_for_layer = []
            for weight_array in layer.get_weights():
                # I assume numpy returns a new array here
                new_weights = weight_array * 1
                new_weights_for_layer.append(new_weights)
            new_model.layers[i].set_weights(new_weights_for_layer)
        return new_model

    def set_snake(self, snake):
        self.snake = snake

    def normalize_distance(self, distance):
        return 1.0 / distance

    def print_inputs_sub(self, val, arr):
        print(val, "Food:", arr[0], "Distance:", arr[1])

    def print_inputs(self):
        print("Direction:", self.snake.direction_dict.get(int(np.argmax(self.inputs[:4], axis=-1))))
        vals = {0: "left-top", 1: "left", 2: "left-bottom", 3: "top", 4: "bottom",
                5: "right-top", 6: "right", 7: "right-bottom"}
        for i in range(8):
            self.print_inputs_sub("Vision, " + vals[i] + ":", self.inputs[4 + i * 2: 7 + i * 2])
        print("------------------------------------------------------------")
