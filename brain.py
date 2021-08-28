import math
import random
import tensorflow.keras.models as models
import tensorflow.keras.initializers as initializers
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow.keras.backend as K

back_color = "#171717"
back_color_2 = "#3c3c3c"
snake_Color = "#4ca3dd"
apple_color = "#ff4040"
line_color = "#2f4454"
food_found_color = "#5cdb95"
grid_size = 30
blob_size = grid_size - 2  # should to be even
grid_count = 20
start_size = 5
apple_boost = 1
input_size = 20
direction_dict = {0: "left", 1: "top", 2: "right", 3: "bottom"}


def create_model():
    model = models.Sequential()
    initializer = initializers.RandomNormal(mean=0., stddev=(1.0 / math.sqrt(16 / 2)))
    initializer2 = initializers.RandomNormal(mean=0., stddev=1.0)
    model.add(Dense(16, activation="relu", input_dim=input_size, use_bias=True,
                    kernel_initializer=initializer2, bias_initializer="zeros"))
    model.add(Dense(16, activation="relu", use_bias=True,
                    kernel_initializer=initializer2, bias_initializer="zeros"))
    model.add(Dense(4, activation="softmax", use_bias=False,
                    kernel_initializer=initializer2))
    return model


def wall_collision(x, y):
    if 0 <= x < grid_count and 0 <= y < grid_count:
        return False
    return True


class NNet:

    def __init__(self, net=None):
        self.food_eaten = False
        self.snake_body = []
        self.all_lines = []
        self.current_direction = self.direction = random.randint(0, 3)
        self.create_elements()
        self.food = self.create_food()
        self.dead = False
        self.score = 0
        self.fitness = 0
        self.step_counter = 0
        self.inputs = np.zeros(input_size)
        if net is None:
            self.net = create_model()
        else:
            self.net = create_model()
            self.net.set_weights(net.get_weights())

    def update_fitness(self):
        head_x = self.snake_body[0].get_x()
        head_y = self.snake_body[0].get_y()
        food_x = self.food.get_x()
        food_y = self.food.get_y()
        if self.current_direction == 0 and head_x > food_x:
            self.fitness += 1
        elif self.current_direction == 1 and head_y > food_y:
            self.fitness += 1
        elif self.current_direction == 2 and food_x > head_x:
            self.fitness += 1
        elif self.current_direction == 3 and food_y > head_y:
            self.fitness += 1
        else:
            self.fitness -= 1

    def update(self):
        if not self.dead:
            self.step_counter += 1
            if self.step_counter % 10 == 0:
                self.fitness += 2

            self.update_body_pos()
            self.update_head_pos()
            # self.update_fitness()
            self.update_snake_if_food_eaten()
            self.dead_check()
            # self.get_inputs()
            # self.print_inputs()

    def update_snake_if_food_eaten(self):
        if self.food_collision(self.snake_body[0].get_x(), self.snake_body[0].get_y()):
            self.add_elements()
            self.food_eaten = True
            self.step_counter = 0
            self.score += apple_boost
            self.fitness += 100 * apple_boost

    def update_body_pos(self):
        for i in range(len(self.snake_body) - 1, 0, - 1):
            self.snake_body[i].set_x(self.snake_body[i - 1].get_x())
            self.snake_body[i].set_y(self.snake_body[i - 1].get_y())

    def get_line(self, x):
        pass

    def get_inputs_around_head(self, index):
        temp = 0
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    temp = 1
                    continue
                else:
                    x = self.snake_body[0].get_x() + i - 1
                    y = self.snake_body[0].get_y() + j - 1
                    if self.food_collision(x, y):
                        self.inputs[i * 3 + j - temp + index] = 1
                    elif wall_collision(x, y) or self.body_collision_excluding_head(x, y):
                        self.inputs[i * 3 + j - temp + index] = -1
                    else:
                        self.inputs[i * 3 + j - temp + index] = 0

    def get_net(self):
        return self.net

    def get_direction_input(self, index):
        for i in range(4):
            if self.current_direction == i:
                self.inputs[index + i] = 1
            else:
                self.inputs[index + i] = -1

    def food_collision(self, x, y):
        if self.food.get_x() == x and self.food.get_y() == y:
            return True
        return False

    def body_collision_excluding_head(self, x, y):
        for i in range(1, len(self.snake_body)):
            if self.snake_body[i].get_x() == x and self.snake_body[i].get_y() == y:
                return True
        return False

    def add_to_inputs_consistent(self, food, distance, index):
        self.inputs[index] = food
        self.inputs[index + 1] = distance

    def look_in_direction(self, xx, yy, index):
        x = self.snake_body[0].get_x() + xx
        y = self.snake_body[0].get_y() + yy
        distance = 1
        # distinguish between body == -1, food == 1 and wall == 0
        while not wall_collision(x, y):
            if self.food_collision(x, y):
                self.add_to_inputs_consistent(1, (distance / grid_count), index)
                return distance, 1
            if self.body_collision_excluding_head(x, y):
                self.add_to_inputs_consistent(-1, (distance / grid_count), index)
                return distance, -1
            distance += 1
            x += xx
            y += yy
        self.add_to_inputs_consistent(0, (distance / grid_count), index)
        return distance, 0

    def surroundings_to_inputs(self, index, draw=False):
        temp = 0
        res = []
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    temp = 1
                    continue
                distance, thing_found = self.look_in_direction(i - 1, j - 1, (i * 3 + j - temp) * 2 + index)
                if draw:
                    res.append((i, j, distance, thing_found))
        return res

    def draw_all_lines(self, c1):
        if not self.snake_body: return
        if not self.all_lines:
            for i in range(8):
                self.all_lines.append(Blob(0, 0, False))
        for i, data in enumerate(self.surroundings_to_inputs(0, draw=True)):  # index does not matter, only need data
            self.all_lines[i].del_obj(c1)
            self.all_lines[i].show_line(c1, self.snake_body[0], data)

    def print_inputs_sub(self, val, arr):
        print(val, "Food:", arr[0], "Distance:", arr[1])

    def print_inputs(self):
        print("Direction:", direction_dict.get(int(np.argmax(self.inputs[:4], axis=-1))))
        vals = {0: "left-top", 1: "left", 2: "left-bottom", 3: "top", 4: "bottom",
                5: "right-top", 6: "right", 7: "right-bottom"}
        for i in range(8):
            self.print_inputs_sub("Vision, " + vals[i] + ":", self.inputs[4 + i*2: 7 + i*2])
        print("------------------------------------------------------------")

    def get_inputs(self):
        self.get_direction_input(0)
        self.surroundings_to_inputs(4, draw=False)  # 16 inputs

    def think(self):
        self.get_inputs()
        temp = np.asarray(self.inputs).reshape(-1, input_size)
        g = np.argmax(self.net(temp), axis=-1)[0]
        self.direction = g

    def mutate(self, rate, mutate_at_all, random_mutate=False):
        if random.uniform(0, 1) <= mutate_at_all:
            if random_mutate: rate *= random.random()     # rate becomes the max
            for j, layer in enumerate(self.net.layers):
                new_weights_for_layer = []
                for weight_array in layer.get_weights():
                    weight_mask = np.random.rand(*weight_array.shape) < rate    # array of true and false
                    mutation_update = np.random.randn(*weight_array.shape) / 5    # * "opens" the tuple
                    mutation_update *= weight_mask
                    weight_array += mutation_update     # add mask to weights
                    new_weights_for_layer.append(weight_array)

                self.net.layers[j].set_weights(new_weights_for_layer)

    def create_elements(self):
        global start_size
        if start_size == 0:
            start_size = 1
        # will spawn in a straight line opposite of direction
        x = 1 if self.current_direction == 0 else -1 if self.current_direction == 2 else 0
        y = 1 if self.current_direction == 1 else -1 if self.current_direction == 3 else 0
        for i in range(start_size):
            self.create_element(grid_count // 2 + x * i, grid_count // 2 + y * i)

    def create_element(self, posx, posy):
        blob1 = Blob(posx, posy, False)
        self.snake_body.append(blob1)

    def add_elements(self):
        if len(self.snake_body) != 0:
            temp = self.snake_body[len(self.snake_body) - 1]
            for i in range(apple_boost):
                self.create_element(temp.get_x(), temp.get_y())

    def dead_check(self):
        x = self.snake_body[0].get_x()
        y = self.snake_body[0].get_y()
        if wall_collision(x, y) or self.body_collision_excluding_head(x, y)\
                or self.step_counter > len(self.snake_body) + grid_count * 4:
            self.dead = True
        if self.dead:
            if self.fitness < 0:
                self.fitness = 0

    def update_head_pos(self):
        # cannot be opposite direction, punish if prediction invalid
        if (self.direction + 2) % 4 != self.current_direction:
            self.current_direction = self.direction
        else:
            # maybe instead just kill?
            self.fitness -= 2
        # move head depending on direction
        if self.current_direction % 2 == 0:
            self.snake_body[0].set_x(self.snake_body[0].get_x() + self.current_direction - 1)
        else:
            self.snake_body[0].set_y(self.snake_body[0].get_y() + self.current_direction - 2)

    def create_food(self):
        if len(self.snake_body) != 0:
            x = get_random_grid()
            y = get_random_grid()
            on_snake = True
            while on_snake:
                on_snake = False
                for i in self.snake_body:
                    if i.get_x() == x and i.get_y() == y:
                        x = get_random_grid()
                        y = get_random_grid()
                        on_snake = True
            return Blob(x, y, True)

    def get_score(self):
        return self.score

    def set_score(self, s):
        self.score = s

    def set_fitness(self, f):
        self.fitness = f

    def get_fitness(self):
        return self.fitness

    def get_dead(self):
        return self.dead

    def show_all_elements(self, c1, draw_lines=False):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.show_blob(c1)
        self.food.show_blob(c1)
        if draw_lines: self.draw_all_lines(c1)

    def move_all_elements(self, c1, draw_lines=False):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.move_blob(c1)
        if draw_lines: self.draw_all_lines(c1)
        elif len(self.all_lines) != 0:
            for i in self.all_lines:
                i.del_obj(c1)

    def delete_all_elements(self, c1):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.del_obj(c1)
        self.food.del_obj(c1)
        if len(self.all_lines) != 0:
            for i in self.all_lines:
                i.del_obj(c1)


def get_random_grid():
    return random.randint(0, grid_count - 1)


class Blob:
    def __init__(self, x, y, apple):
        self.x = x
        self.y = y
        self.apple = apple
        self.col = snake_Color
        if apple:
            self.col = apple_color
        self.obj = None
        self.spacing = (grid_size - blob_size) // 2

    def del_obj(self, c1):
        if not self.obj: return
        c1.delete(self.obj)

    def move_blob(self, c1):
        c1.coords(self.obj, self.x * grid_size + self.spacing, self.y * grid_size + self.spacing,
                  self.x * grid_size + blob_size + self.spacing, self.y * grid_size + blob_size + self.spacing)

    def show_blob(self, c1):
        self.obj = c1.create_rectangle(self.x * grid_size + self.spacing, self.y * grid_size + self.spacing,
                                       self.x * grid_size + blob_size + self.spacing,
                                       self.y * grid_size + blob_size + self.spacing,
                                       outline=self.col, fill=self.col)

    def show_line(self, c1, head_blob, data):
        x, y = head_blob.get_x(), head_blob.get_y()
        i, j, distance, thing_found = data
        # assign correct line color based on what was found
        curr_color = food_found_color if thing_found == 1 else line_color if thing_found == 0 else apple_color
        # diagonal
        if i % 2 == 0 and j % 2 == 0:
            self.obj = c1.create_line((x + max(0, i-1)) * grid_size, (y + max(0, j-1)) * grid_size,
                                      (x + max(0, i-1) + max(0, distance-1) * (i-1)) * grid_size,
                                      (y + max(0, j-1) + max(0, distance-1) * (j-1)) * grid_size,
                                      fill=curr_color, dash=(1, 1))
        else:
            self.obj = c1.create_line((x * grid_size + int(i * (grid_size/2))),
                                      (y * grid_size + int(j * (grid_size/2))),
                                      (x + max(0, i - 1) + max(0, distance - 1) * (i - 1))
                                      * grid_size + int((i % 2) * (grid_size / 2)),
                                      (y + max(0, j - 1) + max(0, distance - 1) * (j - 1))
                                      * grid_size + int((j % 2) * (grid_size / 2)),
                                      fill=curr_color, dash=(1, 1))

    def x_plus_one(self):
        self.x = self.x + 1

    def x_minus_one(self):
        self.x = self.x - 1

    def y_plus_one(self):
        self.y = self.y + 1

    def y_minus_one(self):
        self.y = self.y - 1

    def get_col(self):
        return self.col

    def set_col(self, col):
        self.col = col

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_apple(self):
        return self.apple

    def set_apple(self, apple):
        self.apple = apple
