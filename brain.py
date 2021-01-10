import random
import keras.models
from keras.layers import Dense
import numpy as np
import keras.backend as K

back_color = "#171717"
back_color_2 = "#3c3c3c"
snake_Color = "#4ca3dd"
apple_color = "#ff4040"
grid_size = 30
blob_size = grid_size - 2  # should to be even
grid_count = 30
start_size = 5
apple_boost = 1
start_direction = 0
input_size = 24


def create_model():
    model = keras.models.Sequential()
    initializer = keras.initializers.RandomNormal(mean=0., stddev=1.)
    model.add(Dense(16, activation="relu", input_dim=input_size, use_bias=True,
                    kernel_initializer=initializer, bias_initializer=initializer))
    model.add(Dense(16, activation="relu", use_bias=True,
                    kernel_initializer=initializer, bias_initializer=initializer))
    model.add(Dense(4, activation="softmax", use_bias=False,
                    kernel_initializer=initializer))
    return model


def wall_collision(x, y):
    if 0 <= x < grid_count and 0 <= y < grid_count:
        return False
    return True


class NNet:

    def __init__(self, net=None):
        self.food_eaten = False
        self.snake_body = []
        self.create_elements()
        self.food = self.create_food()
        self.dead = False
        self.direction = start_direction
        self.current_direction = start_direction
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
                self.fitness += 1

            for i in range(len(self.snake_body) - 1, 0, - 1):
                self.snake_body[i].set_x(self.snake_body[i - 1].get_x())
                self.snake_body[i].set_y(self.snake_body[i - 1].get_y())

            self.update_head_pos()
            self.update_fitness()
            self.dead_check()

            if self.food_collision(self.snake_body[0].get_x(), self.snake_body[0].get_y()):
                self.add_elements()
                self.food_eaten = True
                self.step_counter = 0
                self.score += apple_boost
                self.fitness += 100 * apple_boost

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
                self.inputs[index + i] = 0

    def food_collision(self, x, y):
        if self.food.get_x() == x and self.food.get_y() == y:
            return True
        return False

    def body_collision_excluding_head(self, x, y):
        for i in range(1, len(self.snake_body)):
            if self.snake_body[i].get_x() == x and self.snake_body[i].get_y() == y:
                return True
        return False

    def add_to_inputs_consistent(self, food, body, distance, index):
        self.inputs[index] = food
        self.inputs[index + 1] = body
        self.inputs[index + 2] = distance

    def look_in_direction(self, xx, yy, index):
        x = self.snake_body[0].get_x() + xx
        y = self.snake_body[0].get_y() + yy
        distance = 1
        while not wall_collision(x, y):
            if self.food_collision(x, y):
                self.add_to_inputs_consistent(1, 0, 1 / distance, index)
                return
            if self.body_collision_excluding_head(x, y):
                self.add_to_inputs_consistent(0, 1, 1 / distance, index)
                return
            distance += 1
            x += xx
            y += yy
        self.add_to_inputs_consistent(0, 0, 1 / distance, index)

    def surroundings_to_inputs(self, index):
        temp = 0
        for i in range(3):
            for j in range(3):
                if i == j == 1:
                    temp = 1
                    continue
                self.look_in_direction(i - 1, j - 1, (i * 3 + j - temp) * 3 + index)

    def get_inputs(self):
        self.surroundings_to_inputs(0)  # 24 inputs

    def think(self):
        self.get_inputs()
        temp = np.asarray(self.inputs).reshape(-1, input_size)
        g = np.argmax(self.net(temp), axis=-1)[0]
        self.direction = g

    def mutate(self, rate, mutate_at_all):
        if random.uniform(0, 1) <= mutate_at_all:
            for j, layer in enumerate(self.net.layers):
                new_weights_for_layer = []
                for weight_array in layer.get_weights():
                    save_shape = weight_array.shape
                    one_dim_weight = weight_array.reshape(-1)

                    for i, weight in enumerate(one_dim_weight):
                        if random.uniform(0, 1) <= rate:
                            one_dim_weight[i] += random.gauss(0, 1) / 5
                            if one_dim_weight[i] < -1:
                                one_dim_weight[i] = -1
                            elif one_dim_weight[i] > 1:
                                one_dim_weight[i] = 1

                    new_weight_array = one_dim_weight.reshape(save_shape)
                    new_weights_for_layer.append(new_weight_array)

                self.net.layers[j].set_weights(new_weights_for_layer)

    def create_elements(self):
        global start_size
        if start_size == 0:
            start_size = 1
        for i in range(start_size):
            self.create_element(grid_count // 2 + i, grid_count // 2)

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
        if wall_collision(x, y):
            self.dead = True
        elif self.body_collision_excluding_head(x, y):
            self.dead = True
        elif self.step_counter > grid_count * grid_count:
            self.dead = True
        if self.dead:
            if self.score == 0:
                self.fitness = self.fitness // 2
            if self.fitness < 0:
                self.fitness = 0

    def update_head_pos(self):
        if self.direction == 0 and self.current_direction != 2:
            self.current_direction = 0
        elif self.direction == 1 and self.current_direction != 3:
            self.current_direction = 1
        elif self.direction == 2 and self.current_direction != 0:
            self.current_direction = 2
        elif self.direction == 3 and self.current_direction != 1:
            self.current_direction = 3

        if self.current_direction == 0:
            self.snake_body[0].x_minus_one()
        elif self.current_direction == 1:
            self.snake_body[0].y_minus_one()
        elif self.current_direction == 2:
            self.snake_body[0].x_plus_one()
        elif self.current_direction == 3:
            self.snake_body[0].y_plus_one()

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

    def show_all_elements(self, c1):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.show_blob(c1)
            self.food.show_blob(c1)

    def move_all_elements(self, c1):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.move_blob(c1)

    def delete_all_elements(self, c1):  # graphics
        if len(self.snake_body) != 0:
            for i in self.snake_body:
                i.del_obj(c1)
        self.food.del_obj(c1)


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
        c1.delete(self.obj)

    def move_blob(self, c1):
        c1.coords(self.obj, self.x * grid_size + self.spacing, self.y * grid_size + self.spacing,
                  self.x * grid_size + blob_size + self.spacing, self.y * grid_size + blob_size + self.spacing)

    def show_blob(self, c1):
        self.obj = c1.create_rectangle(self.x * grid_size + self.spacing, self.y * grid_size + self.spacing,
                                       self.x * grid_size + blob_size + self.spacing,
                                       self.y * grid_size + blob_size + self.spacing,
                                       outline=self.col, fill=self.col)

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
