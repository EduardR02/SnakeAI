import random
import keras.backend as K
import keras.models
from keras.layers import Dense
import numpy as np
import time

back_color = "#171717"
back_color_2 = "#3c3c3c"
snake_Color = "#4ca3dd"
apple_color = "#ff4040"
grid_size = 50
blob_size = grid_size - 2
grid_count = 10
start_size = 5
apple_boost = 1
start_direction = 0


def create_model():
    model = keras.models.Sequential()
    model.add(Dense(20, activation="relu", input_dim = 13))
    model.add(Dense(15, activation="relu"))
    model.add(Dense(4, activation="softmax"))
    return model


class NNet:

    def __init__(self, net = None):
        self.food_eaten = False
        self.my_list = []
        self.create_elements()
        self.food = self.create_food()
        self.dead = False
        self.direction = start_direction
        self.current_direction = start_direction
        self.score = 0
        self.fitness = 0
        self.step_counter = 0
        self.inputs = []
        if net is None:
            self.net = create_model()
        else:
            self.net = create_model()
            self.net.set_weights(net.get_weights())
            """self.net = keras.models.clone_model(net)
            self.net.build((None, 13))  # replace 10 with number of variables in input layer
            self.net.compile(optimizer='adam', loss='categorical_crossentropy')
            self.net.set_weights(net.get_weights())"""

    def update_fitness(self):
        head_x = self.my_list[0].get_x()
        head_y = self.my_list[0].get_y()
        x = head_x - self.my_list[1].get_x()
        y = head_y - self.my_list[1].get_y()
        food_x = self.food.get_x()
        food_y = self.food.get_y()
        if x != 0:
            if x > 0:
                if head_x <= food_x:
                    self.fitness += 5
            else:
                if head_x >= food_x:
                    self.fitness += 5
        else:
            if y > 0:
                if head_y <= food_y:
                    self.fitness += 5
            else:
                if head_y >= food_y:
                    self.fitness += 5

    def update(self):
        if not self.dead:
            self.step_counter += 1
            if self.step_counter % 10 == 0:
                self.fitness += 1

            for i in range(len(self.my_list) - 1, 0, - 1):
                self.my_list[i].set_x(self.my_list[i - 1].get_x())
                self.my_list[i].set_y(self.my_list[i - 1].get_y())

            self.update_head_pos()
            self.update_fitness()
            self.dead_check()

            if self.food.get_x() == self.my_list[0].get_x() and self.food.get_y() == self.my_list[0].get_y():
                self.add_elements()
                self.food_eaten = True
                self.step_counter = 0
                self.score += apple_boost
                self.fitness += 100 * apple_boost

    def get_line(self, x):
        pass

    def get_inputs_around_head(self):
        body_x = []
        body_y = []
        for i in range(len(self.my_list) - 1):
            body_x.append(self.my_list[i + 1].get_x())
            body_y.append(self.my_list[i + 1].get_y())
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                else:
                    x = self.my_list[0].get_x() + i - 1
                    y = self.my_list[0].get_y() + j - 1
                    if x == self.food.get_x() and y == self.food.get_y():
                        self.inputs.append(1)
                    elif x < 0 or x > grid_count - 1 or y < 0 or y > grid_count - 1:
                        self.inputs.append(-1)
                    else:
                        f = False
                        for k in range(len(body_x)):
                            if x == body_x[k] and y == body_y[k]:
                                self.inputs.append(-1)
                                f = True
                                break
                        if not f:
                            self.inputs.append(0)

    def get_net(self):
        return self.net

    def get_inputs(self):
        self.inputs.clear()
        self.inputs.append((self.food.get_x() - self.my_list[0].get_x()) / (grid_count - 1))
        self.inputs.append((self.food.get_y() - self.my_list[0].get_y()) / (grid_count - 1))
        self.inputs.append(self.my_list[0].get_x() / (grid_count - 1))
        self.inputs.append(self.my_list[0].get_y() / (grid_count - 1))
        self.inputs.append(self.current_direction / 3)
        self.get_inputs_around_head()

    def think(self):
        self.get_inputs()
        temp = np.asarray(self.inputs).reshape(-1, 13)
        g = np.argmax(self.net(temp, training = False), axis=-1)
        self.direction = g

    def mutate(self, rate):
        # first itterate through the layers
        for j, layer in enumerate(self.net.layers):
            new_weights_for_layer = []
            # each layer has 2 matrizes, one for connection weights and one for biases
            # then itterate though each matrix

            for weight_array in layer.get_weights():
                # save their shape
                save_shape = weight_array.shape
                # reshape them to one dimension
                one_dim_weight = weight_array.reshape(-1)

                for i, weight in enumerate(one_dim_weight):
                    # mutate them like i want
                    if random.uniform(0, 1) <= rate:
                        # maybe dont use a complete new weigh, but rather just change it a bit
                        one_dim_weight[i] += random.uniform(-0.3, 0.3)
                        if one_dim_weight[i] < -1:
                            one_dim_weight[i] = -1
                        elif one_dim_weight[i] > 1:
                            one_dim_weight[i] = 1

                # reshape them back to the original form
                new_weight_array = one_dim_weight.reshape(save_shape)
                # save them to the weight list for the layer
                new_weights_for_layer.append(new_weight_array)

            # set the new weight list for each layer
            self.net.layers[j].set_weights(new_weights_for_layer)

    def create_elements(self):
        global start_size
        if start_size == 0:
            start_size = 1
        for i in range(start_size):
            self.create_element(grid_count / 2 + i, grid_count / 2)

    def create_element(self, posx, posy):
        blob1 = Blob(posx, posy, False)
        self.my_list.append(blob1)

    def add_elements(self):
        if len(self.my_list) != 0:
            temp = self.my_list[len(self.my_list) - 1]
            for i in range(apple_boost):
                self.create_element(temp.get_x(), temp.get_y())

    def dead_check(self):
        x = self.my_list[0].get_x()
        y = self.my_list[0].get_y()
        if x < 0 or x > grid_count - 1 or y < 0 or y > grid_count - 1:
            self.dead = True
        else:
            for i in range(len(self.my_list) - 1):
                if self.my_list[0].get_x() == self.my_list[i + 1].get_x() and \
                        self.my_list[0].get_y() == self.my_list[i + 1].get_y():
                    self.dead = True
        if self.step_counter > grid_count * grid_count * 2:
            self.dead = True
            self.fitness = 0
        if self.dead:
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
            self.my_list[0].x_minus_one()
        elif self.current_direction == 1:
            self.my_list[0].y_minus_one()
        elif self.current_direction == 2:
            self.my_list[0].x_plus_one()
        elif self.current_direction == 3:
            self.my_list[0].y_plus_one()

    def create_food(self):
        if len(self.my_list) != 0:
            x = get_random_grid()
            y = get_random_grid()
            on_snake = True
            while on_snake:
                on_snake = False
                for i in self.my_list:
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
        if len(self.my_list) != 0:
            for i in self.my_list:
                i.show_blob(c1)
            self.food.show_blob(c1)

    def move_all_elements(self, c1):  # graphics
        if len(self.my_list) != 0:
            for i in self.my_list:
                i.move_blob(c1)

    def delete_all_elements(self, c1):  # graphics
        if len(self.my_list) != 0:
            for i in self.my_list:
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

    def del_obj(self, c1):
        c1.delete(self.obj)

    def move_blob(self, c1):
        c1.coords(self.obj, self.x * grid_size + 1, self.y * grid_size + 1, self.x * grid_size + blob_size + 1,
                  self.y * grid_size + blob_size + 1)

    def show_blob(self, c1):
        self.obj = c1.create_rectangle(self.x * grid_size + 1, self.y * grid_size + 1,
                                       self.x * grid_size + blob_size + 1, self.y * grid_size + blob_size + 1,
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
