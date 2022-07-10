import config
from point import Point
import utils
import random


class Snake:

    direction_dict = {"left": 0, "up": 1, "right": 2, "down": 3}
    reverse_direction_dict = {0: "left", 1: "up", 2: "right", 3: "down"}
    opposite_direction_dict = {"left": "right", "right": "left", "up": "down", "down": "up"}
    delta_dict = {"left": -1, "up": -1, "right": 1, "down": 1}
    snake_start_size = 5    # min 1
    starting_moves = 0      # is initialized by genetic algorithm

    def __init__(self, brain):
        self.brain = brain
        # important, the brain needs to know inputs and stuff
        self.brain.set_snake(self)
        self.fitness = 0
        self.score = 0
        self.total_moves = 0
        # body list for simple drawing, knowing where the head and tail is
        self.body = []
        # body set for fast collision lookup, the head shall not be added here
        self.body_set_without_head = set()
        self.is_dead = False
        self.current_direction = self.get_random_direction()
        self.moves_left = self.starting_moves
        self.fill_body()

    def update(self, food_position):
        new_direction = self.brain.think(food_position)
        self.update_direction(new_direction)
        self.move_snake()
        self.update_moves()
        # is dead check comes last
        self.calc_is_dead()

    def update_moves(self):
        self.moves_left -= 1
        self.total_moves += 1

    def update_direction(self, new_direction):
        # only change direction if it is not the opposite direction
        if self.current_direction == self.direction_dict[self.opposite_direction_dict[self.reverse_direction_dict[new_direction]]]:
            return
        self.current_direction = new_direction

    def mutate(self, rate):
        new_mutated_brain = self.brain.mutate(rate)
        return new_mutated_brain

    def crossover(self, other, mutation_rate):
        child_brain1, child_brain2 = self.brain.crossover(other.brain, mutation_rate)
        return child_brain1, child_brain2

    def move_snake(self):
        # insert head and remove tail
        delta_point = self.get_dx_dy()
        head_pos = self.get_head_position()
        new_head_pos = head_pos + delta_point
        self.insert_new_head(new_head_pos)
        self.remove_tail()

    def calc_is_dead(self):
        head_pos = self.get_head_position()
        if utils.is_wall_collision(head_pos) or self.is_head_with_body_collision() or self.moves_left <= 0:
            self.is_dead = True

    def add_length_to_snake(self, length_to_add):
        assert len(self.body) > 0
        # all elements will be added to the same position. As the snake moves, they start to follow one by one
        tail_pos = self.body[-1]
        for _ in range(length_to_add):
            self.add_point_to_body(tail_pos)

    def fill_body(self):
        assert self.snake_start_size >= 1
        self.body = []
        delta_point = self.get_dx_dy()
        # dx gives direction, but body has to spawn into the OPPOSITE direction
        delta_point = Point(0, 0) - delta_point
        for i in range(self.snake_start_size):
            new_point = (config.grid_count // 2) + delta_point * i
            self.add_point_to_body(new_point)

    def add_point_to_body(self, point):
        self.body.append(point)
        # don't add the head
        if len(self.body) > 1:
            self.body_set_without_head.add(point)

    def insert_new_head(self, new_position):
        # add to set because old head became part of the body
        if len(self.body) > 1:
            old_head_pos = self.get_head_position()
            self.body_set_without_head.add(old_head_pos)
        self.body.insert(0, new_position)

    def remove_tail(self):
        # default pop is last
        tail_pos = self.body.pop()
        # this means that multiple points are stacked on top of each other, for example after adding multiple points.
        # in that case, the point should not be removed from the body dict, because there are still points there
        if not (len(self.body) > 0 and tail_pos == self.body[-1]):
            # set.discard does not raise error as opposed to remove, in case element doesn't exist
            self.body_set_without_head.discard(tail_pos)

    def compare_performance(self, other):
        # return 1 if "better", 0 if "equal" and -1 if "worse"
        if self.score > other.score: return 1
        if self.score < other.score: return -1
        if self.fitness > other.fitness: return 1
        if self.fitness < other.fitness: return -1
        return 0

    def is_head_with_body_collision(self):
        return self.get_head_position() in self.body_set_without_head

    def is_point_with_body_collision(self, point):
        return point in self.body_set_without_head

    def is_point_with_snake_collision(self, point):
        return self.get_head_position() == point or self.is_point_with_body_collision(point)

    def get_random_direction(self):
        return random.choice(list(self.direction_dict.values()))

    def get_head_position(self):
        return self.body[0]

    def get_dx_dy(self):
        dx = dy = 0
        if self.current_direction == self.direction_dict["left"]:
            dx = self.delta_dict["left"]
        elif self.current_direction == self.direction_dict["right"]:
            dx = self.delta_dict["right"]
        elif self.current_direction == self.direction_dict["up"]:
            dy = self.delta_dict["up"]
        elif self.current_direction == self.direction_dict["down"]:
            dy = self.delta_dict["down"]
        return Point(dx, dy)

    def reset(self, new_brain=None):
        self.__init__(new_brain or self.brain)
