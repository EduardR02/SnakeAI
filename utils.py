from point import Point
import config
import random


def is_wall_collision(point):
    return not (0 <= point.x < config.grid_count.x and 0 <= point.y < config.grid_count.y)


def generate_random_point():
    return Point(random.randint(0, config.grid_count.x - 1), random.randint(0, config.grid_count.y - 1))

def get_point(x, y):
    return Point(int(x), int(y))
