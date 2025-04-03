import config
import file_manager
import numpy as np
from point import Point
import snake
import brain


class Replay:

    def __init__(self, ga):
        self.ga = ga
        self.replay_dir = config.replay_dir + config.sub_replay_dir
        self.file_manager = file_manager.ReplayFileManager(self.replay_dir)
        self.current_snake = None
        self.current_moves = []
        self.current_food = []
        self.general_stats = None

    def update(self):
        raise NotImplementedError()


class ReplayCollector(Replay):

    def __init__(self, ga):
        super().__init__(ga)
        self.best_snake = None

    def update(self):
        # there is no update if snakes don't match because genetic_alg has a setup cycle.
        # If this were to add moves, the first move would be the "default" move, the model has not "thought"
        # at that point yet
        # setup cycle is checked with if the snake has moves yet or not
        if self.current_snake != self.ga.current_snake:
            self.current_snake = self.ga.current_snake
            self.general_stats = [self.current_snake.start_direction,
                                  config.food_gain_times, self.current_snake.snake_start_size]
            if self.current_snake.total_moves == 0:
                return
        self.current_food.append(self.ga.current_snakes_food_pos)
        self.current_moves.append(self.current_snake.current_direction)
        if self.current_snake.is_dead:
            self.update_best()

    def update_best(self):
        if not self.best_snake or self.current_snake.compare_performance(self.best_snake) > 0:
            self.best_snake = self.current_snake
            self.save_replay()
        self.current_food = []
        self.current_moves = []
        self.current_snake = None
        self.general_stats = None

    def save_replay(self):
        food_pos = np.asarray([[pos.x, pos.y] for pos in self.current_food])
        general_stats = np.asarray(self.general_stats)
        self.file_manager.save_replay(food_pos, "food_pos")
        self.file_manager.save_replay(self.current_moves, "moves_pos")
        self.file_manager.save_replay(general_stats, "general_stats")


class ReplayPlayer(Replay):

    def __init__(self, ga):
        super().__init__(ga)
        self.curr_index = 0
        self.current_moves = self.file_manager.load_replay("moves_pos")
        self.current_food = self.file_manager.load_replay("food_pos")
        self.general_stats = self.file_manager.load_replay("general_stats")
        config.updates_per_draw = 1
        config.pause_between_frames_ms = 100
        config.food_gain_times = self.general_stats[1]
        self.start_size = self.general_stats[2]
        self.start_direction = self.general_stats[0]
        self.ga.food_gain_times = config.food_gain_times
        snake.Snake.snake_start_size = self.start_size
        # reinit with new params
        self.ga.reset_gen_alg_params()
        self.reset()

    def reset(self):
        self.curr_index = 0
        # don't need the brain here, just because constructor requires
        self.current_snake = snake.Snake(brain.Brain())
        self.current_snake.start_direction = self.start_direction
        self.current_snake.current_direction = self.start_direction
        self.current_snake.fill_body()
        self.ga.current_snake = self.current_snake

    def update(self):
        if self.curr_index >= len(self.current_moves):
            self.reset()
        else:
            food_pos = Point(int(self.current_food[self.curr_index][0]), int(self.current_food[self.curr_index][1]))
            self.ga.current_snakes_food_pos = food_pos
            self.current_snake.current_direction = int(self.current_moves[self.curr_index])
            self.current_snake.move_snake()
            self.update_score()
            self.curr_index += 1

    def update_score(self):
        # update score for gui and to add length
        if self.curr_index > 0:
            prev_food = Point(int(self.current_food[self.curr_index - 1][0]),
                              int(self.current_food[self.curr_index - 1][1]))
            if self.current_snake.get_head_position() == prev_food:
                mult = 5 if self.current_snake.score >= 0 else 1
                self.current_snake.score += config.food_gain_times * mult
                self.current_snake.add_length_to_snake(config.food_gain_times * mult)
