import config
import tkinter as tk
import rectangle
import line
from point import Point

class GUI:

    def __init__(self, ga, title="Snake AI"):
        self.label_height = 52      # can be determined with print(self.label.winfo_height()) (CALL AFTER UPDATE)
        self.ga = ga
        self.root = tk.Tk()
        self.canvas = None
        self.label = None
        # label has to be inited first
        self.init_label()
        self.center()
        self.init_canvas()
        self.root.title(title)
        self.root.resizable(False, False)
        self.init_rect_and_line()
        self.snake_rects = []
        self.lines = []
        self.food = None
        self.create_snake_and_food()

    def update(self):
        self.draw_all()
        self.root.update_idletasks()
        self.root.update()

    def draw_all(self):
        self.update_label()
        self.remove_snake_and_food()
        self.create_snake_and_food()
        if config.draw_lines:
            self.remove_all_lines()
            self.draw_all_lines()

    def create_snake_and_food(self):
        self.food = rectangle.Rect(self.ga.current_snakes_food_pos, True)
        for pos in self.ga.current_snake.body:
            self.snake_rects.append(rectangle.Rect(pos, False))

    def remove_snake_and_food(self):
        for rect in self.snake_rects:
            rect.del_obj()
        self.snake_rects = []
        self.food.del_obj()
        self.food = None

    def draw_all_lines(self):
        head = self.snake_rects[0].grid_pos
        inputs = self.ga.current_snake.brain.inputs
        if not inputs or not self.ga.current_snake or self.ga.current_snake.is_dead:
            return
        dxdy = [Point(x, y) for x in range(3) for y in range(3) if not (x == y == 1)]
        rotated_dxdy = self.ga.current_snake.brain.align_to_direction(-1, dxdy)
        for i, dx_dy in enumerate(rotated_dxdy):
            self.lines.append(line.Line(head, config.line_color))
            idx = i * 3
            closest = max(inputs[idx], inputs[idx+1], inputs[idx+2])
            self.lines[-1].create_line((dx_dy.x, dx_dy.y, closest, 0 if closest == inputs[idx] else 1))

    def remove_all_lines(self):
        for line in self.lines:
            line.del_obj()
        self.lines = []

    def center(self):  # graphics
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (config.window_size.x // 2)
        y = (self.root.winfo_screenheight() // 2) - (config.window_size.y // 2)
        self.root.geometry('{}x{}+{}+{}'.format(config.window_size.x, config.window_size.y + self.label_height, x, y))

    def init_canvas(self):
        self.canvas = tk.Canvas(self.root, height=config.window_size.y, width=config.window_size.x,
                                bg=config.back_color, highlightthickness=0)
        self.canvas.pack()

    def init_label(self):
        self.label = tk.Label(self.root, text=self.label_text(),
                              fg=config.food_color, bg=config.back_color_2, font=config.label_font)
        self.label.pack(fill=tk.BOTH)

    def init_rect_and_line(self):
        rectangle.Rect.canvas = self.canvas
        line.Line.canvas = self.canvas

    def update_label(self):
        self.label.config(text=self.label_text())

    def label_text(self):
        return f"Score: {self.ga.current_snake.score}, Agent: {self.ga.current_snake_idx}," \
               f" FramePause: {config.pause_between_frames_ms},\n Generation: {self.ga.generation}," \
               f" DrawPerUpdate: {config.updates_per_draw} "
