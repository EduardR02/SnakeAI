import config
from point import Point


class Rect:

    # will be set in GUI
    canvas = None

    def __init__(self, position, is_apple):
        self.grid_pos = position
        self.grid_size = config.grid_size
        self.is_apple = is_apple
        self.border_width = config.border_width
        self.color = config.snake_color
        if is_apple:
            self.color = config.food_color
        self.obj = None
        self.create_rect()

    def del_obj(self):
        if not self.obj: return
        self.canvas.delete(self.obj)

    def move_rect(self):
        if not self.obj: return
        self.canvas.coords(self.obj, *self.get_coords())

    def create_rect(self):
        self.obj = self.canvas.create_rectangle(*self.get_coords(), outline=self.color, fill=self.color)

    def get_coords(self):
        return (self.grid_pos.x * self.grid_size.x + self.border_width.x,
                self.grid_pos.y * self.grid_size.y + self.border_width.y,
                (self.grid_pos.x + 1) * self.grid_size.x - self.border_width.x,
                (self.grid_pos.y + 1) * self.grid_size.y - self.border_width.y)

    def move(self, new_coords):
        self.grid_pos = new_coords
        self.move_rect()
