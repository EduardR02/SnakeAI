import config
from point import Point


class Line:

    # will be set in GUI
    canvas = None
    # curr_color = food_found_color if thing_found == object_dict["food"] else line_color if thing_found == object_dict["wall"] else apple_color

    def __init__(self, position, color):
        self.grid_pos = position
        self.color = color
        self.obj = None
        self.grid_size = config.grid_size

    def del_obj(self):
        if not self.obj: return
        self.canvas.delete(self.obj)

    def create_line(self, head_pos, data):
        x, y = head_pos.x, head_pos.y
        dx, dy, distance, thing_found = data
        # diagonal
        if dx % 2 == 0 and dy % 2 == 0:
            self.obj = self.canvas.create_line((x + max(0, i-1)) * grid_size, (y + max(0, j-1)) * grid_size,
                                      (x + max(0, i-1) + max(0, distance-1) * (i-1)) * grid_size,
                                      (y + max(0, j-1) + max(0, distance-1) * (j-1)) * grid_size,
                                      fill=curr_color, dash=(1, 1))
        else:
            self.obj = self.canvas.create_line((x * grid_size + int(i * (grid_size/2))),
                                      (y * grid_size + int(j * (grid_size/2))),
                                      (x + max(0, i - 1) + max(0, distance - 1) * (i - 1))
                                      * grid_size + int((i % 2) * (grid_size / 2)),
                                      (y + max(0, j - 1) + max(0, distance - 1) * (j - 1))
                                      * grid_size + int((j % 2) * (grid_size / 2)),
                                      fill=curr_color, dash=(1, 1))