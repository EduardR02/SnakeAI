import config


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

    def create_line(self, data):
        x, y = self.grid_pos.x, self.grid_pos.y
        dx, dy, distance, thing_found = data
        # inv normalize
        distance = round(config.grid_size.x - (distance * (config.grid_size.x - 1)))
        grid_size = self.grid_size.x
        curr_color = config.food_found_color if thing_found == 0 else config.line_color
        # diagonal
        if dx % 2 == 0 and dy % 2 == 0:
            self.obj = self.canvas.create_line((x + max(0, dx-1)) * grid_size, (y + max(0, dy-1)) * grid_size,
                                      (x + max(0, dx-1) + max(0, distance-1) * (dx-1)) * grid_size,
                                      (y + max(0, dy-1) + max(0, distance-1) * (dy-1)) * grid_size,
                                      fill=curr_color, dash=(1, 1))
        else:
            self.obj = self.canvas.create_line((x * grid_size + int(dx * (grid_size/2))),
                                      (y * grid_size + int(dy * (grid_size/2))),
                                      (x + max(0, dx - 1) + max(0, distance - 1) * (dx - 1))
                                      * grid_size + int((dx % 2) * (grid_size / 2)),
                                      (y + max(0, dy - 1) + max(0, distance - 1) * (dy - 1))
                                      * grid_size + int((dy % 2) * (grid_size / 2)),
                                      fill=curr_color, dash=(1, 1))