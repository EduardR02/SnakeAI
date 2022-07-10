class Point:

    def __init__(self, x, y):
        assert isinstance(x, int) and isinstance(y, int)
        self.x = x
        self.y = y

    # because no type check would be problematic with inheritance, but this is not needed here
    def __eq__(self, other):
        return hasattr(other, "x") and hasattr(other, "y") and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __mul__(self, other):
        if not isinstance(other, Point):
            return Point(int(self.x * other), int(self.y * other))
        else:
            return Point(self.x * other.x, self.y * other.y)

    def __floordiv__(self, other):
        if isinstance(other, Point):
            return Point(int(self.x / other.x), int(self.y / other.y))
        else:
            return Point(int(self.x / other), int(self.y / other))

    # x, y have to remain int at all times, therefore both div types result in floor div
    def __truediv__(self, other):
        return self.__floordiv__(other)

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        else:
            raise IndexError("key out of range")

    def __str__(self):
        return f"{self.x} {self.y}"

    def sum_xy(self):
        return self.x + self.y
