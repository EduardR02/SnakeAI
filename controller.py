import config
from pynput import keyboard


class KeyController:

    def __init__(self):
        self.listener = None

    def key_control(self, key):
        try:
            x = key.char
        except AttributeError:
            x = key

        if x == "-":
            config.key_toggle = not config.key_toggle
            if config.key_toggle:
                print("unlocked")
            elif not config.key_toggle:
                print("locked")

        elif config.key_toggle:
            if x == "d":
                config.pause_between_frames_ms = max(config.pause_between_frames_ms - 10, 0)
            elif x == "a":
                config.pause_between_frames_ms += 10
            elif x == "q":
                config.pause_between_frames_ms += 1
            elif x == "e":
                config.pause_between_frames_ms = max(config.pause_between_frames_ms - 1, 0)
            elif x == "y":
                config.updates_per_draw = max(config.updates_per_draw - 1, 1)
            elif x == "c":
                config.updates_per_draw += 1
            elif x == "l":
                config.load_m = True
            elif x == "s":
                config.save_m = True
            elif x == "w":
                config.draw_lines = not config.draw_lines

    def start(self):
        self.listener = keyboard.Listener(on_press=self.key_control)
        self.listener.start()
