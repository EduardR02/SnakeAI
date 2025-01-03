from point import Point


back_color = "#171717"
back_color_2 = "#3c3c3c"
snake_color = "#4ca3dd"
food_color = "#ff4040"
line_color = "#2f4454"
food_found_color = "#5cdb95"
grid_size = Point(50, 50)
border_width = Point(1, 1)
grid_count = Point(10, 10)
food_gain_times = 1
window_size = grid_size * grid_count
label_font = "Courier 15 bold"
pause_between_frames_ms = 0
updates_per_draw = 10
population_size = 50
mutation_rate = 0.1   # 0.1 seems to work best, can be randomly set to 0 - mutation_rate with random_mutate
mutation_rate_decay = 1
minimum_mutation_rate = 0.03
key_toggle = True
models_file_path = "models1/"  # folder from which all models load and save
graph_file_name = "my_graph_6.png"
replay_dir = "replays/"
sub_replay_dir = "replay2/"
load_m = False
save_m = False
no_graphics = False
draw_lines = True
play_replay = False
auto_save = False
