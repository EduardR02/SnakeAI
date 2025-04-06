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
food_gain_times = 5
window_size = grid_size * grid_count
label_font = "Courier 15 bold"
pause_between_frames_ms = 100
updates_per_draw = 1
population_size = 500
mutation_rate = 0.01   # seemingly the lower the better, and at the end when it's really good anneal to zero with decay. 1% is *high*
mutation_rate_decay = 0.99
minimum_mutation_rate = 0.0
key_toggle = True
model_save_dir = "models/" # folder from which all models load and save
model_load_dir = "models/"
graph_file_name = "graph.png"
replay_dir = "replays/"
sub_replay_dir = "replay10x10/"
load_m = False
save_m = False
no_graphics = True
draw_lines = False
play_replay = False
auto_save = False
