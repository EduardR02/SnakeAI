import random
import tkinter as tk
from pynput import keyboard
import brain
import tensorflow as tf
import matplotlib.pyplot as plt

window_size = brain.grid_size * brain.grid_count
my_font = "Courier 15 bold"
ms_time = 1
draw_for_update = 10
population_size = 1000 * 2
counter = 0
mutation_rate = 0.1
mutation_rate2 = 1
counter_2 = 0
all_snakes = []
saved_all_snakes = []
gen = 1
best_1 = best_2 = best_of_gen_1 = best_of_gen_2 = None
all_fitness = 0
key_toggle = True
models_file_path = "models/"  # folder from which all models load and save
graph_file_name = "my_graph_2.png"
load_m = False
no_graphics = False
graph_best_gen = []


def key_control(key):
    global ms_time, draw_for_update, key_toggle, load_m
    try:
        x = key.char
    except AttributeError:
        x = key

    if x == "-":
        key_toggle = not key_toggle
        if key_toggle:
            print("unlocked")
        elif not key_toggle:
            print("locked")

    elif key_toggle:
        if x == "d":
            if ms_time > 0:
                ms_time -= 10
        elif x == "a":
            ms_time += 10
        elif x == "q":
            ms_time += 1
        elif x == "e":
            if ms_time > 0:
                ms_time -= 1
        elif x == "y":
            if draw_for_update > 1:
                draw_for_update -= 1
        elif x == "c":
            draw_for_update += 1
        elif x == "l":
            load_m = True
        elif x == "s":
            save_models()
        if ms_time < 0:
            ms_time = 0


def save_graph():
    plt.plot(graph_best_gen, color="#61AFEF")
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Snake Neural Network')
    ax = plt.gca()
    ax.set_facecolor('#2b2b2b')
    plt.savefig(graph_file_name, bbox_inches="tight")


def train(loaded=False):
    global counter, best_1, best_2
    if gen == 1 and not loaded:
        for i in range(population_size):
            all_snakes.append(brain.NNet())
    if best_1 is None:
        best_1 = best_2 = all_snakes[0]
    repeat_train()


def repeat_train():
    b = True
    while b:
        a = []
        for i in range(len(all_snakes)):
            if not all_snakes[i].get_dead():
                all_snakes[i].think()
                all_snakes[i].update()
            else:
                a.append(i)
        for i in range(len(all_snakes)):
            if all_snakes[i].food_eaten:
                all_snakes[i].food = all_snakes[i].create_food()
                all_snakes[i].food_eaten = False
        if len(a) != 0:
            for i in range(len(a)):
                saved_all_snakes.append(all_snakes[a[i] - i])
                all_snakes.pop(a[i] - i)
        if len(all_snakes) == 0:
            b = False
    prep_next_gen()


# noinspection PyUnresolvedReferences
def save_models():
    print("saving...")
    if len(all_snakes) == 0:
        for i in range(len(saved_all_snakes)):
            saved_all_snakes[i].net.save(f"{models_file_path}model_nr_{i}.h5")
    elif len(saved_all_snakes) == 0:
        for i in range(len(all_snakes)):
            all_snakes[i].net.save(f"{models_file_path}model_nr_{i}.h5")
    if best_1 is not None:
        best_1.net.save(f"{models_file_path}model_nr_best_1.h5")
        best_2.net.save(f"{models_file_path}model_nr_best_2.h5")
        best_of_gen_1.net.save(f"{models_file_path}model_nr_best_of_gen_1.h5")
        best_of_gen_2.net.save(f"{models_file_path}model_nr_best_of_gen_2.h5")
    print("models saved!")


def load_models():
    global best_1, best_2, best_of_gen_1, best_of_gen_2, gen, counter, counter_2, \
        all_fitness, load_m, ms_time, draw_for_update, graph_best_gen
    print("loading...")
    for i in range(population_size):
        try:
            all_snakes.append(brain.NNet(brain.keras.models.load_model
                                         (f"{models_file_path}model_nr_{i}.h5", compile=False)))
        except OSError:
            all_snakes.append(
                brain.NNet(brain.keras.models.load_model(f"{models_file_path}model_nr_best_1.h5", compile=False)))

    best_1 = brain.NNet(brain.keras.models.load_model(f"{models_file_path}model_nr_best_1.h5", compile=False))
    best_2 = brain.NNet(brain.keras.models.load_model(f"{models_file_path}model_nr_best_2.h5", compile=False))
    best_of_gen_1 = brain.NNet(brain.keras.models.load_model
                               (f"{models_file_path}model_nr_best_of_gen_1.h5", compile=False))
    best_of_gen_2 = brain.NNet(brain.keras.models.load_model
                               (f"{models_file_path}model_nr_best_of_gen_2.h5", compile=False))

    counter_2 = counter = all_fitness = 0
    gen = 1
    saved_all_snakes.clear()
    load_m = False
    ms_time = 100
    draw_for_update = 1
    graph_best_gen.clear()
    print("models loaded")


def load_level(c1, r1, l1):
    global counter
    if counter < population_size:
        init_agent(c1, r1, l1)
    else:
        counter = 0
        prep_next_gen(c1, r1, l1)


def init_agent(c1, r1, l1):
    global counter
    all_snakes[counter].show_all_elements(c1)  # graphics
    update(c1, r1, l1)


def update(c1, r1, l1):
    global counter_2, counter, ms_time
    if not all_snakes[counter].get_dead() and len(all_snakes) != 0:
        all_snakes[counter].think()
        all_snakes[counter].update()
        counter_2 += 1

        if all_snakes[counter].food_eaten:
            all_snakes[counter].delete_all_elements(c1)
            all_snakes[counter].food = all_snakes[counter].create_food()
            all_snakes[counter].show_all_elements(c1)
            all_snakes[counter].food_eaten = False

        # if all_snakes[counter].get_score() == 70:
        # ms_time = 100
        if counter_2 >= draw_for_update:
            all_snakes[counter].move_all_elements(c1)  # graphics
            l1.config(text=f"Sc: {all_snakes[counter].get_score()}, A: {counter},"
                           f" Re: {ms_time}, Gen: {gen}, UpD: {draw_for_update}")
            counter_2 = 0
            r1.after(ms_time, update, c1, r1, l1)
        else:
            update(c1, r1, l1)

    else:
        all_snakes[counter].delete_all_elements(c1)  # graphics
        counter += 1
        load_level(c1, r1, l1)


def prep_next_gen(c1=None, r1=None, l1=None):
    global gen, best_2, best_1
    brain.K.clear_session()
    if load_m:
        all_snakes.clear()
        load_models()
        if no_graphics:
            train(True)
        else:
            load_level(c1, r1, l1)
    else:
        if not no_graphics:
            for i in all_snakes:
                saved_all_snakes.append(i)
            all_snakes.clear()
        mergesort(saved_all_snakes)
        calc_fitness()
        pick_next_gen()
        save_graph()
        gen += 1
        if no_graphics:
            train()
        else:
            load_level(c1, r1, l1)


# noinspection PyUnresolvedReferences
def pick_next_gen():
    global best_of_gen_1, best_of_gen_2, best_1, best_2
    if best_1 is None:
        for i in range(4):
            pick_one()
        best_1 = best_2 = all_snakes[0]
    else:
        all_snakes.insert(0, brain.NNet(best_1.get_net()))
        all_snakes.insert(1, brain.NNet(best_2.get_net()))
        all_snakes.insert(2, brain.NNet(best_of_gen_1.get_net()))
        all_snakes.insert(3, brain.NNet(best_of_gen_2.get_net()))

    crossover((population_size // 2) - 2, 0.5)
    for i in range((population_size // 2) - 2):
        pick_one()
    saved_all_snakes.clear()


def copy_snake(snake):
    x = brain.NNet(snake.get_net())
    x.set_fitness(snake.get_fitness())
    x.set_score(snake.get_score())
    return x


def calc_fitness():
    global best_1, best_2, all_fitness, best_of_gen_1, best_of_gen_2, saved_all_snakes, graph_best_gen
    all_fitness = 0
    for i in saved_all_snakes:
        all_fitness += i.get_fitness()
    best_of_gen_1 = copy_snake(saved_all_snakes[-1])
    best_of_gen_2 = copy_snake(saved_all_snakes[-2])
    if best_of_gen_1.get_fitness() > best_1.get_fitness():
        if best_1.get_fitness() > best_2.get_fitness():
            best_2 = copy_snake(best_1)
        best_1 = copy_snake(best_of_gen_1)
    elif best_of_gen_1.get_fitness() > best_2.get_fitness():
        best_2 = copy_snake(best_of_gen_1)
    if best_of_gen_2.get_fitness() > best_2.get_fitness():
        best_2 = copy_snake(best_of_gen_2)
    graph_best_gen.append(best_of_gen_1.get_score())
    print("Best:", best_1.get_score(), ",", best_1.get_fitness(), ";", best_2.get_score(), ",", best_2.get_fitness())
    print(f"Best of Gen {gen}:", best_of_gen_1.get_score(), ",", best_of_gen_1.get_fitness(), ";",
          best_of_gen_2.get_score(), ",", best_of_gen_2.get_fitness())


def crossover(how_many, bias = 0.5):
    global population_size, best_1, best_2

    if len(saved_all_snakes) != 0:
        parents = []
        for i in range(population_size // 10):  # magic number 10% of pop
            parents.append(saved_all_snakes[-(i + 1)])
        for r in range(how_many):
            p1 = parents[random.randint(0, len(parents) - 1)]
            p2 = parents[random.randint(0, len(parents) - 1)]
            counter_cc = 0
            while p1 == p2 and counter_cc < 5:
                p2 = parents[random.randint(0, len(parents) - 1)]
                counter_cc += 1

            child = brain.NNet(p2.get_net())

            for k, layer in enumerate(child.get_net().layers):
                child_weights = []

                for p, weight_array in enumerate(layer.get_weights()):
                    save_shape = weight_array.shape
                    weight_array2 = brain.np.asarray(p1.get_net().layers[k].get_weights()[p])
                    reshaped_weights = weight_array.reshape(-1)
                    reshaped_weights2 = weight_array2.reshape(-1)
                    for n in range(len(reshaped_weights)):
                        if random.uniform(0, 1) <= bias:
                            reshaped_weights[n] = reshaped_weights2[n]
                    new_weights = reshaped_weights.reshape(save_shape)
                    child_weights.append(new_weights)
                child.net.layers[k].set_weights(child_weights)
            child.mutate(mutation_rate, mutation_rate2)
            all_snakes.append(child)
    else:
        for i in range(how_many):
            all_snakes.append(brain.NNet())


def pick_one_2(i):
    if len(saved_all_snakes) != 0:
        temp = saved_all_snakes[-(i + 1)]
        child = brain.NNet(temp.get_net())
        child.mutate(mutation_rate, mutation_rate2)
        all_snakes.append(child)
    else:
        all_snakes.append(brain.NNet())


def pick_one():
    if len(saved_all_snakes) != 0:
        s = random.random()
        if s <= 0.1:
            temp = best_1
        elif 0.1 < s <= 0.2:
            temp = best_2
        elif 0.2 < s <= 0.3:
            temp = best_of_gen_1
        elif 0.3 < s <= 0.4:
            temp = best_of_gen_2
        else:
            d = random.random()
            i = 0
            while 0 < d:
                if all_fitness == 0:
                    i = random.randint(1, population_size)
                    break
                d -= (saved_all_snakes[-i - 1].get_fitness() / all_fitness)
                i += 1
            i -= 1
            temp = saved_all_snakes[-i - 1]
        net = temp.get_net()
        child = brain.NNet(net)
        child.mutate(mutation_rate, mutation_rate2)
        all_snakes.append(child)

    else:
        all_snakes.append(brain.NNet())


def mergesort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        mergesort(left)
        mergesort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i].get_fitness() < right[j].get_fitness():
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


def center(master):  # graphics
    master.update_idletasks()
    x = (master.winfo_screenwidth() // 2) - (window_size // 2)
    y = (master.winfo_screenheight() // 2) - (window_size // 2)
    master.geometry('{}x{}+{}+{}'.format(window_size, window_size + 29, x, y))  # 29 is label height


def main():
    li = keyboard.Listener(on_press=key_control)
    li.start()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)  # because tf is bugge bugge
    if not no_graphics:
        root = tk.Tk()
        center(root)
        root.title("Snake AI")
        canvas = tk.Canvas(root, height=window_size, width=window_size, bg=brain.back_color, highlightthickness=0)
        l1 = tk.Label(root, text=f"Sc: 0, A: {counter}, Re: {ms_time},"
                                 f" Gen: {gen}, UpD: {draw_for_update}",
                      fg=brain.apple_color, bg=brain.back_color_2, font=my_font)
        l1.pack(fill=tk.BOTH)
        canvas.pack()
        root.resizable(0, 0)
        pick_next_gen()
        load_level(canvas, root, l1)
        root.mainloop()
    else:
        train()


if __name__ == "__main__":
    main()
