import pathlib

HOME_PATH = pathlib.Path(__file__).parents[2].resolve()
DATA_PATH = HOME_PATH / "data/data"
TMP_PATH = HOME_PATH / "/results/tmp"
SAVE_PATH = HOME_PATH / "data/save"
OUT_PATH = HOME_PATH / "results/out"
DRAW_PATH = HOME_PATH / "data/draw_data"
DRAW_OUT_PATH = HOME_PATH / "results/out_draw"
DEMO_DATA_PATH = HOME_PATH / "data/demo_data"
DEMO_OUT_PATH = HOME_PATH / "results/demo_out"