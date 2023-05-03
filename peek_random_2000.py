import os
import shutil
from random import choice, sample

all_path = "real_flat"
selected_path = "real_flat_2000"

all = os.listdir(all_path)
peeked = sample(all, 2000)

if os.path.exists(selected_path):
    shutil.rmtree(selected_path)
os.mkdir(selected_path)

for img_name in peeked:
    src_path = os.path.join(all_path, img_name)
    dest_path = os.path.join(selected_path, img_name)
    shutil.copy(src_path, dest_path)
