import os
import shutil
from uuid import uuid4

path = "real_flat"
if not os.path.exists(path):
    os.mkdir(path)

for root, dirs, files in os.walk("real_img"):
   for name in files:
       current_img_path = os.path.join(root, name)
       new_path = os.path.join(path, f"{uuid4()}.jpg")
       shutil.copy(current_img_path, new_path)

