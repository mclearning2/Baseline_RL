import os

def check_or_make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)