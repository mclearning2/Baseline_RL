import os
import importlib

from common.utils.parse import get_config_and_module

from pyfiglet import Figlet

if __name__ == '__main__':
    f = Figlet(font='slant')
    print(f.renderText("Baseline RL"))

    config, module = get_config_and_module()

    project = module.Project(config)
    project.run()
