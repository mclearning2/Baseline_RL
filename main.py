# Author : mclearning2
# Date : 2019-03-01
# Description : OpenAI gym의 Atari 환경으로 Baseline의 RL 알고리즘들을 최대한 많이
#               다양하게 쉽게 구현하기 위한 프로젝트

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
