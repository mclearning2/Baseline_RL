# Author : mclearning2
# Date : 2019-03-01
# Description : OpenAI gym의 Atari 환경으로 Baseline의 RL 알고리즘들을 최대한 많이
#               다양하게 쉽게 구현하기 위한 프로젝트

import os
import shutil

from common.parse import get_config, select_project, import_module

if __name__ == '__main__':

    config = get_config()

    project_path = select_project() # e.g. projects/policy-based/A2C_CartPole-v1.py

    config.project = os.path.basename(project_path).split(".")[0]
    config.video_dir = os.path.join('report/videos', config.project)
    config.params_path = os.path.join('report/model', config.project, 'model.pt')
    config.hyperparams_path = os.path.join('report/model', config.project, 'hyperparams.pkl')
    config.tensorboard_path = os.path.join('report/tensorboard', config.project)

    if not config.tb_not_force and os.path.isdir(config.tensorboard_path):
        shutil.rmtree(config.tensorboard_path)

    module = import_module(project_path)
    project = module.Project(config)
    project.run() 
