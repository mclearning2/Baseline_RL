import os
import shutil
import logging
from importlib import import_module

from common.parse import get_config
from common.pyinquirer import select_project

if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s",
                        datefmt="%m/%d/%Y %I:%M:%S")
    logger = logging.getLogger(name="logger")

    config = get_config()

    project_path = select_project(projects_dir="projects")

    config.project = os.path.basename(project_path).split(".")[0]

    config.video_dir = os.path.join('report/videos', config.project)
    config.params_path = os.path.join('report/model', config.project,
                                      'model.pt')
    config.hyperparams_path = os.path.join('report/model', config.project,
                                           'hyperparams.pkl')
    config.tensorboard_path = os.path.join('report/tensorboard',
                                           config.project)

    if os.path.isdir(config.tensorboard_path):
        shutil.rmtree(config.tensorboard_path)

    # projects/policybased/A2C_CartPole-v1.py
    # => projects.policybased.A2C_CartPole-v1
    # print(project_path)
    # import sys; sys.exit()
    import_name = project_path.split(".")[0].replace("\\", ".")

    # import projects.policybased.A2C_CartPole-v1
    module = import_module(import_name)

    project = module.Project(config)
    project.run()
