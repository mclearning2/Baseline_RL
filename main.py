import os
import shutil
from importlib import import_module

from common.parse import get_config
from common.pyinquirer import select_project

def main():
    config = get_config()

    # e.g. projects/policy_based/A2C_CartPole-v1.py
    project_path = select_project(projects_dir=config.projects_dir)
    # e.g. projects/policy_based/A2C_CartPole-v1
    non_extention_path = project_path.split(".")[0]
    # e.g. A2C_CartPole-v1
    config.project = non_extention_path.split('/')[-1]
    # e.g. projects.policy_based.A2C_CartPole-v1
    import_name = non_extention_path.replace("/", ".")

    module = import_module(import_name)
    project = module.Project(config)
    project.run()

if __name__ == '__main__':
    main()
    
