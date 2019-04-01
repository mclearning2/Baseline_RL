import os
import argparse
import importlib
from glob import glob
from PyInquirer import style_from_dict, Token, prompt, Separator

def get_config_and_module():
    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

    parser.add_argument("--user_name", type=str, default="mclearning2")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--test", dest="test", action="store_true")
    parser.add_argument("--restore", dest="restore", action="store_true")
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--record", dest="record", action="store_true")

    parser.add_argument("--project_dir",
                        type=str,
                        default="project")
    parser.add_argument("--video_dir",
                        type=str,
                        default="report/videos/")
    parser.add_argument("--models_dir",
                        type=str,
                        default="report/models/")
    parser.add_argument("--hyper_params_dir",
                        type=str,
                        default="report/hyperparams/")

    parser.set_defaults(test=False)
    parser.set_defaults(restore=False)
    parser.set_defaults(render=False)
    parser.set_defaults(record=False)

    config = parser.parse_args()

    # Select module to run
    if not config.project:
        config.project = select_project(config.project_dir)

    # Import module to run
    import_name = config.project_dir + "." + config.project
    module = importlib.import_module(import_name)

    # Initialize path to save information by project name
    config.video_dir = os.path.join(config.video_dir, config.project)
    config.models_path = os.path.join(config.models_dir,
                                      config.project, 'parameters.pt')
    config.hyper_params_path = os.path.join(config.hyper_params_dir,
                                            config.project, 'hyperparameters.pkl')

    return config, module

def select_project(dir_name):
    style = style_from_dict({
        Token.Separator: '#cc0000',
        Token.QuestionMark: '#673ab7 bold',
        Token.Selected: '#cc5454',  # default
        Token.Pointer: '#673ab7 bold',
        Token.Instruction: '',  # default
        Token.Answer: '#f44336 bold',
        Token.Question: '',
    })

    choices = [Separator('== Algorithms with environment ==')]
    for file in glob(os.path.join(dir_name, '*.py')):
        choices.append({'name': file})

    questions = {
        'type': 'list',
        'message': 'Select the algorithm you want to run in directory ',
        'name': dir_name,
        'choices': choices
    }

    experiment = prompt(questions, style=style)[dir_name]
    file_name = os.path.basename(experiment)
    module_name = os.path.splitext(file_name)[0]

    return module_name