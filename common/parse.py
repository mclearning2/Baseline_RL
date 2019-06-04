import os
import argparse
import importlib
from glob import glob
from PyInquirer import style_from_dict, Token, prompt, Separator

def get_config() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

    parser.add_argument("-u", "--user_name", type=str, 
                        default="mclearning2",
                        help="Wandb(Weights & biases)에서 사용하는 user name. \n"
                             "{user_name}/{project}/{run_id}를 통해 Cloud에서 "
                             "불러오거나 저장할 때 쓰인다.")

    parser.add_argument("-p", "--project", type=str, 
                        default=None,
                        help="projects 폴더에 존재하는 파일 이름. ")

    parser.add_argument("-r", "--run_id", type=str, 
                        default=None,
                        help="Wandb로부터 restore할 때 사용.\n"
                             "None일 경우 아무것도 하지 않음.\n"
                             "https://app.wandb.ai/home에서 특정 run_id를 넣을 경우, "
                             "자동으로 report/model 안에 --project 이름의 폴더 안에 "
                             "hyperparams.pkl과 model.pt를 저장")

    parser.add_argument("-s", "--seed", type=int, default=1)

    parser.add_argument("-t", "--test", dest="test", action="store_true")
    parser.add_argument("-l", "--restore", dest="restore", action="store_true")
    parser.add_argument("-d", "--render", dest="render", action="store_true")
    parser.add_argument("-c", "--record", dest="record", action="store_true")
    parser.add_argument("-f", "--tb_not_force", dest="record", action="store_true",
                        help="해당 텐서보드 폴더가 있으면 지우지 않기."
                             "default로 지움")

    parser.add_argument("--report_dir", type=str, default="report")

    parser.set_defaults(test=False)
    parser.set_defaults(restore=False)
    parser.set_defaults(render=False)
    parser.set_defaults(record=False)
    parser.set_defaults(tb_not_force=False)

    config = parser.parse_args()

    return config

def select_project() -> str:
    ''' projects 폴더 내의 하나의 파일을 선택하도록 한다. 

        e.g. projects.policy-based.A2C_CartPole-v1.py    
    '''

    project_dir = "projects"

    # Select folder in projects
    # ==========================================================================
    style = style_from_dict({
        Token.Separator: '#cc0000',
        Token.QuestionMark: '#673ab7 bold',
        Token.Selected: '#cc5454',  # default
        Token.Pointer: '#673ab7 bold',
        Token.Instruction: '',  # default
        Token.Answer: '#f44336 bold',
        Token.Question: '',
    })

    select_dir = [Separator('== Algorithms Type ==')]
    for dir_name in sorted(os.listdir(project_dir)):
        if dir_name != "__pycache__":
            select_dir.append(dir_name)

    questions = {
        'type': 'list',
        'message': 'Select the algorithm type.',
        'name': "selected_dir",
        'choices': select_dir
    }

    selected_dir = prompt(questions, style=style)['selected_dir']

    # ==========================================================================

    # Select project in 
    # ==========================================================================

    choices = [Separator('== Select Algorithms ==')]
    for file in sorted(glob(os.path.join(project_dir, selected_dir, '*.py'))):
        choices.append({'name': file})

    questions = {
        'type': 'list',
        'message': 'Select the project.',
        'name': "selected_project",
        'choices': choices
    }

    selected_project = prompt(questions, style=style)['selected_project']

    # ==========================================================================

    return selected_project

def import_module(project_path: str):

    import_name = project_path.split('.')[0].replace("/", ".")
    module = importlib.import_module(import_name)

    return module
