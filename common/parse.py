import os
import argparse
import importlib
from glob import glob
from PyInquirer import style_from_dict, Token, prompt, Separator

def get_config_and_module():
    ''' 사용자가 고를 수 있게 만든 요소. 
        common.abstract.base_project와 연결되어 있으므로 수정해서는 안된다.

        save and load:
            여기서 말하는 파라미터는 모델 파라미터와 하이퍼파라미터를 의미.
            파라미터 저장은 학습이 끝나거나 중간에 Ctrl+C로 종료되었을 때 이루어짐
            파라미터 복원은 마지막으로 실행했던 학습의 파라미터를 불러옴.
            
            그 외의 파라미터 복원을 위해서는 wandb에 저장했던 결과물을 복원
            해야 한다. user_name/project/run_id에서 restore해야한다.

            ex)
            [train and record]
            python3 main.py --record

            [restore and test and render]
            python3 main.py --restore --test --render

            [restore from wandb and test]
            python3 main.py --uesr_name mclearning2 --project PPO_Pendulum-v0
                            --run_id 1fa63fc3 --restore --test
    '''

    parser = argparse.ArgumentParser(description="Pytorch RL algorithms")

    parser.add_argument("--user_name", type=str, default="mclearning2")
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None)

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--test_mode", dest="test_mode", action="store_true")
    parser.add_argument("--restore", dest="restore", action="store_true")
    parser.add_argument("--render", dest="render", action="store_true")
    parser.add_argument("--record", dest="record", action="store_true")

    parser.add_argument("--report_dir", type=str, default="report")

    parser.set_defaults(test_mode=False)
    parser.set_defaults(restore=False)
    parser.set_defaults(render=False)
    parser.set_defaults(record=False)

    config = parser.parse_args()

    # Select module to run
    if not config.project:
        config.project = select_project("projects")

    # Import module to run
    import_name = "projects." + config.project
    module = importlib.import_module(import_name)

    # Initialize path to save information by project name
    v_dir = os.path.join(config.report_dir, 'videos', config.project)
    p_path = os.path.join(config.report_dir, 'model', config.project, 'model.pt')
    hp_path = os.path.join(config.report_dir, 'model', config.project, 'hyperparams.pkl')

    config.video_dir = v_dir
    config.params_path = p_path
    config.hyperparams_path = hp_path

    return config, module

def select_project(dir_name):
    ''' 실행시 dir_name에 있는 파일들 중 하나를 선택하도록 한다.
        선택한 파일 중 하나를 고르면 그 파일 이름만 반환한다.
    '''
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

    project_file = prompt(questions, style=style)[dir_name]
    file_name = os.path.basename(project_file)
    file_name_without_ext = os.path.splitext(file_name)[0]

    return file_name_without_ext