import os
import argparse
import importlib
from glob import glob
from PyInquirer import style_from_dict, Token, prompt, Separator

def get_config_and_module():
    ''' Parsing and import module
        common.abstract.base_project

        Returns:
            config(Namespace) : It is used in {project_folder}/common/base_project.py
            module(module) : Its 'Project' class is imported and execute run()

        Parser Arguments(config):
            user_name : refer to {run_id}
            project : default None. file name in {project_folder}/projects
                      if you don't set this. You must choose one in interactive command line                      
            run_id : default None. If you input run_id,
                     it restore files from {user_name}/{project}/{run_id} in wandb cloud
            seed : seed for reproduction
            test_mode : test mode on
            restore : restore hyperparameters and model parameter from
                      {project_folder}/record/{project}
            render : render on
            record : record videos in {project_folder}/record/videos/{project}
            video_dir : {project_folder}/record/videos/{project}
            params_path : {project_folder}/record/model/{project}/model.pt
            hyperparams_path : {project_folder}/record/model/{project}/hyperparams.pkl

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

    parser.add_argument("-u", "--user_name", type=str, 
                        default="mclearning2",
                        help="Wandb(Weights & biases)에서 사용하는 user name. \n"
                             "{user_name}/{project}/{run_id}를 통해 Cloud에서 "
                             "불러오거나 저장할 때 쓰인다.")

    parser.add_argument("-p", "--project", type=str, 
                        default=None,
                        help="{project_folder}/projects에 존재하는 파일 이름. \n"
                             "e.g. A2C_CartPole-v1 \n"
                             "{project_folder}/report에서 저장할 폴더 이름으로도 사용")

    parser.add_argument("-r", "--run_id", type=str, 
                        default=None,
                        help="Wandb로부터 restore할 때 사용.\n"
                             "cloud에서 존재하는 run_id를 넣을 경우 report의 model과 video에 "
                             "hyperparameter, model parameter, video를 불러와 저장")

    parser.add_argument("-s", "--seed", type=int, default=1)

    parser.add_argument("-t", "--test_mode", dest="test_mode", action="store_true")
    parser.add_argument("-l", "--restore", dest="restore", action="store_true")
    parser.add_argument("-d", "--render", dest="render", action="store_true")
    parser.add_argument("-c", "--record", dest="record", action="store_true")

    parser.add_argument("--report_dir", type=str, default="report")

    parser.set_defaults(test_mode=False)
    parser.set_defaults(restore=False)
    parser.set_defaults(render=False)
    parser.set_defaults(record=False)

    config = parser.parse_args()

    # Select module to run
    if not config.project:
        config.project = select_project("projects")

    # Import module
    import_name = "projects." + config.project
    module = importlib.import_module(import_name)

    # Initialize path to save information by project name
    config.video_dir = os.path.join(config.report_dir, 'videos', config.project)
    config.params_path = os.path.join(config.report_dir, 'model', config.project, 'model.pt')
    config.hyperparams_path = os.path.join(config.report_dir, 'model', config.project, 'hyperparams.pkl')

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
    for file in sorted(glob(os.path.join(dir_name, '*.py'))):
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