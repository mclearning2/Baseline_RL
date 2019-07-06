import os
from glob import glob
from PyInquirer import style_from_dict, Token, prompt, Separator

style = style_from_dict({
    Token.Separator: '#cc0000',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

def select_project(projects_dir: str = "projects") -> str:
    ''' It makes to users select algorithm(project)
    
    Args:
        projects_dir (str): The directories of projects
    
    Returns:
        str: The relative path of project

        >>> select_project(projects)
        projects/policy-based/A2C_CartPole-v1.py    
    '''

    # Select directories in projects
    # ===================================================================
    select_dir = [Separator('== Algorithms Base ==')]
    for dir_name in sorted(os.listdir(projects_dir)):
        if dir_name != "__pycache__":
            select_dir.append(dir_name)

    questions = {
        'type': 'list',
        'message': 'Select the algorithm type.',
        'name': "selected_dir",
        'choices': select_dir
    }

    selected_dir = prompt(questions, style=style)['selected_dir']

    # Select project in directories
    # ===================================================================

    choices = [Separator('== Select Algorithms ==')]    
    for file in sorted(glob(os.path.join(projects_dir, selected_dir, '*.py'))):
        choices.append({'name': file})

    questions = {
        'type': 'list',
        'message': 'Select the project.',
        'name': "selected_project",
        'choices': choices
    }

    selected_project = prompt(questions, style=style)['selected_project']

    return selected_project