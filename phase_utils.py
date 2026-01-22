from model_call import TEMPERATURE, EFFORT, MAX_TOKENS, STREAM

import inquirer
import os
import os.path as path
import shutil

# Print:
from colorama import Fore, Style
Y = Fore.YELLOW
C = Fore.CYAN
R = Style.RESET_ALL

def select_model():

    models = {
        'model_1': {'model_name': 'deepseek-v3.2', 'model_id': 'deepseek/deepseek-v3.2', 'model_provider': 'atlas-cloud/fp8', 'model_reasoning': 'Token'},
        'model_2': {'model_name': 'gemini-2.5-flash', 'model_id': 'google/gemini-2.5-flash', 'model_provider': 'google-vertex/global', 'model_reasoning': 'Token'},
        'model_3': {'model_name': 'gpt-oss-120b', 'model_id': 'openai/gpt-oss-120b', 'model_provider': 'siliconflow/fp8', 'model_reasoning': 'Effort'}
    }

    choice = inquirer.prompt([inquirer.List('action', 
                                                message = f'Choose a model',
                                                choices = [('deepseek-v3.2','model_1'),
                                                        ('gemini-2.5-flash','model_2'),
                                                        ('gpt-oss-120b','model_3')]
                                    )])['action']
    
    return models.get(choice)


def model_info(model):

    print(f"""
{Y}EXPERIMENTATION{R}

{Y}Model{R}
{C}ID:{R} {model['model_id']}
{C}Name:{R} {model['model_name']}
{C}Provider:{R} {model['model_provider']}

{Y}Parameters{R}
{C}Temperature:{R} {TEMPERATURE}
{C}Reasoning:{R} {EFFORT} effort / {MAX_TOKENS} max tokens
{C}Stream:{R} {STREAM}
    """)


def select_percentage(source_folder):
    
    choice = inquirer.prompt([inquirer.List('action', 
                                                message = f'What percentage of data would you like to process?',
                                                choices = [('20%', 20),
                                                           ('40%', 40),
                                                           ('60%', 60),
                                                           ('80%', 80),
                                                           ('100%', 100)])])
                                
    total_files = len(os.listdir(source_folder))
    return round((choice['action']/100) * total_files)
                                                    

def select_number(raw_folder, processed_folder):
    
    # First run:
    if not os.path.exists(processed_folder):
        return 100
    
    else:
        total_files = len(os.listdir(processed_folder))
        return (100 - total_files) + len(os.listdir(raw_folder))
        

def phase_setup(source_folder, raw_folder, processed_folder, costs_file):
    
    # Pending analysis:
    if path.isdir(raw_folder) and (len(os.listdir(raw_folder)) < len(os.listdir(source_folder))):
        
        choice = inquirer.prompt([inquirer.List('action', 
                                                    message = 'An interrupted analysis was found. How to proceed?',
                                                    choices = [('Resume previous analysis', 'RESUME'),
                                                               ('Start new analysis from scratch', 'RESTART')]
                                       )])['action']

        if os.path.exists(processed_folder):
            shutil.rmtree(processed_folder)

        if choice == 'RESTART':
            shutil.rmtree(processed_folder, ignore_errors=True)
            shutil.rmtree(raw_folder)
            os.makedirs(raw_folder)
            os.remove(costs_file)
            open(costs_file, 'w', encoding='utf-8').close()
            
        return 'RESUME/RESTART'

    # Results already processed:
    elif path.isdir(processed_folder):
        
        choice = inquirer.prompt([inquirer.List('action', 
                                                    message = 'Raw results have already been processed. How to proceed?',
                                                    choices = [('Repeat the processing ', 'REPEAT'),
                                                               ('Go to the next phase', 'SKIP')]
                                    )])['action']

        if choice == 'REPEAT':
            shutil.rmtree(processed_folder)

        return choice
    
    # Completed analysis:
    if path.isdir(raw_folder) and (len(os.listdir(raw_folder)) == len(os.listdir(source_folder))):
        return None

    # First run:
    else:
        os.makedirs(raw_folder)
        open(costs_file, 'w', encoding='utf-8').close()
        return 'FIRST RUN'
