from phase_utils import select_model, model_info, phase_setup, select_number
from model_call import generation_openrouter

from collections import OrderedDict
from tqdm import tqdm

import json
import numpy
import os
import os.path as path

# Print:
from colorama import Fore, Style
Y = Fore.YELLOW
C = Fore.CYAN
R = Style.RESET_ALL


###____________________ PROCESSING _____________________###

def reorder_second_phase(processed_folder):
    
    files_potential_adherence = []

    for filename in os.listdir(processed_folder):
        with open(path.join(processed_folder, filename), 'r', encoding='utf-8') as file:
            current_json = json.load(file)
            files_potential_adherence.append((current_json['storytelling_potential'], current_json['dramatic_situation']['adherence'], filename))

    files_potential_adherence.sort(reverse=True, key=lambda x: (x[0], x[1]))

    for index, (_, _, filename) in enumerate(files_potential_adherence, start=1):
        os.rename(path.join(processed_folder, filename), path.join(processed_folder, f'processed_{index:04d}.json'))


def processing_second_phase(source_folder, raw_folder, processed_folder, situations_file):
    
    # Loading dramatic situations reference:
    with open(situations_file, 'r', encoding='utf-8') as file: 
        situations_reference = json.load(file)

    # Excluding narrative events without dramatic situation:
    file_counter = 1

    for filename in os.listdir(raw_folder):
        with open(path.join(raw_folder, filename), 'r', encoding='utf-8') as file:
            raw_results = json.load(file)

            best_situation_name = next(iter(raw_results))
            best_situation = raw_results[best_situation_name]

            if best_situation['adherence'] >= 60:
                
                # Retrieve narrative unit data:
                with open(path.join(source_folder, filename.replace('raw','processed')), 'r', encoding='utf-8') as file:
                    narrative_unit = json.load(file)

                # Retrieve dramatic situation details:
                dramatic_situation = situations_reference[best_situation_name]

                for original, instantiated in zip(dramatic_situation['dynamic_elements'], best_situation['dynamic_elements']):
                    original['values'] = [value['value'] for value in instantiated['values']]

                narrative_unit['dramatic_situation'] = OrderedDict(
                    [('title', best_situation_name), ('adherence', best_situation['adherence'])] +
                    [(key, value) for key, value in dramatic_situation.items() if key != 'title']
                )

                with open(path.join(processed_folder, f'processed_unordered_{file_counter:04d}.json'), 'w', encoding='utf-8') as output_file:
                    json.dump(narrative_unit, output_file, ensure_ascii=False, indent=4)
                    file_counter += 1   

    reorder_second_phase(processed_folder)            
                

###____________________ GENERATION _____________________###


def validation_second_phase(response, situations_file):
    
    required_keys_situation = {'reasoning', 'adherence', 'dynamic_elements'}
    
    try:
        
        response_json = json.loads(response[response.find('{') : response.rfind('}') + 1])

        # Loading dramatic situations reference:
        with open(situations_file, 'r', encoding='utf-8') as file: 
            reference = json.load(file)
        
        # CHECK DRAMATIC SITUATIONS - Missing situation:
        if len(response_json.keys()) != 36:
            print(f'{C}ERROR:{R} DRAMATIC SITUATIONS - Missing situation')
            return None

        # CHECK DRAMATIC SITUATIONS - Keys:
        if response_json.keys() != reference.keys():
            print(f'{C}ERROR:{R} DRAMATIC SITUATIONS - Wrong name for situation')
            return None
        
        for dramatic_situation_name in response_json.keys():
            
            dramatic_situation = response_json[dramatic_situation_name]
            
            # CHECK DRAMATIC SITUATION - Type:
            if not isinstance(dramatic_situation, dict):
                print(f'{C}ERROR:{R} DRAMATIC SITUATION - Type')
                return None
            
            # CHECK DRAMATIC SITUATION - Keys:
            if not set(dramatic_situation.keys()) == required_keys_situation:
                print('{C}ERROR:{R} DRAMATIC SITUATION - Keys')
                return None
            
            # CHECK DRAMATIC SITUATIONS - Reasoning:
            if not isinstance(dramatic_situation['reasoning'], str):
                 print('{C}ERROR:{R} DRAMATIC SITUATIONS - Reasoning')
                 return None
            
            # CHECK DRAMATIC SITUATIONS - Adherence:
            if not isinstance(dramatic_situation['adherence'], int) or not (0 <= dramatic_situation['adherence'] <= 100):
                print('{C}ERROR:{R} DRAMATIC SITUATIONS - Adherence')
                return None
            
            # CHECK ELEMENTS - Type:
            if not isinstance(dramatic_situation['dynamic_elements'], list):
                print('{C}ERROR:{R} ELEMENTS - Type')
                return None
            
            reference_situation = reference[dramatic_situation_name]
            reference_elements = {item['element'] for item in reference_situation['dynamic_elements']}
            found_elements = set()

            for element in dramatic_situation['dynamic_elements']:

                # CHECK ELEMENTS - Structure & Keys:
                if not isinstance(element, dict) or not set(element.keys()) == {'element', 'values'}:
                    print('{C}ERROR:{R} ELEMENTS - Structure & Keys')
                    return None
                
                # CHECK ELEMENTS - Element:
                if element['element'] not in reference_elements:
                    print('{C}ERROR:{R} ELEMENTS - Element')
                    return None
                
                found_elements.add(element['element'])   

                # CHECK ELEMENTS - Values:
                if not isinstance(element['values'], list):
                    print('{C}ERROR:{R} ELEMENTS - Values')
                    return None
                
                # CHECK ELEMENTS - Adherence value:
                if len(element['values']) == 0 and dramatic_situation['adherence'] >= 60:
                    print('{C}ERROR:{R} ELEMENTS - Adherence value')
                    return None

                # CHECK VALUES:
                if len(element['values']) != 0:
                
                    for value in element['values']:
                        
                        # CHECK VALUES - Keys & Structure:
                        if not isinstance(value, dict) or not set(value.keys()) == {'value', 'source'}:
                            print('{C}ERROR:{R} VALUES - Keys & Structure')
                            return None
                        
                        # CHECK VALUES - Types:
                        if not isinstance(value['value'], str) or not value['value']:
                            print('{C}ERROR:{R} VALUES - Types')
                            return None
                        
                        # CHECK VALUES - Source:
                        if value['source'] not in ('estratto', 'dedotto'):
                            print('{C}ERROR:{R} VALUES - Source')
                            return None

            # CHECK ELEMENTS - All dynamic elements:
            if found_elements != reference_elements:
                    print('{C}ERROR:{R} ELEMENTS - All dynamic elements')
                    return None
        
        return response_json
    
    except json.JSONDecodeError:
        print(f'{C}ERROR:{R} Not a JSON')
        return None


def second_phase(model, files_to_process, source_folder, raw_folder, processed_folder, prompt_file, situations_file, costs_file, final_cost_file):
    
    action = phase_setup(source_folder, raw_folder, processed_folder, costs_file)
    
    # Raw results retrieval:
    if action == 'FIRST RUN' or action == 'RESUME/RESTART':    
        progress_bar = tqdm(sorted(os.listdir(source_folder))[:files_to_process], desc=f'Narrative Units Analysis')

        for filename in progress_bar:
            with open(path.join(source_folder, filename), 'r', encoding='utf-8') as file:
                
                # CHECK - Resumed Analysis:
                if action == 'FIRST RUN' or not path.isfile(path.join(raw_folder, f'raw_{filename.split('_')[-1].split('.')[0]}.json')):
                     
                    narrative_unit = json.load(file)
                    prompt = f'{open(prompt_file, encoding = 'utf-8').read()} \n {narrative_unit}'

                    # Call model:
                    validated_response = None

                    while validated_response is None:
                        progress_bar.set_postfix_str(f'{filename}', refresh=True)
                        response, cost = generation_openrouter(prompt, model['model_id'],  model['model_provider'],  model['model_reasoning'])
                        if response:
                            validated_response = validation_second_phase(response, situations_file)

                    situations_adherence = dict(sorted(
                            validated_response.items(), 
                            key = lambda item: item[1]['adherence'], 
                            reverse=True
                        )
                    )

                    # Saving data:
                    with open(path.join(raw_folder, f'raw_{filename.split('_')[-1].split('.')[0]}.json'), 'w', encoding='utf-8') as out_file:
                        json.dump(situations_adherence, out_file, ensure_ascii=False, indent=4)

                    with open(costs_file, 'a', encoding='utf-8') as out_file:
                        out_file.write(f'{cost}\n')
    
    print('')

    # Raw results processing:
    if action != 'SKIP':
        os.makedirs(processed_folder)
        processing_second_phase(source_folder, raw_folder, processed_folder, situations_file)

        with open(final_cost_file, 'w', encoding='utf-8') as file:
            file.write(f'${numpy.sum(numpy.loadtxt(costs_file))}')
                        

###________________________ MAIN ________________________###

def main():
    
    model = select_model()
    model_info(model)

    print(f'{Y}PHASE 2{R}\nAssessment of the adherence of extracted narrative events to Polti\'s 36 dramatic situations\n')

    # Folders:
    source_folder =  path.join('story_generation', 'results', model['model_name'], 'phase_1', 'processed_results')
    raw_folder = path.join('story_generation', 'results', model['model_name'], 'phase_2', 'raw_results')
    processed_folder = path.join('story_generation', 'results', model['model_name'], 'phase_2', 'processed_results')

    # Files:
    prompt_file = path.join('story_generation', 'resources', 'prompts', 'prompt_phase_2.txt')
    situations_file = path.join('story_generation', 'resources', 'dramatic_situations', 'dramatic_situations.json')
    costs_file = path.join('story_generation', 'results', model['model_name'], 'phase_2', 'Phase 2 - API Costs Detail.txt')
    final_cost_file = path.join('story_generation', 'results', model['model_name'], 'phase_2', 'Phase 2 - API Cost Final.txt')

    files_to_process = select_number(raw_folder,processed_folder)
    second_phase(model, files_to_process, source_folder, raw_folder, processed_folder, prompt_file, situations_file, costs_file, final_cost_file)
    

if __name__ == "__main__":
    main()