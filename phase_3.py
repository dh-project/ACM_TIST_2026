from phase_utils import select_model, model_info, phase_setup
from model_call import generation_openrouter

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

def processing_third_phase(model, source_folder, raw_folder, processed_folder, catalog_file):
    
    model_tags = {
        'gpt-oss-120b': 'gpt',
        'gemini-2.5-flash': 'gem',
        'deepseek-v3.2': 'ds'
    }

    # Retrieve artifact data:
    with open(catalog_file, 'r', encoding='utf-8') as file:
        catalog = json.load(file)

    for filename in os.listdir(raw_folder):
        with open(path.join(raw_folder, filename), 'r', encoding='utf-8') as file:
            raw_results = json.load(file)
 
            file_number = filename.split('_')[-1].split('.')[0]
            story_id = f'{model_tags.get(model['model_name'])}_{file_number}'

            # Retrieve narrative unit data:
            with open(path.join(source_folder, filename.replace('raw','processed')), 'r', encoding='utf-8') as file:
                narrative_unit = json.load(file)

            # Retrieve artifact data:
            catalog_id = narrative_unit['catalog_id']
            artifact = catalog[catalog_id]

            # Base structure:
            processed_story = {
                story_id: {
                    'reperto': {
                        'id_catalogo': catalog_id,
                        'descrizione': artifact['description'],
                        'descrizione_curatoriale': artifact['curator_description']
                    },
                    'fase_1': {
                        'evento_centrale': narrative_unit['core_event'],
                        'descrizione': narrative_unit['event_description'],
                        'potenziale_narrativo': narrative_unit['storytelling_potential']
                    },
                    'fase_2': {
                        'titolo': narrative_unit['dramatic_situation']['title'],
                        'elementi_dinamici': [],
                        'aderenza': narrative_unit['dramatic_situation']['adherence']
                    },
                    'fase_3': {
                        'titolo': raw_results['title'],
                        'luogo': raw_results['place'],
                        'tempo': raw_results['time'],
                        'scene': []
                    },
                    'tabella_riassunto': [
                        ['#', 'Titolo', 'Location', 'Personaggi']
                    ]
                }
            }

            for element in narrative_unit['dramatic_situation']['dynamic_elements']:
                
                processed_story[story_id]['fase_2']['elementi_dinamici'].append({
                    'elemento': element['element'],
                    'valore': ', '.join(element['values'])
                })

            for scene in raw_results['scenes']:
                
                processed_story[story_id]['fase_3']['scene'].append({
                    'num': scene['number'],
                    'titolo': scene['title'],
                    'contenuto': scene['text']
                })

                processed_story[story_id]['tabella_riassunto'].append(
                    [scene['number'], scene['title'], scene['location'], ', '.join(scene['characters'])]
                )

            with open(path.join(processed_folder, f'processed_{file_number}.json'), 'w', encoding='utf-8') as output_file:
                json.dump(processed_story, output_file, ensure_ascii=False, indent=4)


###____________________ GENERATION _____________________###

def validation_third_phase(response):
    
    required_keys = {'title', 'place', 'time', 'scenes'}
    required_keys_scenes = {'number', 'title', 'location', 'characters', 'text'}

    try:
        
        response_json = json.loads(response[response.find('{') : response.rfind('}') + 1])

        # CHECK - Keys:
        if not set(response_json.keys()) == required_keys:
            print(f'{C}ERROR:{R} Keys')
            return None
        
        # CHECK - Keys Types and Length:
        if not(isinstance(response_json['title'], str) and
               isinstance(response_json['place'], str) and
               isinstance(response_json['time'], str) and
               isinstance(response_json['scenes'], list) and
               len(response_json['scenes']) >= 6 and
               len(response_json['scenes']) <= 8): 
            print(f'{C}ERROR:{R} Keys types and Length')
            return None

        # CHECK SCENES
        previous_number = 0

        for item in response_json['scenes']:

            # CHECK SCENES - Keys:
            if not set(item.keys()) == required_keys_scenes:
                print(f'{C}ERROR:{R} SCENES - Keys')
                return None
            
            # CHECK SCENES - Keys Types:
            if not(type(item['number']) is int and
                   isinstance(item['title'], str) and
                   isinstance(item['location'], str) and
                   isinstance(item['characters'], list) and
                   isinstance(item['text'], str)): 
                print(f'{C}ERROR:{R} SCENES - Keys Types')
                return None

            # CHECK SCENES - Number:
            if item['number'] != (previous_number + 1):
                print(f'{C}ERROR:{R} SCENES - Number')
                return None
            
            previous_number = item['number']
            
            # CHECK SCENES - Characters:
            if not item['characters']:
                print(f'{C}ERROR:{R} SCENES - Characters')
                return None

            for character in item['characters']:
                if not isinstance(character, str):
                    print(f'{C}ERROR:{R} SCENES - Characters')
                    return None

        return response_json  

    except json.JSONDecodeError:
        print(f'{C}ERROR:{R} Not a JSON')
        return None


def third_phase(model, source_folder, raw_folder, processed_folder, prompt_file, costs_file, final_cost_file, catalog_file):
    
    action = phase_setup(source_folder, raw_folder, processed_folder, costs_file)
    
    # Raw results retrieval:
    if action == 'FIRST RUN' or action == 'RESUME/RESTART':
        progress_bar = tqdm(sorted(os.listdir(source_folder)), desc=f'Story Generation')

        for filename in progress_bar:
            with open(path.join(source_folder, filename), 'r', encoding='utf-8') as file:
                
                file_number = filename.split('_')[-1].split('.')[0]
                
                # CHECK - Resumed Analysis:
                if action == 'FIRST RUN' or not path.isfile(path.join(raw_folder, f'raw_{file_number}.json')):
              
                    
                    narrative_unit = json.load(file)
                    prompt = f'{open(prompt_file, encoding="utf-8").read()} \n {narrative_unit}'

                    # Call model:
                    validated_response = None

                    while validated_response is None:
                        progress_bar.set_postfix_str(filename, refresh=True)
                        response, cost = generation_openrouter(prompt, model['model_id'],  model['model_provider'],  model['model_reasoning'])
                        if response:
                            validated_response = validation_third_phase(response)

                    # Saving data:
                    with open(path.join(raw_folder, f'raw_{file_number}.json'), 'w', encoding='utf-8') as out_file:
                        json.dump(validated_response, out_file, ensure_ascii=False, indent=4)

                    with open(costs_file, 'a', encoding='utf-8') as out_file:
                        out_file.write(f'{str(cost)}\n')

    print('')

    # Raw results processing:
    if action != 'SKIP':
        os.makedirs(processed_folder)
        processing_third_phase(model, source_folder, raw_folder, processed_folder, catalog_file)

        with open(final_cost_file, 'w', encoding='utf-8') as file:
            file.write(f'${numpy.sum(numpy.loadtxt(costs_file))}')



###________________________ MAIN ________________________###

def main():
    
    model = select_model()
    model_info(model)

    print(f'{Y}PHASE 3{R}\nCreation of curatorial stories from the assessed narrative events, following the dramatic situation with the highest adherence\n')

    # Folders:
    source_folder =  path.join('results', model['model_name'], 'phase_2', 'processed_results')
    raw_folder = path.join('results', model['model_name'], 'phase_3', 'raw_results')
    processed_folder = path.join('results', model['model_name'], 'phase_3', 'processed_results')
    
    # Files:
    prompt_file = path.join('resources', 'prompts', 'prompt_phase_3.txt')
    costs_file = path.join('results', model['model_name'], 'phase_3', 'Phase 3 - API Costs Detail.txt')
    final_cost_file = path.join('results', model['model_name'], 'phase_3', 'Phase 3 - API Cost Final.txt')
    catalog_file = path.join('resources', 'catalog', 'filtered', 'filtered_catalog.json')

    third_phase(model, source_folder, raw_folder, processed_folder, prompt_file, costs_file, final_cost_file, catalog_file)

if __name__ == "__main__":
    main()