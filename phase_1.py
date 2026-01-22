from phase_utils import select_model, model_info, select_percentage, phase_setup
from model_call import generation_openrouter
from tqdm import tqdm

import json
import Levenshtein
import numpy
import os
import os.path as path

# Print:
from colorama import Fore, Style
Y = Fore.YELLOW
C = Fore.CYAN
R = Style.RESET_ALL


###____________________ PROCESSING _____________________###


def remove_duplicates(processed_folder):
    
    descriptions = []
    duplicates = []
    
    # Retrieve:
    for filename in os.listdir(processed_folder):
        with open(path.join(processed_folder, filename), 'r', encoding='utf-8') as file:
            current_json = json.load(file)
            descriptions.append((filename, current_json['event_description']))

    # Confront:
    for index in range(len(descriptions)):
        if descriptions[index][0] not in duplicates:
            for inner in range(index + 1, len(descriptions)):
                if descriptions[inner][0] not in duplicates:
                    if Levenshtein.ratio(descriptions[index][1], descriptions[inner][1]) >= 0.85:
                        duplicates.append(descriptions[inner][0])
                        os.remove(path.join(processed_folder, descriptions[inner][0]))           


def reorder_first_phase(processed_folder):
    
    files_potential = []
    
    for filename in os.listdir(processed_folder):
        with open(path.join(processed_folder, filename), 'r', encoding='utf-8') as file:
            current_json = json.load(file)
            files_potential.append((current_json['storytelling_potential'],filename))
            
    files_potential.sort(reverse = True, key = lambda x: x[0])

    for index, (_, filename) in enumerate(files_potential, start = 1):
        os.rename(path.join(processed_folder, filename), path.join(processed_folder, f'processed_{index:04d}.json'))


def processing_first_phase(raw_folder, processed_folder):

    # Excluding item without narrative event:
    file_counter = 1

    for filename in os.listdir(raw_folder):
        with open(path.join(raw_folder, filename), 'r', encoding='utf-8') as file:
            raw_results = json.load(file)

        for item in raw_results:
            if item['storytelling_potential'] > 0:
                with open(path.join(processed_folder, f'processed_unordered_{file_counter:04d}.json'), 'w', encoding='utf-8') as output_file:
                    json.dump(item, output_file, ensure_ascii=False, indent=4)
                    file_counter += 1    

    # Cleaning and reordering results:
    remove_duplicates(processed_folder)
    reorder_first_phase(processed_folder)        


###____________________ GENERATION _____________________###

def validation_first_phase(response, input_size, catalog_ids):

    required_keys = {'catalog_id', 'has_narrative_event', 'core_event', 'event_description', 'storytelling_potential'}
    
    try:
        response_json = json.loads(response[response.find('[') : response.rfind(']') + 1])

        # CHECK - Catalog items:
        if not isinstance(response_json, list) or len(response_json) != input_size:
            print(f'{C}ERROR:{R} Missing catalog items')
            return None

        # CHECK - Catalog IDs:
        if [item['catalog_id'] for item in response_json] != catalog_ids:
            print(f'{C}ERROR:{R} Wrong catalog IDs')
            return None

        for item in response_json:
            
            # CHECK - Keys:
            if not (isinstance(item, dict) and set(item.keys()) == required_keys):
                print(f'{C}ERROR:{R} Keys')
                return None 
            
            # CHECK - Keys types:
            if not(isinstance(item['catalog_id'], str) and
                   isinstance(item['has_narrative_event'], bool) and
                   (isinstance(item['core_event'], str) or item['core_event'] is None) and
                   (isinstance(item['event_description'], str) or item['event_description'] is None) and
                   type(item['storytelling_potential']) is int):
                print(f'{C}ERROR:{R} Keys types')
                return None
            
            # CHECK - Storytelling potential range:
            if not (0 <= item['storytelling_potential'] <= 100):
                print(f'{C}ERROR:{R} Storytelling potential range')
                return None
            
            # CHECK - Consistency for missing narrative events:
            if not item['has_narrative_event'] and (item['core_event'] or item['event_description'] or item['storytelling_potential'] != 0):
                print(f'{C}ERROR:{R} Consistency for missing narrative events')
                return None
            
            # CHECK - Consistency for found narrative events:
            if item['has_narrative_event'] and (not item['core_event'] or not item['event_description'] or item['storytelling_potential'] == 0):
                print(f'{C}ERROR: {R} Consistency for found narrative events')
                return None 

        return response_json
    
    except json.JSONDecodeError:
        print(f'{C}ERROR:{R} Not a JSON')
        return None


def first_phase(model, files_to_process, source_folder, raw_folder, processed_folder, prompt_file, costs_file, final_cost_file):
    
    action = phase_setup(source_folder, raw_folder, processed_folder, costs_file)
    
    # Raw results retrieval:
    if action == 'FIRST RUN' or action == 'RESUME/RESTART':
        progress_bar = tqdm(sorted(os.listdir(source_folder))[:files_to_process], desc=f'Catalog Analysis')

        for filename in progress_bar:
            with open(path.join(source_folder, filename), 'r', encoding='utf-8') as file:
                
                # CHECK - Resumed Analysis:
                if action == 'FIRST RUN' or not path.isfile(path.join(raw_folder, f'raw_{filename.split('_')[-1].split('.')[0]}.json')):
                    
                    catalog_portion = json.load(file)
                    prompt = f'{open(prompt_file, encoding = 'utf-8').read()} \n {catalog_portion}'

                    # Call model:
                    validated_response = None

                    while validated_response is None:
                        progress_bar.set_postfix_str(f'{filename}', refresh=True)
                        response, cost = generation_openrouter(prompt, model['model_id'],  model['model_provider'],  model['model_reasoning'])
                        if response:
                            validated_response = validation_first_phase(response, len(catalog_portion), [item['catalog_id'] for item in catalog_portion])

                    # Saving data:
                    with open(path.join(raw_folder, f'raw_{filename.split('_')[-1].split('.')[0]}.json'), 'w', encoding='utf-8') as out_file:
                        json.dump(validated_response, out_file, ensure_ascii=False, indent=4)

                    with open(costs_file, 'a', encoding='utf-8') as out_file:
                        out_file.write(f'{str(cost)}\n')
                           
    print('')

    # Raw results processing:
    if action != 'SKIP':
        os.makedirs(processed_folder)
        processing_first_phase(raw_folder, processed_folder)

        with open(final_cost_file, 'w', encoding='utf-8') as file:
            file.write(f'${numpy.sum(numpy.loadtxt(costs_file))}')
                
                

###________________________ MAIN ________________________###

def main():
    
    data_source = 'Catalogo Egizio ITA - Max 5000 Tokens'
    
    model = select_model()
    model_info(model)

    print(f'{Y}PHASE 1{R}\nIdentification and extraction of events with narrative potential from the catalog\n')

    # Folders:
    source_folder = path.join('resources', 'catalog', 'processed', data_source)
    raw_folder = path.join('results', model['model_name'], 'phase_1', 'raw_results')
    processed_folder = path.join('results', model['model_name'], 'phase_1', 'processed_results')

    # Files:
    prompt_file = path.join('resources', 'prompts', 'prompt_phase_1.txt')
    costs_file = path.join('results', model['model_name'], 'phase_1', 'Phase 1 - API Costs Detail.txt')
    final_cost_file = path.join('results', model['model_name'], 'phase_1', 'Phase 1 - API Cost Final.txt')

    files_to_process = select_percentage(source_folder)
    first_phase(model, files_to_process, source_folder, raw_folder, processed_folder, prompt_file, costs_file, final_cost_file)

if __name__ == "__main__":
    main()