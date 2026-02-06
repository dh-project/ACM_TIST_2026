import json
import numpy
import os
import os.path as path
import pandas
import tiktoken

# Print:
from colorama import Fore, Style
Y = Fore.YELLOW
C = Fore.CYAN
R = Style.RESET_ALL

def catalog_filtering(dataframe, filtered_folder):
    
    # Folder Creation:
    os.makedirs(filtered_folder, exist_ok=True)
    
    # Rows without 'Description' or 'Curator_Description':
    rows_original = len(dataframe)
    dataframe = dataframe.dropna(subset = ['Description', 'Curator_Description'], how='all')
    rows_no_description = rows_original - len(dataframe)

    # Rows with technical description:
    rows_before = len(dataframe)
    dataframe = dataframe[~dataframe['Type_Description'].isin(['Informazioni tecniche', 'Informazioni tecnologiche', 'Note tecniche', 'Informazioni teniche', 'Informazioni tecnniche'])]
    rows_technical_description = rows_before - len(dataframe)

    # Normalizing description types:
    corrections = {
        numpy.nan: 'Descrizione Catalogo',
        'Descrizione catalogo mummie animali Torino': 'Descrizione Catalogo',
        'Descrizione per sito collezioni': 'Descrizione Catalogo',
        'Descrizione': 'Descrizione Catalogo',
        'Descrizione  dettagliata': 'Descrizione dettagliata',
        'Decsrizione breve': 'Descrizione breve',
        'Descrizione breve ': 'Descrizione breve'
    }

    dataframe['Type_Description'] = dataframe['Type_Description'].replace(corrections)

    # Rows with redundant and less informative description:
    rows_before = len(dataframe)
    dataframe['Priority'] = dataframe['Type_Description'].map({'Descrizione dettagliata': 1, 'Descrizione breve': 2, 'Descrizione Catalogo': 3}).fillna(4)
    dataframe = dataframe.sort_values(by=['Inv', 'Priority'], ascending=[True, True])
    dataframe = dataframe.drop_duplicates(subset=['Inv'], keep='first')
    rows_redundant = rows_before - len(dataframe)

    # Rows with duplicate descriptions:
    rows_before = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset = [col for col in dataframe.columns if col != 'Inv'], keep = 'first')
    rows_duplicates = rows_before - len(dataframe)

    # Rows with a non meaningful description:
    rows_before = len(dataframe)
    words_description = dataframe['Description'].fillna('').str.split().str.len()
    words_curator_description = dataframe['Curator_Description'].fillna('').str.split().str.len()
    dataframe = dataframe[(words_description >= 30) | (words_curator_description >= 30)]
    row_short_description = rows_before - len(dataframe)

    # Statistics:
    print(f"""
          
{Y}CATALOG FILTERING{R}
          
Rows in the original dataset: {C}{rows_original}{R}
Rows in the filtered dataset: {C}{len(dataframe)}{R}

Removals:
• {C}{rows_no_description}{R} rows removed due to missing values in both Description and Curator_Description
• {C}{rows_technical_description}{R} rows removed due to technical description
• {C}{rows_redundant}{R} rows removed due to redundant descriptions
• {C}{rows_duplicates}{R} rows removed due to duplicate descriptions
• {C}{row_short_description}{R} rows removed due to non-meaningful descriptions
    """)

    # Columns adjustments:
    dataframe = dataframe.rename(columns={'Inv': 'Catalog_ID'})
    dataframe = dataframe.drop(columns=['ObjectName', 'TitleDescription', 'Material', 'Date_Start', 'Date_End', 'Dynasty', 'Pharaoh', 'Acquisition', 'Type_Description', 'Priority', 'Epoch', 'Provenance'])

    # Error adjustments:
    dataframe['Catalog_ID'] = dataframe['Catalog_ID'].str.replace('  v.n.', ' v.n.', regex=False)

    # Adjustments for JSON conversion:
    dataframe = dataframe.where(pandas.notna(dataframe), None)
    dataframe.columns = dataframe.columns.str.lower()

    file_path = path.join(filtered_folder,'Catalogo Egizio Filtered.xlsx')

    if path.exists(file_path):
        os.remove(file_path)

    dataframe.to_excel(file_path, index=False)

    return dataframe


def catalog_conversion(dataframe, processed_folder, max_tokens):
    
    # Folder Creation:
    os.makedirs(processed_folder, exist_ok=True)
    
    encoder = tiktoken.get_encoding('cl100k_base')
    
    # JSON Conversion:
    dataframe = dataframe.replace('"', "'", regex=True) 
    data = dataframe.to_dict(orient='records') 

    current_count = 0
    current_chunk = []
    file_counter = 1

    for record in data:
        
        current_tokens = len(encoder.encode(json.dumps(record, ensure_ascii=False)))

        # Chuck - Full with addition:
        if current_count + current_tokens > max_tokens:
            
            if current_chunk:
            
                with open(path.join(processed_folder, f'catalog_{file_counter:04d}.json'), 'w', encoding='utf-8') as file:
                    json.dump(current_chunk, file, ensure_ascii=False, indent=4)
                    
                file_counter += 1
            
            current_chunk = [record]
            current_count = current_tokens

        # Chuck - Space available:
        else:
            current_chunk.append(record)
            current_count += current_tokens
            
    if current_chunk:
        with open(path.join(processed_folder, f'catalog_{file_counter:04d}.json'), 'w', encoding='utf-8') as file:
            json.dump(current_chunk, file, ensure_ascii=False, indent=4)
      

def catalog_ids(dataframe, filtered_folder):

    # JSON Conversion:
    dataframe = dataframe.replace('"', "'", regex=True) 
    data = dataframe.set_index('catalog_id').T.to_dict() 
    
    with open(path.join(filtered_folder, 'filtered_catalog.json'), 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    
    sheet_name = 'Estraz_ITA'
    original_catalog = path.join('story_generation', 'resources', 'catalog', 'original', 'Catalogo Egizio.xlsx')
    filtered_folder = path.join('story_generation', 'resources', 'catalog', 'filtered')
    processed_folder = path.join('story_generation', 'resources', 'catalog', 'processed')
    
    tokens_per_file = 5000    
    processed_name = f'Catalogo Egizio ITA - Max {tokens_per_file} Tokens'

    # Catalog ITA:
    dataframe_ita = pandas.read_excel(original_catalog, sheet_name=sheet_name)
    dataframe_ita_filtered = catalog_filtering(dataframe_ita, filtered_folder)
    catalog_ids(dataframe_ita_filtered, filtered_folder)

    # Conversion:
    catalog_conversion(dataframe_ita_filtered, path.join(processed_folder,processed_name), tokens_per_file)
    

if __name__ == "__main__":
    main()