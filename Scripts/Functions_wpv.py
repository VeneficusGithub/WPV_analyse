# -------------------
# Import packages 
# -------------------

import PyPDF2 as pypdf
import re
import pandas as pd
import uuid
import hashlib
import os

# -------------------
# Functions to extract scores 
# -------------------

def generate_uuid_for_pdf(pdf):
    """
    Generates a UUID (Universally Unique Identifier) for a PDF file based on its content hash.

    Parameters:
    - pdf (str): Path to the PDF file.

    Returns:
    - str: UUID generated based on the PDF content hash.

    Example:
    >>> generate_uuid_for_pdf('example.pdf')
    '2c7bcea5-c5cd-467e-bd6f-d2e13604c0c1'
    """
    # Open PDF file in binary read mode
    with open(pdf, "rb") as f: 

        # Reads the content of the PDF file
        pdf_content = f.read() 

        # Calculate the MD5 hash of the content
        hash_value = hashlib.md5(pdf_content).hexdigest() 

        # Generate UUID based on the hash
        return str(uuid.uuid5(uuid.NAMESPACE_OID, hash_value))

# extract_name_from_file extracts the name of a Trainee 
def extract_name_from_file(pdf):
    """
    Extracts the name of a Trainee from the filename of a PDF.

    Parameters:
    - pdf (str): Path to the PDF file.

    Returns:
    - tuple: A tuple containing the first name, last name, full name, and folder name.

    """
    #extracts the filename from the provided path
    filename = os.path.basename(pdf) 

    #extracts the folder name containing the pdf file
    folder_name = os.path.basename(os.path.dirname(pdf)) 

    #pattern to select names, based on first name, last name and last name prefix. Defines a list of regular expressions patterns to match names based on various naming conventions
    name_patterns = [r'\b[A-Z][a-z]+(?:_[A-Z][a-z]+)+|[A-Z][a-z]+(?:__[A-Z][a-z]+)+|[A-Z][a-z]+(?:_[a-z][a-z]+)(?:_[A-Z][a-z]+)+|[A-Z][a-z]+(?:_[a-z][a-z]+)(?:__[A-Z][a-z]+)+|[A-Z][a-z]+(?:__[a-z][a-z]+)(?:_[A-Z][a-z]+)+|[A-Z][a-z]+(?:_[a-z][a-z]+)+(?:_[a-z][a-z]+)(?:_[A-Z][a-z]+)+|[A-Z][a-z]+(?:_[a-z][a-z]+)+(?:_[a-z][a-z]+)(?:__[A-Z][a-z]+)+|[A-Z][a-z]+(?:-[A-Z][a-z]+)(?:_[A-Z][a-z]+)+|[A-Z][a-z]+(?:-[A-Z][a-z]+)(?:_[a-z][a-z]+)(?:_[A-Z][a-z]+)+\b']
    
    #iterates over each pattern to search for a match in the filename
    for pattern in name_patterns: 
        #searches for a match of the current pattern in the filename 
        match = re.search(pattern, filename) 
        if match:
            name_part = match.group() # Extract the matched name part
            full_name = [name for name in name_part.split('_') if name] #splits the matched name into parts and removes empty strings.
            names = ' '.join(full_name) #Join the parts to form the full name
            first_name = full_name[0] # Extract the first name
            last_name = full_name[-1] # Extract the last name
            return first_name, last_name, names, folder_name #returns the first name, last name and folder name

def extract_stenscores_and_traits(text):
    """
    Searches for the stenscores pattern in the text and returns the stenscores in a list.

    Parameters:
    - text (str): Text to search for stenscores.

    Returns:
    - list: List of lists containing stenscores.
    """
    # Pattern to recognize scores after the word Stenscores
    stenscores_pattern = r'Stenscores\s*((?:\d+\s*)+)' 

    # Finds all matches of stenscores pattern in the text
    stenscores_match = re.findall(stenscores_pattern, text, re.IGNORECASE) 

    #initializes an empty list
    stenscores = [] 

    # Iterates over each match found
    for match in stenscores_match: 
        stenscores_string = match
        #splits the matched stenscores string into individual scores, converts them to integers, and adds them to the stenscores_page lists
        stenscores_page = [int(score) for score in stenscores_string.split('\n') if score.strip()] 
        stenscores.append(stenscores_page) #appends the list of stenscores for each match to the stenscores list
    return stenscores


def extract_stenscores_from_pdf(pdf):
    """
    Opens the PDF and extracts stenscores and file ID if there are stenscores found.

    Parameters:
    - pdf (str): Path to the PDF file.

    Returns:
    - tuple or None: A tuple containing file ID and stenscores if found, None otherwise.
    """
    # Generates file ID for the PDF
    file_id = generate_uuid_for_pdf(pdf)
    with open(pdf, "rb") as file: 
        pdf_reader = pypdf.PdfReader(file)
        text = ''
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            if page_number == 1 or page_number == 2:
                continue
            page_text = page.extract_text()
            text += page_text

        # Extract stenscores from the text
        stenscores = extract_stenscores_and_traits(text)
        
        # Return file ID and stenscores if found
        if stenscores:
            return file_id, stenscores

        if not stenscores:
            print("Geen stenscores gevonden in de PDF.")   


def extract_percentages(text):
    """
    Searches for percentages pattern in the text and returns the percentages in a list.

    Parameters:
    - text (str): Text to search for percentages.

    Returns:
    - list: List of lists containing percentages.
    """
    # Pattern to recognize percentages
    percentages_pattern = r'Percentages\s*((?:\d+\s*)+)'

    # Find all matches of percentages pattern in the text
    percentages_match = re.findall(percentages_pattern, text, re.IGNORECASE)

    # Initialize an empty list for percentages
    percentages = []

    # Iterate over each match found
    for match in percentages_match:
        percentages_string = match
        # Split the matched percentages string into individual scores, convert to integer and add them to the percentages_page list
        percentages_page = [int(score) for score in percentages_string.split('\n') if score.strip()]

        # Append the list of percentages for each match to the percentage lsit
        percentages.append(percentages_page)

    return percentages

def extract_left_over_percentages(text_extra):
    """
    Searches for remaining percentages pattern in the additional text and returns the percentages in a list.

    Parameters:
    - text_extra (str): Additional text to search for remaining percentages.

    Returns:
    - list: List of lists containing remaining percentages.
    """
    # Pattern to recognize remaining percentages
    pattern = r'((?:\d+)+)'
    # Find all matches of the pattern in the additional text
    pattern_match = re.findall(pattern, text_extra, re.IGNORECASE)

    # Initialize an empty list for remaining percentages
    left_over_percentages = []

    # Iterate over each match found
    for match in pattern_match:
        string = match
        # Split the matched string into individual scores, convert them to integers, and add them to the match_page list
        match_page = [int(score) for score in string.split('\n') if score.strip()]
        # Append the list of remaining percentages for each match to the left_over_percentages list
        left_over_percentages.append(match_page)

    return left_over_percentages


def extract_competencies_from_pdf(pdf):
    """
    Opens the PDF and extracts percentages and page number where they start.

    Parameters:
    - pdf (str): Path to the PDF file.

    Returns:
    - tuple or None: A tuple containing percentages and the page number where they start if found, None otherwise. 
    """
    # Open PDF
    with open(pdf, "rb") as file: 
        pdf_reader = pypdf.PdfReader(file)
        text = ''
        start_keyword = re.compile(r'.*?Percentages.*?')
    
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            text += page_text

            # Store the page number where "Percentages" are found
            if start_keyword.search(page_text):
                percentages_page_number = page_number   

        # Extract percentages from the text
        percentages = extract_percentages(text)
        
        # Return percentages and the page number where they start
        if percentages:
            return percentages, percentages_page_number
    

        if not percentages:
            print("No percentages found within the specified pages.")

def extract_extra_percentages_from_pdf(pdf, percentages_page_number):
    """
    Opens the PDF and extracts remaining percentages from the next page after the given page number.

    Parameters:
    - pdf (str): Path to the PDF file.
    - percentages_page_number (int): Page number where percentages start.

    Returns:
    - list: List of lists containing remaining percentages.
    """
    with open(pdf, "rb") as file: 
        pdf_readers = pypdf.PdfReader(file)
        text_extra = ''
         
        for page_number, page_extra in enumerate(pdf_readers.pages, start=0):
                 # Check the next page after the given page number
                if page_number == percentages_page_number:
                    page_extra_text = page_extra.extract_text()
                    text_extra += page_extra_text

        # Extract remaining percentages from the additional text
        left_over_percentages = extract_left_over_percentages(text_extra)

    return left_over_percentages

def extract_zelfbeeld(text):
    """
    Extracts the zelfbeeld score from the text using regular expressions.

    Parameters:
    - text (str): Text to search for the zelfbeeld score.

    Returns:
    - tuple or None: A tuple containing the zelfbeeld header and score if found, None otherwise.
    """

    # Use regular expressions to extract the relevant information
    match = re.search(r'([1-9]|10)(?=Zelfbeeld)', text)
    if match:
        zelfbeeldscore = match.group(1)
        zelfbeeldhead = 'Zelfbeeld'
        return (zelfbeeldhead, zelfbeeldscore)
    else:
        return None

def read_zelfbeeld(pdf):
    """
    Reads the PDF and extracts the zelfbeeld score.

    Parameters:
    - pdf (str): Path to the PDF file.

    Returns:
    - tuple or None: A tuple containing the zelfbeeld header and score if found, None otherwise.

    """
    with open(pdf, "rb") as file: 
        pdf_reader = pypdf.PdfReader(file)
        text = ''
        for page_number, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            text += page_text

        # Extract zelfbeeld score using extract_zelfbeeld function
        zelfbeeldhead, zelfbeeldscore = extract_zelfbeeld(text)

    return zelfbeeldhead, zelfbeeldscore

# -------------------
# Functions to use all above functions and put the results in multiple CSV files
# -------------------

def Stenscores_Identity_to_csv(identity_df, headgroup_df, subgroup_df, competencies_df, zelfbeeld_df, pdf_files): 
    """
    Extracts data from PDF files and writes it to CSV files.

    Parameters:
    - identity_df (DataFrame): DataFrame containing identity information.
    - headgroup_df (DataFrame): DataFrame containing headgroup information.
    - subgroup_df (DataFrame): DataFrame containing subgroup information.
    - competencies_df (DataFrame): DataFrame containing competencies information.
    - zelfbeeld_df (DataFrame): DataFrame containing zelfbeeld information.
    - pdf_files (list): List of PDF files to process.

    Example:
    >>> Stenscores_Identity_to_csv(identity_df, headgroup_df, subgroup_df, competencies_df, zelfbeeld_df, pdf_files)
    """
    
    output_dir = '../Output'
    output_dir_private = "../Output_private"
    os.makedirs(output_dir, exist_ok=True)

    # Write data to CSV
    identity_df.to_csv(os.path.join(output_dir_private,'Identity.csv'), index=False)
    headgroup_df.to_csv(os.path.join(output_dir,'Headgroup.csv'), index=False) 
    subgroup_df.to_csv(os.path.join(output_dir,'Subgroup.csv'), index=False)
    competencies_df.to_csv(os.path.join(output_dir,'Competencies.csv'), index=False)
    zelfbeeld_df.to_csv(os.path.join(output_dir,'Zelfbeeld.csv'), index=False)
      
    #Looping extraction for all files in folder, adding ID, writing to CSV
    for pdf in pdf_files:
        # Extract name information from file
        first_name, last_name, names, folder_name = extract_name_from_file(pdf)
        
         # Extract stenscores from PDF
        file_id, stenscores = extract_stenscores_from_pdf(pdf)

        # Extract competencies from PDF
        percentages, percentages_page_number = extract_competencies_from_pdf(pdf)

        # Extract additional percentages from PDF
        left_over_percentages = extract_extra_percentages_from_pdf(pdf, percentages_page_number)
        
        # Flatten the list of left-over percentages
        left_over_percentages_list = []
        for sublist in left_over_percentages:
            left_over_percentages_list.extend(sublist)
        left_over_percentages_list = left_over_percentages_list[:-2] # Remove last two elements

        # Extract zelfbeeld score from PDF
        zelfbeeldhead, zelfbeeldscore = read_zelfbeeld(pdf)

        # Handle different stenscores layout scenarios
        matches_list1 = len(stenscores) == 1 and all(isinstance(sublist, list) for sublist in stenscores)
        matches_list2 = len(stenscores) == 5

        if matches_list1 and not matches_list2:
            group_lengths = [5, 7, 7, 6, 5]
            stenscores=stenscores[0]
            result = []
            start_index = 0
            for length in group_lengths:
                group = stenscores[start_index:start_index+length]
                result.append(group)
                start_index+= length
        elif matches_list2:
            result = stenscores
        else:
            print("Layout komt niet overeen met de aangegeven lijsten.")
        
        # Define personality traits and their structure
        persoonlijkheidskenmerken = [["Invloed", "Status", "Dominantie", "Competitie", "Zelfvertoon"],
                                ["Sociabiliteit", "Contactbehoefte", "Sociaal Ontspannen", "Zelfonthulling", "Vertrouwen", "Hartelijkheid", "Zorgzaamheid"],
                                ["Gedrevenheid", "Energie", "Zelfontwikkeling", "Volharding", "Variatiebehoefte", "Originaliteit", "Onafhankelijkheid"],
                                ["Structuur", "Ordelijkheid", "Nauwkeurigheid", "Regelmaat", "Conformisme", "Weloverwogen"],
                                ["Stabiliteit", "Zelfvertrouwen", "Positivisme", "Frustratietolerantie", "Incasseringsvermogen"]]
        # Create dictionary of personality traits
        personality_traits = {}
        for headgroup, scores, subgroups in zip(persoonlijkheidskenmerken, result, range(1, len(result) + 1)):
            head = headgroup[0]
            head_score = scores[0]
            subs = headgroup[1:]
            personality_traits[head] = {"Score": head_score, "Subgroups": dict(zip(subs, scores[1:]))}

       
        # Define competencies and their structure
        compgroup_lengths = [11, 4, 5, 3, 6]

        total_length = sum(compgroup_lengths)
        left_over_count = total_length - len(percentages)
       
        # Process competencies data
        for numbers in percentages:
            if len(numbers) != total_length:
                percentages = percentages[0]
                resultcomp = []
                start_index = 0
                for length in compgroup_lengths:
                    compgroup = percentages[start_index:start_index+length]
                    resultcomp.append(compgroup)
                    start_index+= length
                if left_over_count > 0:
                    resultcomp_min = resultcomp[-1]
                    resultcomp[-1]= resultcomp_min + left_over_percentages_list                    
                else:
                    left_over_percentages_list = []
                        
            if len(numbers) == total_length:
                percentages = percentages[0]
                resultcomp = []
                start_index = 0
                for length in compgroup_lengths:
                    compgroup = percentages[start_index:start_index+length]
                    resultcomp.append(compgroup)
                    start_index+= length
                
            
         # Define competencies and their structure       
        competencies_list = [["Persoonlijke gerichtheid", "Initiatief", "Besluitvaardigheid", "Flexibiliteit", "Stressbestendigheid", "Ambitie", "Zelfstandigheid", "Doorzettingsvermogen", "Resultaatgerichtheid", "Leerbereidheid", "Inzet", "Nauwkeurigheid"],
                        ["Organisatiegerichtheid", "Organisatiesensitiviteit", "Klantoriëntatie", "Kwaliteitsgerichtheid", "Ondernemerschap"],
                        ["Beïnvloedend vermogen", "Overtuigingskracht", "Aansturen van groepen", "Aansturen van individuen", "Coachen van medewerkers", "Onderhandelen"],
                        ["Organisatie vermogen", "Plannen en organiseren", "Delegeren", "Voortgangscontrole"],
                        ["Relationeel vermogen", "Sensitiviteit", "Samenwerking", "Optreden", "Relatiebeheer", "Sociabiliteit", "Assertiviteit"]]
        
        # Create dictionary of competencies
        competencies_dict = {}
        for competencies, compscores, subcompetencies in zip(competencies_list, resultcomp, range(1, len(resultcomp) + 1)):
            headcomp = competencies[0]
            subcompetencies = competencies[1:]
            competencies_dict[headcomp] = {"Subcompetencies": dict(zip(subcompetencies, compscores))}

        # Create DataFrame for ID and names
        identity_df = pd.DataFrame([(file_id, first_name, last_name, names, folder_name)], columns=['File ID', 'Voornaam', 'Achternaam', 'Volledige naam', 'Groep ID'])
        identity_df.to_csv(os.path.join(output_dir_private, 'Identity.csv'), mode='a', index=False, header = False)

        # Create DataFrame for headgroups and scores
        headgroup_df = pd.DataFrame([(file_id, head, data['Score']) for head, data in personality_traits.items()], columns=['File ID', 'Hoofdkenmerk', 'Score'])
        headgroup_df.to_csv(os.path.join(output_dir,'Headgroup.csv'), mode='a', index=False, header = False)

        # Create DataFrame for subgroups and subscores
        subgroup_data = [(file_id, head, subgroup, score) for head, data in personality_traits.items() for subgroup, score in data['Subgroups'].items()]
        subgroup_df = pd.DataFrame(subgroup_data, columns=['File ID','Hoofdkenmerk', 'Subkenmerk', 'Subscore'])
        subgroup_df.to_csv(os.path.join(output_dir, 'Subgroup.csv'), mode='a', index=False, header=False)

        # Create DataFrame for competencies
        competencies_data = [(file_id, headcomp, subcompetencies, compscores) for headcomp, data in competencies_dict.items() for subcompetencies, compscores in data['Subcompetencies'].items()]
        competencies_df = pd.DataFrame(competencies_data, columns=['File ID', 'Hoofdcompetentie', 'Subcompetentie', 'Percentage'])
        competencies_df.to_csv('../Output/Competencies.csv', mode='a', index=False, header=False)

        # Create DataFrame for ID and names
        zelfbeeld_df = pd.DataFrame([(file_id, zelfbeeldhead, zelfbeeldscore)], columns=['File ID', 'Hoofdkenmerk', 'Score'])
        zelfbeeld_df.to_csv(os.path.join(output_dir, 'Zelfbeeld.csv'), mode='a', index=False, header = False)


def merge_excel_with_csv(output_csv_path, csv_file_path=None, excel_file_path=None):
    """
    Merges data from an Excel file with data from a CSV file based on a common column 'Volledige naam',
    and writes the merged data to a new CSV file.

    Parameters:
    - output_csv_path (str): Output CSV file path for merged data.
    - csv_file_path (str): Input CSV file path.
    - excel_file_path (str): Input Excel file path.

    Example:
    >>> merge_excel_with_csv("merged_data.csv", "identity.csv", "name_info.xlsx")
    """

    # Ensures at least one file path is provided
    if not csv_file_path or not excel_file_path:
        if not csv_file_path:
            print("CSV file path not provided")
        if not excel_file_path:
            print("Excel file path not provided")
        print("Both file paths must be provided to proceed.")
        return
    
    # Read Excel and CSV files
    df = pd.read_excel(excel_file_path)
    csv_df = pd.read_csv(csv_file_path)
    
    # Merge dataframes
    merged_df = pd.merge(csv_df, df, on="Volledige naam", how="outer")
    
    # Select specific columns
    transformed_data = merged_df[["File ID", "Groep ID", "Richting"]]
    
    # Set output directory
    output_dir = os.path.dirname(output_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Write to CSV
    transformed_data.to_csv(output_csv_path, index=False)
    print(f"Merged data has been saved to {output_csv_path}")

