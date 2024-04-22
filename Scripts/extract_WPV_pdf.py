# -------------------
# Import packages 
# -------------------

import pandas as pd
from pathlib import Path
import Functions_wpv as functions

# -------------------
# Directory paths
# -------------------
DIRECTORY_PDF = "C:/Users/LukaHoebenVFTalentAC/WPV_Trainees"
DIRECTORY_EXCEL = "C:/Users/LukaHoebenVFTalentAc/OneDrive - Veneficus B.V/Documenten"
DIRECTORY_OUTPUT_PRIVATE = "../Output_private"
DIRECTORY_OUTPUT = "../Output"

# -------------------
# Main Function
# -------------------
def main():

    """
    Main function to process PDF files for WPV Trainees, extract data into various dataframes,
    and merge additional identity information from an Excel file.

    This function handles:
    - Searching for PDF files in a specified directory.
    - Creating and populating dataframes for various types of data extracted from PDF files.
    - Merging data from an Excel file into the existing CSV data framework.
    """

    pdf_search = Path(DIRECTORY_PDF).rglob("*.pdf")
    pdf_files = [str(file.absolute()) for file in pdf_search]

    #indicating the different dataframes which are then exported to csv
    identity_df = pd.DataFrame(columns=['File ID', 'Voornaam', 'Achternaam', 'Volledige naam', 'Groep ID']) #identity_df which contains the given file ID, including the First Name, Last Name, Full Name of the trainee and the group ID. 
    headgroup_df = pd.DataFrame(columns=['File ID', 'Hoofdkenmerk', 'Score']) #headgroup_df contains the File ID per trainee, the score for 'persoonlijkheidskenmerken'. the persoonlijkheidskenmerken are under headgroup
    subgroup_df = pd.DataFrame(columns=['File ID','Hoofdkenmerk', 'Subkenmerk', 'Subscore']) #subgroup_df contains file id, headgroup from headgroup_df and the subgroups are the 'persoonlijheidskenmerken' per headgroup and the score from this subgroup
    competencies_df = pd.DataFrame(columns=['File ID', 'Hoofdcompetentie', 'Subcompetentie', 'Percentage']) #competencies_df contains the percentages per subcompetency and the headgroup(competency) of this subcompetency
    zelfbeeld_df = pd.DataFrame(columns=['File ID', 'Hoofdkenmerk', 'Score']) #zelfbeeld_df contains the score for "zelfbeeld"
    functions.Stenscores_Identity_to_csv(identity_df, headgroup_df, subgroup_df, competencies_df, zelfbeeld_df, pdf_files)

    
    # File path to name_info_wpv.xlsx 
    excel_file_path = f"{DIRECTORY_EXCEL}/name_info_wpv.xlsx" 

    #path to identity_df 
    csv_file_path = f"{DIRECTORY_OUTPUT_PRIVATE}/identity.csv" 
    
    #output info_job.csv
    output_csv_path = f"{DIRECTORY_OUTPUT}/info_job.csv"

    # Call the function which merges name_info_wpv with Identity.csv if name_info or identity_df not present it returns that the file does not exist
    functions.merge_excel_with_csv(output_csv_path, csv_file_path, excel_file_path) 

if __name__ == "__main__":
    main()
