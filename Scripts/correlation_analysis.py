#%%
# -------------------
# Import packages 
# -------------------

import pandas as pd
import functions_correlation_clusters as correlation
import json
import sys
import os



def load_configuration(config_path):
    """
    Load configuration from a JSON file and adjust file paths based on the environment.

    Parameters:
    - config_path (str): Path to the configuration JSON file.

    Returns:
    - config t: Modified configuration parameters.
    """
    # Load configuration from JSON file
    with open(config_path) as f:
        config = json.load(f)
    
    # Modify paths based on environment
    if not hasattr(sys, 'ps1'):  # Not running in interactive mode
        # Get the directory of the configuration file
        config_dir = os.path.dirname(os.path.abspath(config_path))

        # Modify all file paths in config
        for section in config:
            for key, value in config[section].items():
                if isinstance(value, str) and value.startswith('../'):
                    config[section][key] = os.path.normpath(os.path.join(config_dir, value))
    
    return config


def main():
    """
    Main function to perform correlation analysis using the provided configuration.

    """
    # Define the default configuration file path
    DEFAULT_CONFIG_PATH = '../Scripts/config.json'

    # Check if running in interactive mode or terminal
    if hasattr(sys, 'ps1'):  # Running in interactive mode
        config_path = DEFAULT_CONFIG_PATH
    else:  # Running in terminal
        if len(sys.argv) < 2:
            print("Usage: python correlation_analysis.py <config_file_path>")
            sys.exit(1)
        config_path = sys.argv[1]

    # Load configuration
    config = load_configuration(config_path)


    # Access configuration parameters
    file_paths = config["file_paths"] #input files
    correlation_base = config["correlation_base"] #output correlation subkenmerken of competenties
    correlation_per_direction = config["correlation_per_direction"] #output correlation per richting
    correlation_difference = config["correlation_difference"] #output correlation verschil tussen richtingen
    outliers = config["outliers"] #outliers
    correlation_without_outliers = config["correlation_without_outliers"] #correlatie zonder de outliers
    correlation_difference_outliers = config["correlation_difference_outliers"]  #correlatie verschil voor of na outliers

    # Access file paths from each section
    csv_headgroup = file_paths["csv_headgroup"] #input headgroup scores, scores per hoofdkenmerk per persoon(File ID)
    csv_subgroup = file_paths["csv_subgroup"] #input subkenmerken
    csv_competencies = file_paths["csv_competencies"] #input competenties
    csv_info = file_paths["csv_info"] #input info per trainee (groep, richting)
    pdf_path_hoofd = correlation_base["pdf_path_hoofd"] #pdf correlatie hoofdkenmerken
    csv_sub_cor_file = correlation_base["csv_sub_cor_file"] #output correlatie subkenmerken
    csv_sub_cor_filtered = correlation_base["csv_sub_cor_filtered"] #correlatie tussen subkenmerken waarbij de correlatie > 0.6 is (absoluut)
    pdf_path_sub = correlation_base["pdf_path_sub"] #pdf met 3 heatmaps van subkenmerken, 1. gefiltered op >0.6 2. alles met labels. 3. alles zonder labels
    csv_comp_cor_file = correlation_base["csv_comp_cor_file"]   #correlatie tussen subcompetenties
    csv_comp_cor_filtered = correlation_base["csv_comp_cor_filtered"] #correlatie tussen subcompetenties waarbij de correlatie > 0.6 is (absoluut)
    pdf_path_comp = correlation_base["pdf_path_comp"] #pdf met 3 heatmaps van competenties, 1. gefiltered op >0.6 2. alles met labels. 3. alles zonder labels

    csv_all_richtingen = correlation_per_direction["csv_all_richtingen"] # correlatie tussen subkenmerken gesplitst per Richting (Analist, Engineer, PO, Scientist)
    csv_sub_cor_filtered_richting = correlation_per_direction["csv_sub_cor_filtered_richting"] # correlatie tussen subkenmerken gesplitst per Richting, waarbij correlatie > 0.6 is
    pdf_richting_sub = correlation_per_direction["pdf_richting_sub"] #pdf correlatie subkenmerken per richting
    csv_all_richtingen_comp = correlation_per_direction["csv_all_richtingen_comp"] #correlatie tussen subcompetenties gesplitst per Richting
    csv_comp_cor_filtered_richting = correlation_per_direction["csv_comp_cor_filtered_richting"] #correlatie tussen subcompetenties gesplitst per Richting, waarbij correlatie > 0.6 is
    pdf_richting_comp = correlation_per_direction["pdf_richting_comp"] #pdf correlatie competentie per richting

    csv_correlation_richting = correlation_difference["csv_correlation_richting"] #correlatie verschil tussen Analist en Engineer per combinatie van 2 subkenmerken
    csv_correlation_richting_filtered = correlation_difference["csv_correlation_richting_filtered"] #De combinaties van correlation_difference.csv die een P_val hebben van < 0.05 = significant verschil
    csv_correlation_richting_comp = correlation_difference["csv_correlation_richting_comp"] #correlatie verschil tussen Analist en Engineer per combinatie van 2 subcompetenties 
    csv_correlation_richting_filtered_comp = correlation_difference["csv_correlation_richting_filtered_comp"] #De combinaties van correlation_difference_comp.csv die een P_val hebben van < 0.05 = significant verschil

    csv_filtered_outliers_subkenmerken = outliers["csv_filtered_outliers_subkenmerken"] #correlatie voor alles behalve (excluded_id), waarbij verschil tussen correlatie met en correlatie zonder excluded_id de contributie tot totale correlatie geeft. Stdev berekent en dit zijn de excluded_ids > 2x stdev (threshold)
    csv_filtered_outliers_comp = outliers["csv_filtered_outliers_comp"] #correlatie voor alles behalve (excluded_id), waarbij verschil tussen correlatie met en correlatie zonder excluded_id de contributie tot totale correlatie geeft. Stdev berekent en dit zijn de excluded_ids > 2x stdev (threshold)
    csv_filtered_outliers_subkenmerken_Analist = outliers["csv_filtered_outliers_subkenmerken_Analist"] #Hetzelfde als filtered_subkenmerken_outliers.csv dat berekent op basis van Analist alleen. 
    csv_filtered_outliers_subkenmerken_Engineer = outliers["csv_filtered_outliers_subkenmerken_Engineer"] #Hetzelfde als filtered_subkenmerken_outliers.csv dat berekent op basis van Engineer alleen. 
    csv_filtered_outliers_comp_Analist = outliers["csv_filtered_outliers_comp_Analist"] #Hetzelfde als filtered_comp_outliers.csv dat berekent op basis van Analist alleen. 
    csv_filtered_outliers_comp_Engineer = outliers["csv_filtered_outliers_comp_Engineer"] #Hetzelfde als filtered_comp_outliers.csv dat berekent op basis van Engineer alleen. 

    csv_correlation_without_outliers_subkenmerken = correlation_without_outliers["csv_correlation_without_outliers_subkenmerken"] #correlatie berekent per combinatie subkenmerken, waarbij de subscores van de outliers niet zijn meegenomen. 
    pdf_outliers_sub = correlation_without_outliers["pdf_outliers_sub"] #heatmap correlatie zonder outliers, subkenmerken
    csv_correlation_without_outliers_subkenmerken_A = correlation_without_outliers["csv_correlation_without_outliers_subkenmerken_A"] #correlatie berekent per combinatie subkenmerken, waarbij de subscores van de outliers niet zijn meegenomen, Voor Analist alleen
    csv_correlation_without_outliers_subkenmerken_E = correlation_without_outliers["csv_correlation_without_outliers_subkenmerken_E"] #correlatie berekent per combinatie subkenmerken, waarbij de subscores van de outliers niet zijn meegenomen, Voor Engineer alleen
    pdf_outliers_sub_A = correlation_without_outliers["pdf_outliers_sub_A"] #heatmap correlatie zonder outliers Analist, subkenmerken
    pdf_outliers_sub_E = correlation_without_outliers["pdf_outliers_sub_E"] #heatmap correlatie zonder outliers Engineer, subkenmerken

    csv_correlation_without_outliers_comp = correlation_without_outliers["csv_correlation_without_outliers_comp"] #correlatie berekent per combinatie subcompetenties, waarbij de percentages van de outliers niet zijn meegenomen
    pdf_outliers_comp = correlation_without_outliers["pdf_outliers_comp"] #heatmap correlatie zonder outliers, competenties
    csv_correlation_without_outliers_comp_A = correlation_without_outliers["csv_correlation_without_outliers_comp_A"] #correlatie berekent per combinatie subcompetenties, waarbij de percentages van de outliers niet zijn meegenomen, Analist alleen
    csv_correlation_without_outliers_comp_E = correlation_without_outliers["csv_correlation_without_outliers_comp_E"] #correlatie berekent per combinatie subcompetenties, waarbij de percentages van de outliers niet zijn meegenomen, Engineer alleen
    pdf_outliers_comp_A = correlation_without_outliers["pdf_outliers_comp_A"] #heatmap correlatie zonder outliers Analist, competenties
    pdf_outliers_comp_E = correlation_without_outliers["pdf_outliers_comp_E"] #heatmap correlatie zonder outliers Engineer, competenties

    csv_correlation_richting_without_outliers = correlation_difference_outliers["csv_correlation_richting_without_outliers"] #verschil tussen correlatie Analist en Engineer, zonder outliers, met p value, subkenmerken
    csv_correlation_richting_without_outliers_comp = correlation_difference_outliers["csv_correlation_richting_without_outliers_comp"] #verschil tussen correlatie Analist en Engineer, zonder outliers, met p value, competenties
    csv_correlation_difference_outlier = correlation_difference_outliers["csv_correlation_difference_outlier"] #verschil tussen totale correlatie met en zonder outliers, met p value, subkenmerken
    csv_correlation_difference_outlier_comp = correlation_difference_outliers["csv_correlation_difference_outlier_comp"] #verschil tussen totale correlatie met en zonder outliers, met p value, competenties
    
    #calculation for all correlations Hoofdkenmerk 
    correlation.correlation_hoofdkenmerk(csv_headgroup, pdf_path_hoofd, output_to_pdf=True) #correlation headgroup = 'hoofdkenmerken'

    # Subkenmerken
    # -------------------  
    df_cor = correlation.correlation_total(csv_subgroup, csv_sub_cor_file, csv_sub_cor_filtered,'Subkenmerk', 'Subscore', 'Hoofdkenmerk', pdf_path_sub) #correlatie subgroup = "subkenmerken"
    
    #correlation per function  "Richting"
    df_all_richtingen = correlation.correlation_per_function(csv_subgroup, csv_info, csv_all_richtingen, csv_sub_cor_filtered_richting, 'Subkenmerk', 'Subscore', 'Hoofdkenmerk', pdf_richting_sub) #berekent de correlatie per richting 
    correlation.compare_richtingen(df_all_richtingen, csv_correlation_richting, csv_correlation_richting_filtered, "Subkenmerk", "Hoofdkenmerk")  # amount of Trainees per richting is hardcoded -> change this - vergelijkt de correlatie van Analist en Engineer
    
    #outliers
    subgroup_df = pd.read_csv(csv_subgroup) #opnieuw input subgroup_df(buiten de gegeven functies)
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(subgroup_df, 2, csv_filtered_outliers_subkenmerken, 'Subkenmerk', 'Subscore') #berekent de contributie van een persoon tot de correlatie. output is de outliers
    correlation_df= correlation.correlation_without_outliers(subgroup_df, df_outliers, csv_correlation_without_outliers_subkenmerken, 'Subkenmerk', 'Subscore', 'Hoofdkenmerk', pdf_outliers_sub) # de correlatie zonder de outliers
    correlation.compare_correlation_after_outliers(df_cor, correlation_df, csv_correlation_difference_outlier, "Subkenmerk") #verschil correlatie voor en na outliers
    
    #outlier analysis per richting
    info_job_df = pd.read_csv(csv_info) #informatie per persoon(richting)
    merged_df = pd.merge(subgroup_df, info_job_df, on= "File ID", how = 'inner') #voegt de informatie per persoon toe aan de subgroup_df
    subgroup_A = merged_df[merged_df["Richting"] == "Analist"] #kiest alleen de Analisten
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(subgroup_A, 2, csv_filtered_outliers_subkenmerken_Analist, 'Subkenmerk', 'Subscore') #berekent de contributie van een persoon tot de correlatie, van alleen analisten
    correlation_df_A = correlation.correlation_without_outliers(subgroup_A, df_outliers, csv_correlation_without_outliers_subkenmerken_A, 'Subkenmerk', 'Subscore', 'Hoofdkenmerk', pdf_outliers_sub_A) #correlatie zonder de outliers(Analisten)

    subgroup_E = merged_df[merged_df["Richting"] == "Engineer"] #filtert op Engineers
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(subgroup_E, 2, csv_filtered_outliers_subkenmerken_Engineer, 'Subkenmerk', 'Subscore') #berekent de contributie van een persoon tot de correlatie, van alleen Engineers
    correlation_df_E = correlation.correlation_without_outliers(subgroup_E, df_outliers, csv_correlation_without_outliers_subkenmerken_E, 'Subkenmerk', 'Subscore', 'Hoofdkenmerk', pdf_outliers_sub_E) #correlatie zonder de outliers(Engineers)
    correlation.compare_richtingen_zonder_outliers(correlation_df_A, correlation_df_E, csv_correlation_richting_without_outliers) #verschil tussen Analist en Engineer in correlatie zonder outliers. 


    # Competenties 
    # -------------------  
    df_cor = correlation.correlation_total(csv_competencies, csv_comp_cor_file, csv_comp_cor_filtered,'Subcompetentie', 'Percentage', 'Hoofdcompetentie', pdf_path_comp) #correlatie competentie 
    
    #correlation per function "Richting"  
    df_all_richtingen = correlation.correlation_per_function(csv_competencies, csv_info, csv_all_richtingen_comp, csv_comp_cor_filtered_richting, 'Subcompetentie', 'Percentage', 'Hoofdcompetentie', pdf_richting_comp) #berekent de correlatie per richting
    correlation.compare_richtingen(df_all_richtingen, csv_correlation_richting_comp, csv_correlation_richting_filtered_comp, "Subcompetentie", "Hoofdcompetentie") #vergelijkt de correlatie van Analist en Engineer
    
    #Outliers    
    competencies_df = pd.read_csv(csv_competencies)#opnieuw input competencies_df(buiten de gegeven functies)
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(competencies_df, 2, csv_filtered_outliers_comp, 'Subcompetentie', 'Percentage') #berekent de contributie van een persoon tot de correlatie. output is de outliers
    correlation_df = correlation.correlation_without_outliers(competencies_df, df_outliers, csv_correlation_without_outliers_comp, 'Subcompetentie', 'Percentage', 'Hoofdcompetentie', pdf_outliers_comp) #correlatie zonder de outliers
    correlation.compare_correlation_after_outliers(df_cor, correlation_df, csv_correlation_difference_outlier_comp, "Subcompetentie") # total of trainees is hardcoded in function = NEEDS TO BE CHANGED BY HAND, #verschil correlatie voor en na outliers
    
    #outlier analysis per richting
    merged_df = pd.merge(competencies_df, info_job_df, on= "File ID", how = 'inner') #voegt de informatie per persoon toe aan de subgroup_df
    competencies_df1 = merged_df[merged_df["Richting"] == "Analist"] #kiest alleen de Analisten
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(competencies_df1, 2, csv_filtered_outliers_comp_Analist, 'Subcompetentie', 'Percentage') #berekent de contributie van een persoon tot de correlatie, van alleen analisten
    correlation_df_A = correlation.correlation_without_outliers(competencies_df1, df_outliers, csv_correlation_without_outliers_comp_A, 'Subcompetentie', 'Percentage', 'Hoofdcompetentie', pdf_outliers_comp_A) #correlatie zonder de outliers (Analisten)

    competencies_df2 = merged_df[merged_df["Richting"] == "Engineer"] #kiest alleen de Engineers
    df_outliers, base_df = correlation.calculate_correlation_excluding_individual(competencies_df2, 2, csv_filtered_outliers_comp_Engineer, 'Subcompetentie', 'Percentage') #berekent de contributie van een persoon tot de correlatie, van alleen Engineers
    correlation_df_E = correlation.correlation_without_outliers(competencies_df2, df_outliers, csv_correlation_without_outliers_comp_E, 'Subcompetentie', 'Percentage', 'Hoofdcompetentie', pdf_outliers_comp_E) #correlatie zonder de outliers (Engineers)
    correlation.compare_richtingen_zonder_outliers(correlation_df_A, correlation_df_E, csv_correlation_richting_without_outliers_comp) #verschil tussen Analist en Engineer in correlatie zonder outliers. 

if __name__ == "__main__":
    main()






