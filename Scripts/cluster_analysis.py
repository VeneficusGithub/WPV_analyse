#import functions
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

#main function that calculates different cluster analysis, also a part that is based on interpetation and only needed values with current data
def main():
    DEFAULT_CONFIG_PATH = '../Scripts/config.json'

    # Check if running in interactive mode or terminal
    if hasattr(sys, 'ps1'):  # Running in interactive mode
        config_path = DEFAULT_CONFIG_PATH
    else:  # Running in terminal
        if len(sys.argv) < 2:
            print("Usage: python correlation_analysis.py <config_file_path>")
            sys.exit(1)
        config_path = sys.argv[1]

    config = load_configuration(config_path)

    # Access configuration parameters
    file_paths = config["file_paths"] #input files
    clusters = config["clusters"] 

    # Access file paths from each section
    csv_headgroup = file_paths["csv_headgroup"] #input headgroup scores, scores per hoofdkenmerk per persoon(File ID)
    csv_subgroup = file_paths["csv_subgroup"] #input subkenmerken
    csv_competencies = file_paths["csv_competencies"] #input competenties
    pdf_amount_clusters = clusters["pdf_amount_clusters"] #pdf beste aantal clusters (hoofdkenmerken, competenties, subkenmerken)
    pdf_clusters_total = clusters["pdf_clusters_total"] #heatmap van scores per persoon ingedeeld op clusters (2) voor hoofdkenmerken, competenties, subkenmerken
    pdf_amount_clusters_kenmerken = clusters["pdf_amount_clusters_kenmerken"] #beste aantal clusters voor subkenmerken (per hoofdkenmerk)
    pdf_amount_clusters_competencies = clusters["pdf_amount_clusters_competencies"] #beste aantal clusters voor subcompetenties (per hoofdcompetentie)
    pdf_clusters_kenmerken = clusters["pdf_clusters_kenmerken"] #heatmap van scores per persoon ingedeeld per clusters (2) voor elk hoofdkenmerk 
    pdf_clusters_competencies = clusters["pdf_clusters_competencies"] #heatmap van scores per persoon ingedeeld per clusters (2) voor elk hoofdcompetentie
    pdf_clusters_kenmerken_structuur = clusters["pdf_clusters_kenmerken_structuur"] #heatmap van scores per persoon ingedeeld per clusters voor het hoofdkenmerk structuur (7 clusters)
    pdf_clusters_competencies_organisatiegerichtheid = clusters["pdf_clusters_competencies_organisatiegerichtheid"] #heatmap van scores per persoon ingedeeld per clusters voor hoofdcompetente organisatiegerichtheid (9 clusters)
    csv_clusters_kenmerken = clusters["csv_clusters_kenmerken"] #csv met persoon en cluster nummer voor alle hoofdkenmerken (2 clusters)
    csv_clusters_competencies = clusters["csv_clusters_competencies"] #csv met persoon en cluster nummer voor alle hoofdcompetentie (2 clusters)
    csv_clusters_kenmerken_structuur = clusters["csv_clusters_kenmerken_structuur"] #csv met persoon en cluster nummer voor het hoofdkenmerk structuur (7 clusters)
    csv_clusters_competencies_organisatiegerichtheid = clusters["csv_clusters_competencies_organisatiegerichtheid"] #csv met persoon en cluster nummer voor hoofdcompetentie organisatiegerichtheid (9 clusters)
    
    #Cluster analysis total
    fig1 = correlation.amount_of_clusters(csv_headgroup, 'Hoofdkenmerk', 'Score') #determines optimal number of students for classifying trainees - hoofdkenmerken
    fig2 = correlation.amount_of_clusters(csv_subgroup, 'Subkenmerk', 'Subscore') #determines optimal number of students for classifying trainees - subkenmerken
    fig3 = correlation.amount_of_clusters(csv_competencies,'Subcompetentie', 'Percentage') #determines optimal number of students for classifying trainees - competenties

    # merging the figures together in pdf
    with PdfPages(pdf_amount_clusters) as pdf:
            for fig in [fig1, fig2, fig3]:
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to free memory

    show_plot = False #for interactive mode this can be True otherwise always False
    #making heatmap 
    fig1 = correlation.clustering_analysis(csv_headgroup, 'Hoofdkenmerk', 'Score', num_clusters=2, label= "Hoofdkenmerk") #plots the scores in a heatmap and orders according the clusters - hoofdkenmerken
    fig2 = correlation.clustering_analysis(csv_subgroup, 'Subkenmerk', 'Subscore', num_clusters=2) #plots the scores in a heatmap and orders according the clusters - subkenmerken
    fig3 = correlation.clustering_analysis(csv_competencies,'Subcompetentie', 'Percentage', num_clusters=2, label= "Subcompetentie") #plots the scores in a heatmap and orders according the clusters - competenties
    
    #merges the figures above together in one pdf if show plot is false otherwise nothing happens. 
    if show_plot: 
        None
    else: 
        with PdfPages(pdf_clusters_total) as pdf: 
            for fig in [fig1, fig2, fig3]:
                pdf.savefig(fig)
                plt.close(fig)  # Close the figure to free memory

    correlation.amount_of_clusters_per_hoofdkenmerk(csv_subgroup, 'Hoofdkenmerk', 'Subkenmerk', 'Subscore', pdf_amount_clusters_kenmerken) #determines optimal number of students for classifying trainees - per hoofdkenmerk
    correlation.amount_of_clusters_per_hoofdkenmerk(csv_competencies, 'Hoofdcompetentie', 'Subcompetentie', 'Percentage', pdf_amount_clusters_competencies) #determines optimal number of students for classifying trainees - per hoofdcompetentie
    correlation.clustering_analysis_per_hoofdkenmerk(csv_subgroup, 'Hoofdkenmerk', 'Subkenmerk', 'Subscore', pdf_clusters_kenmerken, csv_clusters_kenmerken, num_clusters=2)  #plots the scores in a heatmap and orders according the clusters - per hoofdkenmerken 1 figure
    correlation.clustering_analysis_per_hoofdkenmerk(csv_competencies, 'Hoofdcompetentie', 'Subcompetentie', 'Percentage', pdf_clusters_competencies, csv_clusters_competencies, num_clusters=2) #plots the scores in a heatmap and orders according the clusters - per hoofdcompetentie 1 figure

    #onderstaande functies zijn op interpretatie van amount of clusters per hoofdkenmerk, handmatig gekozen
    hoofdkenmerk = "Structuur"
    correlation.clustering_analysis_per_hoofdkenmerk_gesplitst(csv_subgroup, 'Hoofdkenmerk', 'Subkenmerk', 'Subscore', pdf_clusters_kenmerken_structuur, csv_clusters_kenmerken_structuur, hoofdkenmerk, num_clusters=7)
    hoofdkenmerk = "Organisatiegerichtheid"
    correlation.clustering_analysis_per_hoofdkenmerk_gesplitst(csv_competencies, 'Hoofdcompetentie', 'Subcompetentie', 'Percentage', pdf_clusters_competencies_organisatiegerichtheid, csv_clusters_competencies_organisatiegerichtheid, hoofdkenmerk, num_clusters=9)

if __name__ == "__main__":
    main()

