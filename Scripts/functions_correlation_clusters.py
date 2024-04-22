# -------------------
# Import packages 
# -------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import t

# -------------------
# Correlation Analysis
# -------------------

def plot_heatmap(correlation_df, mask=None, annot=True, title='Correlatie tussen persoonlijkheidskenmerken', label= "kenmerk", figsize=(10, 8), output_pdf=None): 
    """
    Function to plot a heatmap of the correlation data.

    Parameters:
    - correlation_df (DataFrame): Correlation matrix
    - mask (pandas.DataFrame, optional): A DataFrame of boolean values where True indicates the cells that should be masked (hidden) in the heatmap. Typically used to hide weaker correlations for better visual emphasis on stronger correlations. Defaults to None.
    - annot(bool): If True, the correlation values are shown in the heatmap; if False they are not displayed. Default is True
    - title(str): Title for the heatmap.
    - label (str): Label for the heatmap axes.
    - output_pdf(PdfPages object, optional): PdfPages object to save plots into a PDF

    Returns:
    None

    """
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 6})
    sns.heatmap(correlation_df, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black', annot=annot, fmt="0.2f", mask=mask)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel(label)
    #makes sure that the x-axis labels fit when using output to pdf. 
    plt.xticks(rotation=20, ha='right') 
    
    #plot headmap or save to pdf
    if output_pdf:
        output_pdf.savefig()
        plt.close()
    else:
        plt.show()

def correlation_hoofdkenmerk(csv_headgroup, output_pdf_path, output_to_pdf=True):
    """
    Calculates and displays the Pearson correlation between scores of various 'Hoofdkenmerk' groups from a CSV file.
    
    Parameters:
    - csv_headgroup (str): Path to the CSV file containing the 'Hoofdkenmerk' and 'Score' columns.
    - output_pdf_path (str): Path where the PDF of the correlation heatmap will be saved.
    - output_to_pdf (bool, optional): If True, outputs the heatmap to a PDF file; otherwise, displays the heatmap. Defaults to True.

    Returns:
    None
    """

    # Load dataframe from CSV
    headgroup_df = pd.read_csv(csv_headgroup)
    
    # Group the dataframe by 'Hoofdkenmerk'
    grouped = headgroup_df.groupby('Hoofdkenmerk')
    
    # Extract unique hoofdkenmerk values
    groups = list(grouped.groups.keys())

    # Dictionary to store pairwise correlations
    correlations = {}

    # Iterate over all pairs of groups and calculate correlations 
    for group1, group2 in combinations(groups, 2): 
        # Extract scores for each group
        data_group1 = grouped.get_group(group1)['Score']  
        data_group2 = grouped.get_group(group2)['Score'] 

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(data_group1, data_group2) 

        # Store correlation in dictionary 
        correlations[(group1, group2)] = correlation 

    # Print correlations
    print("Correlation between each pair of groups in column 2 ('kenmerk'):")
    for pair, correlation in correlations.items():
        print(f"Groups {pair}: {correlation}")
    
    # Initialize a dataframe to store correlation values for heatmap
    correlation_df = pd.DataFrame(index=groups, columns=groups)

    # Fill the correlation dataframe symmetrically using dictionary
    for (group1, group2), correlation in correlations.items():
        correlation_df.at[group1, group2] = correlation
        correlation_df.at[group2, group1] = correlation

    # Convert scores to float for consistent formatting
    correlation_df = correlation_df.astype(float)

    # Plot heatmap or save to PDF 
    if output_to_pdf:
        with PdfPages(output_pdf_path) as pdf:
            plot_heatmap(correlation_df, mask=None, title='Correlation between groups', label= "kenmerk", output_pdf=pdf)
    else:
        plot_heatmap(correlation_df, mask=None, title='Correlation between groups', label= "kenmerk")
             

def correlation_total(csv_input, csv_cor_file, csv_cor_filtered, sub_column_name, score_column_name, main_column_name, output_pdf_path, output_to_pdf=True):
    """
    Calculate correlations between subgroups and generate correlation heatmaps.

    Parameters:
    - csv_input (str) : Path to input CSV file
    - csv_cor_file (str): Path to save correlations CSV file
    - csv_cor_filtered (str): Path to save the filtered correlations CSV file.
    - sub_column_name (str): Name of the column containing subgroup identifiers. (Subkenmerk or Subcompetentie)
    - score_column_name (str): Name of the column containing scores. (Subscore or Percentage)
    - main_column_name (str): Name of the column containing main identifiers (Hoofdkenmerk or Hoofdcompetentie).
    - output_pdf_path (str): Path to save the PDF with heatmaps.
    - output_to_pdf (bool, optional): Whether to output heatmaps to a PDF. Default is True.

    Returns: 
    - df_cor (DataFrame): DataFrame containing correlation values

    """
    # Read input CSV
    df_input = pd.read_csv(csv_input)

    # Group data by subgroup
    grouped = df_input.groupby(sub_column_name)

    # Get unique groups
    groups = list(grouped.groups.keys())

    # Dictionary to store correlations
    correlations = {}

    # Iterate over all pairs of groups and calculate correlations
    for group1, group2 in combinations(groups, 2):
        # Extract scores for each group
        data_group1 = grouped.get_group(group1)[score_column_name]
        data_group2 = grouped.get_group(group2)[score_column_name]

        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(data_group1, data_group2)

        # Store correlation in dictionary 
        correlations[(group1, group2)] = correlation

    # Drop the original "Pair" column
    grouped.head()

    # Initialize a dataframe to store correlation values for heatmap
    correlation_df = pd.DataFrame(index=groups, columns=groups)

    # Fill the correlation dataframe symmetrically using dictionary
    for (group1, group2), correlation in correlations.items():
        correlation_df.at[group1, group2] = correlation
        correlation_df.at[group2, group1] = correlation
    
    # Convert correlation DataFrame to float type
    correlation_df = correlation_df.astype(float)
    
    # Create DataFrame for correlations
    df_cor= pd.DataFrame(correlations.items(), columns =["Pair", "Correlation"])
    df_cor[[f'{sub_column_name} 1', f'{sub_column_name} 2']] = df_cor['Pair'].apply(pd.Series)
    df_unique = df_input.drop_duplicates(subset=[sub_column_name])
    df_cor[f'{main_column_name} 1'] = df_cor[f'{sub_column_name} 1'].map(df_unique.set_index(sub_column_name)[main_column_name])
    df_cor[f'{main_column_name} 2'] = df_cor[f'{sub_column_name} 2'].map(df_unique.set_index(sub_column_name)[main_column_name])
    df_cor.drop(columns=["Pair"], inplace=True)

    # Save all correlations to a CSV file
    df_cor.to_csv(csv_cor_file, index= False)

    # Filter correlations based on threshold and save to a CSV file
    filtered_df = df_cor[(df_cor['Correlation'] < -0.59) | (df_cor['Correlation'] > 0.59)]
    filtered_df.to_csv(csv_cor_filtered, index= False)

    # Plot heatmaps
    if output_to_pdf:
        with PdfPages(output_pdf_path) as pdf:
            # Plotting heatmap for >0.6
            mask = correlation_df.abs() <= 0.6
            plot_heatmap(correlation_df, mask=mask, title='Correlation Heatmap for >0.6', label= sub_column_name, output_pdf=pdf)

            # Plotting heatmap for all
            plot_heatmap(correlation_df, title='Correlation Heatmap for All', label= sub_column_name, output_pdf=pdf)

            # Plotting heatmap without annotations
            plot_heatmap(correlation_df, annot=False, title='Correlation Heatmap without Annotations', label= sub_column_name, output_pdf=pdf)
            
    else:
        # Plotting heatmap for >0.6
        mask = correlation_df.abs() <= 0.6
        plot_heatmap(correlation_df, mask=mask, title='Correlation Heatmap for >0.6', label= sub_column_name)

        # Plotting heatmap for all
        plot_heatmap(correlation_df, title='Correlation Heatmap for All', label= sub_column_name)

        # Plotting heatmap without annotations
        plot_heatmap(correlation_df, annot=False, title='Correlation Heatmap without Annotations',  label= sub_column_name)
    
    return df_cor

def correlation_per_function(csv_input, csv_info, csv_all_richtingen, csv_sub_cor_filtered, sub_column_name, score_column_name, main_column_name, output_pdf_path, output_to_pdf = True):
    """
    Calculate correlations between subgroups and generate correlation heatmaps per function (Analist/Engineer/Scientist/Productowner).

    Parameters:
    - csv_input (str) : Path to input CSV file
    - csv_info (str): Path to info CSV file (info_job.csv)
    - csv_all_richtingen (str) : Path to save the correlations CSV file
    - csv_sub_cor_filtered (str): Path to save the filtered correlations CSV file.
    - sub_column_name (str): Name of the column containing subgroup identifiers. (Subkenmerk or Subcompetentie)
    - score_column_name (str): Name of the column containing scores. (Subscore or Percentage)
    - main_column_name (str): Name of the column containing main identifiers (Hoofdkenmerk or Hoofdcompetentie).
    - output_pdf_path (str): Path to save the PDF with heatmaps.
    - output_to_pdf (bool, optional): Whether to output heatmaps to a PDF. Default is True.

    Returns: 
    - df_all_richtingen (DataFrame): DataFrame containing correlation values per function ()
    """

    # Read input CSV with scores
    df = pd.read_csv(csv_input)

    # Read info CSV containing job-related information
    info_job_df = pd.read_csv(csv_info)

    # Merge score input with job-related information
    merged_df = pd.merge(df, info_job_df, on= "File ID", how = 'inner')
    
    # Get unique function values (Richting)
    richtingen = merged_df['Richting'].unique()

    # List to store correlation DataFrames for each Richting
    richting_correlations = []
    # Dictionary to store correlations for each Richting
    correlation_richting = {}

    # Iterate over each unique Richting value
    for richting_value in richtingen:
        # Filter DataFrame for the specific Richting value
        subgroup_richting = merged_df[merged_df['Richting'] == richting_value]
        
        # Group by sub_column_name
        grouped = subgroup_richting.groupby(sub_column_name)

        # Get unique groups
        groups = list(grouped.groups.keys())

        # Dictionary to store correlations
        correlations = {}

        # Iterate over all pairs of groups and calculate correlations
        for group1, group2 in combinations(groups, 2):
            # Extract scores for each group
            data_group1 = grouped.get_group(group1)[score_column_name]
            data_group2 = grouped.get_group(group2)[score_column_name]
            
            if not data_group1.var() == 0 and not data_group2.var() == 0:
                correlation, _ = pearsonr(data_group1, data_group2)
                correlations[(group1, group2)] = correlation
            else:
                # If either array is constant, set correlation to 0
                correlations[(group1, group2)] = None

        # Create DataFrame to store correlations for this Richting      
        df_sub_cor_richting= pd.DataFrame(correlations.items(), columns =["Pair", "Correlation"])
        df_sub_cor_richting[[f'{sub_column_name} 1', f'{sub_column_name} 2']] = df_sub_cor_richting['Pair'].apply(pd.Series)
        subgroup_df_unique = df.drop_duplicates(subset=[sub_column_name])
        df_sub_cor_richting[f'{main_column_name} 1'] = df_sub_cor_richting[f'{sub_column_name} 1'].map(subgroup_df_unique.set_index(sub_column_name)[main_column_name])
        df_sub_cor_richting[f'{main_column_name} 2'] = df_sub_cor_richting[f'{sub_column_name} 2'].map(subgroup_df_unique.set_index(sub_column_name)[main_column_name])
        df_sub_cor_richting['Richting'] = richting_value
        df_sub_cor_richting.drop(columns=["Pair"], inplace=True)

        # Store correlations for this Richting value
        richting_correlations.append(df_sub_cor_richting)
        correlation_richting[richting_value] = correlations

    # Combine all Richting correlation DataFrames into one DataFrame
    df_all_richtingen = pd.concat(richting_correlations)

    # Save all correlations to a CSV file
    df_all_richtingen.to_csv(csv_all_richtingen, index=False)

    # Filter significant correlations based on a threshold and save to a CSV file
    filtered_df = df_all_richtingen[(df_all_richtingen['Correlation'] < -0.59) | (df_all_richtingen['Correlation'] > 0.59)]
    filtered_df.to_csv(csv_sub_cor_filtered, index= False)

    # Generate correlation heatmaps per function and save to a PDF
    with PdfPages(output_pdf_path) as pdf:
        # Create 4 plots, one for each Richting value
        for i, (richting_value, correlations) in enumerate(correlation_richting.items(), start=1):
            correlation_df = pd.DataFrame(index=groups, columns=groups)
            
            # Fill the correlation DataFrame
            for (group1, group2), correlation in correlations.items():
                correlation_df.at[group1, group2] = correlation
                correlation_df.at[group2, group1] = correlation

            # Convert the correlation values to float
            correlation_df = correlation_df.astype(float)

            # Plot heatmap for the current Richting value
            if output_to_pdf:
                plot_heatmap(correlation_df, mask=None, title= f'Correlation Between Groups (Richting: {richting_value})', label= sub_column_name, output_pdf=pdf)
            else:
                plot_heatmap(correlation_df, mask=None, title= f'Correlation Between Groups (Richting: {richting_value})', label= sub_column_name)

            
    return df_all_richtingen


def calculate_correlation_excluding_individual(subgroup_df, threshold_factor, csv_filtered_outliers_subkenmerken, sub_column_name, score_column_name):
    """
    Calculate correlations excluding individual observations and identify outliers.

    Parameters:
    - subgroup_df (DataFrame): DataFrame containing subgroup and score data.
    - threshold_factor (float): Factor to determine the threshold for identifying outliers.
    - csv_filtered_outliers_subkenmerken (str): Path to save the CSV file containing identified outliers.
    - sub_column_name (str): Name of the column containing subgroup identifiers.
    - score_column_name (str): Name of the column containing scores.

    Returns:
    - df_outliers (DataFrame): DataFrame containing identified outliers.
    - base_df (DataFrame): DataFrame containing trainees except outliers.
    """
    
    # Extract subgroup data
    data = subgroup_df
    file_ids = data["File ID"].unique()
    
    # List to store correlations excluding individual observations
    correlations_excluded_individual = []
    
    #Calculate correlations excluding each individual file ID
    for excluded_id in file_ids:
        subgroup_excluded_individual = data[data["File ID"] != excluded_id]
        grouped = subgroup_excluded_individual.groupby(sub_column_name)

        # Get unique groups
        groups = list(grouped.groups.keys())

        # Iterate over all pairs of groups and calculate correlations
        for group1, group2 in combinations(groups, 2):
            # Extract scores for each group
            data_group1 = grouped.get_group(group1)[score_column_name]
            data_group2 = grouped.get_group(group2)[score_column_name]
            # Calculate Pearson correlation coefficient
            correlation, _ = pearsonr(data_group1, data_group2)
            correlations_excluded_individual.append({
                "excluded_id": excluded_id,
                "group1": group1,
                "group2": group2,
                "correlation": correlation
            })

    # Convert list of correlations to DataFrame
    correlation_excluded_df = pd.DataFrame(correlations_excluded_individual)

    # Calculate correlations for all observations
    grouped_all = data.groupby(sub_column_name)
    groups_all = list(grouped_all.groups.keys())
    
    correlations_all = []

    for group1, group2 in combinations(groups_all, 2):
        # Extract scores for each group
        data_group1 = grouped_all.get_group(group1)[score_column_name]
        data_group2 = grouped_all.get_group(group2)[score_column_name]
        # Calculate Pearson correlation coefficient
        correlation, _ = pearsonr(data_group1, data_group2)
        correlations_all.append({
            "group1": group1,
            "group2": group2,
            "correlation": correlation
        })

    # Convert list of correlations to DataFrame
    correlation_all_df = pd.DataFrame(correlations_all)

    # Merge correlation DataFrames
    merged_df = pd.merge(correlation_excluded_df, correlation_all_df, on=['group1', 'group2'], suffixes=('_excluded', '_all'))

    # Calculate the difference between correlations
    merged_df['correlation_difference'] = merged_df['correlation_all'] - merged_df['correlation_excluded']
    # Calculate standard deviation of correlation differences
    std_deviation_df = merged_df.groupby(['group1', 'group2'])['correlation_difference'].std().reset_index()
    std_merged_df = pd.merge(merged_df, std_deviation_df, on=['group1', 'group2'], how='left')
    std_merged_df.rename(columns={"correlation_difference_x": 'correlation_difference', "correlation_difference_y": "stdev"}, inplace = True)
    
    # Calculate threshold for identifying outliers
    std_merged_df['threshold'] = threshold_factor * std_merged_df['stdev']
    # Identify outliers
    df_outliers = std_merged_df[abs( std_merged_df['correlation_difference']) > abs( std_merged_df['threshold'])]
    # Select all except outliers
    base_df =  std_merged_df[abs( std_merged_df['correlation_difference']) < abs( std_merged_df['threshold'])]
    # Save identified outliers to a CSV file
    if csv_filtered_outliers_subkenmerken:
        df_outliers.to_csv(csv_filtered_outliers_subkenmerken)
    
    return df_outliers, base_df


def correlation_without_outliers(subgroup_df, outliers_df, csv_correlation_without_outliers_subkenmerken, sub_column_name, score_column_name, main_column_name, output_pdf_path, output_to_pdf = True):
    """
    Calculate correlation between subgroups without outliers and generate a heatmap plot.

    Parameters:
    - subgroup_df (DataFrame): DataFrame containing subgroup data.
    - outliers_df (DataFrame): DataFrame containing outlier data.
    - csv_correlation_without_outliers_subkenmerken (str): Path to save the correlation data CSV file.
    - sub_column_name (str): Name of the column containing subgroup identifiers.
    - score_column_name (str): Name of the column containing scores.
    - main_column_name (str): Name of the main column to be dropped from subgroup_df.
    - output_pdf_path (str): Path to save the PDF file of the heatmap plot.
    - output_to_pdf (bool, optional): Flag to control whether to output the heatmap plot to PDF. Default is True.

    Returns:
    - correlation_df(DataFrame): DataFrame containing correlation data without outliers between subgroups 

    """
    # Drop the main column
    subgroup_df = subgroup_df.drop(columns=[main_column_name])

    # Pivot subgroup_df based on sub_column_name
    paired_df = subgroup_df.pivot(index='File ID', columns=sub_column_name, values=score_column_name)

    # Generate column pairs for correlation calculation
    column_pairs = [(col1, col2) for col1 in paired_df.columns for col2 in paired_df.columns if col1 != col2]

    # Create a list to store DataFrames for each column pair
    dfs = []

    # Iterate through each column pair and extract scores
    for col1, col2 in column_pairs:
        scores = paired_df[[col1, col2]].dropna()
        scores.columns = ['Score_1', 'Score_2']
        scores[f'{sub_column_name}_1'] = col1
        scores[f'{sub_column_name}_2'] = col2
        dfs.append(scores[[f'{sub_column_name}_1', 'Score_1', f'{sub_column_name}_2', 'Score_2']])

    # Concatenate the DataFrames in the list
    paired_scores_df = pd.concat(dfs, ignore_index=False)
    paired_scores_df.reset_index(inplace=True)
    paired_scores_df.columns = ["File ID",'group1', 'Score_1', 'group2', 'Score_2']
    # Create a unique identifier for each pair of subgroups
    paired_scores_df["unique_id"] = paired_scores_df.apply(lambda row: '_'.join(sorted([row['group1'], row["group2"]])), axis = 1)
    paired_scores_df = paired_scores_df.drop_duplicates(subset = ['File ID', 'unique_id'])

    # Merge the two dataframes based on the 'File ID' column for both groups
    merged_df = pd.merge(paired_scores_df, outliers_df, left_on=['File ID', 'group1', 'group2'], right_on=['excluded_id', 'group1', 'group2'], how='left', suffixes=('_group1', '_group2'))
        
    # Filter out rows where 'File ID' is present in both groups
    merged_df = merged_df[merged_df['excluded_id'].isna()]
    merged_df = merged_df.iloc[:, :5]
        
    #Group the pairs together and apply pearsonr on this group
    correlation_scores = merged_df.groupby(['group1', 'group2']).apply(lambda x: pearsonr(x['Score_1'], x['Score_2'])[0])
    correlation_scores = pd.DataFrame(correlation_scores)
    correlation_data = []

    for (group1, group2), group_data in merged_df.groupby(['group1', 'group2']):
        correlation = pearsonr(group_data['Score_1'], group_data['Score_2'])[0]
        correlation_data.append([group1, group2, correlation])
    
    # Create DataFrame to store correlation data
    correlation_df = pd.DataFrame(correlation_data, columns=['Group 1', 'Group 2', 'Correlation'])
   
    # Create inverted DataFrame for correlation_df (means Group 1 and 2 are switched)
    inverted_df = correlation_df.copy()
    inverted_df.columns = ["Group 2", "Group 1", "Correlation"]
   
    # Merge normal and inverted together
    correlation_df= pd.concat([correlation_df, inverted_df], ignore_index=True)
   
    # Save correlation data to CSV
    correlation_df.to_csv(csv_correlation_without_outliers_subkenmerken)

    # Pivot correlation data for heatmap plot
    pivot_correlation = correlation_df.pivot(index = "Group 1", columns = 'Group 2', values = 'Correlation')
    
    # Generate heatmap plot and save to PDF if specified
    if output_to_pdf:
        with PdfPages(output_pdf_path) as pdf:
            plot_heatmap(pivot_correlation, title='Correlation Heatmap for All without outliers', label= sub_column_name, output_pdf=pdf) 
    else: 
        plot_heatmap(pivot_correlation, title='Correlation Heatmap for All without outliers', label= sub_column_name) 

    return correlation_df

def calculate_t_test_outliers(row):
    """
    Calculate the t-test for paired samples comparing correlations before and after removing outliers.

    Parameters:
    - row (Series): A row from a DataFrame containing correlation data.

    Returns:
    - Series: Series containing the difference in correlations and the p-value of the t-test.

    """
    # Extract correlation values before and after removing outliers
    cor_before = row['before']
    cor_after = row['without outliers']
    
    # Calculate the difference in correlation
    diff_cor = cor_before - cor_after
    # Sample size
    n = 46

    # Calculate standard error of the difference
    se_diff = ((1/n)+(1/ n)) ** 0.5

    # Calculate degrees of freedom
    dof = n - 2
    
    # Calculate t-statistic
    t_statistic = diff_cor / se_diff
    # Calculate two-tailed p-value for the t-test
    p_val_diff = 2*(1- t.cdf(abs(t_statistic), dof))
    
    # Return Series containing difference in correlations and p-value
    return pd.Series({'diff_cor': diff_cor, 'P_val': p_val_diff})
    
def compare_correlation_after_outliers(df_sub_cor , correlation_df, csv_correlation_difference_outliers, sub_column_name):
    """
    Compare correlations before and after removing outliers and calculate the t-test for paired samples.

    Parameters:
    - df_sub_cor (DataFrame): DataFrame containing correlation data before outliers removal.
    - correlation_df (DataFrame): DataFrame containing correlation data after outliers removal.
    - csv_correlation_difference_outliers (str): Path to save the CSV file of correlation differences and p-values.
    - sub_column_name (str): Name of the column containing subgroup identifiers.

    Returns:
    - df (DataFrame): DataFrame containing correlation data before outliers removal.
    - df1 (DataFrame) : DataFrame containing correlation data after outliers removal.
    """

    # Convert df_sub_cor to df, drop unnecessary columns and change order 
    df = pd.DataFrame(df_sub_cor)
    df.drop(df.columns[[3, 4]], axis = 1, inplace = True)
    df.columns = ["Correlation", "Group 1", "Group 2"]
    desired_order = ["Group 1", "Group 2", "Correlation"]
    df = df[desired_order]

    #convert correlation_df to DataFrame
    df1 = pd.DataFrame(correlation_df)
    
    # Merge correlation dataframes on group identifiers
    df = df.merge(df1, on=['Group 1', 'Group 2'], how='inner')
    df.columns = ["Group 1", "Group 2", "before", "without outliers"]

    # Calculate t-test for paired samples
    p_values = df.apply(calculate_t_test_outliers, axis=1)
    p_values_df = pd.concat([df, p_values], axis=1)

    # Save results to CSV
    p_values_df.to_csv(csv_correlation_difference_outliers, index=True)
    
    return df, df1

     
def calculate_student_t(row):
    """
    Calculate the t-test for independent samples comparing correlations between two groups.

    Parameters:
    - row (Series): A row from a DataFrame containing correlation data.

    Returns:
    - Series: Series containing the difference in correlations and the p-value of the t-test.

    """
    # Extract correlation values for analyst and engineer groups
    analist_corr = row['Analist']
    engineer_corr = row['Engineer']

    # Sample sizes for analyst and engineer groups
    n_analist = 31
    n_engineer = 10

    # Calculate total sample size
    sample_size = n_analist + n_engineer

    # Calculate difference in correlations
    diff_cor = analist_corr - engineer_corr

    # Calculate standard error of the difference
    se_diff = ((1/n_analist)+(1/ n_engineer)) ** 0.5

    # Calculate degrees of freedom
    dof = sample_size - 2

    # Calculate t-statistic
    t_statistic = diff_cor / se_diff

     # Calculate two-tailed p-value for the t-test
    p_val_diff = 2*(1- t.cdf(abs(t_statistic), dof))

    # Return Series containing difference in correlations and p-value
    return pd.Series({'diff_cor': diff_cor, 'P_val': p_val_diff})

def compare_richtingen(df_all_richtingen, csv_correlation_richting, csv_correlation_richting_filtered, sub_column_name, main_column_name):
    """
    Compare correlations between two directions (e.g., Analyst and Engineer) and filter significant differences.

    Parameters:
    - df_all_richtingen (DataFrame): DataFrame containing correlation data for all directions.
    - csv_correlation_richting (str): Path to save the CSV file of correlation data between directions.
    - csv_correlation_richting_filtered (str): Path to save the CSV file of filtered correlation data.
    - sub_column_name (str): Name of the column containing subgroup identifiers
    - main_column_name (str): Name of the main column (Hoofdkenmerk - Hoofdcompetentie).

    """

    # Convert df_all_richtingen to DataFrame and filter rows for only Analyst and Engineer directions
    df = pd.DataFrame(df_all_richtingen)
    df = df[df['Richting'].isin(['Analist', 'Engineer'])]
       
    # Pivot DataFrame to create a table for correlation data between directions
    pivot_df = df.pivot_table(index=[f'{sub_column_name} 1', f'{sub_column_name} 2', f'{main_column_name} 1', f'{main_column_name} 2'], columns='Richting', values='Correlation')

    # Calculate t-test for independent samples for each row in the pivot DataFrame
    p_values = pivot_df.apply(calculate_student_t, axis=1)
    p_values_df = pd.concat([pivot_df, p_values], axis=1)

    # Filter DataFrame to include only rows with significant differences (p-value < 0.05)
    filtered_df = p_values_df[p_values_df['P_val'] < 0.05]

    # Save correlation data and filtered correlation data to CSV files
    p_values_df.to_csv(csv_correlation_richting, index=True)
    filtered_df.to_csv(csv_correlation_richting_filtered, index= True)

def compare_richtingen_zonder_outliers(correlation_df_A, correlation_df_E, csv_correlation_richting_without_outliers):
    """
    Compare correlations between two directions (e.g., Analyst and Engineer) without outliers and calculate the t-test for independent samples.

    Parameters:
    - correlation_df_A (DataFrame): DataFrame containing correlation data for Analyst direction without outliers.
    - correlation_df_E (DataFrame): DataFrame containing correlation data for Engineer direction without outliers.
    - csv_correlation_richting_without_outliers (str): Path to save the CSV file of correlation data between functions without outliers.

    Returns:
    - p_values_df (DataFrame): DataFrame containing correlation data and t-test results between functions without outliers.

    """

    # Convert correlation_df_A and correlation_df_E to DataFrames
    df = pd.DataFrame(correlation_df_A)
    df1 = pd.DataFrame(correlation_df_E)

    # Merge correlation DataFrames on group identifiers
    df = df.merge(df1, on=['Group 1', 'Group 2'], how='inner')
    df.columns = ["Group 1", "Group 2", "Analist", "Engineer"]

    # Calculate t-test for independent samples for each row in the merged DataFrame
    p_values = df.apply(calculate_student_t, axis=1)
    p_values_df = pd.concat([df, p_values], axis=1)
    
    # Save correlation data and t-test results to CSV file
    p_values_df.to_csv(csv_correlation_richting_without_outliers, index=True)
    
    return p_values_df

# -------------------
# Cluster Analysis
# -------------------


def amount_of_clusters(csv_data, key_column, value_column):
    """
    Determine the optimal number of clusters for clustering by evaluating silhouette scores.

    This function reads a CSV dataset, pivots it based on specified columns, scales the data,
    performs K-means clustering for a range of cluster counts, calculates silhouette scores,
    and plots the scores against the number of clusters to identify the optimal count.

    Parameters:
    - csv_data (str): The file path to the CSV data.
    - key_column (str): The name of the column to pivot on which will become the columns in the pivoted DataFrame.(Hoofdkenmerk - Subkenmerk - Subcompetentie)
    - value_column (str): The name of the column whose values will be rearranged and used for clustering. (Score - Subscore - Percentage)

    Returns:
    - matplotlib.figure.Figure: A matplotlib figure object that contains the plot of silhouette scores against cluster counts.
    """
    
    #load data from a CSV file into a DataFrame 
    df = pd.read_csv(csv_data)

    # Set the "File ID" as the index of the DataFrame
    df.set_index('File ID', inplace=True)

    # Pivot the DataFrame using the specified key and value columns
    pivot_data = df.pivot(columns=key_column, values=value_column)
    pivot_data.columns = pivot_data.columns.rename(None) 

    # Reset index to turn the pivot table back to DataFrame
    pivot_data.reset_index(inplace=True) 
    
    # Standardize the data to have a mean of zero and a standard deviation of one
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:]) 
    
    # Define the range of possible cluster counts
    k_values = range(2, 11)

    # Create a list to store silhouette scores per k value
    silhouette_scores = []
    
    # Compute the silhouette score for each K value
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled_data)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    # Determine the K value that maximizes the silhouette score
    optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
    print(f'Optimal number of clusters (K) {key_column} =', optimal_k)

    # Plot the silhouette scores against the number of clusters.
    fig, ax = plt.subplots()
    ax.plot(k_values, silhouette_scores, marker='o')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title(f'Silhouette Score for Different Values of K - {key_column}')
    ax.set_xticks(k_values)
    ax.grid(True)
    
    return fig

def clustering_analysis(csv_data, key_column, value_column, num_clusters, label= "Subkenmerk", show_plot=False):

    """
    Performs clustering analysis on the specified columns of a CSV file and can visualize the results
    as a heatmap with clusters.

    Parameters:
    - csv_data (str): Path to the CSV file.
    - key_column (str): The column to be pivoted and become the new columns in the DataFrame.
    - value_column (str): The column whose values are to be used for clustering after pivot.
    - num_clusters (int): Number of clusters for the KMeans algorithm.
    - label (str, optional): Label for the x-axis in the plot. Defaults to "Subkenmerk".
    - show_plot (bool, optional): If True, show the plot; otherwise, return the plot object. Defaults to False.

    Returns:
    - matplotlib.figure.Figure or None: Returns the plot figure if show_plot is False, otherwise displays the plot.
    """

    # Load data
    df = pd.read_csv(csv_data)

    # Set the "File ID" as the index of the DataFrame
    df.set_index('File ID', inplace=True)

    # Pivot the DataFrame using the specified key and value columns
    pivot_data = df.pivot(columns=key_column, values=value_column)
    pivot_data.columns = pivot_data.columns.rename(None)

    # Reset index to turn the pivot table back to DataFrame
    pivot_data.reset_index(inplace=True)

    # Standardize the data to have a mean of zero and a standard deviation of one
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:]) 
    
    # Set a seed to create the same plot when repeating the function
    Myseed = np.random.seed(123)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=25, random_state=Myseed)
    clusters = kmeans.fit_predict(scaled_data)
    kmeans_score = silhouette_score(scaled_data, clusters)
    
    # Prepare results DataFrame
    result_df = pd.DataFrame(index=pivot_data.index, columns=['File ID', 'Kmean_Cluster_hoofdkenmerk'])
    result_df['File ID'] = pivot_data['File ID']  # Assuming 'File ID' is the column name storing full names
    result_df['Kmean_Cluster_hoofdkenmerk'] = clusters
    pivot_data_clustered = pivot_data.copy()

    # Convert 'File ID' column to categorical
    pivot_data_clustered['File ID'] = pd.Categorical(pivot_data_clustered['File ID'])

    # Use cluster_indices to reorder the DataFrame
    cluster_indices = [np.where(clusters == cluster_id)[0] for cluster_id in range(num_clusters)]
    clustered_indices = [index for indices in cluster_indices for index in indices]  # Flatten the list
    pivot_data_clustered = pivot_data_clustered.iloc[clustered_indices]

    shortened_labels = [label[:8] + "..." for label in pivot_data_clustered['File ID']] 

    # Plot heatmap using Seaborn
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 6})
    sns.heatmap(pivot_data_clustered.set_index('File ID'), cmap='viridis')

    # Add horizontal lines to separate clusters
    for cluster_indices_array in cluster_indices:
        cluster_start_index = pivot_data_clustered.index.get_loc(cluster_indices_array[0])
        plt.axhline(y=cluster_start_index, color='black', linestyle="--")
         
    # Add labels and title
    plt.xlabel(label)
    plt.ylabel('File ID')
    plt.yticks(ticks=plt.yticks()[0], labels=shortened_labels)
    plt.xticks(rotation=20, ha='right')
    plt.title('Clustered Heatmap')
    
    print("Silhouette Score for KMeans:", kmeans_score)

    if show_plot:
        plt.show()
    else:
        return plt.gcf()


def amount_of_clusters_per_hoofdkenmerk(csv_data, main_column, key_column, value_column, output_pdf_path):
    """
    Determines the optimal number of clusters for each unique value in a specified column of a CSV dataset, 
    using silhouette scores for K-means clustering, and saves the plots to a PDF file.

    Parameters:
    - csv_data (str): Path to the CSV file.
    - main_column (str): The column based on which unique values data will be segmented.
    - key_column (str): The column to pivot on.
    - value_column (str): The column whose values are aggregated during pivoting and used for clustering.
    - output_pdf_path (str): Path to save the output PDF containing the plots of silhouette scores.

    Return:
    - None
    
    """
    # Load data
    df = pd.read_csv(csv_data)

    # Set the "File ID" as the index of the DataFrame
    df.set_index('File ID', inplace=True)
    
    # Get unique hoofdkenmerk values
    hoofdkenmerk_values = df[main_column].unique()
    
    # Initialize a PDF file to store all plots
    with PdfPages(output_pdf_path) as pdf:
        for hk in hoofdkenmerk_values:
            # Filter DataFrame based on each unique value of the main column
            subset_df = df[df[main_column] == hk]

            # Pivot the subset DataFrame (based on key_column and value_column)
            pivot_data = subset_df.pivot(columns=key_column, values=value_column)
            pivot_data.columns = pivot_data.columns.rename(None)
            pivot_data.reset_index(inplace=True)
            
            # Scale data to normalize the feature scale
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:]) 
            
            # Range of K values for K-means
            k_values = range(2, 11)
            silhouette_scores = []
            
            for k in k_values:
                # Fit K-means clustering to the data
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(scaled_data)
                
                # Calculate the silhouette score for each K
                silhouette_avg = silhouette_score(scaled_data, kmeans.labels_)
                silhouette_scores.append(silhouette_avg)

            # Determine the optimal number of clusters by finding the max silhouette score 
            optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
            print(f"Optimal number of clusters (K) for {main_column} '{hk}' =", optimal_k)

            # Plot silhouette scores
            plt.plot(k_values, silhouette_scores, marker='o')
            plt.xlabel('Number of Clusters (K)')
            plt.ylabel('Silhouette Score')
            plt.title(f'Silhouette Score for Different Values of K {main_column} {hk})')
            plt.xticks(k_values)
            plt.grid(True)
            pdf.savefig()
            plt.close()

def clustering_analysis_per_hoofdkenmerk(csv_data, main_column, key_column, value_column, output_pdf_path, csv_clusters, num_clusters):
    """
    Performs KMeans clustering analysis on a dataset, grouped by a specified main column, 
    and saves both a summary of clusters to a CSV and visualizations to a PDF.

    Parameters:
    - csv_data (str): Path to the CSV file containing the dataset.
    - main_column (str): Column used to group the data before performing clustering.
    - key_column (str): Column used as a pivot during clustering.
    - value_column (str): Column whose values are used for clustering.
    - output_pdf_path (str): Path to save the PDF containing cluster visualizations.
    - csv_clusters (str): Path to save the CSV containing clustering results.
    - num_clusters (int): Number of clusters to form in KMeans clustering.

    Returns:
    - None
    """

    # Load data
    df = pd.read_csv(csv_data)

    # Set the "File ID" as the index of the DataFrame
    df.set_index('File ID', inplace=True)
    
    # Get unique hoofdkenmerk values
    hoofdkenmerk_values = df[main_column].unique()

    # Initialize an empty dictionary to store results for each hoofdkenmerk
    results = {}
    results_total = []

    label = key_column

    for hk in hoofdkenmerk_values:
        # Filter DataFrame for the current hoofdkenmerk value
        subset_df = df[df[main_column] == hk]

        # Pivot the subset DataFrame
        pivot_data = subset_df.pivot(columns=key_column, values=value_column)
        pivot_data.columns = pivot_data.columns.rename(None)
        pivot_data.reset_index(inplace=True)
        
        # Scaling data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:]) 

        # Setting a seed 
        Myseed = np.random.seed(123)
        kmeans = KMeans(n_clusters=num_clusters, n_init=25, random_state=Myseed)

        # Clustering
        clusters = kmeans.fit_predict(scaled_data)
        kmeans_score = silhouette_score(scaled_data, clusters)
        
        # Preparing results DataFrame
        result_df = pd.DataFrame(index=pivot_data.index, columns=['File ID', 'Cluster_group', main_column])
        result_df['File ID'] = pivot_data['File ID']  
        result_df["Cluster_group"] = clusters
        result_df[main_column] = hk

        results_total.append(result_df)

        # Store the results for the current hoofdkenmerk value
        results[hk] = {
            'pivot_data': pivot_data,
            'kmeans_score': kmeans_score,
            'result_df': result_df
        }
    
    # Concat List to DataFrame
    final_result_df = pd.concat(results_total)

    # Save results to PDF
    final_result_df.to_csv(csv_clusters, index = True)

    with PdfPages(output_pdf_path) as pdf:
    # Plot heatmap for each hoofdkenmerk
        for hk, result in results.items():
            pivot_data_clustered = result['pivot_data'].copy()

            # Convert 'File ID' column to categorical
            pivot_data_clustered['File ID'] = pd.Categorical(pivot_data_clustered['File ID'])

            clusters = result['result_df']['Cluster_group']

            # Use cluster_indices to reorder the DataFrame
            cluster_indices = [np.where(clusters == cluster_id)[0] for cluster_id in range(num_clusters)]
            clustered_indices = [index for indices in cluster_indices for index in indices]  # Flatten the list
            pivot_data_clustered = pivot_data_clustered.iloc[clustered_indices]

            shortened_labels = [label[:8] + "..." for label in pivot_data_clustered['File ID']] 

            # Print silhouette score for KMeans
            print(f"Silhouette Score for KMeans ({main_column} {hk}):", result['kmeans_score'])

            # Plot heatmap using Seaborn
            plt.figure(figsize=(10, 8))
            plt.rcParams.update({'font.size': 8})
            sns.heatmap(pivot_data_clustered.set_index('File ID'), cmap='viridis')

            # Add horizontal lines to separate clusters
            for cluster_indices_array in cluster_indices:
                cluster_start_index = pivot_data_clustered.index.get_loc(cluster_indices_array[0])
                plt.axhline(y=cluster_start_index, color='black', linestyle="--")

            # Add labels and title
            plt.xlabel(label)
            plt.ylabel('File ID')
            plt.xticks(rotation=20, ha='right')
            plt.yticks(ticks=plt.yticks()[0], labels=shortened_labels)
            plt.title(f'Clustered Heatmap for {hk}')

            # Save plot to pdf 
            pdf.savefig()
            plt.close() 

    
def clustering_analysis_per_hoofdkenmerk_gesplitst(csv_data, main_column, key_column, value_column, output_pdf_path, csv_clusters, hoofdkenmerk, num_clusters):
    """
    Performs KMeans clustering analysis on a specific hoofdkenmerk value within a dataset, 
    and saves both a summary of clusters to a CSV and visualizations to a PDF.

    Parameters:
    - csv_data (str): Path to the CSV file containing the dataset.
    - main_column (str): Column used to group the data before performing clustering.
    - key_column (str): Column used as a pivot during clustering.
    - value_column (str): Column whose values are used for clustering.
    - output_pdf_path (str): Path to save the PDF containing cluster visualizations.
    - csv_clusters (str): Path to save the CSV containing clustering results.
    - hoofdkenmerk (str): The specific value of hoofdkenmerk to analyze.
    - num_clusters (int): Number of clusters to form in KMeans clustering.

    Returns:
    - None
    """
    # Load Data
    df = pd.read_csv(csv_data)

    # Set the "File ID" as the index of the DataFrame
    df.set_index('File ID', inplace=True)
    
    # Get unique hoofdkenmerk values
    hoofdkenmerk_values = df[main_column].unique()

    # Initialize an empty dictionary to store results for each hoofdkenmerk
    results = {}
    results_total = []

    label = key_column
    selected_hk_found = False

    for hk in hoofdkenmerk_values:
        if not selected_hk_found:
            # Check if current hoofdkenmerk matches the selected hoofdkenmerk
            if hk == hoofdkenmerk:
                selected_hk_found = True
            else:
                continue
        # Filter DataFrame for the current hoofdkenmerk value
        subset_df = df[df[main_column] == hk]

        # Pivot the subset DataFrame
        pivot_data = subset_df.pivot(columns=key_column, values=value_column)
        pivot_data.columns = pivot_data.columns.rename(None)
        pivot_data.reset_index(inplace=True)
        # Scale Data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pivot_data.iloc[:, 1:]) 

        # Set a seed
        Myseed = np.random.seed(123)
        kmeans = KMeans(n_clusters=num_clusters, n_init=25, random_state=Myseed)

        # Clustering
        clusters = kmeans.fit_predict(scaled_data)
        kmeans_score = silhouette_score(scaled_data, clusters)
        
        #prepare results DataFrame
        result_df = pd.DataFrame(index=pivot_data.index, columns=['File ID', 'Cluster_group', main_column])
        result_df['File ID'] = pivot_data['File ID']  # Assuming 'File ID' is the column name storing full names
        result_df["Cluster_group"] = clusters
        result_df[main_column] = hk

        #append results to one DataFrame
        results_total.append(result_df)

        # Store the result for the current hoofdkenmerk value
        results[hk] = {
            'pivot_data': pivot_data,
            'kmeans_score': kmeans_score,
            'result_df': result_df
        }

        if hk == hoofdkenmerk:
            final_result_df = result_df
            final_result_df.to_csv(csv_clusters, index = True)
 
            with PdfPages(output_pdf_path) as pdf:
            # Plot heatmap for each hoofdkenmerk
                for hk, result in results.items():
                    pivot_data_clustered = result['pivot_data'].copy()

                    # Convert 'File ID' column to categorical
                    pivot_data_clustered['File ID'] = pd.Categorical(pivot_data_clustered['File ID'])

                    clusters = result['result_df']['Cluster_group']

                    # Use cluster_indices to reorder the DataFrame
                    cluster_indices = [np.where(clusters == cluster_id)[0] for cluster_id in range(num_clusters)]
                    clustered_indices = [index for indices in cluster_indices for index in indices]  # Flatten the list
                    pivot_data_clustered = pivot_data_clustered.iloc[clustered_indices]

                    shortened_labels = [label[:8] + "..." for label in pivot_data_clustered['File ID']] 

                    # Print silhouette score for KMeans
                    print(f"Silhouette Score for KMeans ({main_column} {hk}):", result['kmeans_score'])

                    # Plot heatmap using Seaborn
                    plt.figure(figsize=(10, 8))
                    plt.rcParams.update({'font.size': 8})
                    sns.heatmap(pivot_data_clustered.set_index('File ID'), cmap='viridis')

                    # Add horizontal lines to separate clusters
                    for cluster_indices_array in cluster_indices:
                        cluster_start_index = pivot_data_clustered.index.get_loc(cluster_indices_array[0])
                        plt.axhline(y=cluster_start_index, color='black', linestyle="--")

                    # Add labels and title
                    plt.xlabel(label)
                    plt.ylabel('File ID')
                    plt.xticks(rotation=20, ha='right')
                    plt.yticks(ticks=plt.yticks()[0], labels=shortened_labels)
                    plt.title(f'Clustered Heatmap for {hk}')

                    # Save plot to PDF
                    pdf.savefig()
                    plt.close() 

    





# %%
