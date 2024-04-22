# WPV Analysis
In this project we extracted all result from multiple PDF files. With these results correlation analysis and cluster analysis are performed. These results include scores for personality traits, competencies and self-image.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
## Installation
Provide instructions on how to install your project and any dependencies.

```bash
# Clone the repository
git clone https://github.com/your_username/your_project.git

# Install dependencies
pip install -r requirements.txt
```

Keep in mind you cloned the git which means that everything that needs to be published needs to be added to the github account you cloned it from at the end.

## Usage
Before running the `extract_WPV_pdf.py` script, ensure that all PDF files follow a specific naming format:

Everything before WPV in the name does not matter.

- Each PDF file should contain the name of the trainee.
- The name should be in the format: `WPV-FirstName_LastName.pdf`

Example of valid PDF names:
- `report_Werkgerelateerde_Persoonlijkheidsvragenlijst_Adaptief-WPV-Luka_Hoeben.pdf`
Invalid PDF names may result in incorrect extraction of the names for identity.csv.

The second step is to change the directory names in the extract_WPV_pdf.

- **DIRECTORY_PDF and DIRECTORY_EXCEL need to be changed !!**

Once you have ensured that all PDF files meet this naming convention, you can run the script as follows:

```bash
python extract_WPV_pdf.py 
```

the correlation_analysis.py and cluster_analysis.py can be run without any changes.

```bash
# Run correlation analysis
python Scripts/correlation_analysis.py Scripts/config.json 
# Run cluster analysis
python Scripts/cluster_analysis.py Scripts/config.json 
```

## Documentation
For more detailed documentation on each function and how to use them, please refer to the docstrings within the Python code.

