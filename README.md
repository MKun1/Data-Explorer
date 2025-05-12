# Data-Explorer
Data Explorer is a comprehensive Python toolkit designed to simplify basic data analysis.
Data Explorer - Automated Data Analysis Toolkit
**Overview**
Data Explorer is a comprehensive Python toolkit designed to simplify basic data analysis tasks through an automated workflow. The project combines data cleaning capabilities with statistical analysis and visualization tools, while maintaining detailed logs and generating professional PDF reports.

**Key Features**
Data Cleaning Module (dynamic_dataset_loading_and_cleaning.py)
Smart Dataset Loading
Handles CSV files with informative feedback

Comprehensive Issue Detection:

•	Missing values identification

•	Duplicate row detection

•	Data type analysis

•	Negative value detection

•	Automated Cleaning Workflow:

•	Date column parsing with multiple format support

•	Text column standardization

•	Price/discount/phone number formatting

•	Outlier handling (capping or removal)

•	Missing value imputation strategies

•	Duplicate removal

•	Type conversion with error handling

•	Interactive Cleaning: User-guided decisions at each cleaning step

•	Report Generation: Detailed PDF logs of all cleaning operations

**Statistical Analysis Module (statisticsmodule.py)
Descriptive Statistics:**

•	Comprehensive measures (mean, median, mode, range, variance, etc.)

•	Automatic data insights generation

•	Distribution Analysis:

•	Skewness and kurtosis calculations

•	Normal, uniform, and exponential distribution fitting

•	Statistical tests (Kolmogorov-Smirnov, Shapiro-Wilk)

•	Relationship Analysis:

•	Correlation and covariance matrices

•	Simple linear regression

•	Visualizations:

•	Distribution plots (histograms, boxplots, density plots)

•	Correlation heatmaps

•	Pairwise scatterplots

•	Regression visualizations

•	Interactive Workflow: User-selectable columns and options

•	Automated Reporting: PDF generation with all analysis results



# Installation
Clone the repository:

bash
git clone https://github.com/yourusername/data-explorer.git
cd data-explorer
Install required dependencies:

bash
pip install -r requirements.txt
(Create a requirements.txt file with these packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn)

# Usage
Data Cleaning Module
python
from dynamic_dataset_loading_and_cleaning import AutomatedPDFReport, upload_dataset, detect_issues, clean_data

# Initialize reporting
report = AutomatedPDFReport()

# Load and analyze data
dataset = upload_dataset(report)
if dataset is not None:
    detect_issues(dataset, report)
    cleaned_data = clean_data(dataset, report)
    report.generate_pdf("cleaning_report.pdf")
Statistical Analysis Module
python
from statisticsmodule import main

# Run complete statistical analysis workflow
main()  # Follow interactive prompts
Example Workflow
o	Load your dataset (CSV format)

o	Review automatically detected issues

o	Make cleaning choices through interactive prompts

o	Perform statistical analysis on cleaned data:

o	Descriptive statistics

o	Distribution fitting

o	Correlation analysis

o	Regression modeling

o	Generate comprehensive PDF reports

o	Output Samples
o	The system generates detailed PDF reports containing:

o	All user inputs and choices (in blue)

o	Statistical results and insights

o	Visualizations of distributions and relationships

o	Cleaning operation logs

o	Final dataset previews

# Dependencies
o	Python 3.7+

o	pandas

o	numpy

o	matplotlib

o	seaborn

o	scipy

o	scikit-learn

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
MIT License

