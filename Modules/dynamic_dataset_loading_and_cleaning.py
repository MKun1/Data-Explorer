import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages
from io import StringIO
import sys



class AutomatedPDFReport:
    def __init__(self):
        self.output_buffer = StringIO()  # To store logged messages for the PDF
        self.visualizations = []  # To store Matplotlib figures for visualizations
        self.vis_counter = 0  # Counter for visualizations

    def log_terminal_output(self, message, is_user_input=False):
        """Logs terminal output for inclusion in the PDF."""
        formatted_message = f"> {message}" if is_user_input else message
        print(message)  # Display in terminal
        self.output_buffer.write(formatted_message + "\n")  # Save to buffer

    def save_visualization(self, fig, title=None):
        """Saves a visualization and returns a marker for placement in the PDF."""
        vis_id = len(self.visualizations)
        self.visualizations.append((fig, title))  # Append figure and optional title
        marker = f"%%VIS_{vis_id}%%"
        self.log_terminal_output(marker)  # Insert marker into log
        return marker

    def generate_pdf(self, output_file="automated_report.pdf"):
        """Generates a PDF report summarizing all logged actions and visualizations."""
        with PdfPages(output_file) as pdf:
            print("\n--- Generating PDF ---")
            text_content = self.output_buffer.getvalue()  # Get all logged text
            lines = text_content.split("\n")  # Split into lines for processing

            plt.figure(figsize=(10, 8))  # Start a new figure for the first page
            plt.axis('off')  # Remove axes for text rendering
            y_position = 0.95  # Initial vertical position for rendering text

            for line in lines:
                # Check for visualization markers and handle them
                if "%%VIS_" in line and "%%" in line:
                    try:
                        vis_id = int(line.split("%%VIS_")[1].split("%%")[0])
                        print(f"Inserting visualization {vis_id}")
                        
                        pdf.savefig()  # Save the current text page
                        plt.close()  # Close text rendering figure

                        if vis_id < len(self.visualizations):  # Ensure the visualization exists
                            fig, title = self.visualizations[vis_id]
                            if title:
                                plt.figure(figsize=(10, 1))  # Add title as a separate page
                                plt.axis('off')
                                plt.text(0.5, 0.5, title, fontsize=12, va='center', ha='center', color='black')
                                pdf.savefig()
                                plt.close()
                            pdf.savefig(fig)  # Save the visualization
                            plt.close()

                        plt.figure(figsize=(10, 8))  # Start a new page for text
                        plt.axis('off')
                        y_position = 0.95  # Reset vertical position
                        continue
                    except Exception as e:
                        print(f"Error processing marker: {e}")

                # Handle user input or prompts in blue color
                color = 'blue' if line.startswith("> ") else 'black'
                
                # Page management for text
                if y_position < 0.1:  # If text overflows the page, start a new page
                    pdf.savefig()
                    plt.close()
                    plt.figure(figsize=(10, 8))
                    plt.axis('off')
                    y_position = 0.95

                # Render text
                plt.text(0.1, y_position, line, 
                         fontsize=10, wrap=True, ha='left', va='top', color=color)
                y_position -= 0.05  # Adjust vertical spacing

            # Save the last page
            pdf.savefig()
            plt.close()

        print(f"\nPDF generated successfully: {output_file}")




# Function to upload dataset
def upload_dataset(report):
    """Uploads the dataset and logs the process."""
    file_path = input("Enter the file path for your dataset (CSV file): ")
    report.log_terminal_output(f"User provided file path: {file_path}", is_user_input=True)
    
    try:
        data = pd.read_csv(file_path)
        report.log_terminal_output(f"Dataset loaded successfully! It has {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        report.log_terminal_output(f"Error: Could not load dataset. {e}")
        return None


# Function to detect issues in the dataset
def detect_issues(data, report):
    """Detects and logs dataset issues."""
    report.log_terminal_output("\n--- Detecting Issues ---")
    issues = {
        "missing_values": data.isnull().sum(),
        "duplicates": data.duplicated().sum(),
        "data_types": data.dtypes,
        "negative_values": {}
    }

    # Log missing values
    report.log_terminal_output("Missing Values Per Column:")
    report.log_terminal_output(issues["missing_values"].to_string())

    # Log duplicates
    report.log_terminal_output(f"Duplicate Rows: {issues['duplicates']}")

    # Log data types
    report.log_terminal_output("Column Data Types:")
    report.log_terminal_output(issues["data_types"].to_string())

    # Log negative values
    for col in data.select_dtypes(include=[np.number]).columns:
        negative_count = (data[col] < 0).sum()
        if negative_count > 0:
            issues["negative_values"][col] = negative_count
    if issues["negative_values"]:
        report.log_terminal_output("Columns with Negative Values:")
        report.log_terminal_output(str(issues["negative_values"]))
    else:
        report.log_terminal_output("No negative values detected.")
    
    return issues



#################################################################


def detect_boolean_columns(data, report):
    """Identify and log boolean-like columns."""
    boolean_columns = [
        col for col in data.columns
        if data[col].dropna().isin([True, False]).all()
    ]
    report.log_terminal_output(f"Detected boolean-like columns: {boolean_columns}")
    return boolean_columns




#################################################################


def clean_data(data, report):
    report.log_terminal_output("\n--- Cleaning Data ---")

    # Step 0: Detect Boolean-Like Columns Early
    boolean_columns = detect_boolean_columns(data, report)
    if boolean_columns:
        report.log_terminal_output(f"Detected boolean-like columns: {boolean_columns}")

    # Identify other column types for additional cleaning
    text_columns = data.select_dtypes(include=['object']).columns
    price_columns = [col for col in data.columns if 'price' in col.lower() or 'cost' in col.lower() or 'amount' in col.lower()]
    discount_columns = [col for col in data.columns if 'discount' in col.lower()]
    phone_columns = [col for col in data.columns if 'phone' in col.lower() or 'contact' in col.lower()]

    # Automatically detect date columns
    date_columns = [
        col for col in data.columns
        if pd.api.types.is_datetime64_any_dtype(data[col])
        or col.lower().startswith("date")
        or col.lower().endswith("date")
    ]

    if date_columns:
        for date_col in date_columns:
            report.log_terminal_output(f"\nDetected date-like column: '{date_col}'")

            # Flexible Parsing with Multiple Formats
            def parse_dates(value):
                formats_to_try = [
                    "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"
                ]
                for fmt in formats_to_try:
                    try:
                        return pd.to_datetime(value, format=fmt)
                    except ValueError:
                        continue
                return pd.NaT

            # Apply parsing logic
            data[date_col] = data[date_col].apply(parse_dates)
            report.log_terminal_output(f"Attempted to parse '{date_col}' with multiple formats. Invalid entries replaced with NaT.")

            # Handle Missing or Invalid Dates
            missing_dates = data[date_col].isnull().sum()
            if missing_dates > 0:
                report.log_terminal_output(f"The column '{date_col}' has {missing_dates} missing or invalid entries.")
                choice = input(f"How would you like to handle missing or invalid entries in '{date_col}'?\n"
                               "1. Drop rows with missing dates\n"
                               "2. Fill with a default placeholder (e.g., '1900-01-01')\n"
                               "3. Fill based on sequential interpolation (forward/backward fill)\n"
                               "4. Replace invalid entries with a user-defined date format\n"
                               "Enter your choice: ")
                report.log_terminal_output(f"User chose: {choice}", is_user_input=True)
                if choice == "1":
                    data = data.dropna(subset=[date_col])
                    report.log_terminal_output(f"Rows with missing or invalid dates in '{date_col}' have been dropped.")
                elif choice == "2":
                    placeholder = input(f"Enter the placeholder date for '{date_col}' (default: '1900-01-01'): ")
                    placeholder = pd.to_datetime(placeholder) if placeholder else pd.to_datetime("1900-01-01")
                    data[date_col] = data[date_col].fillna(placeholder)
                    report.log_terminal_output(f"Missing dates in '{date_col}' have been filled with '{placeholder}'.")
                elif choice == "3":
                    method = input(f"Choose interpolation method for '{date_col}' ('ffill' or 'bfill'): ")
                    if method in ['ffill', 'bfill']:
                        data[date_col] = data[date_col].fillna(method=method)
                        report.log_terminal_output(f"Missing dates in '{date_col}' filled using '{method}' interpolation.")
                    else:
                        report.log_terminal_output(f"Invalid interpolation method for '{date_col}'. Skipping.")
                elif choice == "4":
                    user_date = input(f"Enter a valid date to replace invalid entries in '{date_col}': ")
                    try:
                        user_date = pd.to_datetime(user_date)
                        data[date_col] = data[date_col].fillna(user_date)
                        report.log_terminal_output(f"Invalid entries in '{date_col}' replaced with '{user_date}'.")
                    except Exception as e:
                        report.log_terminal_output(f"Error: Unable to process user-defined date. {e}. Skipping.")
                else:
                    report.log_terminal_output(f"Invalid choice for '{date_col}'. Skipping handling of dates.")

            # Validate and Standardize Format
            if not data[date_col].isnull().sum():
                desired_format = '%Y-%m-%d'
                data[date_col] = data[date_col].dt.strftime(desired_format)
                report.log_terminal_output(f"'{date_col}' standardized to '{desired_format}' format.")

            # Ensure Chronological Order
            data = data.sort_values(by=date_col).reset_index(drop=True)
            report.log_terminal_output(f"'{date_col}' cleaned and sorted chronologically.")
    else:
        report.log_terminal_output("No date-like columns detected.")

    # Clean Text Columns
    for col in text_columns:
        if data[col].dropna().isin([True, False]).all():
            report.log_terminal_output(f"Skipping boolean column: '{col}'")
            continue
        report.log_terminal_output(f"Cleaning text-based column: '{col}'")
        data[col] = data[col].astype(str).str.replace(r'[^\w\s]', '', regex=True).str.strip().str.lower()
        new_missing_values = data[col].isnull().sum()
        if new_missing_values > 0:
            report.log_terminal_output(f"\nColumn '{col}' now has {new_missing_values} missing values after cleaning.")
            choice = input(f"How would you like to handle missing values in '{col}'?\n"
                           "1. Drop rows\n2. Fill with a placeholder\nEnter your choice: ")
            report.log_terminal_output(f"User chose: {choice}", is_user_input=True)
            if choice == "1":
                data = data.dropna(subset=[col])
                report.log_terminal_output(f"Rows with missing values in '{col}' dropped.")
            elif choice == "2":
                placeholder = input(f"Enter a placeholder value for '{col}': ")
                data[col] = data[col].fillna(placeholder)
                report.log_terminal_output(f"Missing values in '{col}' filled with '{placeholder}'.")

    # Clean Price Columns
    for col in price_columns:
        report.log_terminal_output(f"Cleaning price column: '{col}'")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce').round(2)

    # Clean Discount Columns
    for col in discount_columns:
        report.log_terminal_output(f"Cleaning discount column: '{col}'")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace('%', '', regex=True), errors='coerce')

    # Clean Phone Number Columns
    for col in phone_columns:
        report.log_terminal_output(f"Cleaning phone number column: '{col}'")
        data[col] = data[col].astype(str).str.replace(r'[^\d]', '', regex=True).apply(lambda x: x if len(x) == 10 else None)

    

    
    #Handle Outliers
    for col in data.select_dtypes(include=[np.number]).columns:
        report.log_terminal_output(f"\nAnalyzing column '{col}' for outliers...")
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        report.log_terminal_output(f"Column '{col}' contains {outliers} outliers (below {lower_bound} or above {upper_bound}).")
        
        if outliers > 0:
            choice = input(f"How would you like to handle outliers in '{col}'?\n"
                           "1. Replace with bounds (cap values to lower/upper thresholds)\n"
                           "2. Remove rows containing outliers\n"
                           "3. Skip\n"
                           "Enter your choice: ")
            report.log_terminal_output(f"User chose option {choice}.", is_user_input=True)
            if choice == "1":
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                report.log_terminal_output(f"Outliers in '{col}' capped to range [{lower_bound}, {upper_bound}].")
            elif choice == "2":
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                report.log_terminal_output(f"Rows containing outliers in '{col}' have been removed.")
            elif choice == "3":
                report.log_terminal_output(f"Skipping outlier handling for '{col}'.")
            else:
                report.log_terminal_output(f"Invalid choice. Skipping outlier handling for '{col}'.")



    # Handle missing values
  
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            report.log_terminal_output(f"\nColumn '{col}' has {data[col].isnull().sum()} missing values.")
            choice = input(f"How would you like to handle missing values in '{col}'?\n"
                           "1. Drop rows with missing values in this column\n"
                           "2. Fill with mean (for numerical columns only)\n"
                           "3. Fill with median (for numerical columns only)\n"
                           "4. Fill with a placeholder (for categorical columns only)\n"
                           "Enter your choice: ")
            report.log_terminal_output(f"User chose option {choice}.", is_user_input=True)
            
            if choice == "1":
                data = data.dropna(subset=[col])
                report.log_terminal_output(f"Rows with missing values in '{col}' have been dropped.")
            elif choice == "2" and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].mean())
                report.log_terminal_output(f"Missing values in '{col}' filled with mean value.")
            elif choice == "3" and pd.api.types.is_numeric_dtype(data[col]):
                data[col] = data[col].fillna(data[col].median())
                report.log_terminal_output(f"Missing values in '{col}' filled with median value.")
            elif choice == "4":
                placeholder = input(f"Enter a placeholder value for '{col}': ")
                data[col] = data[col].fillna(placeholder)
                report.log_terminal_output(f"Missing values in '{col}' filled with placeholder: '{placeholder}'.")
            else:
                report.log_terminal_output(f"Invalid choice for column '{col}'. Skipping handling of missing values.")




    # Handle duplicate rows
   
    duplicates_count = data.duplicated().sum()
    report.log_terminal_output(f"\nThe dataset has {duplicates_count} duplicate rows.")
    if duplicates_count > 0:
        choice = input("Do you want to remove duplicate rows? (yes/no): ")
        report.log_terminal_output(f"User chose: {choice}", is_user_input=True)
        if choice.lower() == "yes":
            data = data.drop_duplicates()
            report.log_terminal_output("Duplicate rows have been removed.")
  


    # Handle type conflicts
   
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            report.log_terminal_output(f"\nColumn '{col}' contains non-numeric values.")
            choice = input(f"Do you want to convert '{col}' to numeric by coercing errors? (yes/no): ")
            report.log_terminal_output(f"User chose: {choice}", is_user_input=True)
            
            if choice.lower() == "yes":
                data[col] = pd.to_numeric(data[col], errors='coerce')
                report.log_terminal_output(f"Converted '{col}' to numeric. Invalid entries replaced with NaN.")
                
                # Handle newly generated NaN values
                new_missing_values = data[col].isnull().sum()
                if new_missing_values > 0:
                    report.log_terminal_output(f"Column '{col}' now has {new_missing_values} NaN values due to coercion.")
                    choice = input(f"How would you like to handle these NaN values in '{col}'?\n"
                                   "1. Drop rows with NaN values\n"
                                   "2. Fill with mean\n"
                                   "3. Fill with median\n"
                                   "4. Fill with a placeholder\n"
                                   "Enter your choice: ")
                    report.log_terminal_output(f"User chose option {choice}.", is_user_input=True)
                    
                    if choice == "1":
                        data = data.dropna(subset=[col])
                        report.log_terminal_output(f"Rows with NaN values in '{col}' have been dropped.")
                    elif choice == "2":
                        data[col] = data[col].fillna(data[col].mean())
                        report.log_terminal_output(f"NaN values in '{col}' filled with mean value.")
                    elif choice == "3":
                        data[col] = data[col].fillna(data[col].median())
                        report.log_terminal_output(f"NaN values in '{col}' filled with median value.")
                    elif choice == "4":
                        placeholder = input(f"Enter a placeholder value for '{col}': ")
                        data[col] = data[col].fillna(placeholder)
                        report.log_terminal_output(f"NaN values in '{col}' filled with placeholder: '{placeholder}'.")
                    else:
                        report.log_terminal_output(f"Invalid choice for column '{col}'. Skipping handling of NaN values.")
    


    # Handle negative values
 
    for col in data.select_dtypes(include=[np.number]).columns:
        negative_count = (data[col] < 0).sum()
        if negative_count > 0:
            report.log_terminal_output(f"\nColumn '{col}' has {negative_count} negative values.")
            choice = input(f"How would you like to handle negative values in '{col}'?\n"
                           "1. Replace with 0\n"
                           "2. Replace with absolute values\n"
                           "3. Drop rows with negative values\n"
                           "Enter your choice: ")
            report.log_terminal_output(f"User chose option {choice}.", is_user_input=True)
            
            if choice == "1":
                data[col] = data[col].apply(lambda x: 0 if x < 0 else x)
                report.log_terminal_output(f"Negative values in '{col}' replaced with 0.")
            elif choice == "2":
                data[col] = data[col].abs()
                report.log_terminal_output(f"Negative values in '{col}' replaced with absolute values.")
            elif choice == "3":
                data = data[data[col] >= 0]
                report.log_terminal_output(f"Rows with negative values in '{col}' have been removed.")
            else:
                report.log_terminal_output(f"Invalid choice. Skipping handling of negative values for '{col}'.")
   


    # Save the cleaned dataset
    save_choice = input("\nDo you want to save the cleaned dataset to a new file? (yes/no): ")
    if save_choice.lower() == "yes":
        file_name = input("Enter the filename to save the cleaned dataset (default: 'cleaned_dataset.csv'): ")
        file_name = file_name if file_name else "cleaned_dataset.csv"
        try:
            data.to_csv(file_name, index=False)
            print(f"Cleaned dataset saved as '{file_name}'.")
        except Exception as e:
            print(f"Error: Could not save the file. {e}")
    
    # Final dataset preview
    print("\n--- Cleaned Dataset Preview ---")
    print(data.head())

    return data




##########################################################33

# Main program flow
if __name__ == "__main__":
    # Initialize the PDF report
    report = AutomatedPDFReport()
    
    # Step 1: Upload the dataset
    dataset = upload_dataset(report)  # Pass the report object to log actions
    if dataset is not None:
        # Step 2: Detect issues in the dataset
        detect_issues(dataset, report)  # Log detected issues
        
        # Step 3: Clean the dataset
        cleaned_dataset = clean_data(dataset, report)  # Log cleaning process
        
        # Step 9: Save final cleaned dataset (if user chooses)
        report.log_terminal_output("\n--- Final Cleaned Dataset ---")
        report.log_terminal_output(cleaned_dataset.head().to_string())  # Log a preview of the cleaned dataset

        # Step 10: Generate the PDF report
        print("\nGenerating PDF summary...")
        report.generate_pdf("cleaning_summary_report.pdf")  # Save the PDF summary
        print("PDF report generated successfully as 'cleaning_summary_report.pdf'!")

