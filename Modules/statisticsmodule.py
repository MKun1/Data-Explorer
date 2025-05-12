import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew


##################################
def load_dataset():
    """Loads a dataset provided by the user."""
    print("\n--- Dataset Loader ---")
    file_path = input("Enter the path to the CSV file: ")
    try:
        data = pd.read_csv(file_path)
        print("\nDataset loaded successfully.")
        print(f"Preview:\n{data.head()}")
        return data
    except Exception as e:
        print(f"Error: Could not load the dataset. {e}")
        return None

##################################################


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

# Define the AutomatedPDFReport class

class AutomatedPDFReport:
    def __init__(self):
        self.output_buffer = StringIO()
        self.visualizations = []
        self.vis_counter = 0

    def log_terminal_output(self, message, is_user_input=False):
        formatted_message = f"> {message}" if is_user_input else message
        print(message)
        self.output_buffer.write(formatted_message + "\n")

    def save_visualization(self, fig, title=None):
        """Returns and inserts a visualization marker"""
        vis_id = len(self.visualizations)
        self.visualizations.append((fig, title))
        marker = f"%%VIS_{vis_id}%%"  # Simplified marker format
        self.log_terminal_output(marker)  # Insert into output
        return marker

    def generate_pdf(self, output_file="automated_report.pdf"):
        """Generates PDF with user input in blue color"""
        with PdfPages(output_file) as pdf:
            print("\n--- Generating PDF ---")
            text_content = self.output_buffer.getvalue()
            lines = text_content.split("\n")
            
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            y_position = 0.95
            
            for line in lines:
                # Visualization marker handling (keep existing code)
                if "%%VIS_" in line and "%%" in line:
                    try:
                        vis_id = int(line.split("%%VIS_")[1].split("%%")[0])
                        print(f"Inserting visualization {vis_id}")
                        
                        pdf.savefig()
                        plt.close()
                        
                        if vis_id < len(self.visualizations):
                            fig, _ = self.visualizations[vis_id]
                            pdf.savefig(fig)
                            plt.close()
                        
                        plt.figure(figsize=(10, 8))
                        plt.axis('off')
                        y_position = 0.95
                        continue
                    except Exception as e:
                        print(f"Error processing marker: {e}")
                
                # New: Color handling for prompts/user input
                if line.startswith("> ") or "<USER INPUT>" in line:
                    color = 'blue'  # User input color
                    line = line.replace("<USER INPUT>", "").replace("</USER INPUT>", "")
                else:
                    color = 'black'  # Default color
                
                # Page management
                if y_position < 0.1:
                    pdf.savefig()
                    plt.close()
                    plt.figure(figsize=(10, 8))
                    plt.axis('off')
                    y_position = 0.95
                
                plt.text(0.1, y_position, line, 
                        fontsize=10,
                        wrap=True,
                        ha='left',
                        va='top',
                        color=color)  # Apply color here
                y_position -= 0.05
            
            pdf.savefig()
            plt.close()

        print(f"\nPDF generated with blue prompts: {output_file}")






###################################################################################



def descriptive_statistics(data, report):
    """Generates descriptive statistics for numeric columns with automatic insights."""
    report.log_terminal_output("\n--- Descriptive Statistics ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    # Ask the user if they want insights for specific columns
    report.log_terminal_output(f"\nAvailable numeric columns: {list(numeric_columns)}")
    prompt = "Enter column names (comma-separated) for which to generate insights, or press Enter to include all: "
    print(prompt)
    selected_columns = input().split(',')
    report.log_terminal_output(f"{prompt}{', '.join(selected_columns)}", is_user_input=True)

    # If specific columns are chosen, filter them
    selected_columns = [col.strip() for col in selected_columns] if selected_columns != [''] else numeric_columns

    for col in selected_columns:
        if col not in numeric_columns:
            report.log_terminal_output(f"Column '{col}' is not numeric or does not exist. Skipping...")
            continue

        report.log_terminal_output(f"\nColumn: {col}")
        col_data = data[col].dropna()

        # Basic measures
        mean = col_data.mean()
        median = col_data.median()
        mode = col_data.mode().iloc[0] if not col_data.mode().empty else None
        data_range = col_data.max() - col_data.min()
        variance = col_data.var()
        std_dev = col_data.std()
        iqr = col_data.quantile(0.75) - col_data.quantile(0.25)
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)

        # Log basic measures
        measures = f"""
        Mean: {mean:.2f}
        Median: {median:.2f}
        Mode: {mode:.2f} {"(No mode)" if mode is None else ""}
        Range: {data_range:.2f}
        Variance: {variance:.2f}
        Standard Deviation: {std_dev:.2f}
        Interquartile Range (IQR): {iqr:.2f}
        25th Percentile (Q1): {q1:.2f}
        75th Percentile (Q3): {q3:.2f}
        """
        report.log_terminal_output(measures)

        # Generate insights
        insights = "\n--- Insights ---\n"
        if abs(mean - median) > std_dev / 2:
            insights += "The mean and median differ significantly, suggesting skewness in the data.\n"
        if data_range > 4 * std_dev:
            insights += "The range is notably large compared to the standard deviation, indicating potential outliers.\n"
        if iqr < std_dev:
            insights += "The interquartile range is smaller than the standard deviation, suggesting most data points are close to the center.\n"
        if variance == 0:
            insights += f"The variance is zero, indicating all values in '{col}' are identical.\n"
        else:
            insights += "The variance indicates the spread of data points around the mean.\n"
        if std_dev / mean > 0.5:
            insights += "The coefficient of variation is high, suggesting considerable variability in the data.\n"
        else:
            insights += "The data shows moderate variability with respect to the mean.\n"
        report.log_terminal_output(insights)

    report.log_terminal_output("\nSummary statistics and insights generated successfully.")



##########################################
def visualize_distributions(data, report):
    """Visualizes data distributions for numeric columns, allowing the user to choose columns interactively."""
    report.log_terminal_output("\n--- Data Distribution Visualization ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    for col in numeric_columns:
        prompt = f"Do you want to generate visualizations for '{col}'? (yes/no): "
        print(prompt)
        choice = input().strip().lower()
        report.log_terminal_output(f"{prompt}{choice}", is_user_input=True)

        if choice != 'yes':
            report.log_terminal_output(f"Skipping visualizations for column: {col}")
            continue

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Histogram
        sns.histplot(data[col].dropna(), kde=False, ax=axes[0], bins=20, color='blue')
        axes[0].set_title(f"Histogram of {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frequency")

        # Boxplot
        sns.boxplot(x=data[col].dropna(), ax=axes[1], color='green')
        axes[1].set_title(f"Boxplot of {col}")
        axes[1].set_xlabel(col)

        # Density Plot
        sns.kdeplot(data[col].dropna(), ax=axes[2], color='red')
        axes[2].set_title(f"Density Plot of {col}")
        axes[2].set_xlabel(col)
        axes[2].set_ylabel("Density")

        plt.tight_layout()
        plt.show()  # Display the visualization
        report.save_visualization(fig, f"Distributions of {col}")

    report.log_terminal_output("\nVisualizations generated successfully.")



################################################

def calculate_distribution_stats(data, report):
    """Calculates skewness and kurtosis for numeric columns."""
    report.log_terminal_output("\n--- Skewness and Kurtosis Analysis ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    for col in numeric_columns:
        report.log_terminal_output(f"\nColumn: {col}")
        col_data = data[col].dropna()

        # Calculate skewness and kurtosis
        col_skewness = skew(col_data)
        col_kurtosis = kurtosis(col_data)

        # Log results
        results = f"""
        Skewness: {col_skewness:.2f}
        Kurtosis: {col_kurtosis:.2f}
        """
        if col_skewness > 1:
            results += "The data is highly positively skewed (right-skewed).\n"
        elif col_skewness < -1:
            results += "The data is highly negatively skewed (left-skewed).\n"
        else:
            results += "The data is approximately symmetric.\n"

        if col_kurtosis > 3:
            results += "The data has heavy tails (leptokurtic).\n"
        elif col_kurtosis < 3:
            results += "The data has light tails (platykurtic).\n"
        else:
            results += "The data is mesokurtic (normal-like).\n"

        report.log_terminal_output(results)

    report.log_terminal_output("\nSkewness and kurtosis analysis completed successfully.")




##################################################################

from scipy.stats import norm, kstest, shapiro
import numpy as np
def fit_normal_distribution(data, report):
    """Fits numeric columns to a normal distribution with visualization and statistical tests."""
    report.log_terminal_output("\n--- Fit Data to Normal Distribution ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    # Loop through numeric columns
    for col in numeric_columns:
        report.log_terminal_output(f"\nAnalyzing column: {col}")
        col_data = data[col].dropna()  # Drop missing values for analysis

        # Fit to normal distribution
        mean, std_dev = norm.fit(col_data)  # Estimate parameters of normal distribution
        report.log_terminal_output(f"Fitted Normal Parameters - Mean: {mean:.2f}, Std Dev: {std_dev:.2f}")

        # Ask the user if they want the visualization
        prompt = f"Do you want to generate the visualization for '{col}'? (yes/no): "
        print(prompt)  # Display prompt to the user
        choice = input().strip().lower()  # Capture user input
        report.log_terminal_output(f"{prompt}{choice}", is_user_input=True)  # Log both prompt and user input

        if choice == 'yes':
            # Visualize histogram with normal curve
            sns.histplot(col_data, kde=False, bins=20, color='blue', label='Observed Data', stat='density')
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mean, std_dev)
            plt.plot(x, p, 'r-', label='Normal Fit')
            plt.title(f"Histogram with Normal Fit - {col}")
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            
            # Save visualization explicitly
            print(f"Saving visualization for column: {col}")  # Debugging log
            report.save_visualization(plt.gcf(), f"Normal Fit Visualization - {col}")  # Save before closing
            plt.close()  # Ensure the plot is closed after saving

        # Perform statistical tests
        ks_stat, ks_p_value = kstest(col_data, 'norm', args=(mean, std_dev))
        shapiro_stat, shapiro_p_value = shapiro(col_data)

        # Log statistical test results
        stats_results = f"""
        Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, P-value={ks_p_value:.4f}
        {"KS Test: The data follows a normal distribution (fail to reject null hypothesis)." if ks_p_value > 0.05 else "KS Test: The data does not follow a normal distribution (reject null hypothesis)."}

        Shapiro-Wilk Test: Statistic={shapiro_stat:.4f}, P-value={shapiro_p_value:.4f}
        {"Shapiro-Wilk Test: The data follows a normal distribution (fail to reject null hypothesis)." if shapiro_p_value > 0.05 else "Shapiro-Wilk Test: The data does not follow a normal distribution (reject null hypothesis)."}
        """
        report.log_terminal_output(stats_results)

    report.log_terminal_output("\nNormal distribution analysis completed successfully.")

############################################################

from scipy.stats import uniform, expon
def fit_uniform_distribution(data, report):
    """Fits numeric columns to a uniform distribution and performs visual and statistical tests."""
    report.log_terminal_output("\n--- Fit Data to Uniform Distribution ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    for col in numeric_columns:
        report.log_terminal_output(f"\nAnalyzing column: {col}")
        col_data = data[col].dropna()

        # Fit to uniform distribution
        min_val, max_val = col_data.min(), col_data.max()
        report.log_terminal_output(f"Fitted Uniform Parameters - Min: {min_val:.2f}, Max: {max_val:.2f}")

        # Ask the user if they want the visualization
        prompt = f"Do you want to generate the visualization for '{col}'? (yes/no): "
        print(prompt)
        choice = input().strip().lower()
        report.log_terminal_output(f"{prompt}{choice}", is_user_input=True)

        if choice == 'yes':
            sns.histplot(col_data, kde=False, bins=20, color='blue', label='Observed Data', stat='density')
            x = np.linspace(min_val, max_val, 100)
            p = uniform.pdf(x, loc=min_val, scale=max_val - min_val)
            plt.plot(x, p, 'r-', label='Uniform Fit')
            plt.title(f"Histogram with Uniform Fit - {col}")
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.legend()
            
        # Save visualization explicitly
            print(f"Saving visualization for column: {col}")  # Debugging log
            report.save_visualization(plt.gcf(), f"Uniform Fit Visualization - {col}")  # Save before closing
            plt.close()  # Ensure the plot is closed after saving

       

        # Perform KS test
        ks_stat, ks_p_value = kstest(col_data, 'uniform', args=(min_val, max_val - min_val))
        ks_results = f"""
        Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, P-value={ks_p_value:.4f}
        {"KS Test: The data follows a uniform distribution (fail to reject null hypothesis)." if ks_p_value > 0.05 else "KS Test: The data does not follow a uniform distribution (reject null hypothesis)."}
        """
    report.log_terminal_output(ks_results)


##############################################
def fit_exponential_distribution(data, report):
    """Fits numeric columns to an exponential distribution and performs visual and statistical tests."""
    report.log_terminal_output("\n--- Fit Data to Exponential Distribution ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) == 0:
        report.log_terminal_output("No numeric columns found in the dataset.")
        return

    for col in numeric_columns:
        report.log_terminal_output(f"\nAnalyzing column: {col}")
        col_data = data[col].dropna()

        # Fit to exponential distribution
        loc, scale = expon.fit(col_data)
        report.log_terminal_output(f"Fitted Exponential Parameters - Loc: {loc:.2f}, Scale (1/Lambda): {scale:.2f}")

        # Ask the user if they want the visualization
        prompt = f"Do you want to generate the visualization for '{col}'? (yes/no): "
        print(prompt)
        choice = input().strip().lower()
        report.log_terminal_output(f"{prompt}{choice}", is_user_input=True)

        if choice == 'yes':
            sns.histplot(col_data, kde=False, bins=20, color='blue', label='Observed Data', stat='density')
            x = np.linspace(col_data.min(), col_data.max(), 100)
            p = expon.pdf(x, loc=loc, scale=scale)
            plt.plot(x, p, 'r-', label='Exponential Fit')
            plt.title(f"Histogram with Exponential Fit - {col}")
            plt.xlabel(col)
            plt.ylabel('Density')
            


        # Save visualization explicitly
            print(f"Saving visualization for column: {col}")  # Debugging log
            report.save_visualization(plt.gcf(), f"Exponential Fit Visualization - {col}")  # Save before closing
            plt.close()  # Ensure the plot is closed after saving
   


        

        # Perform KS test
        ks_stat, ks_p_value = kstest(col_data, 'expon', args=(loc, scale))
        ks_results = f"""
        Kolmogorov-Smirnov Test: Statistic={ks_stat:.4f}, P-value={ks_p_value:.4f}
        {"KS Test: The data follows an exponential distribution (fail to reject null hypothesis)." if ks_p_value > 0.05 else "KS Test: The data does not follow an exponential distribution (reject null hypothesis)."}
        """
        report.log_terminal_output(ks_results)


###################################################################

def analyze_correlation_and_covariance(data, report):
    """Computes correlation and covariance for numeric columns and provides insights."""
    report.log_terminal_output("\n--- Correlation and Covariance Analysis ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) < 2:
        report.log_terminal_output("Insufficient numeric columns for correlation or covariance analysis (at least 2 are needed).")
        return

    # Compute correlation matrix
    report.log_terminal_output("\nPearson Correlation Matrix:")
    correlation_matrix = data[numeric_columns].corr(method='pearson')
    report.log_terminal_output(correlation_matrix.to_string())

    # Compute covariance matrix
    report.log_terminal_output("\nCovariance Matrix:")
    covariance_matrix = data[numeric_columns].cov()
    report.log_terminal_output(covariance_matrix.to_string())

    # Provide insights on strong correlations
    report.log_terminal_output("\n--- Insights on Correlation ---")
    threshold = 0.7  # Define a threshold for strong correlation
    insights = ""
    for col1 in numeric_columns:
        for col2 in numeric_columns:
            if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold:
                direction = "positively" if correlation_matrix.loc[col1, col2] > 0 else "negatively"
                insights += f"'{col1}' and '{col2}' are strongly {direction} correlated (Correlation = {correlation_matrix.loc[col1, col2]:.2f}).\n"

    report.log_terminal_output(insights)


###################################################################


###################################################################



# Simple Linear Regression
def simple_linear_regression(data, report):
    """Performs simple linear regression on user-selected variables."""
    report.log_terminal_output("\n--- Simple Linear Regression ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) < 2:
        report.log_terminal_output("Insufficient numeric columns for regression analysis (at least 2 are needed).")
        return

    print(f"\nAvailable numeric columns: {list(numeric_columns)}")
    prompt_target = "Enter the dependent (target) variable: "
    print(prompt_target)  # Display the prompt to the user
    target = input().strip()  # Capture user input directly
    report.log_terminal_output(f"{prompt_target}{target}", is_user_input=True)  # Log both the prompt and the input

    prompt_predictor = "Enter the independent (predictor) variable: "
    print(prompt_predictor)  # Display the prompt to the user
    predictor = input().strip()  # Capture user input directly
    report.log_terminal_output(f"{prompt_predictor}{predictor}", is_user_input=True)  # Log both the prompt and the input

    if target not in numeric_columns or predictor not in numeric_columns:
        report.log_terminal_output("Invalid variable(s) selected. Please ensure both are numeric columns.")
        return

    X = data[[predictor]].dropna()
    y = data[target].dropna()
    X, y = X.align(y, join='inner', axis=0)

    model = LinearRegression()
    model.fit(X, y)

    intercept, coefficient = model.intercept_, model.coef_[0]
    regression_equation = f"{target} = {intercept:.2f} + {coefficient:.2f} * {predictor}"
    report.log_terminal_output(f"\nRegression Equation: {regression_equation}")

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)

    metrics = f"""
    Model Performance Metrics:
    R²: {r2:.2f}
    MSE: {mse:.2f}
    RMSE: {rmse:.2f}
    MAE: {mae:.2f}
    """
    report.log_terminal_output(metrics)

    prompt_choice = "Do you want to visualize the regression line and residuals? (yes/no): "
    print(prompt_choice)  # Display the prompt to the user
    choice = input().strip().lower()  # Capture user input directly
    report.log_terminal_output(f"{prompt_choice}{choice}", is_user_input=True)  # Log both the prompt and the input

    if choice == 'yes':
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X[predictor], y=y, label="Actual Data", color="blue")
        sns.lineplot(x=X[predictor], y=y_pred, label="Regression Line", color="red")
        plt.title(f"Regression Line: {target} vs {predictor}")
        plt.xlabel(predictor)
        plt.ylabel(target)
        plt.legend()
        plt.show()  # Display the visualization
        report.save_visualization(fig, "Regression Line")


###################################################################



# Heatmap Visualization
def visualize_correlation_heatmap(data, report):
    """Generates a heatmap for the correlation matrix with user consent."""
    report.log_terminal_output("\n--- Correlation Heatmap ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) < 2:
        report.log_terminal_output("Insufficient numeric columns for a correlation heatmap (at least 2 are needed).")
        return

    # Ask user if they want the heatmap
    prompt_choice = "Do you want to generate a correlation heatmap? (yes/no): "
    print(prompt_choice)  # Display the prompt to the user
    choice = input().strip().lower()  # Capture user input directly
    report.log_terminal_output(f"{prompt_choice}{choice}", is_user_input=True)  # Log both the prompt and the input

    if choice == 'yes':
        correlation_matrix = data[numeric_columns].corr()
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()  # Display the visualization
        report.save_visualization(fig, "Correlation Heatmap")  # Save for the report


###################################################################



# Scatterplot Visualization
def visualize_scatterplots(data, report):
    """Generates scatterplots for pairwise relationships with user consent."""
    report.log_terminal_output("\n--- Pairwise Scatterplots ---")
    numeric_columns = data.select_dtypes(include=['number']).columns

    if len(numeric_columns) < 2:
        report.log_terminal_output("Insufficient numeric columns for scatterplots (at least 2 are needed).")
        return

    prompt_choice = "Do you want to generate scatterplots for numeric columns? (yes/no): "
    print(prompt_choice)  # Display the prompt to the user
    choice = input().strip().lower()  # Capture user input directly
    report.log_terminal_output(f"{prompt_choice}{choice}", is_user_input=True)  # Log both the prompt and the input

    if choice == 'yes':
        fig = sns.pairplot(data[numeric_columns])
        plt.show()  # Display the visualization
        report.save_visualization(fig.fig, "Pairwise Scatterplots")  # Save for the report




###################################################################



def main():
    """Main function to load a dataset and perform descriptive statistics and visualizations."""
    print("\n--- Statistics Module ---")
    data = load_dataset()
    if data is not None:
        report = AutomatedPDFReport()  # Initialize the report generator
        descriptive_statistics(data, report)
        visualize_distributions(data, report)
        calculate_distribution_stats(data, report)
        fit_normal_distribution(data, report)
        fit_uniform_distribution(data, report)
        fit_exponential_distribution(data, report)
        analyze_correlation_and_covariance(data, report)
        simple_linear_regression(data, report) 
        visualize_correlation_heatmap(data, report)
        visualize_scatterplots(data, report)
        report.generate_pdf("final_statistics_report.pdf")  # Generate the PDF report


if __name__ == "__main__":
    main()
