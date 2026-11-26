## 1000411-sarveshwaran-Hospital-Inpatient-Discharges

# 🏥 MediScope

## 3D Hospital Inpatient Discharge Analytics Dashboard

MediScope is an interactive hospital data analytics dashboard designed to explore and analyze inpatient discharge records.

The application uses **Python, Pandas, NumPy, Plotly, Streamlit, scikit-learn, and Three.js** to process hospital data and present meaningful insights through interactive visualizations, statistical analysis, machine learning, and 3D graphics.

The dashboard helps users investigate patterns related to **patient demographics, diagnosis categories, length of hospital stay, healthcare charges, illness severity, payment types, hospitals, and counties**.

MediScope also includes interactive 3D charts, an animated medical background, and an offline virtual assistant named Dr. MediBot.

---

## 📌 Project Overview

Hospital inpatient discharge data contains information about patients who have been discharged from hospitals, including their demographics, diagnosis categories, length of stay, healthcare charges, and payment information.

Analyzing this information can help users understand patterns in hospital discharge records and compare different categories of healthcare data.

MediScope transforms the dataset into an interactive analytics platform.

The system provides:

- Total number of discharge records
- Average and median length of hospital stay
- Average and total recorded healthcare charges
- Number of hospitals and counties represented
- Patient age-group distribution
- Payment-type distribution
- Average length of stay by severity
- Diagnosis-based length-of-stay analysis
- Healthcare charge comparisons
- Facility and county comparisons
- Missing-value analysis
- Potential length-of-stay outlier detection
- Mathematical and statistical analysis
- Interactive 3D visualizations
- Machine-learning severity prediction experiment
- Interactive dashboard filters
- Raw and cleaned dataset viewing
- Filtered dataset downloads
- Graph explanations through `?` help buttons
- Dr. MediBot virtual assistant
- Dark and light-mix dashboard themes
- Animated Three.js medical background

---

## 🎯 Project Objective

The main objective of MediScope is to develop an interactive hospital data analytics dashboard that allows users to investigate patterns within inpatient discharge records.

The project focuses on:

1. Loading and processing hospital inpatient discharge data.
2. Detecting and standardizing relevant dataset columns.
3. Cleaning and preparing numerical and categorical information.
4. Applying interactive filters to hospital records.
5. Calculating statistical measures such as mean, median, and standard deviation.
6. Investigating relationships between hospital stay duration and healthcare charges.
7. Comparing patient demographics and illness severity.
8. Analyzing discharge records across hospitals and counties.
9. Presenting data through interactive 2D and 3D visualizations.
10. Demonstrating machine-learning classification using scikit-learn.
11. Explaining the mathematical calculations behind the visualizations.
12. Providing an interactive platform for exploring hospital data.

---

## 📊 Dataset

The application uses the following dataset:

**Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv**

The dataset contains de-identified hospital inpatient discharge information.

The application can process up to **250,000 rows** from the supplied CSV dataset for analysis.

### Dataset Information

The application identifies and uses information such as:

- Record ID
- Patient Age Group
- Gender
- Race
- Severity of Illness
- Diagnosis Description
- Length of Stay
- Total Charges
- Payment Typology
- Facility Name
- Facility County
- Health Service Area / Region
- Discharge Year
- Discharge Date

The application automatically detects several alternative column names so that the dataset can be processed even when the original column headings differ.

### Missing Data

The source dataset may contain missing or invalid values.

The application handles missing information during preprocessing.

Examples include:

- Missing categorical values are represented as `"Unknown"`.
- Invalid numerical values are converted into missing numerical values.
- Records with missing or non-positive length-of-stay values are removed.
- Records containing negative total charges are removed.
- Non-finite charge values are handled during cleaning.

The application also records the number of original rows, cleaned rows, removed rows, and detected source columns.

---

## 🧹 Data Processing

Before performing analysis, the hospital dataset goes through a preprocessing stage.

The application:

1. Loads the CSV dataset using Pandas.
2. Reads up to 250,000 rows.
3. Handles different CSV encodings.
4. Normalizes column names.
5. Detects relevant source columns using alternative column names.
6. Converts length-of-stay values into numerical formats.
7. Converts total charges into numerical formats.
8. Standardizes patient age groups.
9. Standardizes gender categories.
10. Standardizes severity categories into Minor, Moderate, Major, Extreme, and Unknown.
11. Standardizes payment categories.
12. Handles missing categorical information using `"Unknown"`.
13. Removes records with invalid length-of-stay values.
14. Removes records with negative charges.
15. Limits extremely high length-of-stay values to 3,650 days.
16. Prepares standardized columns for dashboard analysis.
17. Applies the selected sidebar filters.

The processed dataset is then used by the dashboard visualizations, mathematical calculations, and AI experiment.

---

## 📐 Statistical Analysis

MediScope uses statistical calculations to investigate hospital discharge patterns.

### 1. Mean

The mean represents the arithmetic average of a numerical dataset.

For length of stay:

\[
\bar{x}=\frac{\sum_{i=1}^{n}x_i}{n}
\]

Where:

- \(x_i\) = Individual length of stay.
- \(n\) = Number of valid records.
- \(\bar{x}\) = Average length of stay.

### 2. Median

The median represents the middle value when all numerical observations are arranged in ascending order.

It helps identify the central value without being as strongly affected by extreme observations as the mean.

### 3. Standard Deviation

Standard deviation measures the spread of values around the mean.

The application uses the sample standard deviation:

\[
s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar{x})^2}{n-1}}
\]

Where:

- \(s\) = Sample standard deviation.
- \(x_i\) = Individual observation.
- \(\bar{x}\) = Mean.
- \(n\) = Number of valid observations.

### 4. Payment-Type Percentage

The percentage distribution identifies the proportion of discharge records belonging to each payment category.

\[
\text{Percentage}=
\frac{\text{Category Count}}{\text{Total Records}}\times100
\]

### 5. Pearson Correlation

Pearson's correlation coefficient measures the direction and strength of a linear relationship between length of stay and total charges.

\[
r=\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
\]

Where:

- \(X\) = Length of stay.
- \(Y\) = Total recorded charges.
- \(r\) = Pearson correlation coefficient.

The value of \(r\) ranges from -1 to +1.

A positive value indicates a positive linear association, while a negative value indicates a negative linear association.

Correlation does not prove causation.

### 6. Interquartile Range (IQR)

The application uses the interquartile range to identify potential length-of-stay outliers.

\[
IQR=Q_3-Q_1
\]

Upper fence:

\[
Q_3+1.5(IQR)
\]

Lower fence:

\[
\max(0,Q_1-1.5(IQR))
\]

Records outside these fences are flagged as potential statistical outliers. They are not automatically removed.

---

## 📊 Dashboard Visualizations

MediScope contains multiple visualizations organized into different dashboard pages.

### 01 · Overview

The Overview page provides a general summary of the currently filtered hospital discharge records.

It displays four main KPI cards:

- Discharge Records
- Average Length of Stay
- Average Total Charges
- Facilities Represented

It also displays three additional insights:

- Median Stay
- Total Recorded Charges
- Counties Represented

The page includes four visualizations.

**Patient Records by Age Group**

Uses a bar chart to display the number of discharge records in each age group.

**Primary Payment Type**

Uses a donut chart to show the distribution of records across payment categories.

**Length-of-Stay Distribution**

Uses a histogram to display the frequency of different hospital stay durations.

**Average Stay by Severity**

Uses a bar chart to compare average length of stay across illness severity categories.

Purpose:

The Overview page gives users a quick understanding of the hospital discharge dataset and its major patterns.

### 02 · Diagnosis & Stay

The Diagnosis & Stay page investigates how hospital stay duration varies across diagnosis categories and severity groups.

It includes:

**Average Stay per Diagnosis**

Uses a horizontal bar chart to compare the average length of stay across selected diagnosis categories.

Users can adjust the number of diagnosis categories displayed using a slider.

**Length of Stay by Severity**

Uses a boxplot to compare the distribution of hospital stay duration across severity categories.

The boxplot helps examine:

- Median
- First quartile
- Third quartile
- Interquartile range
- Potential outliers

**Diagnosis Summary Table**

Displays:

- Number of records
- Average stay
- Median stay
- Average charges

The diagnosis summary can also be downloaded as a CSV file.

Purpose:

This page helps users examine differences in hospital stay duration across diagnosis and severity categories.

### 03 · Charges & Demographics

The Charges & Demographics page explores healthcare charges across different patient groups.

It displays three main charge statistics:

- Average Charges
- Median Charges
- Total Charges

The page contains three visualizations.

**Charges by Severity**

Uses a boxplot to compare the distribution of total recorded charges across severity categories.

**Average Charges by Age Group and Gender**

Uses a grouped bar chart to compare average charges across patient age groups and gender categories.

**Charge Distribution by Age Group**

Uses a violin plot to display the distribution and density of recorded healthcare charges across age groups.

It also provides a demographic summary table.

Purpose:

This page helps investigate how recorded healthcare charges vary across patient demographics and illness severity.

### 04 · Facilities & Counties

The Facilities & Counties page compares hospital discharge volume and length of stay across healthcare facilities and geographical areas.

It contains three main visualizations.

**Facilities with the Most Discharge Records**

Uses a bar chart to rank facilities by discharge volume.

**Average Length of Stay by Facility**

Compares average length of stay across selected high-volume facilities.

**County and Facility Heatmap**

Uses a heatmap to display average length of stay for combinations of counties and facilities.

The page also provides a facility summary table that can be downloaded as a CSV file.

Purpose:

This page helps users compare the distribution of discharge records and average hospital stay across different facilities and counties.

### 05 · Data Quality

The Data Quality page examines the condition of the source dataset and the preprocessing process.

It displays:

- Original number of rows
- Number of cleaned rows
- Number of removed rows
- Detected source columns
- Preprocessing steps
- Missing-value percentages
- Potential length-of-stay outliers
- Cleaned dataset preview

**Missing-Value Profile**

Uses a bar chart to show the percentage of missing values across dataset columns.

**Length-of-Stay Outliers**

Uses the Interquartile Range method to identify potential outlier records.

The cleaned dataset sample can be downloaded as a CSV file.

Purpose:

This page helps users understand the quality of the source data and how the application prepares it for analysis.

### 06 · Mathematical Analysis

The Mathematical Analysis page explains the statistical calculations used by MediScope.

It contains six main sections:

1. Descriptive statistics for length of stay.
2. Descriptive statistics for healthcare charges.
3. Percentage distribution by payment type.
4. Comparison of discharge records across severity categories.
5. Pearson correlation between length of stay and total charges.
6. Explanation of how mathematical calculations are used by the dashboard.

The results are recalculated using the currently filtered records.

Purpose:

This page demonstrates how mathematical and statistical concepts are applied to real-world hospital discharge data.

### 07 · Math Behind the Graphs

The Math Behind the Graphs page explains how the dashboard converts raw records into mathematical calculations and 3D visualizations.

It contains ten calculation steps.

**Step 1 · Counting Records**

Explains how category counts are calculated and used in bar charts and KPI cards.

\[
n_g=\sum_{i=1}^{n}\mathbf{1}[g_i=g]
\]

**Step 2 · Payment Shares and Angles**

Explains how payment-category percentages and donut chart angles are calculated.

\[
p_g=\frac{n_g}{n}\times100
\]

\[
\theta_g=\frac{n_g}{n}\times360^\circ
\]

**Step 3 · Mean, Median, Standard Deviation and Histogram Bins**

Explains how length-of-stay values are grouped into histogram intervals and how the mean, median, standard deviation, and bin width are calculated.

**Step 4 · Grouped Means**

Explains how average length of stay and charges are calculated for each severity group.

\[
\bar{x}_g=\frac{1}{n_g}\sum_{i\in g}x_i
\]

**Step 5 · Boxplot Quartiles**

Explains how quartiles and the interquartile range represent the distribution of hospital stay durations.

**Step 6 · Violin and Density Distributions**

Explains how recorded healthcare charges are distributed across age groups using density-based visualization.

**Step 7 · Pearson Correlation**

Explains the relationship between length of stay and total charges.

It also displays the correlation value, least-squares regression line, and an interactive 3D scatter plot.

**Step 8 · County and Facility Heatmap**

Explains how the average length of stay is calculated for each county and facility combination.

**Step 9 · AI Model Evaluation**

Explains the mathematical calculations behind classification accuracy, precision, recall, and the confusion matrix.

**Step 10 · Missing-Value Percentages**

Explains how the percentage of missing values is calculated for individual dataset columns.

\[
\text{Missing Percentage}=
\frac{\text{Missing Cells}}{\text{Total Records}}\times100
\]

Purpose:

This page connects mathematical concepts to the actual visualizations produced by the application. Users can examine the formulas, live calculated values, and their corresponding 3D graphs.

### 08 · AI Experiment

The AI Experiment page demonstrates machine-learning classification using scikit-learn.

The purpose is to predict standardized illness severity categories using demographic and administrative information.

Users can select input features such as:

- Age Group
- Gender
- Race
- Payment Type
- Diagnosis
- County

Two model options are available:

**Random Forest**

Uses a Random Forest classification model containing 100 decision trees.

**Dummy Baseline**

Uses a simple most-frequent-class prediction strategy as a baseline for comparison.

Users can also select the percentage of records used for testing.

The application:

1. Prepares the selected input features.
2. Removes records with unknown severity targets.
3. Selects eligible severity classes.
4. Splits the records into training and testing datasets.
5. Applies categorical preprocessing.
6. Trains the selected classification model.
7. Predicts severity categories.
8. Calculates test accuracy.
9. Displays a confusion matrix.
10. Generates a classification report.

The results include:

- Test Accuracy
- Training Records
- Testing Records
- Actual vs Predicted Severity
- Confusion Matrix
- Precision
- Recall
- F1-score

Purpose:

This page demonstrates how machine learning can be applied to structured hospital discharge data.

The experiment is educational and is not validated for clinical use.

### 09 · Data Explorer

The Data Explorer page allows users to search and examine processed hospital discharge records.

Users can search using:

- Facility
- Diagnosis
- County
- Region
- Record ID

The page also provides filters for:

- Length-of-stay range
- Maximum total charges

The filtered records are displayed in an interactive data table.

Users can download the displayed records as a CSV file.

The page also provides a numerical summary of selected columns.

Output file:

`mediscope_filtered_records.csv`

Purpose:

This page allows users to examine individual records and export selected data for further analysis.

---

## 🔎 Interactive Filters

The dashboard provides interactive filters through the sidebar.

Users can filter the dataset using:

- Age Group
- Gender
- Severity of Illness
- Payment Type

The application also includes a 3D Charts toggle.

When filters are changed:

- KPI values update.
- Graphs update.
- Statistical information updates.
- Diagnosis comparisons update.
- Facility and county comparisons update.
- Data Explorer records update.
- Mathematical calculations update.

This allows users to investigate different subsets of the hospital discharge dataset without manually editing the source CSV.

---

## 📈 Key Performance Indicators

The Overview page displays four primary KPIs.

### Discharge Records

Shows the number of hospital discharge records included after applying the selected filters.

### Average Length of Stay

Shows the average number of inpatient days across the currently filtered records.

\[
\text{Average Stay}=\frac{\sum LOS}{n}
\]

### Average Total Charges

Shows the average recorded healthcare charges for the currently filtered records.

### Facilities Represented

Shows the number of distinct healthcare facilities represented in the filtered dataset.

The dashboard also displays median stay, total recorded charges, and the number of counties represented.

---

## 🧊 3D Visualization System

MediScope includes a dedicated 3D visualization engine.

The system uses **Plotly 3D graphics and Three.js** to provide a more interactive visual experience.

The 3D chart engine converts supported Plotly visualizations into 3D representations.

The visualizations include:

- Extruded 3D bar charts
- Solid 3D donut charts
- 3D boxplots
- 3D violin plots
- 3D heatmaps
- 3D scatter plots
- 3D density surfaces

### 3D Chart Controls

Users can interact with the 3D visualizations through:

- Dragging to rotate the graph.
- Scrolling to zoom.
- Hovering over objects to inspect exact values.
- Using the automatic 360-degree rotation control where available.

The 3D Charts toggle allows users to switch between the supported 3D chart presentation and traditional 2D charts.

The Math Behind the Graphs page uses 3D visualizations for its supported graph demonstrations.

### Animated Background

The application also includes a Three.js animated background.

It contains:

- Animated particle effects.
- Floating 3D medical crosses.
- Rotating wireframe geometric objects.
- Animated gradient lighting.
- Mouse-responsive background effects.
- Animated DNA visualization in the main hero section.
- A floating 3D Dr. MediBot character.

The background is designed to remain behind the main dashboard interface.

---

## 🤖 Dr. MediBot

MediScope includes an interactive virtual assistant named **Dr. MediBot**.

Dr. MediBot provides explanations and guidance about the dashboard.

Users can ask questions about:

- Dashboard pages.
- Graph interpretations.
- Mathematical calculations.
- Statistical concepts.
- Interactive filters.
- 3D chart controls.
- Current filtered dataset values.
- AI model accuracy and evaluation.

The assistant includes suggested questions to help users get started.

The chat engine runs locally within the application and does not require an external AI API key.

The floating 3D doctor character provides an interactive entry point to the assistant.

Purpose:

Dr. MediBot helps users understand the dashboard and its analytical features.

---

## ❓ Graph Explanations

Major dashboard visualizations include a `?` help button.

The help button provides an explanation of:

- What the graph represents.
- Which variables are being compared.
- What the axes represent.
- How the graph should be interpreted.
- What patterns users can investigate.

For supported 3D graphs, the explanation also describes how to rotate, zoom, and inspect values.

This feature makes the dashboard easier to understand for users who may not have previous experience with data visualization.

---

## 🎨 Dashboard Customization

MediScope includes a theme-switching feature.

Users can switch between:

**Full Dark Theme**

Displays the interface using a dark appearance.

**Light + Dark Mix**

Combines the animated light-toned background with a dark glass-style dashboard interface.

The application also includes:

- Animated gradient effects.
- Glass-style dashboard panels.
- Animated chart borders.
- Interactive hover effects.
- 3D depth effects.
- Responsive layout.
- Animated medical background objects.

The design aims to maintain readability while presenting an interactive dashboard.

---

## 🛠️ Technologies Used

### Programming Language

**Python**

Python is used for the main application logic, dataset processing, calculations, and machine-learning implementation.

### Data Processing

**Pandas**

Pandas is used for:

- Loading CSV datasets.
- Cleaning data.
- Detecting and standardizing columns.
- Filtering records.
- Grouping data.
- Calculating statistics.
- Preparing summary tables.

**NumPy**

NumPy is used for:

- Numerical calculations.
- Array operations.
- Statistical processing.
- Handling numerical values.
- Preparing visualization data.

### Data Visualization

**Plotly**

Plotly is used to create:

- Bar charts.
- Pie and donut charts.
- Histograms.
- Scatter plots.
- Boxplots.
- Violin plots.
- Heatmaps.
- Interactive 3D visualizations.

### Machine Learning

**scikit-learn**

scikit-learn is used for:

- Data preprocessing.
- Missing-value handling.
- One-hot encoding.
- Training and testing data splitting.
- Random Forest classification.
- Dummy baseline classification.
- Accuracy calculation.
- Classification reports.
- Model evaluation.

### Web Application

**Streamlit**

Streamlit is used to build the interactive dashboard and provide:

- Sidebar navigation.
- Dashboard filters.
- KPI cards.
- Interactive graphs.
- Data tables.
- CSV downloads.
- Mathematical analysis.
- AI experiment controls.
- Dr. MediBot interface.

### 3D Graphics and Animation

**Three.js**

Three.js is used for:

- Animated backgrounds.
- Floating medical objects.
- Particle systems.
- DNA visualization.
- 3D Dr. MediBot character.

**JavaScript and CSS**

JavaScript and CSS are used for:

- 3D scene initialization.
- Animation loops.
- Interactive background effects.
- Theme switching.
- Hover effects.
- Dashboard styling.
- Responsive visual elements.

---

## 📦 Project Structure

MediScope/

```text
MediScope/
│
├── app.py
│
├── Hospital_Inpatient_Discharges__SPARCS_De-Identified___2021_20231012.csv
│
├── requirements.txt
│
├── README.md
│
└── .streamlit/
    └── config.toml
```

The `.streamlit/config.toml` file is created automatically by the application if it does not already exist.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Move into the project folder:

```bash
cd MediScope
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Make sure the hospital discharge CSV file is available in the same folder as `app.py` or inside a supported `data/` or `dataset/` folder.

---

## 📦 Requirements

The project requires the following Python libraries:

```text
streamlit
pandas
numpy
plotly
scikit-learn
```

These dependencies should also be provided in:

`requirements.txt`

The 3D visual layer loads Three.js through a CDN and requires browser access to an available CDN.

---

## ▶️ Running the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in a web browser.

After opening the dashboard:

1. Explore the Overview page.
2. Use the sidebar filters to select patient groups.
3. Open the Diagnosis & Stay page.
4. Compare healthcare charges and demographics.
5. Explore Facilities & Counties.
6. Examine data preprocessing in Data Quality.
7. Open Mathematical Analysis to inspect statistical calculations.
8. Explore Math Behind the Graphs to understand the formulas.
9. Open AI Experiment to train and evaluate a classification model.
10. Use Data Explorer to search and download records.
11. Switch between 2D and 3D charts using the sidebar toggle.
12. Rotate and zoom the interactive 3D graphs.
13. Use the `?` buttons to understand the visualizations.
14. Interact with Dr. MediBot.
15. Change the dashboard theme using the theme switch.

---

## 🔬 Testing

The application should be tested using different filter combinations and dashboard features.

### Filter Testing

Examples include:

- Different age groups.
- Different genders.
- Different severity categories.
- Different payment types.
- Multiple filters at the same time.

Testing should verify that:

- KPI values update correctly.
- Graphs respond to selected filters.
- Statistical calculations update.
- Diagnosis comparisons reflect the selected data.
- Facility and county summaries update.
- Data Explorer records match the filters.

### Visualization Testing

Testing should verify that:

- All dashboard graphs render correctly.
- The 2D and 3D chart toggle works.
- 3D graphs can be rotated.
- Zooming works correctly.
- Hover information displays the correct values.
- Graph explanation buttons display relevant information.
- The animated background does not cover dashboard content.

### Mathematical Testing

Testing should verify that:

- Mean values are calculated correctly.
- Median values are calculated correctly.
- Standard deviation is calculated correctly.
- Payment percentages are calculated correctly.
- Pearson correlation is calculated correctly.
- IQR outlier limits are calculated correctly.
- Mathematical calculations respond to filters.

### AI Experiment Testing

Testing should verify that:

- Feature selection works.
- Random Forest training runs.
- Dummy baseline training runs.
- Training and testing records are separated.
- Accuracy is calculated correctly.
- The confusion matrix displays correctly.
- The classification report is generated.

### Data Export Testing

Testing should verify that:

- Cleaned data can be downloaded.
- Diagnosis summaries can be downloaded.
- Facility summaries can be downloaded.
- Data Explorer records can be downloaded.
- Exported CSV files contain the expected filtered records.

### Interface Testing

Testing should verify that:

- Sidebar navigation works.
- Dr. MediBot opens correctly.
- Theme switching works.
- Text remains readable.
- Dashboard elements remain visible.
- The interface remains usable at different screen sizes.

---

## ⚠️ Limitations

MediScope has several limitations.

### Dataset Dependency

The quality of the analysis depends on the completeness and accuracy of the supplied hospital discharge dataset.

### Sample Size

The application processes up to 250,000 rows from the CSV dataset.

The displayed results therefore represent the loaded sample and may not describe every record in the original dataset.

### Static Dataset

The application uses a supplied CSV file and does not automatically receive live hospital information.

### Correlation Does Not Prove Causation

The statistical analysis identifies associations between variables such as length of stay and total charges.

These relationships do not prove that one variable directly causes changes in another.

### Healthcare Charges

Recorded total charges should not automatically be interpreted as the actual amount paid by a patient or as a measure of healthcare quality.

### Machine-Learning Limitations

The AI experiment is an educational demonstration using the available dataset sample.

Its predictions are not clinically validated and should not be used to make medical decisions.

### 3D Graphics Requirements

The animated background and some 3D features require browser WebGL support and access to a working Three.js CDN.

Performance may vary depending on the device and browser.

### Missing Information

The analysis is limited to the variables included in the dataset.

Additional clinical, operational, and contextual information may be required for more detailed hospital analysis.

---

## 🚀 Future Improvements

Possible future improvements include:

- Adding support for additional hospital datasets.
- Adding date-range filters.
- Adding year-by-year comparisons.
- Adding more advanced statistical analysis.
- Adding additional machine-learning models.
- Improving model evaluation and validation.
- Adding predictive analytics for hospital stay duration.
- Adding more 3D chart types.
- Adding downloadable analytical reports.
- Improving performance with larger datasets.
- Adding additional interactive geographical visualizations.
- Improving mobile responsiveness.
- Adding more advanced data-quality checks.
- Adding a persistent dataset-management system.
- Expanding Dr. MediBot with additional analytical explanations.

---

## 📚 Research and Learning

This project demonstrates the application of several data analytics and computer science concepts.

**Data Processing**

- CSV Data Loading
- Data Cleaning
- Column Detection
- Data Standardization
- Missing-Value Handling
- Data Preprocessing

**Mathematics and Statistics**

- Mean
- Median
- Standard Deviation
- Percentages
- Frequency Distribution
- Quartiles
- Interquartile Range
- Pearson Correlation
- Linear Regression
- Grouped Statistical Analysis

**Data Visualization**

- Bar Chart Analysis
- Donut Chart Analysis
- Histogram Analysis
- Boxplot Analysis
- Violin Plot Analysis
- Heatmap Analysis
- Scatter Plot Analysis
- Interactive 3D Visualization

**Artificial Intelligence and Machine Learning**

- Categorical Data Encoding
- Data Imputation
- Training and Testing Splits
- Random Forest Classification
- Baseline Classification
- Confusion Matrices
- Accuracy
- Precision
- Recall
- F1-score

**Computer Science and Interface Development**

- Python Programming
- JavaScript
- CSS
- Streamlit Application Development
- Three.js
- WebGL
- Interactive User Interfaces
- 3D Graphics and Animation

The project demonstrates how raw hospital discharge information can be transformed into an interactive analytics platform by combining data processing, mathematics, statistics, machine learning, and 3D visualization.

---

## 🌐 Deployment

The application is designed to run using Streamlit.

It can be hosted on a Streamlit-compatible deployment platform.

The deployment environment requires:

- `app.py`
- `requirements.txt`
- The hospital discharge CSV dataset

The dataset must be accessible to the application at runtime.

Because the source CSV is large, the deployment process must account for file-size restrictions. Git LFS or an appropriate external storage solution may be required to provide the actual dataset to the application.

The browser must also be able to load Three.js from an available CDN for the animated 3D background and related visual elements.

**GitHub Repository:**

Add the MediScope repository URL after publishing the project.

**Live Demo:**

Add the deployed Streamlit application URL after deployment.

---

## 👨‍💻 Project Information

**Project:** MediScope

**Title:** 3D Hospital Inpatient Discharge Analytics Dashboard

**Assessment:** Summative Assessment (SA)

**Course:** Data Analytics

**Programming Language:** Python

**Data Processing:** Pandas and NumPy

**Visualization:** Plotly

**Machine Learning:** scikit-learn

**3D Graphics:** Three.js and WebGL

**Application Framework:** Streamlit

**Dataset:** Hospital Inpatient Discharges (SPARCS De-Identified) – 2021

---

## 📌 Conclusion

MediScope demonstrates how Python-based data analytics can be applied to hospital inpatient discharge information.

By combining data preprocessing, statistical calculations, interactive filtering, machine learning, and 3D visualization, the application provides a platform for exploring relationships between patient demographics, diagnosis categories, length of hospital stay, healthcare charges, illness severity, and healthcare facilities.

The nine dashboard pages provide different perspectives on the dataset.

The Mathematical Analysis page explains the statistical methods used by the application, while Math Behind the Graphs connects mathematical formulas to the actual 3D visualizations.

The AI Experiment demonstrates machine-learning classification using scikit-learn, and Data Explorer allows users to search, inspect, and download processed records.

The 3D visualization system, animated medical background, and Dr. MediBot assistant provide additional interactive features that improve the overall dashboard experience.

Overall, MediScope demonstrates how raw hospital discharge records can be transformed into an interactive hospital analytics application using Python, Pandas, NumPy, Plotly, Streamlit, scikit-learn, and Three.js.

---

## Credits

**Name:** Sarveshwaran K

**No:** 1000411

**School Name:** Jain Vidyalaya IB World School

**Mentor's Name:** Syed Ali Beema
