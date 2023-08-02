# DHHS Chronic Disease Indicators - Cancer  Prevalence Analysis
# <bu>CLASSIFICATION PROJECT</bu>
by Annie Carter
Sourced by U.S. Department of Health & Human Services

![image.png](attachment:image.png)

## <u>Project Description</u>
The dataset "'U.S._Chronic_Disease_Indicators__CDI_.csv'" contains information about various chronic disease indicators in the United States, including data on risk factors, prevalence, and health outcomes. It also includes demographic information such as age, gender, race, and geographical location.



## <u>Project Goal</u>

The goal of this machine learning project is to build a predictive model that can accurately forecast the prevalence of the chronic disease cancer based on demographics and comparison to diabetes. This project determines factors contributing to the prevalence of the chronic disease of cancer in the United States.  The data collected from this project can help with future DHHS planning on distribution of resources for primary and secondary preventions (e.g. cancer awareness, screening) and improved research to decrease prevalence and improve state outcomes 


## <u>Initial Questions</u>
1. Does category of  "male or female" have a relationship to cancer?
2. Does state have relationship with cancer?
3. Does year have a relationship with cancer?
4. Does race have a relationship to cancer  prevalence?

## Data Dictionary

There were 34  columns in the initial data and 9 columns after preparation; 1185676 rows in the intial data set. A random sample of 100000 rows using a random state of 42 was selected for this project. After preparation the central target column of cancer_Yes was created by combining the top for prevlant chronic diseases (Cancer, Cardiovascular Disase, Diabetes and Chronic Obstruction Pulmonary Disease (COPD). Columns were renamed to enhance readibaility and cleaned for integrity with 34973. Some definitions from original column names were derived from Center for Disease Control and Prevention (CDC) Morbidity and Mortaliy Weekly Report (MMWR) https://www.cdc.gov/mmwr/pdf/rr/rr6401.pdf 

| Original                    |   Target    |       Datatype          |       Definition             |
|-----------------------------|-------------|-------------------------|------------------------------|
|Topic (Cancer, Cardiovascular|             |                         |                              |
|Disease, Diabetes, Chronic   |Yes_cancer   | 34973 non-null  int64   |  target variable             |
|Obstructive Pulmonary Disease|             |                         |                              |


|     Original                |   Feature    |       Datatype         |     Definition               |
|-----------------------------|--------------|------------------------|------------------------------|
|YearStart                    |  Year        | 34973 non-null  int64  | Year of observations         |
|LocationAbbr                 |State (Abbr)  | 34973 non-null  object | State Abbreviation           |
|StratificationCategory1      |Gender        | 34973 non-null  object | Male or Female               |
|Stratification1              |Race/Ethnicity| 34973 non-null  object | Race or Ethnicity            |
|GeoLocation                  |lat/long      | 34973 non-null  object | Latitude and Longituted      |
|DataValue                    |Data Value    | 34973 non-null  object | Number of Occurence          |
|Stratification1              |Race/Ethnicity| 34973 non-null  object | Race or Ethnicity            |
|Yes_female                   |Yes_Female    | 34973 non-null  int64  | Female =1 Male=0 Other       |

Hypothesis 1 - 

alpha = .05
H0 =  Category of "male or female" gender has no relationship to cancer
Ha = Category of "male or female" gender has a relationship to cancer
Outcome: We accept or reject the Null Hypothesis.

Hypothesis 2 - 

alpha = .05
H0 = Race has no relationship to cancer  prevalence
Ha = Race has a relationship to cancer  prevalence
Outcome: We reject the Null Hypothesis.

## <u>Planning Process</u>
#### Planning
1. Articulate a well-defined problem statement that pertains to the DHHS Chronic Disease Indicator dataset, encompassing the relevant information essential for addressing the chronic disease of interest.

2. Develop a comprehensive README.md file to meticulously document all essential aspects of the project, including the problem's context, dataset characteristics, and a detailed procedure for reproducing the analysis.

#### Acquisition and Preparation
3. Execute data preprocessing techniques, encompassing data loading and cleansing methodologies to address missing values and outliers effectively.

4. Engage in rigorous feature selection methodologies, relying on meticulous analysis of the dataset to identify influential features that may significantly impact the prevalence of the chronic disease. Employ various approaches such as correlation analysis, feature importance estimation, or domain expertise-based selection criteria.

5. Create specialized scripts(e.g acquire.py and wrangle.py or prepare.py), to streamline the data acquisition, preparation, and data splitting processes for improved efficiency and consistency.

6. Ensure confidentiality and data security through the implementation of a .gitignore file to exclude sensitive information and files, thereby safeguarding proprietary aspects of the project.

#### Exploratory Analysis
7. Employ exploratory data analysis techniques to gain profound insights into the dataset, utilizing compelling visualizations and relevant statistical tests (e.g., chi-square test, t-test) to extract meaningful patterns and relationships within the data.

#### Modeling
8. Conscientiously choose a suitable machine learning algorithm tailored for regression tasks, carefully evaluating potential options such as Logistic Regression, Decision Trees, Random Forests, or K Nearest Neighbor based on their appropriateness for the given problem.

9. Implement the selected machine learning model using a robust library such as scikit-learn, systematically evaluating multiple models, including Decision Trees, Logistic Regression, and Random Forests, while ensuring reproducibility through a fixed Random Seed value 42.

10. Rigorously train the models using the available dataset to ensure optimal learning and model performance.

11. Perform rigorous model validation techniques to assess the generalization capability and reliability of the models.

12. Carefully select the most effective model, e.g., Logistic Regression, based on thorough evaluation metrics for further analysis.

#### Product Delivery
13. Assemble a final notebook that effectively amalgamates superior visualizations, well-trained models, and pertinent data, presenting comprehensive insights and conclusions with scientific rigor.

14. Generate a Prediction.csv file containing the predictions derived from the test data using the chosen model, thus facilitating further evaluation and utilization of the model's outcomes.

15. Ensure meticulous attention to project documentation, adhering to scientific and professional standards, to prepare for successful presentation or seamless deployment.


Instructions to Reproduce the Final Project Notebook
To successfully run/reproduce the final project notebook, please follow these steps:

Read this README.md document to familiarize yourself with the project details and key findings.
Before proceeding, ensure that you have the necessary database credentials. Get data set from https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi Create .gitignore for privacy if necessary
Clone the classification_project repository from my GitHub or download the following files: aquire.py, wrange.py or prepare.py, and final_report.ipynb. You can find these files in the project repository.
Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
Ensure that all necessary libraries or dependent programs are installed. You may need to install additional packages if they are not already present in your environment.
Run the final_report.ipynb notebook to execute the project code and generate the results.
By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>
TBD

## <u>Conclusion</u>
TBD

## <u>Next Steps</u>
TBD
## <u>Recommendations</u>
TBD
