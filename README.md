# DHHS Chronic Disease Indicators - Cancer  Prevalence Analysis
# <bu>CLASSIFICATION PROJECT</bu>
by Annie Carter
Sourced by U.S. Department of Health & Human Services
![image](https://github.com/annie-carter/DHHS_chronic_disease_indicators/assets/131205837/627005b8-5424-4961-85d0-f61b213445b1)


## <u>Project Description</u>
This machine learning classification project seeks to conduct an in-depth analysis and predictive assessment of cancer prevalence in the United States using the dataset "U.S. Chronic Disease Indicators (CDI)." The dataset encompasses a rich array of chronic disease indicators, including cancer-related data, risk factors, prevalence rates, and health outcomes. Additionally, it comprises crucial demographic information, such as age, gender, race, and geographical location.

* Cancer Prevalence: Notably, cancer stands as a primary cause of morbidity and mortality within the United States. Its impact extends to a substantial population across diverse age groups and geographical regions.

By employing advanced classification techniques, our objective is to develop a robust predictive model that leverages the available demographic and chronic disease indicators to address the significant variations in cancer rates, often influenced by demographic factors like gender and race. This investigation aims to identify intricate patterns and critical risk factors associated with cancer. The valuable insights derived from this analysis will contribute to targeted interventions, early detection strategies, and enhanced healthcare planning, thereby aiding in the mitigation of cancer's impact on public health in the US.



## <u>Project Goal</u>

TThe primary goal of this project is to develop a robust predictive model using advanced classification techniques that effectively utilizes demographic and chronic disease indicators to address variations in cancer rates, with a particular focus on demographic factors such as gender and race. Through this investigation, we aim to identify intricate patterns and critical risk factors associated with cancer prevalence. The valuable insights gained from this analysis will inform targeted interventions, early detection strategies, and improved healthcare planning, ultimately contributing to mitigating the impact of cancer on public health in the United States.


## <u>Initial Questions</u>
1. Is there a relationship between the category of "male or female" and cancer prevalence?
2. Is there a relationship between the state and cancer prevalence?
3. Is there a relationship between the year and cancer prevalence?
4. Is there a relationship between race and cancer prevalence?

## Data Dictionary

The initial dataset comprised 34 columns, which reduced to 9 columns after preparation. It contained 1,185,676 rows, but a random sample of 1,000,000 rows was chosen for this project using a random state of 42. During preparation, the central target column "cancer_Yes" was created by combining the top four prevalent chronic diseases: Cancer, Cardiovascular Disease, Diabetes, and Chronic Obstructive Pulmonary Disease (COPD). Column names were renamed for improved readability and cleaned for data integrity, resulting in 537,407 rows. Some column definitions were obtained from the Center for Disease Control and Prevention (CDC) Morbidity and Mortality Weekly Report (MMWR) at https://www.cdc.gov/mmwr/pdf/rr/rr6401.pdf. 

| Original                    |   Target    |       Datatype          |       Definition              |
|-----------------------------|-------------|-------------------------|------------------------------ |
|Topic (Cancer, Cardiovascular|             |                         |                               |
|Disease, Diabetes, Chronic   |Yes_cancer   | 537407 non-null  int64   |  target variable              |
|Obstructive Pulmonary Disease|             |                         |                               |


|     Original                |   Feature    |       Datatype         |     Definition                |
|-----------------------------|--------------|------------------------|------------------------------ |
|YearStart                    |Year          | 537407 non-null  int64  | Year of observations          |
|LocationAbbr                 |State (Abbr)  | 537407 non-null  object | State Abbreviation            |
|Gender                       |Gender        | 537407 non-null  object  | Male or Female                | 
|Stratification1              |Demographics  | 537407 non-null  object | Race or Ethnicity             |    
|GeoLocation                  |Geo Location  | 537407 non-null  object | Latitude and Longituted       |
|Race/Ethnicity               |Race/Ethnicity| 537407 non-null  object | Race or Ethnicity             |
|StratificationCategory1      |"same"        | 537407 non-null  object | Used for feature engineering  |
|Longitude                    |Longitude     | 537407 non-null  float64| Longitude                     |
|Latitude                     |Latitude      | 537407 non-null  float64| Latitude                      |
|Yes_female                   |Yes_Female    | 537407 non-null  int64  | Female =1 Male=0 Other        |

Hypothesis 1 - 

alpha = .05
* H0 =  Category of "male or female" gender has no relationship to cancer
* Ha = Category of "male or female" gender has a relationship to cancer
* Outcome: We reject the Null Hypothesis.

Hypothesis 2 - 

alpha = .05
* H0 = Race has no relationship to cancer  prevalence
* Ha = Race has a relationship to cancer  prevalence
* Outcome: We reject the Null Hypothesis.

Hypothesis 3 -

alpha = .05
H0 = Year has no relationship to cancer prevalence
Ha = Year has a relationship to cancer prevalence
* Outcome: We reject the Null Hypothesis.

## <u>Planning Process</u>
#### Planning
1. Clearly define the problem statement related to the DHHS Chronic Disease Indicator dataset, including essential information for addressing the chronic disease of interest.

2. Create a detailed README.md file documenting the project's context, dataset characteristics, and analysis procedure for easy reproducibility.

#### Acquisition and Preparation
3. Preprocess the data, handling missing values and outliers effectively during data loading and cleansing.

4. Perform feature selection meticulously, identifying influential features impacting the prevalence of the chronic disease through correlation analysis, feature importance estimation, or domain expertise-based selection criteria.

5. Develop specialized scripts (e.g., acquire.py and wrangle.py) for efficient and consistent data acquisition, preparation, and data splitting.

6. Safeguard proprietary aspects of the project by implementing confidentiality and data security measures, using .gitignore to exclude sensitive information.

#### Exploratory Analysis
7. Utilize exploratory data analysis techniques, employing compelling visualizations and relevant statistical tests to extract meaningful patterns and relationships within the dataset.

#### Modeling
8. Carefully choose a suitable machine learning algorithm, evaluating options like Logistic Regression, Decision Trees, Random Forests, or K Nearest Neighbor tailored for the regression task.

9. Implement the selected machine learning models using robust libraries (e.g., scikit-learn), systematically evaluating multiple models, including Decision Trees, Logistic Regression, and Random Forests, with a fixed Random Seed value 42 for reproducibility.

10. Train the models rigorously to ensure optimal learning and model performance.

11. Conduct rigorous model validation techniques to assess model generalization capability and reliability.

12. Select the most effective model, such as Logistic Regression, based on thorough evaluation metrics for further analysis.

#### Product Delivery
13. Assemble a final notebook, combining superior visualizations, well-trained models, and pertinent data to present comprehensive insights and conclusions with scientific rigor.

14. Generate a Prediction.csv file containing predictions from the chosen model on test data for further evaluation and utilization.

15. Maintain meticulous project documentation, adhering to scientific and professional standards, to ensure successful presentation or seamless deployment.

## <u>Instructions to Reproduce the Final Project Notebook</u> 
To successfully run/reproduce the final project notebook, please follow these steps:

1.  Read this README.md document to familiarize yourself with the project details and key findings.
2. Before proceeding, ensure that you have the necessary database credentials. Get data set from https://catalog.data.gov/dataset/u-s-chronic-disease-indicators-cdi Create .gitignore for privacy if necessary
3. Clone the classification_project repository from my GitHub or download the following files: aquire.py, wrange.py or prepare.py, and final_report.ipynb. You can find these files in the project repository.
4. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
5. Ensure that all necessary libraries or dependent programs are installed. You may need to install additional packages if they are not already present in your environment.
6. Run the final_report.ipynb notebook to execute the project code and generate the results.
By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>
<span style ='color:#1F456E'>Relationship between Gender, Race/Ethnicity, and US Locations and Cancer:
The analysis revealed significant relationships between gender, race/ethnicity, and US locations with cancer prevalence. Certain demographics showed higher susceptibility to cancer, indicating potential disparities in cancer rates.

<span style ='color:#1F456E'>Decision Tree Model Outperformed Other Classification Models:
After evaluating multiple classification models, the Decision Tree model consistently outperformed Logistic Regression and Random Forest models in all phases (train, validate, and test). The Decision Tree model achieved an average accuracy of 80%, surpassing the baseline accuracy of 69%.

## <u>Conclusion</u>
<span style ='color:#1F456E'>The analysis of the DHHS Chronic Disease Indicators dataset revealed a significant relationship between gender, race/ethnicity, US locations, and cancer prevalence. Among the classification models used, the Decision Tree Model consistently outperformed other models in all three evaluation metrics: train, validate, and test data. The Decision Tree Model's performance averaged at 80%, surpassing the baseline of 69%. This suggests that the Decision Tree Model is a reliable approach for predicting cancer prevalence.

<span style ='color:#1F456E'>Notably, gender and race/ethnicity showed significant associations with cancer prevalence. Prostate, lung, and colorectal cancers were most prevalent in men, accounting for 42% of cases, while breast, lung, and colorectal cancers were more prevalent in women, comprising half of all cases. Additionally, lung cancer death rates declined substantially in both men and women between 1990 and 2016, indicating the success of aggressive awareness campaigns and increased breast cancer screening. Breast cancer death rates also showed a significant decline during this period, along with prostate and colorectal cancer death rates.

## <u>Next Steps</u>
<span style ='color:#1F456E'> 1. **Time-Series Analysis:** If time permits, conducting a time-series analysis could provide valuable insights into the trends and patterns of cancer prevalence over the years, especially the drop observed between 2008 and 2016. Exploring this time range in more detail may reveal underlying factors or interventions that contributed to the decrease in cancer rates.

<span style ='color:#1F456E'>2. **Melt Observation Data:** To refine the observations and potentially uncover more meaningful relationships, consider using the 'melt' operation to reshape the data. This process can transform the data from a wide format to a long format, making it easier to analyze and visualize the relationships between different variables and cancer prevalence.

<span style ='color:#1F456E'>3. **Geo-Location Clustering:** Investigate the spatial distribution of cancer prevalence by selecting specific geographic areas. Perform clustering analysis to identify regions with similar cancer patterns based on geo-location data. This can help in understanding whether certain locations are more susceptible to higher or lower cancer rates and could guide targeted intervention strategies.
    
<span style ='color:#1F456E'>4. **Feature Engeneer for Specific Cancers :** Enhance predictions and insights through targeted feature engineering for specific cancer types in the DHHS Chronic Disease Indicators analysis. Capture unique characteristics and risk factors, improving accuracy in prevalence predictions.
    
## <u>Recommendations</u>
<span style ='color:#1F456E'>- **Targeted Awareness Campaigns:** Focus on raising awareness about specific cancer types that are most prevalent in certain gender and race/ethnicity groups. Tailored awareness campaigns can improve early detection and prompt appropriate interventions.

<span style ='color:#1F456E'>- **Geographical Interventions:** Based on the identified clusters of cancer prevalence in certain geographic areas, implement targeted interventions and healthcare initiatives to address regional disparities in cancer rates.

<span style ='color:#1F456E'>- **Further Research:** Conduct further research to understand the reasons behind the decline in lung, breast, prostate, and colorectal cancer death rates. Identify factors contributing to the rise in colorectal cancer cases in younger adults to develop effective prevention strategies.

<span style ='color:#1F456E'>- **Long-Term Monitoring:** Continuously monitor cancer prevalence trends over time to identify any emerging patterns and respond promptly to potential changes in cancer rates.

<span style ='color:#1F456E'>By implementing these recommendations and conducting additional research, we can gain deeper insights into cancer prevalence, improve early detection, and implement effective interventions, ultimately leading to better cancer outcomes and improved public health.
