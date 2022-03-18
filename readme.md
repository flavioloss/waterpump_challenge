## Data Mining Water Pump Data: Project Overview
### [GitHub Project Page](https://github.com/flavioloss/waterpump_challenge/blob/main/)
### This project is from the Machine Learning challenge hosted in Driven Data [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/) where i placed in rank 903 out of 13.000 competitors
#### What was developed:
- Estimator to predict if the water pump is functional, needs repair or non-function: configures a multi label classification model with accuracy score of 82.15%
- Exploratory Data Analisys: The data contains a lot of features, so it was essential to visualize every distribution, relation and correlation between variables
- Feature Engineering and Importance: using random forest, we can see each feature importance to the estimator. Also, feature engineering was extremely import to process non-numerical features, along with the creation of new features using the spatial ones that the data contains (latitude, longitude)
- Several models where used, including Random Forests, Neural Network, and CatBoost, with the last being the best fit and accuracy score

#### After the modeling i have build a deploy for identifying in which category the water pump fits. The deploy is a map visualization of Tanzania, using the Plotly Python library. 
#### For building and maintaining the app online, i used Streamlit and Streamlit Cloud.
#### [Deploy Project Page](https://github.com/flavioloss/waterpump_deploy)
