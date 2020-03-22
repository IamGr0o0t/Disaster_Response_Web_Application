# Disaster Response Project

### Table of content
---------------
* [Motivation](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#motivation)
* [Installation](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#installation)
* [Software versions](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#software-versions)
* [File descriptions](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#file-descriptions)
* [How to Execute](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#how-to-execute)
* [Results and Findings](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#results-and-findings)
* [Licensing, Authors, Acknowledgements](https://github.com/IamGr0o0t/Disaster_Response_Web_Application#licensing-authors-acknowledgements)

### Motivation
---------------
The purpose of the disaster response application is to process raw text data and predict the disaster response category that message might fall into.
- See images below.
![file1](https://github.com/IamGr0o0t/Disaster_Response_Web_Application/blob/master/images/DR_App_Classification.png)

![file1](https://github.com/IamGr0o0t/Disaster_Response_Web_Application/blob/master/images/Genre_Message_Distribution.png)

![file1](https://github.com/IamGr0o0t/Disaster_Response_Web_Application/blob/master/images/Direct_Word_Cloud.png)

![file1](https://github.com/IamGr0o0t/Disaster_Response_Web_Application/blob/master/images/Social_Word_Cloud.png)

![file1](https://github.com/IamGr0o0t/Disaster_Response_Web_Application/blob/master/images/News_Word_Cloud.png)

### Installation
---------------
Most of packages used were installed via most recent conda distribution.
```bash
conda 4.8.2
```
To install extra packages please do it with conda-forge.
```bash
conda install -c conda-forge (Insert Package Here)
```

### Software versions
---------------
* Anaconda == 4.8.2
* Python == 3.7.6
* Flask == 1.1.1    
* matplotlib == 3.1.1    
* nltk == 3.4.5    
* numpy == 1.17.2   
* pandas == 0.25.1   
* plotly == 4.5.2    
* scikit-learn == 0.21.3   
* scipy == 1.3.1    
* seaborn == 0.9.0    
* SQLAlchemy == 1.3.9 

### File descriptions
---------------
- app<br>
| - template<br>
| |- master.html (main page of web app)<br>
| |- go.html (classification result page of web app)<br>
|- run.py (flask file that runs the app)<br>

- data<br>
|- categories.csv (data to process, containing message categories)<br>
|- messages.csv (data to process, containing text data of the messages)<br>
|- DisasterResponse.db (sqlite database containing merged and cleaned data from<br>
                        csv files)<br>
|- process_data.py (ETL pipeline that extracts and transforms data from message<br>
                    & categories csv's and load clean data into sqlite database) <br>

- images<br>
|- screenshots of the web application.<br>

- models<br>
|- train_classifier.py (ML pipeline code which takes data from sqlite database<br>
                        as input. Performs text preprocessing with NLTK, train and test classification model and returns pickle file with fitted model.)<br>
|- classifier.pkl (!IMPORTANT this file is missing due to large format. One has to run train_classifier.py to have it visible. <br>
                  Pickle file containing fitted model from train_classifier.py)<br>

- notebooks<br>
|- ETL_Pipeline_Preparation.ipynb (Notebook containing code used for<br>
                                  process_data.py)<br>
|- ML_Pipeline_Preparation.ipynb (Notebook containing code used for<br>
                                  train_classifier.py)<br>


### How to Execute
---------------
1. ETL Pipeline (preprocess_data.py)
* Open terminal window and navigate to project directory
  > $ cd FigureEight_Disaster_Response<br>
  > $ python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db<br>

2. ML Pipeline (train_classifier.py) 
* Open terminal window and navigate to project directory
  > $ cd FigureEight_Disaster_Response<br>
  > $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl<br>

3. Web Application (train_classifier.py) 
* Open terminal window and navigate to project directory
  > $ cd FigureEight_Disaster_Response<br>
  > $ python app/run.py<br>

### Results and Findings
---------------
* Data provided contains 36 different categories, but most of them are highly
  imbalanced which has large effect on prediction score measures. The solution
  to that problem could be to over or under sample if data volume provided allows it. Another potential solution to imbalanced categories could be to generate synthetic samples (SMOTE) for our training set.

### Licensing, Authors, Acknowledgements
---------------
This datasets was provided by [Figure Eight](https://www.figure-eight.com/).<br>
Thank you to [Udacity.com](https://classroom.udacity.com) for an amazing Data Science Nanodegree programme.



