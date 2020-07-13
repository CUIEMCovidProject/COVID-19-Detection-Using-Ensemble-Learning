# COVID-19-Detection-Using-Ensemble-Learning

This repository contains all the necessary scripts used for data preparation, model training and building the GUI App.

-----

## Dependencies:
- Tensorflow 2.2.0
- numpy
- opencv
- PyQt5
- scikit-learn

-----


Link to our [npy files](https://drive.google.com/drive/folders/1yRlHtGmDKXHYzEPMUQAi1_JMB9nqXjuX?usp=sharing)



-----

## Repository Structure:

The repository has four directories: 
- DataPreparation Script : Contains [CreateNPY_Files.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/DataPreparation%20Script/CreateNPY_Files.py) script for making npy arrays out of the images<br><br>
- Model Training Script : <br>
    1. [Model.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/Model%20Training%20Script/Model.py) : Contains function for model preparation
    2. [Ensembling.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/Model%20Training%20Script/Ensembling.py) : Contains functions for ensembling and function for measuring the performance of the ensembler
    3. [TrainModel.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/Model%20Training%20Script/TrainModel.py) : Script for training the models
    4. [Performance](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/Model%20Training%20Script/Performance.py) : Script for getting the performance metrics
    <br><br>
- GUI Application : This directory contains Desktop Application made using Qt which uses the trained models and ensembling for COVID-19 detection.<br><br>
![GUI APP](https://raw.githubusercontent.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/master/Screenshot/python_GqQXhc1Erf.png)<br><br>
    1. [app.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/GUI%20Application/app.py) : Main Python application
    2. [utils.py](https://github.com/CUIEMCovidProject/COVID-19-Detection-Using-Ensemble-Learning/blob/master/GUI%20Application/utils.py) : Contains utility functions that is used by the application.

-----
## Steps to prepare data:
    1. The Image data are kept into separate directories as COVID_19 +ve and COVID_19 -ve.
    2. These images are split into separate diretories as train and test

    Thus the directory structure is as:

    Images
        |
        ----Train
        |       |
        |       ---- COVID_19 +ve
        |       |
        |       ---- COVID_19 -ve
        |
        -----Test
                |
                ---- COVID_19 +ve
                |
                ---- COVID_19 -ve
        
    3. Then run the CreateNPY_Files.py script and the enter the paths according to the prompt.


## Steps to train the model:
    1. Run the TrainModel.py script and the enter the paths.
    2. The models will be saved in the directory.
    
## Check the performance of the ensembling:
    1. Run Performance.py to get the performance of the ensembler and enter the paths.    
    2. And the accuracy, confusion matrix will be printed in the console


## Run the GUI Application    
    1. Run the app.py script



