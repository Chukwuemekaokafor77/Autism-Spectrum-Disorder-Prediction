# Autism Spectrum Disorder Prediction
# Autism Spectrum Disorder Prediction

## Overview

Autism Spectrum Disorder (ASD) is a neurodevelopmental disorder characterized by challenges with social interaction, communication, and repetitive behaviors. Early diagnosis is crucial for effective intervention and support. This project aims to develop a machine learning model that can predict the likelihood of ASD in toddlers based on behavioral features. The prediction tool is designed to assist healthcare professionals and parents in making informed decisions about pursuing formal clinical diagnosis.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Deployment](#deployment)
- [Results](#results)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Dataset

The dataset used in this project was curated by Dr. Fadi Thabtah and is titled "Autistic Spectrum Disorder Screening Data for Toddlers." This dataset contains 1054 records and 18 attributes, including 10 key behavioral features (Q-Chat-10) and other demographic information.

- **Title:** Autistic Spectrum Disorder Screening Data for Toddlers
- **Curator:** Dr. Fadi Thabtah
- **Source:** [ASDTests.com](http://www.asdtests.com)
- **Attributes:** Behavioral features (A1 to A10), Age in months, Sex, Ethnicity, Jaundice history, Family history of ASD, etc.

The dataset plays a critical role in training and testing the machine learning model developed in this project.

## Project Structure

├── app │ ├── init.py │ ├── model.py │ ├── routes.py │ ├── static │ │ └── style.css │ └── templates │ └── index.html ├── data │ ├── Toddler Autism dataset July 2018.csv │ ├── Toddler data description.docx │ └── main.ipynb ├── model │ ├── feature_names.pkl │ ├── model.pkl │ └── scaler.pkl ├── venv │ └── ... (virtual environment files) ├── .gitignore ├── dockerfile ├── requirements.txt └── run.py


- **app/**: Contains the Flask application with templates, static files, and routes.
- **data/**: Dataset and Jupyter notebook for data exploration.
- **model/**: Serialized machine learning model and scaler files.
- **venv/**: Virtual environment (excluded from the repository).
- **.gitignore**: Specifies files and directories to ignore in the repository.
- **dockerfile**: Docker configuration for containerization.
- **requirements.txt**: Python dependencies for the project.
- **run.py**: Entry point for the Flask application.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chukwuemekaokafor77/Autism-Spectrum-Disorder-Prediction.git
   cd Autism-Spectrum-Disorder-Prediction

2. **Create and activate a virtual environment:**
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install the dependencies:**
    pip install -r requirements.txt
4. **Run the Flask application:**
    Run the Flask application:
5. **Open your web browser and navigate to:**
    http://127.0.0.1:5000/

Usage
Web Interface: The application provides a simple web interface where users can input behavioral data and receive a prediction on whether the child is likely to have ASD traits.

Docker Deployment: The application can be containerized using Docker. Build and run the Docker container as follows:
docker build -t autism_spectrum_disorder_prediction .
docker run -p 5000:5000 autism_spectrum_disorder_prediction

Model
Model Used: Logistic Regression was selected for its interpretability and effectiveness in binary classification tasks.

Cross-Validation: The model was evaluated using 5-fold cross-validation, achieving a high accuracy of 99.7%.

Feature Engineering: Categorical variables were one-hot encoded, and the data was standardized before training.

Deployment
The application has been containerized using Docker for easy deployment across different environments. The dockerfile includes all necessary steps to build and run the application in a containerized environment.

Results
The model demonstrated high accuracy and reliability in predicting ASD traits based on the input features. This tool can be a valuable asset in preliminary ASD screening, helping to identify children who may benefit from further clinical evaluation.

Acknowledgments
I would like to extend my gratitude to Dr. Fadi Thabtah for curating the "Autistic Spectrum Disorder Screening Data for Toddlers" dataset. The insights provided by this dataset were invaluable in the development of this project.

License
This project is licensed under the MIT License - see the LICENSE file for details.

About the Developer
Chukwuemeka Michael Okafor is a data scientist passionate about applying machine learning and AI to solve real-world problems. With experience in software development and data analysis, Chukwuemeka focuses on creating impactful tools that improve lives. Connect on LinkedIn or explore more projects on the portfolio page.
