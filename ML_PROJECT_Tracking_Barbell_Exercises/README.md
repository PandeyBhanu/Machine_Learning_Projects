# Tracking Barbell Exercises: Exploring Context-Aware Applications for Strength Training

## Project Overview
The goal of this project is to harness the power of machine learning to recognize exercise patterns from data collected through gyroscopes and accelerometers. We aim to unlock the commercial potential of this technology, particularly for health and fitness trackers. The project leverages data collected from a wristband device designed to mimic the placement and orientation of a watch, allowing for controlled experiments.

## Data Sources
We gathered data using the default settings of the sensors on the wristband, which recorded at the following frequencies:

## Ingredients
Accelerometer: 12,500Hz
Gyroscope: 25,000Hz
### Exercise Routines
The project focuses on five core weightlifting exercises: Bench Press, Deadlift, Overhead Press, Barbell Row, and Squat. These exercises are performed in a standardized routine of 3 sets, each comprising 5 repetitions.

### Participant Information
A total of five participants were involved in the data collection process.

### Additional Data Collection
In a separate training session, participants repeated the same exercises, but this time in 3 sets of 10 repetitions. This variation in sets and reps was introduced to evaluate how machine learning models generalize over different weights. Consequently, we obtained a rich dataset, totaling 150 sets of data (5 exercises x 5 participants x 6 sets). Furthermore, data was collected during 'rest' periods between exercise sets. To ensure data fidelity, participants were not restricted during these resting periods, resulting in a diverse range of activities, including standing, walking, and sitting. This data will be crucial for studying state transitions from rest to exercise.

### Analytical Tools
The project was executed using Visual Studio Code (VS Code) for code development and analysis.

### Project Structure
To maintain a structured and organized approach, the research is compartmentalized into folders. This approach allows for a clear and insightful demonstration of the thought process behind each step, including the utilization of multiple machine learning models informed by external research.

## Folder Structure:
src: The primary directory.
data: Contains scripts and files related to data collection and preprocessing.
features: Encompasses feature engineering and any associated scripts.
models: Holds machine learning model development and related code.
visualization: This section includes code and outputs for data visualization and exploration.
Within each of these subfolders, you will find Python scripts (.py) and Jupyter Notebook files (.ipynb) for transparency and reproducibility.

This project was meticulously organized and structured to encourage transparency, showcase the development process, and facilitate future research and development in the field of context-aware applications for strength training.
