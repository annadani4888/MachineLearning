## **Overview**

This script processes geospatial data to classify agricultural crops within a specific Area of Interest (AOI) in Iowa, USA. It integrates satellite imagery from PlanetScope (PS) and the USDA's Crop Data Layer (CDL), ingests them into Sentinel Hub for processing, and trains a machine learning (ML) model to identify pixels representing Corn or Soybeans. The goal is to create a predictive ML model using the features derived from the satellite data.

**Key Steps**
 1. **Configuring Sentinel Hub** 
 The script initializes a Sentinel Hub configuration using a specific user profile (`annatesting`) and sets up a bounding box (BBox) for the AOI based on a .geojson file.
 2. **CDL and PlanetScope Data Retrieval**
    - **CDL Data (Raster Classifications)**:
    - Data is requested via SentinelHubRequest from a user-defined BYOC (Bring Your Own COG) collection. (The collection was created by the user, by ingesting the CDL raster into Sentinel Hub).
    - The CDL Evalscript extracts class-specific crop information (Corn = 255, Soybeans = 36) and a data mask.
      
    - **PlanetScope Data (Satellite Imagery)**:
	- This data comes from a separate BYOC collection. The data is requested from the SentinelHub API. The request includes a specified time interval.
	- The PlanetScope Evalscript retrieves multi-band data (Blue, Green, Red, NIR) and a scene mask to filter valid pixels.
	
 3. **Data Preprocessing**
 
	 The CDL and PlanetScope responses are converted into NumPy arrays and filtered:
	 
	 - **CDL**: Only valid crop classes (Corn and Soybeans) are retained.
	 - **PlanetScope**: Bands are filtered using the scene mask to exclude invalid pixels.


 4. **Data Handling and Output**:
    - The DataFrame is exported to a CSV file for persistence and further analysis.
    - Basic descriptive statistics and troubleshooting logs are included, such as class distribution and pixel validity checks.

5. **Machine Learning Model Training**
   
    **Preprocessing**:
      - Missing values (NaN) in the dataset are imputed using column means.
      - Class imbalance is addressed with SMOTE (Synthetic Minority Oversampling Technique) to ensure equal representation of Corn and Soybean pixels in the training data.
               
    **Model:**
   - A Decision Tree Classifier is trained using the spectral bands (Blue, Green, Red, NIR) as features and the crop classes as labels.
   - Hyperparameter tuning is performed with GridSearchCV to optimize tree depth and splitting criteria.

6. **Evaluation:**
The trained model is evaluated using:
    - Classification metrics (accuracy, precision, recall, F1-score).
    - Cross-validation scores to assess model generalization.
 
 

## Technical Highlights
-   **Sentinel Hub Integration**:
    
    -   The script leverages Sentinel Hub's Python package for seamless interaction with geospatial data repositories and custom Evalscripts for raster-based classification and feature extraction.
-   **Data Masking and Cleaning**:
    
    -   Scene masks and class filtering ensure that only meaningful pixels are used for training, mitigating noise from invalid or irrelevant data.
-   **Machine Learning with Imbalance Handling**:
    
    -   Incorporates SMOTE to address class imbalance between Corn and Soybean pixels.
    -   Uses hyperparameter tuning (via GridSearchCV) and cross-validation to ensure robust model training and evaluation.

## Use Case

This script is intended for land classification in precision agriculture, providing a pipeline for:

- Processing satellite and crop-specific data layers.
- Preparing a training dataset with meaningful spectral features.
- Developing predictive models to classify agricultural crops in a specific AOI.
