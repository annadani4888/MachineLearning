import json
import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentinel Hub configuration
config = SHConfig(profile="annatesting")

# Define my AOI bounding box
# Load the GeoJSON file
with open("/home/anna.rouse/Downloads/iowa-aoi/iowa_arps_box.geojson") as f:
    aoi_geojson = json.load(f)

# Extract the coordinates of the polygon
coordinates = aoi_geojson["features"][0]["geometry"]["coordinates"][0]

# Find the bounding box for the polygon (min lon, min lat, max lon, max lat)
min_lon = min([point[0] for point in coordinates])
max_lon = max([point[0] for point in coordinates])
min_lat = min([point[1] for point in coordinates])
max_lat = max([point[1] for point in coordinates])

# Create a BBox object
bounding_box = BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)
resolution = 10
width, height = bbox_to_dimensions(bounding_box, resolution=resolution)
# Now `bounding_box` can be used in the script
logger.info(1)

# CDL Evalscript
evalscript_byoc = """
//VERSION=3

function setup() {
  return {
    input: ["Class", "dataMask"],
    output: { bands: 4 }
  };
}

function evaluatePixel(sample) {
  switch (sample.Class) {
    case 1: return [255 / 255, 210 / 255, 0 / 255, sample.dataMask];
    case 5: return [36 / 255, 110 / 255, 0 / 255, 1];
  }
}
"""

# PlanetScope ARPS Evalscript
evalscript_planet = """
//VERSION=3

function setup() {
    return {
        input: ["blue", "green", "red", "nir", "scene_mask"],
        output: { bands: 5 }
    };
}

let factor = 1 / 2000;
function evaluatePixel(sample) {
    return [
         factor * sample.blue,
         factor * sample.green,
         factor * sample.red,
         factor * sample.nir,
         sample.scene_mask
    ];
}
"""

# CDL Data Request
request_byoc = SentinelHubRequest(
    evalscript=evalscript_byoc,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.define_byoc(
                "c57c4f30-d5b5-4fdb-bf5e-af20d51f301a"
            )
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bounding_box,
    size=(width, height),
    config=config,
)

# PlanetScope ARPS Data Request
request_planet = SentinelHubRequest(
    evalscript=evalscript_planet,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.define_byoc(
                "a3729550-2ba2-4ae5-8f1e-e8ac62dd050d"
            ),
            time_interval=("2023-04-10", "2023-04-24"),
        )
    ],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bounding_box,
    size=(width, height),
    config=config,
)

# Execute the requests and retrieve data
response_byoc = request_byoc.get_data()
response_planet = request_planet.get_data()

# Convert responses to NumPy arrays
class_map = np.array(response_byoc[0][..., 0])  # CDL class band
data_mask_byoc = np.array(response_byoc[0][..., 1])  # CDL data mask

# Mask out invalid data in CDL (eg: 0 and 210)
valid_classes = [36, 255]  # Soybeans (36) and Corn (255)
class_map = np.where(np.isin(class_map, valid_classes), class_map, np.nan)

# Extract PlanetScope bands and apply the scene mask
blue_band = np.array(response_planet[0][..., 0])
green_band = np.array(response_planet[0][..., 1])
red_band = np.array(response_planet[0][..., 2])
nir_band = np.array(response_planet[0][..., 3])
scene_mask_planet = np.array(response_planet[0][..., 4])


# Inspecting scene mask

# Apply scene mask to PlanetScope data to filter invalid pixels
blue_band = np.where(scene_mask_planet == 0, np.nan, blue_band)
green_band = np.where(scene_mask_planet == 0, np.nan, green_band)
red_band = np.where(scene_mask_planet == 0, np.nan, red_band)
nir_band = np.where(scene_mask_planet == 0, np.nan, nir_band)

# # Flatten the arrays into 1D arrays
class_flat = class_map.flatten()
blue_flat = blue_band.flatten()
green_flat = green_band.flatten()
red_flat = red_band.flatten()
nir_flat = nir_band.flatten()

# # Filter out NaN values
valid_indices = ~np.isnan(class_flat)  # Retain only non-NaN pixels

class_flat = class_flat[valid_indices]
blue_flat = blue_flat[valid_indices]
green_flat = green_flat[valid_indices]
red_flat = red_flat[valid_indices]
nir_flat = nir_flat[valid_indices]

# #Create DataFrame from the valid data
data = {
    "Class": class_flat,
    "Blue": blue_flat,
    "Green": green_flat,
    "Red": red_flat,
    "NIR": nir_flat,
}

# DEBUGGING
# data_frame = pd.DataFrame(data).dropna()
data_frame = pd.DataFrame(data)

# Save DataFrame to a CSV file
output_path = "/home/anna.rouse/Downloads/iowa-aoi/arps_combined_data_frame.csv"
data_frame.to_csv(output_path, index=False)

## TRAIN ML MODEL

# Step 1: Preprocessing
logger.info("Starting data preprocessing...")
X = data_frame[["Blue", "Green", "Red", "NIR"]].values  # Extract features
y = data_frame["Class"].values  # Extract labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 2: Handle NaNs # Eliminate NaNs before applying SMOTE
logger.info("Handling NaNs in the data...")
imputer = SimpleImputer(strategy="mean")  # Replace NaNs with the column mean
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Step 3: Balance the dataset
logger.info("Applying SMOTE to balance the dataset")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

logger.info(
    "Class distribution after SMOTE:\n%s", pd.Series(y_train_balanced).value_counts()
)

# Step 4: Train the initial model
logger.info("Training the Decision Tree model...")
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Step 5: Evaluate the model
logger.info("Evaluating the model on the test set...")
y_pred = model.predict(X_test)
logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
logger.info("Test Accuracy: %f", accuracy_score(y_test, y_pred))

# Step 6: Cross-validation
logger.info("Performing cross-validation...")
scores = cross_val_score(
    model, X_train_balanced, y_train_balanced, cv=5
)  # 5-fold cross-validation
logger.info("Cross-validation scores: %s", scores)
logger.info("Mean CV accuracy: %f", scores.mean())

# Step 7: Hyperparameter Tuning
logger.info("Starting hyperparameter tuning...")
param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_balanced, y_train_balanced)

# Print the best parameters from GridSearchCV
logger.info("Best parameters found during tuning: %s", grid_search.best_params_)

# Re-train the model using the best parameters
logger.info("Re-training the model with best parameters...")
best_model = grid_search.best_estimator_
best_model.fit(X_train_balanced, y_train_balanced)

logger.info("Training complete.")
