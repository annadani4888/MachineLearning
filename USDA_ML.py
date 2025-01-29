import numpy as np
import pandas as pd
import json
from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SHConfig,
    bbox_to_dimensions,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Sentinel Hub configuration
config = SHConfig(profile="annatesting")

# Define my AOI bounding box
# Load the GeoJSON file
with open('/home/anna.rouse/Downloads/iowa-aoi/iowa_arps_box.geojson') as f:
    aoi_geojson = json.load(f)

# Extract the coordinates of the polygon
coordinates = aoi_geojson['features'][0]['geometry']['coordinates'][0]

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
print("Bounding Box:", bounding_box)

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
            data_collection=DataCollection.define_byoc("c57c4f30-d5b5-4fdb-bf5e-af20d51f301a")
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
            data_collection=DataCollection.define_byoc("a3729550-2ba2-4ae5-8f1e-e8ac62dd050d"),
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

# Troubleshooting
# Log dimensions of the responses to ensure they are non-empty
print("CDL Data Request shape:", response_byoc[0].shape if response_byoc else "No data")
print("PlanetScope Data Request shape:", response_planet[0].shape if response_planet else "No data")

# Inspect data content
print("CDL Data Request unique values:", np.unique(response_byoc[0]) if response_byoc else "No data")
print("CDL Data Request unique values:", np.unique(response_byoc[0]) if response_byoc else "No data")

# Convert responses to NumPy arrays
class_map = np.array(response_byoc[0][..., 0])  # CDL class band
data_mask_byoc = np.array(response_byoc[0][..., 1]) # CDL data mask

# Mask out invalid data in CDL (eg: 0 and 210)
valid_classes = [36, 255] # Soybeans (36) and Corn (255)
class_map = np.where(np.isin(class_map, valid_classes), class_map, np.nan)

# Debugging: Verify unique values in the raw and unmasked CDL class maps
print("Raw CDL class map unique values (before masking):", np.unique(class_map))
print("Masked CDL class map unique values:", np.unique(class_map))

# Extract PlanetScope bands and apply the scene mask
blue_band = np.array(response_planet[0][..., 0])
green_band = np.array(response_planet[0][..., 1])
red_band = np.array(response_planet[0][..., 2])
nir_band = np.array(response_planet[0][..., 3])
scene_mask_planet = np.array(response_planet[0][..., 4])

# Troubleshooting: Log raw PlanetScope band values

#Inspecting scene mask
print("Scene Mask Shape:", scene_mask_planet.shape)
print("Scene Mask Unique Values:", np.unique(scene_mask_planet))
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
valid_indices = ~np.isnan(class_flat) #Retain only non-NaN pixels
print("Number of valid pixels (non-NaN):", np.sum(valid_indices))
print("Length of class_flat after filtering:", len(class_flat))

class_flat = class_flat[valid_indices]
blue_flat = blue_flat[valid_indices]
green_flat = green_flat[valid_indices]
red_flat = red_flat[valid_indices]
nir_flat = nir_flat[valid_indices]

print("Valid indices shape:", valid_indices.shape)
print("Number of valid pixels in class_flat:", np.sum(~np.isnan(class_flat)))
print("Number of valid pixels in blue_flat:", np.sum(~np.isnan(blue_flat)))
print("Number of valid pixels in green_flat:", np.sum(~np.isnan(green_flat)))
print("Number of valid pixels in red_flat:", np.sum(~np.isnan(red_flat)))
print("Number of valid pixels in nir_flat:", np.sum(~np.isnan(nir_flat)))
# #Create DataFrame from the valid data
data = {
     "Class": class_flat,
     "Blue": blue_flat,
     "Green": green_flat,
     "Red": red_flat,
     "NIR": nir_flat,
 }

# DEBUGGING
#df = pd.DataFrame(data).dropna()
df = pd.DataFrame(data)
print("DataFrame Shape:", df.shape)
print("DataFrame Head:")
print(df.head())

# Save DataFrame to a CSV file
output_path = "/home/anna.rouse/Downloads/iowa-aoi/arps_combined_df.csv"
df.to_csv(output_path, index=False)
print(f"DataFrame saved to {output_path}")

## TRAIN ML MODEL

# Preprocessing
X = df[["Blue", "Green", "Red", "NIR"]].values  # Extract features
y = df["Class"].values  # Extract labels

# Check for class imbalance
print("Class distribution:\n", df["Class"].value_counts())

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the initial model
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Eliminate NaNs before applying SMOTE

imputer = SimpleImputer(strategy="mean") # Replace NaNs with the column mean
X = imputer.fit_transform(X)

# Verify NaNs are handled
print("Are there NaNs in X?", pd.DataFrame(X).isnull().sum().sum())

print("Shape of X before SMOTE:", X.shape)
print("Shape of y before SMOTE:", y.shape)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X, y)
print("Shape of X_train_balanced:", X_train_balanced.shape)
print("Shape of y_train_balanced:", y_train_balanced.shape)
print("Class distribution after oversampling:", dict(pd.Series(y_train_balanced).value_counts()))

#weights = {36: 2, 255: 1} # Assign more weights to Corn (255)
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate the model on the test set
from sklearn.metrics import classification_report, accuracy_score

y_pred = model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Cross-validation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters from GridSearchCV
print("Best parameters:", grid_search.best_params_)

# Re-train the model using the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Visualize the Decision Tree
#from sklearn.tree import export_text

#tree_rules = export_text(best_model, feature_names=["Blue", "Green", "Red", "NIR"])
#print(tree_rules)

