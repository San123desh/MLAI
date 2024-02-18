# Import necessary libraries and modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the concrete data from the provided URL
url = "https://cocl.us/concrete_data"
concrete_data = pd.read_csv(url)
# Define predictors and target variable
predictors = concrete_data.drop(columns=['Strength'])
target = concrete_data['Strength']
# A. Build a baseline model
# Randomly split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)
# Initialize variables to store mean squared errors
mean_squared_errors = []
# Repeat steps 2-3 for 50 epochs
for _ in range(50):
    # Build a neural network model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, verbose=0)
    
    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Append the mean squared error to the list
    mean_squared_errors.append(mse)
# Calculate and report the mean and standard deviation of mean squared errors
mean_mse = pd.Series(mean_squared_errors).mean()
std_mse = pd.Series(mean_squared_errors).std()

print("Mean of Mean Squared Errors:", mean_mse)
print("Standard Deviation of Mean Squared Errors:", std_mse)
# B. Normalize the data
# Normalize the predictors by subtracting the mean and dividing by the standard deviation
predictors_normalized = (predictors - predictors.mean()) / predictors.std()

# Repeat steps A using the normalized data
X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(predictors_normalized, target, test_size=0.3, random_state=42)

mean_squared_errors_norm = []

for _ in range(50):
    model_norm = Sequential()
    model_norm.add(Dense(10, activation='relu', input_shape=(X_train_norm.shape[1],)))
    model_norm.add(Dense(1))
    
    model_norm.compile(optimizer='adam', loss='mean_squared_error')
    
    model_norm.fit(X_train_norm, y_train_norm, epochs=50, verbose=0)
    
    y_pred_norm = model_norm.predict(X_test_norm)
    mse_norm = mean_squared_error(y_test_norm, y_pred_norm)
    
    mean_squared_errors_norm.append(mse_norm)

mean_mse_norm = pd.Series(mean_squared_errors_norm).mean()
std_mse_norm = pd.Series(mean_squared_errors_norm).std()

print("\nMean of Mean Squared Errors (Normalized):", mean_mse_norm)
print("Standard Deviation of Mean Squared Errors (Normalized):", std_mse_norm)

# Increase the number of hidden layers
# Repeat steps B using a neural network with three hidden layers
mean_squared_errors_hidden_layers = []

for _ in range(50):
    model_hidden_layers = Sequential()
    model_hidden_layers.add(Dense(10, activation='relu', input_shape=(X_train_norm.shape[1],)))
    model_hidden_layers.add(Dense(10, activation='relu'))
    model_hidden_layers.add(Dense(10, activation='relu'))
    model_hidden_layers.add(Dense(1))
    
    model_hidden_layers.compile(optimizer='adam', loss='mean_squared_error')
    
    model_hidden_layers.fit(X_train_norm, y_train_norm, epochs=50, verbose=0)
    
    y_pred_hidden_layers = model_hidden_layers.predict(X_test_norm)
    mse_hidden_layers = mean_squared_error(y_test_norm, y_pred_hidden_layers)
    
    mean_squared_errors_hidden_layers.append(mse_hidden_layers)

mean_mse_hidden_layers = pd.Series(mean_squared_errors_hidden_layers).mean()
std_mse_hidden_layers = pd.Series(mean_squared_errors_hidden_layers).std()

print("\nMean of Mean Squared Errors (Hidden Layers):", mean_mse_hidden_layers)
print("Standard Deviation of Mean Squared Errors (Hidden Layers):", std_mse_hidden_layers)
