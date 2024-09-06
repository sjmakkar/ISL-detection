import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract features (data) and labels from the data dictionary
data_list = data_dict['data']
labels = data_dict['labels']

# Determine the maximum length of sequences
max_length = max(len(seq) for seq in data_list)

# Pad sequences to the maximum length
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_list]

# Convert the padded data list into a NumPy array
data = np.array(padded_data)

# Preprocess the data (Example: Scaling numerical features)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Evaluate the model's accuracy
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
