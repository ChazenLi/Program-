import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check the current working directory
print(f"Current working directory: {os.getcwd()}")

# Ensure the file path is correct
file_path = 'E:/Python-projects/pytorch/练习/diabetes.csv'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"{file_path} not found in the current directory.")

# Prepare dataset
xy = np.genfromtxt(file_path, delimiter=',', dtype=np.float32, skip_header=1)
x_data = xy[:, :-1]
y_data = xy[:, [-1]]

# Standardize features
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# Split dataset into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Convert to torch tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).float()

# Check if CUDA is available and use GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move data to the device
x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

# Design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 128)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.linear2 = torch.nn.Linear(128, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.linear3 = torch.nn.Linear(64, 32)
        self.bn3 = torch.nn.BatchNorm1d(32)
        self.linear4 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.bn1(self.linear1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.linear2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.linear3(x)))
        x = self.dropout(x)
        x = self.sigmoid(self.linear4(x))
        return x

model = Model().to(device)  # Move model to the device

# Construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch_list = []
train_loss_list = []
val_loss_list = []
nums_epoch = 100  # Set the number of epochs for training

# Training cycle: forward, backward, update
for epoch in range(nums_epoch):
    model.train()  # Set the model to training mode
    y_train_pred = model(x_train)
    train_loss = criterion(y_train_pred, y_train)
    
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        y_val_pred = model(x_val)
        val_loss = criterion(y_val_pred, y_val)
    
    epoch_list.append(epoch)
    train_loss_list.append(train_loss.item())
    val_loss_list.append(val_loss.item())
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{nums_epoch}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Plot the loss curves
plt.plot(epoch_list, train_loss_list, label='Train Loss')
plt.plot(epoch_list, val_loss_list, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curves')
plt.legend()
plt.show()

# Function to predict the probability of having the disease
def predict(model, input_data, device):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
        output = model(input_tensor)
        return output.item()

# Example input data (modify this with actual input data)
example_input = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]  # Example data point

# Normalize the input data
example_input = scaler.transform([example_input])  # Reshape to (1, 8)

# Predict the probability
predicted_prob = predict(model, example_input, device)
print(f"Predicted probability of having the disease: {predicted_prob:.4f}")

# Print the final diagnosis based on a threshold (e.g., 0.5)
threshold = 0.5
if predicted_prob >= threshold:
    print("The model predicts that there is a high probability of having the disease.")
else:
    print("The model predicts that there is a low probability of having the disease.")