import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from Models.NARNET import NARNET
from Utils.data_generator import generate_synthetic_data


def train_model():
    seq_len = 10
    num_samples = 1000
    x_data, y_data = generate_synthetic_data(seq_len, num_samples)

    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    input_size = 1
    hidden_size = 50
    num_layers = 8
    output_size = 1
    model = NARNET(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(x_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

    model.load_state_dict(best_model)
    return model, x_test, y_test, criterion
