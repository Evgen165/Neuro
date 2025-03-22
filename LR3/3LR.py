import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Загрузка данных
data = pd.read_csv("data.csv", header=None)
features = data.iloc[:, :-1].values.astype(np.float32)
labels = data.iloc[:, -1].values
labels = np.where(labels == 'Iris-setosa', 0, 1)  
labels = labels.astype(np.longlong)

# Разделение данных на обучающую и тестовую выборки
train_size = int(0.7 * len(features))
train_features, test_features = features[:train_size], features[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Преобразование данных в тензоры
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Определение модели, функции потерь и оптимизатора
model = nn.Linear(4, 2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Обучение модели
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    predictions = model(train_features_tensor)
    loss = loss_function(predictions, train_labels_tensor)
    loss.backward()
    optimizer.step()
    print(f'Эпоха {epoch + 1}/{num_epochs}, Потеря: {loss.item():.4f}')

# Оценка модели на тестовых данных
with torch.no_grad():
    predictions = model(test_features_tensor)
    _, predicted_labels = torch.max(predictions, 1)
    correct_predictions = (predicted_labels == test_labels_tensor).sum().item()
    total_samples = len(test_labels_tensor)
    accuracy = 100 * correct_predictions / total_samples
    print(f'Точность: {accuracy:.2f}%')