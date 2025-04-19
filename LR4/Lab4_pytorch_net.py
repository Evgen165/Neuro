import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data_frame = pd.read_csv('dataset_simple.csv')
input_features = torch.tensor(data_frame[['age', 'income']].values, dtype=torch.float32)
target_values = torch.tensor(data_frame['will_buy'].values, dtype=torch.float32).reshape(-1, 1)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Первый полносвязный слой: 2 входа, 3 выхода
        self.hidden_layer = nn.Linear(2, 3)
        # Функция активации Tanh
        self.activation_tanh = nn.Tanh()
        # Второй полносвязный слой: 3 входа, 1 выход
        self.output_layer = nn.Linear(3, 1)
        # Сигмоида для получения вероятности
        self.activation_sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Прямой проход через сеть
        x = self.activation_tanh(self.hidden_layer(x))
        x = self.activation_sigmoid(self.output_layer(x))
        return x

model = SimpleClassifier()
criterion = nn.BCELoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  

num_epochs = 100
loss_history = []
for epoch in range(num_epochs):
    predictions = model(input_features)
    # Вычисление потерь
    current_loss = criterion(predictions, target_values)
    
    optimizer.zero_grad()
    current_loss.backward()
    optimizer.step()
    
    # Сохранение и вывод потерь
    loss_history.append(current_loss.item())
    if epoch % 10 == 0:
        print(f'Эпоха {epoch+1}, Потери: {current_loss.item():.4f}')

with torch.no_grad():
    final_predictions = model(input_features)
    predicted_labels = (final_predictions >= 0.5).float()
    # Вычисление точности
    accuracy = (predicted_labels == target_values).float().mean()
    print(f'\nТочность: {accuracy.item():.4f}')

plt.figure(figsize=(10, 5))
# График изменения функции потерь
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_history)
plt.xlabel("Номер эпохи")
plt.ylabel("Значение функции потерь")
plt.title("Динамика потерь в процессе обучения")