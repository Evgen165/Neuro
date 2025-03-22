import torch

# 1. Создаем тензор целочисленного типа со случайным значением
random_tensor = torch.randint(1, 10, (1,)) 

# 2. Преобразуем тензор к типу float32 и включаем вычисление градиента
random_tensor = random_tensor.to(torch.float32)
random_tensor.requires_grad = True

# 3. Проводим ряд операций 
power = 2
tensor_powered = random_tensor**power
random_multiplier = torch.randint(1, 11, (1,)).to(torch.float32) # случайное число от 1 до 10
tensor_scaled = tensor_powered * random_multiplier
tensor_exponential = torch.exp(tensor_scaled)

# 4. Вычисляем и выводим производную 
tensor_exponential.backward()
print(f"Производная для степени n={power}: {random_tensor.grad}")