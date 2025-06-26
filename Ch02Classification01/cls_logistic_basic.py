import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Simüle veri: 100 kişi, rastgele boy (cm) ve kilo (kg)
np.random.seed(0)
boy = np.random.normal(170, 10, 100)  # Ortalama 170 cm, std=10
kilo = np.random.normal(70, 12, 100)  # Ortalama 70 kg, std=12

# Vücut kitle indeksi (BMI) hesapla
bmi = kilo / ((boy / 100) ** 2)

# Obezite: BMI > 30 ise 1, değilse 0
obez = (bmi > 30).astype(int)

X = np.column_stack([boy, kilo])
y = obez

# Model kur ve eğit
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
print("Doğruluk:", (y == y_pred).mean())

# Sınırı görselleştir (sadece 2 özellik olduğu için kolay)
plt.scatter(boy, kilo, c=y, cmap="coolwarm", label="Gerçek Sınıf", alpha=0.7, marker="o")
plt.xlabel("Boy (cm)")
plt.ylabel("Kilo (kg)")
plt.title("Obezite Sınıflandırması")
plt.show()
