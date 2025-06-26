import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Basit veri seti oluşturalım
X = np.arange(0, 10, 0.5).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# SVR modeli (RBF kernel ile)
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X, y)
y_pred = svr.predict(X)

plt.scatter(X, y, color='blue', label='Gerçek')
plt.plot(X, y_pred, color='red', label='SVR Tahmin')
plt.legend()
plt.title("Support Vector Regression (SVR)")
plt.show()


