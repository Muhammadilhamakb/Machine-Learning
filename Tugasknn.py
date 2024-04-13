import numpy as np

x_train = np.array([[40, 5, 60],
                   [50, 8, 40],
                   [50, 7, 30],
                   [70, 4, 60],
                   [80, 4, 80],
                   [60, 6, 60]])
y_train = np.array(['Jelek', 'Bagus', 'Jelek','Bagus','Bagus','Bagus'])

x_test = np.array([50, 3, 40])

distances = np.sum(np.abs(x_train - x_test), axis=1)  # Jarak Manhattan

k = 3
nearest_indices = np.argsort(distances)[:k]
nearest_neighbors = y_train[nearest_indices]
unique_classes, counts = np.unique(nearest_neighbors, return_counts=True)
prediction_k3 = unique_classes[np.argmax(counts)]
print(f"Kelas prediksi untuk data uji dengan K = 3 adalah :'{prediction_k3}'.")


k = 4
nearest_indices = np.argsort(distances)[:k]
nearest_neighbors = y_train[nearest_indices]
unique_classes, counts = np.unique(nearest_neighbors, return_counts=True)
prediction_k4 = unique_classes[np.argmax(counts)]
print(f"Kelas prediksi untuk data uji dengan K = 4 adalah :'{prediction_k4}'.")


k = 5
nearest_indices = np.argsort(distances)[:k]
nearest_neighbors = y_train[nearest_indices]
unique_classes, counts = np.unique(nearest_neighbors, return_counts=True)
prediction_k5 = unique_classes[np.argmax(counts)]
print(f"Kelas prediksi untuk data uji dengan K = 5 adalah :'{prediction_k5}'.")
