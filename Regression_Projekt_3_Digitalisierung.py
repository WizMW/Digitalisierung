import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
import matplotlib.pyplot as plt


# Weibull Parameter einlesen
data = pd.read_csv('Weibull_dist_mit_label.csv')
# Generierung von x- und y-Werten aus Weibull-Verteilungsparametern
def generate_data(shape, loc, scale, size=1500):
    #x = np.linspace(0, 10, size)
    x = np.linspace(0, weibull_min.ppf(0.999, shape, loc=loc, scale=scale), size)
    y = stats.weibull_min.cdf(x, c=shape, loc=loc, scale=scale)
    data = np.column_stack((x, y))
    return data

#Labels initialisieren
X = []
y = []

for index, row in data.iterrows():
    distribution = generate_data(row['shape'], row['loc'], row['scale'])
    X.append(distribution)
    y.append(row['label'])

# In Numpy-Arrays umwandeln
X = np.array(X)
y = np.array(y)

# Aus zweidimensionalem Array, eindimensonaler Array für Regressor
X = X.reshape(X.shape[0], -1)

X_save = np.array(X).reshape(data.shape[0], -1)  
y_save = np.array(y).reshape(-1, 1)  

combined_data = np.hstack((X_save, y_save))

np.savetxt('combined_weibull_data.csv', combined_data, delimiter=',', header='features,label', comments='', fmt='%f')

print("Combined data saved to combined_weibull_data.csv")


# MinMaxScaler für X
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modell initialisieren und trainieren
model = MLPRegressor(hidden_layer_sizes=(100,100,50), activation='relu', solver='adam', alpha=0.001, random_state=42, max_iter=2000, tol=1e-6)
model.fit(X_train, y_train)

# Vorhersagen auf dem Testset machen
y_pred = model.predict(X_test)

# Mean Squared Error berechnen
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error auf dem Testset:", mse)

error_test = 0

# Ausgabe der Vorhersagen
for label, prediction in zip(y_test, y_pred):
    print("Label:", label, "Vorhergesagtes Label:", prediction)
    error_test += np.square(label-prediction)
error_test = error_test/len(y_test)

# Vorhersagen für alle Daten machen
all_predictions = model.predict(X_scaled)

error = 0

# Durchschnittliche Vorhersagen für verschiedene Labels berechnen
average_predictions = {}
for label, prediction in zip(y, all_predictions):
    if label in average_predictions:
        average_predictions[label].append(prediction)
    else:
        average_predictions[label] = [prediction]

average_predictions = {label: np.mean(predictions) for label, predictions in average_predictions.items()}

print(average_predictions)

# Kennlinie plotten
plt.scatter(average_predictions.keys(), average_predictions.values(), marker='o')
plt.xlabel('Label')
plt.ylabel('Durchschnittliche Vorhersage')
plt.title('Kennlinie der durchschnittlichen Vorhersagen')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)
plt.show()