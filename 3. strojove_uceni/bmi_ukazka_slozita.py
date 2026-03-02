# Neuronová síť predikující BMI kategorii
# Jedná se pouze o učební ukázku - pro BMi je jinak využití neuronky nevhodné

import csv

# ZMĚNA: Nutno načíst nové knihovny
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ---------- Načtení CSV a úprava dat ----------
X = []  # = vstupy
Y = []  # = výstupy

with open(r"3. strojove_uceni\data\bmi.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        height = float(row["Height"])
        weight = float(row["Weight"])
        # Na vstupu mohou být jen číselné vstupy:
        if row["Gender"] == "Male":
            gender = 0
        else:
            gender = 1

        bmi_category = int(row["Index"])

        X.append([gender, height, weight])
        Y.append(bmi_category)


# ---------- Rozdělení na trénování a testování ----------
# ZMĚNA: split pomocí funkce
#   funcke má tu výhodu, že dataset zároveň promíchá

trening_X, test_X, trening_Y, test_Y  = train_test_split(
        X, Y,
        test_size=0.2,
        random_state=42)

# ---------- Neuronová síť ----------
neural_network = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation="relu",
    max_iter=2000,
    verbose=True,
    random_state=4
)

neural_network.fit(trening_X, trening_Y)

# ---------- Vyhodnocení ----------
results = neural_network.predict(test_X)

correct = 0
for i in range(len(results)):
    if test_Y[i] == results[i]:
        correct += 1
print("Přesnost:", correct / len(results))

# ---------- Confusion matrix ----------
# zobrazuje jaké odpovědi dává neuronka pro dané vstupy
ConfusionMatrixDisplay.from_predictions(
    test_Y, results)
plt.show()
