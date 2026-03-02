import csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ---------- Načtení CSV a úprava dat ----------
X = []  # = vstupy
Y = []  # = výstupy

with open(r"3. strojove_uceni\data\Smartphone_Usage_Productivity_Dataset_50000.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Převeďte věk na číselný typ
        Age = float(row["Age"])
        
        # Úprava pro pohlaví
        if row["Gender"] == "Male":
            gender = 0
        elif row["Gender"] == "Female":
            gender = 1
        else:
            gender = 2  # Může být další kategorie, pokud je v souboru nějaké jiné pohlaví
        
        # Povolíme jen číselné hodnoty pro telefonní hodiny
        phone_hours = float(row["Daily_Phone_Hours"])
        
        # Pokud je potřeba, přidejte kódování pro profesi
        if row["Occupation"] == "Student":
            occupation = 0
        elif row["Occupation"] == "Professional":
            occupation = 1
        elif row["Occupation"] == "Freelancer":
            occupation = 2
        else:
            occupation = 3  # Například Business Owner nebo jiné profesní kategorie

        # Přidejte nové vstupy
        X.append([Age, gender, occupation])
        Y.append(phone_hours)

# ---------- Kontrola formátu Y ----------
print("První 5 hodnot Y:", Y[:5])  # Tiskněte první 5 hodnot pro diagnostiku

# ---------- Rozdělení na trénování a testování ----------
trening_X, test_X, trening_Y, test_Y = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# ---------- Neuronová síť ----------
neural_network = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation="relu",
    max_iter=2000,
    verbose=True,
    random_state=4
)

# Trénování modelu
neural_network.fit(trening_X, trening_Y)

# ---------- Vyhodnocení ----------
results = neural_network.predict(test_X)

# Vyhodnocení přesnosti
correct = 0
for i in range(len(results)):
    if test_Y[i] == results[i]:
        correct += 1
print("Přesnost:", correct / len(results))

# ---------- Confusion matrix ----------
ConfusionMatrixDisplay.from_predictions(
    test_Y, results)
plt.show()