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
   
        age = float(row["Age"]) / 30
        sleep = float(row["Sleep_Hours"])
        stres = float(row["Stress_Level"])
        socky = float(row["Social_Media_Hours"])
       
        if row["Gender"] == "Male":
            gender = 0
        elif row["Gender"] == "Female":
            gender = 1
        else:
            gender = 2 
        
        if row["Device_Type"] == "Android":
            device = 1
        elif row["Device_Type"] == "iOS":
            device = 2
        else:
            device = 3

        

        phone_hours_raw = float(row["Daily_Phone_Hours"])
        if phone_hours_raw < 4:
            phone_hours = 0
        elif phone_hours_raw > 4 and phone_hours_raw < 8:
            phone_hours = 1
        elif phone_hours_raw > 8 and phone_hours_raw < 12:
            phone_hours = 2
        else:   
            phone_hours = 3
       
        

        if row["Occupation"] == "Student":
            occupation = 0
        elif row["Occupation"] == "Professional":
            occupation = 1
        elif row["Occupation"] == "Freelancer":
            occupation = 2
        else:
            occupation = 3 

        X.append([age, gender, occupation, device, sleep, stres, socky])
        Y.append(phone_hours)

# ---------- Rozdělení na trénování a testování ----------
trening_X, test_X, trening_Y, test_Y = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# ---------- Neuronová síť ----------
neural_network = MLPClassifier(
    hidden_layer_sizes=(32, 8),
    activation="relu",
    max_iter=2000,
    verbose=True,
    random_state=4,
    n_iter_no_change=40
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