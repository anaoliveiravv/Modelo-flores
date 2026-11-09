import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

flores = pd.read_csv("flores.csv")

X = flores.drop(columns=["especie"])
y = flores["especie"]

preprocessamento = ColumnTransformer(
    transformers=[
        ("cores", OneHotEncoder(handle_unknown="ignore"), ["cor"])
    ],
    remainder="passthrough"
)

modelo = Pipeline([
    ("preprocessamento", preprocessamento),
    ("classificador", DecisionTreeClassifier(random_state=42))
])

modelo.fit(X, y)

print("Modelo treinado!")
print("Espécies:", ", ".join(modelo.classes_))

while True:
    print("\n--- Identificador de Flores ---")

    diametro = float(input("Diâmetro da flor (cm): "))
    comprimento = float(input("Comprimento da pétala (cm): "))
    petalas = int(input("Quantidade de pétalas: "))
    cor = input("Cor da flor: ").lower().strip()

    nova_flor = pd.DataFrame([{
        "diametro_flor_cm": diametro,
        "comprimento_petala_cm": comprimento,
        "numero_petalas": petalas,
        "cor": cor
    }])

    resultado = modelo.predict(nova_flor)

    print("\nEspécie prevista:", resultado[0])

    continuar = input("\nDeseja testar outra flor? (s/n): ").lower()

    if continuar != "s":
        print("Programa encerrado!")
        break
