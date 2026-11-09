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

nova_flor = pd.DataFrame([{
    "diametro_flor_cm": 19.0,
    "comprimento_petala_cm": 7.2,
    "numero_petalas": 34,
    "cor": "amarela"
}])

resultado = modelo.predict(nova_flor)
print("Espécie prevista:", resultado[0])
