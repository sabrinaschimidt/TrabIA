import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# Lista com os caminhos dos arquivos CSV
caminhos_dos_arquivos = glob('/home/sabrina/projects/TrabIA/*.csv')

# Inicializa um DataFrame vazio para armazenar os dados combinados
dados_combinados = pd.DataFrame()

# Itera sobre os arquivos e carrega os dados
for caminho in caminhos_dos_arquivos:
    # Carrega o arquivo CSV em um DataFrame
    dados = pd.read_csv(caminho)

    # Adiciona os dados ao DataFrame combinado usando concat
    dados_combinados = pd.concat([dados_combinados, dados], ignore_index=True)

# Verifica se o DataFrame combinado não está vazio
if not dados_combinados.empty:
    # Todas as colunas, exceto a última, são atributos
    X = dados_combinados.iloc[:, :-1].copy()

    # A última coluna dos dados originais é assumida como a classe (y)
    y = dados_combinados.iloc[:, -1]

    # Lida com valores ausentes usando SimpleImputer
    imputer = SimpleImputer(strategy='most_frequent')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Codifica as variáveis categóricas em X usando LabelEncoder
    label_encoder = LabelEncoder()
    X_encoded = X.apply(label_encoder.fit_transform)

    # Codifica as variáveis categóricas na coluna de classes (y)
    label_encoder_y = LabelEncoder()
    y_encoded = label_encoder_y.fit_transform(y)

    # Divide os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

    # 1. K Nearest Neighbors (KNN)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y_train)
    knn_predictions = knn_classifier.predict(X_test)

    # Avalie a precisão do modelo KNN
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"Acurácia do KNN: {knn_accuracy:.2f}")

    # Resto do código...
else:
    print("Atenção: Nenhum dado foi carregado. Verifique se os arquivos CSV estão corretamente especificados.")
