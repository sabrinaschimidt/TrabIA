import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregue seu conjunto de dados
dados = pd.read_csv('*.csv')

# Exiba informações básicas sobre o conjunto de dados
print("Informações Básicas:")
print(dados.info())

# Visualize as primeiras linhas do conjunto de dados
print("\nPrimeiras linhas:")
print(dados.head())

# Estatísticas resumidas das colunas numéricas
print("\nEstatísticas resumidas:")
print(dados.describe())

# Verifique valores ausentes
print("\nValores ausentes:")
print(dados.isnull().sum())

# Visualize a distribuição das características numéricas
colunas_numericas = dados.select_dtypes(include=[np.number])
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
for coluna in colunas_numericas.columns:
    sns.histplot(dados[coluna], kde=True, label=coluna)
plt.legend()
plt.title("Distribuição das Características Numéricas")
plt.show()

# Visualize a correlação entre características numéricas
matriz_correlacao = colunas_numericas.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacao, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mapa de Calor de Correlação")
plt.show()

# Visualize a relação entre características categóricas e a variável alvo
colunas_categoricas = dados.select_dtypes(exclude=[np.number])
plt.figure(figsize=(12, 6))
for coluna in colunas_categoricas.columns:
    sns.boxplot(x=coluna, y='coluna_alvo', data=dados)
    plt.title(f"{coluna} vs Variável Alvo")
    plt.xticks(rotation=45)
    plt.show()