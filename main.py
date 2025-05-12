import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import KFold, train_test_split, ParameterGrid
from imblearn.under_sampling import NearMiss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy.stats import wilcoxon

def data_imputation(df):
    df['DATE_DIED'] = df['DATE_DIED'].replace('9999-99-99', 1)
    df['DATE_DIED'] = df['DATE_DIED'].apply(lambda x: 0 if x != 1 else 1)

    categorical_cols = [
        'SEX', 'CLASIFFICATION_FINAL', 'PATIENT_TYPE', 'DATE_DIED', 'INTUBED', 'PNEUMONIA',
        'PREGNANT', 'DIABETES', 'USMER', 'MEDICAL_UNIT', 'COPD', 'ASTHMA', 'ICU',
        'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY', 'RENAL_CHRONIC', 'TOBACCO'
    ]

    numerical_cols = ['AGE']

    df.replace([97, 98, 99], np.nan, inplace=True)

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

def dataframe_analysis(df):
  plt.figure(figsize=(12, 10))  # Ajusta o tamanho da figura

  # Histograma da distribuição das idades por Sexo
  plt.subplot(2, 2, 1)
  sns.histplot(df[df['SEX'] == 1]['AGE'], bins=30, kde=True, color='blue', label="Feminino", alpha=0.6)
  sns.histplot(df[df['SEX'] == 2]['AGE'], bins=30, kde=True, color='red', label="Masculino", alpha=0.6)
  plt.title('Distribuição das Idades por Sexo')
  plt.xlabel('Idade')
  plt.ylabel('Frequência')
  plt.legend()

  # Scatterplot: Idade x Gravidade do COVID-19 (Classificação)
  plt.subplot(2, 2, 2)
  sns.scatterplot(x=df['AGE'], y=df['CLASIFFICATION_FINAL'], alpha=0.5, color='red')
  plt.title('Idade vs Classificação do COVID-19')
  plt.xlabel('Idade')
  plt.ylabel('Classificação (1-3: Positivo, 4+: Negativo/Inconclusivo)')

  # Line plot: Mortalidade por idade
  plt.subplot(2, 2, 3)
  mortalidade_por_idade = df.groupby('AGE')['DATE_DIED'].mean()
  sns.lineplot(x=mortalidade_por_idade.index, y=mortalidade_por_idade.values, color='black')
  plt.title('Taxa de Mortalidade por Idade')
  plt.xlabel('Idade')
  plt.ylabel('Proporção de óbitos')

  # Boxplot: Distribuição da Idade dos Pacientes
  plt.subplot(2, 2, 4)
  sns.boxplot(y=df['AGE'], color='purple', width=0.3)
  plt.title('Distribuição da Idade dos Pacientes')
  plt.ylabel('Idade')

  # Ajustar os limites do eixo Y para evitar distorções visuais
  plt.ylim(df['AGE'].min() - 5, df['AGE'].max() + 5)

  # Ajustar layout e exibir
  plt.tight_layout()
  plt.show()

def dataset_normalization(df):
  # Teste de Shapiro-Wilk para verificar a distribuição dos dados na coluna 'AGE'
  stat, p_value = stats.shapiro(df['AGE'])

  alpha = 0.05  # Nível de significância
  if p_value > alpha: # Segue uma Distribuição Normal
    scaler = StandardScaler()

    df['AGE'] = scaler.fit_transform(df[['AGE']])
  else: # Não Segue uma Distribuição Normal
    scaler = MinMaxScaler()

    df['AGE'] = scaler.fit_transform(df[['AGE']])

def kFold_KNN_division(df):
  df_sample = df.sample(frac=0.01, random_state=42)

  # print(f"Tamanho do DataFrame Original: {len(df)}")
  # print(f"Tamanho do DataFrame Amostrado: {len(df_sample)}")

  x = df_sample.drop('DATE_DIED', axis=1)
  y = df_sample['DATE_DIED']

  kf = KFold(n_splits=10, shuffle=True, random_state=42)

  hyperParameter = {
      'n_neighbors': [3, 5, 7, 9, 11, 15],
      'metrics': ['euclidean', 'manhattan', 'chebyshev', 'cosine']
  }

  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_list = []
  wilcoxon_list = []

  for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    x_train_divided, x_val, y_train_divided, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_divided = scaler.fit_transform(x_train_divided)
    x_val = scaler.transform(x_val)

    undersampler = NearMiss(version=3)
    x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train_divided, y_train_divided)

    best_accuracy = 0
    best_params = None

    for params in ParameterGrid(hyperParameter):
      knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric=params['metrics'])
      knn.fit(x_train_resampled, y_train_resampled)
      y_pred = knn.predict(x_val)
      accuracy = accuracy_score(y_val, y_pred)

      if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params

    print("\nMelhor Parâmetro:")
    print(best_params)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)

    knn_best = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metrics'])
    knn_best.fit(x_train_resampled, y_train_resampled)
    y_pred = knn_best.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    # Teste de Wilcoxon para comparar previsões com valores reais
    try:
        wilcoxon_test = wilcoxon(y_test, y_pred)[1]  # Retorna o valor p
    except ValueError:
        wilcoxon_test = np.nan  # O teste não pode ser calculado se não houver variação nos dados

    # Armazenar resultados
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    wilcoxon_list.append(wilcoxon_test)

    # print(f"\nFold Resultados:")
    # print(f"Acurácia: {accuracy:.4f}")
    # print(f"Precisão: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1-score: {f1:.4f}")
    # print(f"Wilcoxon p-valor: {wilcoxon_test:.4f}")

  print("\nMédia e Desvio Padrão das Métricas - Modelo KNN")
  print(f"Acurácia média: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
  print(f"Precisão média: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
  print(f"Recall médio: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
  print(f"F1-score médio: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
  print(f"Wilcoxon p-valor médio: {np.nanmean(wilcoxon_list):.4f}")

def kFold_DecisionTree_division(df):
    df_sample = df.sample(frac=0.01, random_state=42)

    # print(f"Tamanho do DataFrame Original: {len(df)}")
    # print(f"Tamanho do DataFrame Amostrado: {len(df_sample)}")

    x = df_sample.drop('DATE_DIED', axis=1)
    y = df_sample['DATE_DIED']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    hyperParameter = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    wilcoxon_list = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_divided, x_val, y_train_divided, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_divided = scaler.fit_transform(x_train_divided)
        x_val = scaler.transform(x_val)

        undersampler = NearMiss(version=3)
        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train_divided, y_train_divided)

        best_accuracy = 0
        best_params = None

        for params in ParameterGrid(hyperParameter):
            tree = DecisionTreeClassifier(
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42
            )
            tree.fit(x_train_resampled, y_train_resampled)
            y_pred = tree.predict(x_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        print("\nMelhor Parâmetro:")
        print(best_params)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)

        tree_best = DecisionTreeClassifier(
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42
        )
        tree_best.fit(x_train_resampled, y_train_resampled)
        y_pred = tree_best.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

        # Teste de Wilcoxon para comparar previsões com valores reais
        try:
            wilcoxon_test = wilcoxon(y_test, y_pred)[1]  # Retorna o valor p
        except ValueError:
            wilcoxon_test = np.nan  # O teste não pode ser calculado se não houver variação nos dados

        # Armazenar resultados
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        wilcoxon_list.append(wilcoxon_test)

        # print(f"\nFold Resultados:")
        # print(f"Acurácia: {accuracy:.4f}")
        # print(f"Precisão: {precision:.4f}")
        # print(f"Recall: {recall:.4f}")
        # print(f"F1-score: {f1:.4f}")
        # print(f"Wilcoxon p-valor: {wilcoxon_test:.4f}")

    print("\nMédia e Desvio Padrão das Métricas - Modelo Árvore de Decisão")
    print(f"Acurácia média: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"Precisão média: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall médio: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1-score médio: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Wilcoxon p-valor médio: {np.nanmean(wilcoxon_list):.4f}")

def kFold_RandomForest_division(df):
    df_sample = df.sample(frac=0.01, random_state=42)

    # print(f"Tamanho do DataFrame Original: {len(df)}")
    # print(f"Tamanho do DataFrame Amostrado: {len(df_sample)}")

    x = df_sample.drop('DATE_DIED', axis=1)
    y = df_sample['DATE_DIED']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    hyperParameter = {
        'n_estimators': [50, 100, 200],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    wilcoxon_list = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_divided, x_val, y_train_divided, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_divided = scaler.fit_transform(x_train_divided)
        x_val = scaler.transform(x_val)

        undersampler = NearMiss(version=3)
        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train_divided, y_train_divided)

        best_accuracy = 0
        best_params = None

        for params in ParameterGrid(hyperParameter):
            rf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            rf.fit(x_train_resampled, y_train_resampled)
            y_pred = rf.predict(x_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        print("\nMelhor Parâmetro:")
        print(best_params)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)

        rf_best = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            criterion=best_params['criterion'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
        rf_best.fit(x_train_resampled, y_train_resampled)
        y_pred = rf_best.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

        try:
            wilcoxon_test = wilcoxon(y_test, y_pred)[1]
        except ValueError:
            wilcoxon_test = np.nan

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        wilcoxon_list.append(wilcoxon_test)

    print("\nMédia e Desvio Padrão das Métricas - Modelo Random Forest")
    print(f"Acurácia média: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"Precisão média: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall médio: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1-score médio: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Wilcoxon p-valor médio: {np.nanmean(wilcoxon_list):.4f}")

def kFold_CatBoost_division(df):
    df_sample = df.sample(frac=0.01, random_state=42)

    x = df_sample.drop('DATE_DIED', axis=1)
    y = df_sample['DATE_DIED']

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    hyperParameter = {
        'iterations': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 10]
    }

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    wilcoxon_list = []

    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        x_train_divided, x_val, y_train_divided, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        x_train_divided = scaler.fit_transform(x_train_divided)
        x_val = scaler.transform(x_val)

        undersampler = NearMiss(version=3)
        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train_divided, y_train_divided)

        best_accuracy = 0
        best_params = None

        for params in ParameterGrid(hyperParameter):
            cb = CatBoostClassifier(
                iterations=params['iterations'],
                learning_rate=params['learning_rate'],
                depth=params['depth'],
                loss_function='Logloss',
                verbose=0,
                random_seed=42
            )
            cb.fit(x_train_resampled, y_train_resampled)
            y_pred = cb.predict(x_val)
            accuracy = accuracy_score(y_val, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params

        print("\nMelhor Parâmetro:")
        print(best_params)

        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_resampled, y_train_resampled = undersampler.fit_resample(x_train, y_train)

        cb_best = CatBoostClassifier(
            iterations=best_params['iterations'],
            learning_rate=best_params['learning_rate'],
            depth=best_params['depth'],
            loss_function='Logloss',
            verbose=0,
            random_seed=42
        )
        cb_best.fit(x_train_resampled, y_train_resampled)
        y_pred = cb_best.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

        try:
            wilcoxon_test = wilcoxon(y_test, y_pred)[1]
        except ValueError:
            wilcoxon_test = np.nan

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        wilcoxon_list.append(wilcoxon_test)

    print("\nMédia e Desvio Padrão das Métricas - Modelo CatBoost")
    print(f"Acurácia média: {np.mean(accuracy_list):.4f} ± {np.std(accuracy_list):.4f}")
    print(f"Precisão média: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    print(f"Recall médio: {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")
    print(f"F1-score médio: {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    print(f"Wilcoxon p-valor médio: {np.nanmean(wilcoxon_list):.4f}")

df = pd.read_csv('./Covid_Data.csv')

# Preparação e análise inicial dos dados e amostras
data_imputation(df)
# dataframe_analysis(df)
# dataset_normalization(df)

# Divisão das amostras e treinamento com os modelos
kFold_KNN_division(df)
kFold_DecisionTree_division(df)
kFold_RandomForest_division(df)
kFold_CatBoost_division(df)