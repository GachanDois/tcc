# %%
import pandas as pd
import seaborn as sn
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.optimizers
from xgboost import XGBClassifier
import joblib
from python_speech_features import mfcc
from sklearn.metrics import f1_score, confusion_matrix
import librosa
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%

# Função para extrair as características de áudio
def extrair_caracteristicas(audio, taxa_amostragem):
    chroma_stft = librosa.feature.chroma_stft(y=audio, sr=taxa_amostragem)
    chroma_mean = chroma_stft.mean(axis=1)
    chroma_over_mean = chroma_mean.mean()

    rms = librosa.feature.rms(y=audio)
    rms_mean = rms.mean()

    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=taxa_amostragem)
    spectral_centroid_mean = spectral_centroid.mean()

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=taxa_amostragem)
    spectral_bandwidth_mean = spectral_bandwidth.mean()

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=taxa_amostragem, roll_percent=0.85)
    spectral_rolloff_mean = rolloff.mean()

    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
    zero_crossing_rate_mean = zero_crossing_rate.mean()

    mfccs = librosa.feature.mfcc(y=audio, sr=taxa_amostragem, n_mfcc=20)
    mfccs_mean = mfccs.mean(axis=1)

    return (chroma_over_mean, rms_mean, spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_mean, zero_crossing_rate_mean, *mfccs_mean)

# Função para criar o DataFrame a partir dos dados
def criar_dataframe(caminho_audio):
    
    audio, taxa_amostragem = librosa.load(caminho_audio, sr=None)
    tamanho_janela = taxa_amostragem  # 1 segundo
    label = 'REAL'  # Substitua 'LABEL' pelo rótulo apropriado
    passo = tamanho_janela
    inicio = 0
    fim = tamanho_janela
    dados = []

    while fim <= len(audio):
        segmento = audio[inicio:fim]
        caracteristicas = extrair_caracteristicas(segmento, taxa_amostragem)
        dados.append(caracteristicas + (label,))
        inicio += passo
        fim += passo

    colunas = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
               'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10',
               'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20',
               'LABEL']
    df = pd.DataFrame(data=dados, columns=colunas)
    
    
    return df

# Caminho para o arquivo de áudio
#caminho_audio = 'Audios_WAV/Yato_Viva.wav'


# Crie o DataFrame a partir dos dados
#df = criar_dataframe(caminho_audio)

# Visualize o DataFrame
#print(df)



import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(LabelEncoder):
    def fit(self, y):
        return super().fit(y)

    def transform(self, y):
        return super().transform(y)

    def inverse_transform(self, y):
        return super().inverse_transform(y)

def avaliar_modelo(caminho_audio):
    # Carregue o modelo salvo com pickle
    with open('xgboost_model.pkl', 'rb') as file:
        modelo_carregado = pickle.load(file)

    # Realize as operações que você deseja
    df = criar_dataframe(caminho_audio)  # Carregue o DataFrame

    x = df.drop(columns=["LABEL"])
    y = df['LABEL']

    label_encoder = CustomLabelEncoder()
    label_encoder.fit(["REAL", "FALSE"])

    transformed_labels = label_encoder.transform(y)

    original_labels = label_encoder.inverse_transform(transformed_labels)

    previsoes = modelo_carregado.predict(x)

    previsoes_str = label_encoder.inverse_transform(previsoes)

    # Calcule a acurácia
    acuracia = accuracy_score(y, previsoes_str)
    
    print("Previsões:", previsoes_str)
    print("Acurácia do modelo XGBoost: {:.2f}%".format(acuracia * 100))

# Chame a função para executar a avaliação



