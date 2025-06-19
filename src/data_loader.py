import os
import glob
import numpy as np
from typing import List, Tuple
from .feature_extractor import extract
from tensorflow.keras.models import Model

def load_data(paths: List[str], model: Model, preprocess_func) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carrega imagens de diretórios e extrai vetores de características para cada uma.

    Args:
        paths (List[str]): Lista de caminhos para os diretórios (ex: train e test).
        model (Model): Modelo CNN usado para extração.
        preprocess_func (function): Função de pré-processamento da imagem.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: Matriz de vetores de características extraídos.
            - y: Vetor de rótulos (0 = saudável, 1 = Parkinson).
    """
    X, y = [], []
    for path in paths:
        for label, class_name in [(0, "healthy"), (1, "parkinson")]:
            folder = os.path.join(path, class_name)
            for img_file in glob.glob(os.path.join(folder, '*.png')):
                features = extract(img_file, model, preprocess_func)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)