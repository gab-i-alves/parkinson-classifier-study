from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Model

def get_feature_extractor() -> Model:
    """
    Retorna uma instância do modelo ResNet50 pré-treinado sem a camada de classificação final. \n
    Utiliza pooling global average para gerar vetores de características.

    Returns:
        Model: Modelo Keras ResNet50 com saída de pooling 'avg'
    """
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract(img_path: str, model: Model, preprocess_func, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extrai um vetor de características de uma imagem usando o modelo fornecido.

    Args:
        img_path (str): Caminho da imagem.
        model (Model): Modelo CNN (ex: ResNet50).
        preprocess_func (function): Função de pré-processamento (ex: preprocess_input).
        target_size (tuple): Tamanho para redimensionar a imagem (default: (224, 224)).

    Returns:
        np.ndarray: Vetor de características extraído da imagem (shape: [2048,])
    """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = preprocess_func(np.expand_dims(x, axis=0))
    features = model.predict(x, verbose=0)
    return features[0]
