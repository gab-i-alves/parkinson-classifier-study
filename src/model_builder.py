from sklearn.neural_network import MLPClassifier

def build_mlp() -> MLPClassifier:
    """
    Constrói um modelo MLP (Multilayer Perceptron) com hiperparâmetros definidos.

    Returns:
        MLPClassifier: Classificador MLP configurado com uma camada escondida e ativação ReLU.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        alpha=0.0001,
        solver='adam',
        learning_rate='constant',
        max_iter=500,
        random_state=42
    )
