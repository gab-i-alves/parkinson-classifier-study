import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List

def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str = "Matriz de Confusão") -> None:
    """
    Plota uma matriz de confusão com rótulos personalizados.

    Args:
        cm (np.ndarray): Matriz de confusão.
        labels (List[str]): Lista de nomes das classes.
        title (str): Título do gráfico (default: "Matriz de Confusão").

    Returns:
        None
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()