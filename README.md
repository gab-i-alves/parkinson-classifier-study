

<img src="https://www.researchgate.net/publication/310666616/figure/fig2/AS:11431281118555374@1675792775695/Archimedes-spirals-drawn-by-a-normal-volunteer-and-patients-with-Parkinsons-disease.png" alt="Exemplo de espirais" width="1900" height=4>


# Parkinson Classifier

Sistema para classificação de imagens spiral/wave em pacientes com ou sem Parkinson usando extração de características com ResNet50 e classificação com redes neurais (MLP).

## ⚠️ Aviso Legal
Este projeto foi desenvolvido **exclusivamente** para fins de **estudo** e pesquisa acadêmica.

Não deve ser utilizado para fins **clínicos**, **médicos** ou **diagnósticos**.

Os resultados produzidos pelo modelo não substituem avaliação profissional e não devem ser interpretados como conclusivos sobre a presença ou ausência da Doença de Parkinson.

---

## Visão Geral

Este projeto tem como objetivo detectar a presença de Parkinson com base em imagens de testes motores (spiral/wave). A pipeline envolve:

1. Extração de características visuais com ResNet50
2. Classificação com MLP (Multilayer Perceptron)
3. Avaliação com acurácia, matriz de confusão e relatório de desempenho

---

## Estrutura do Projeto

```
Parkinson-Classifier/
├── data/                    
|   |──teste (healthy/parkinson)
├── src/                     
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_extractor.py
│   ├── model_builder.py
│   ├── trainer.py
│   └── utils.py
├── Parkinson_Classifier.ipynb
├── requirements.txt
└── README.md
```

---

## Requisitos

* Python 3.8+
* TensorFlow
* NumPy
* OpenCV
* scikit-learn
* Matplotlib
* Seaborn

Instale com:

```bash
pip install -r requirements.txt
```
### Dataset

Existem várias fontes que disponibilizam datasets com imagens. Neste caso, foi utilizado o seguinte:

> https://www.kaggle.com/datasets/kmader/parkinsons-drawings

---

## Como Usar

### 1. Estrutura de Dados

Organize suas imagens da seguinte forma:

```
data/
  train/
    healthy/
    parkinson/
  test/
    healthy/
    parkinson/
```

### 2. Execute o notebook

Abra e execute passo a passo o arquivo:

```
Parkinson_Classifier.ipynb
```

Ele executa:

* Extração de características com ResNet50
* Treinamento da MLP
* Avaliação com matriz de confusão e métricas


---

## Resultados Esperados

O sistema alcança boa performance ao distinguir entre pacientes saudáveis e parkinsonianos com base em padrões das imagens.

---

## Futuras Melhorias

* Fine-tuning da ResNet50
* Cross-validation k-fold
* Teste com outras CNNs (EfficientNet, MobileNet)
* Geração de relatório PDF/HTML automatizado
