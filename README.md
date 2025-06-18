# 🤖 Projeto de Reconhecimento de Sinais em Libras com CNN

Este projeto utiliza **Deep Learning** (mais especificamente Redes Neurais Convolucionais - CNNs) para reconhecer três sinais de Libras: **OI**, **BOM** e **TCHAU** a partir da câmera do usuário.

## 📁 Estrutura do Projeto


## 📷 1. Coleta de Amostras (`captura.py`)

Captura imagens da sua webcam e salva nas pastas correspondentes às classes. Pressione `s` para salvar uma imagem. Ideal capturar **1000 imagens por classe** com variações de ângulo, distância e iluminação.

## 🧠 2. Treinamento da Rede Neural (`treinamentoCNN.py`)

- Usa `TensorFlow` e `Keras` para criar uma CNN baseada na arquitetura **LeNet**.
- Pré-processamento inclui: escala de cinza, normalização, equalização.
- Utiliza `Data Augmentation` para melhorar a generalização.
- Treinamento com 10 épocas e visualização das curvas de perda e acurácia.
- Salva o modelo final como `modelo.keras`.

## 🧪 3. Teste em Tempo Real (`testeModelo.py`)

- Carrega o modelo treinado.
- Captura imagens da webcam e retorna a **classe prevista** (OI, BOM ou TCHAU) com a **probabilidade** da predição.

## ⚙️ Requisitos

Instale os pacotes necessários com:

```bash
pip install opencv-python tensorflow==2.19.0 scikit-learn matplotlib
````

## 🚀 Como Usar

Clone ou baixe este repositório.

Execute captura.py para coletar dados.

Execute treinamentoCNN.py para treinar o modelo.

Execute testeModelo.py para ver a mágica acontecer! 🪄

## 💬 Créditos

Projeto feito por Blanca adaptado com base em aulas práticas do professor Márcio sobre
Redes Neurais Convolucionais aplicadas à classificação de imagens.
Sinais em Libras escolhidos: OI, BOM e TCHAU.

## 📌 Observações

- Recomenda-se uso de fundo uniforme e iluminação constante.

- A acurácia do modelo pode ser melhorada com mais amostras de diferentes pessoas.

- 🗂️ Este repositório representa o conteúdo da pasta `DeepLearning` do projeto original.
