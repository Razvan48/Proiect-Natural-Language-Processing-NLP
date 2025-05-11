{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install tensorflow\n",
        "!pip install keras"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZbV-NTATrVLS",
        "outputId": "acda6781-fa78-49fd-8f88-37ce46394f13"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: tensorflow in /usr/local/lib/python3.11/dist-packages (2.18.0)\n",
            "Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.4.0)\n",
            "Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.6.3)\n",
            "Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (25.2.10)\n",
            "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.6.0)\n",
            "Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.2.0)\n",
            "Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (18.1.1)\n",
            "Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.4.0)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from tensorflow) (24.2)\n",
            "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (5.29.4)\n",
            "Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.32.3)\n",
            "Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from tensorflow) (75.2.0)\n",
            "Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.0)\n",
            "Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.1.0)\n",
            "Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (4.13.2)\n",
            "Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.17.2)\n",
            "Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (1.71.0)\n",
            "Requirement already satisfied: tensorboard<2.19,>=2.18 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.18.0)\n",
            "Requirement already satisfied: keras>=3.5.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.8.0)\n",
            "Requirement already satisfied: numpy<2.1.0,>=1.26.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (2.0.2)\n",
            "Requirement already satisfied: h5py>=3.11.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (3.13.0)\n",
            "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.4.1)\n",
            "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.11/dist-packages (from tensorflow) (0.37.1)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse>=1.6.0->tensorflow) (0.45.1)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.0.9)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras>=3.5.0->tensorflow) (0.15.0)\n",
            "Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.4.1)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.10)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.4.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3,>=2.21.0->tensorflow) (2025.4.26)\n",
            "Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.8)\n",
            "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (0.7.2)\n",
            "Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from tensorboard<2.19,>=2.18->tensorflow) (3.1.3)\n",
            "Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.11/dist-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow) (3.0.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras>=3.5.0->tensorflow) (2.19.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow) (0.1.2)\n",
            "Requirement already satisfied: keras in /usr/local/lib/python3.11/dist-packages (3.8.0)\n",
            "Requirement already satisfied: absl-py in /usr/local/lib/python3.11/dist-packages (from keras) (1.4.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from keras) (2.0.2)\n",
            "Requirement already satisfied: rich in /usr/local/lib/python3.11/dist-packages (from keras) (13.9.4)\n",
            "Requirement already satisfied: namex in /usr/local/lib/python3.11/dist-packages (from keras) (0.0.9)\n",
            "Requirement already satisfied: h5py in /usr/local/lib/python3.11/dist-packages (from keras) (3.13.0)\n",
            "Requirement already satisfied: optree in /usr/local/lib/python3.11/dist-packages (from keras) (0.15.0)\n",
            "Requirement already satisfied: ml-dtypes in /usr/local/lib/python3.11/dist-packages (from keras) (0.4.1)\n",
            "Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from keras) (24.2)\n",
            "Requirement already satisfied: typing-extensions>=4.5.0 in /usr/local/lib/python3.11/dist-packages (from optree->keras) (4.13.2)\n",
            "Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras) (3.0.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.11/dist-packages (from rich->keras) (2.19.1)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.11/dist-packages (from markdown-it-py>=2.2.0->rich->keras) (0.1.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 45,
      "metadata": {
        "id": "cSt0erWpqoWz"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras import layers, Model, Input\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "import pandas as pd\n",
        "import time\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "\n",
        "def Positional_Encoding(dim_subsecventa, dim_embedding):\n",
        "    factori_scalare = np.array([1 / (10000 ** (2 * (pozitie_embedding // 2) / dim_embedding)) for pozitie_embedding in range(dim_embedding)])  # (1, dim_embedding)\n",
        "    pozitii_initiale = np.array([[p] for p in range(dim_subsecventa)])  # (dim_subsecventa, 1)\n",
        "\n",
        "    valori = pozitii_initiale * factori_scalare  # (dim_subsecventa, dim_embedding)\n",
        "    # esantioanele au initial pozitiile 0, 1, 2, etc,\n",
        "    # pozitiile vor deveni arrays de dimensiune dim_embedding,\n",
        "    # fiecare element din embedding fiind pozitia initiala a esantionului * factor de scalare\n",
        "    rezultat = np.zeros((dim_subsecventa, dim_embedding))\n",
        "    rezultat[:, 0::2] = np.sin(valori[:, 0::2])\n",
        "    rezultat[:, 1::2] = np.cos(valori[:, 1::2])\n",
        "    return rezultat\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def Self_Attention(layer_precedent, dim_embedding):\n",
        "    # Q = ce informatie cauta un esantion de la altele, K = ce informatie detine fiecare, V = informatie deitnuta in detaliu\n",
        "    Q = layers.Dense(dim_embedding)(layer_precedent)\n",
        "    #print(Q.shape)\n",
        "    # transofrmam datele din stratul precedent pentru dea aprofunda informatia deja existenta\n",
        "    K = layers.Dense(dim_embedding)(layer_precedent)\n",
        "    V = layers.Dense(dim_embedding)(layer_precedent)\n",
        "\n",
        "    scoruri_atentie = layers.Lambda(lambda x: tf.matmul(x[0], x[1], transpose_b=True))([Q, K])\n",
        "    # scoruri_atentie e de dimensiune (lungime_secventa, lungime_secventa)\n",
        "    # deci fiecare esantion din secventa are un scor de atentie fata de restul\n",
        "    # prin Q * K.T fiecare esantion vede daca are ce obtine de la restul\n",
        "    ponderi_atentie = layers.Softmax(axis=-1)(scoruri_atentie)\n",
        "    # fiecare esantion primeste de la fiecare ce a cautat\n",
        "    rezultat = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([ponderi_atentie, V])\n",
        "    return rezultat"
      ],
      "metadata": {
        "id": "U2_FsxWDqy8i"
      },
      "execution_count": 46,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def Encoder(layer_precedent, dim_embedding, dim_feed_forward):\n",
        "    self_attention = Self_Attention(layer_precedent, dim_embedding)\n",
        "    self_attention += layer_precedent\n",
        "    self_attention = layers.LayerNormalization()(self_attention)\n",
        "\n",
        "    feed_forward = layers.Dense(dim_feed_forward, activation='relu')(self_attention)\n",
        "    # primul strat din ff mareste dimensiunea pentru a aprofunda informatia din self_attention, iar al doilea aduce dimensiunea la loc pentru a se potrivi cu dim_encoder\n",
        "    feed_forward = layers.Dense(dim_embedding)(feed_forward)\n",
        "\n",
        "    encoder = feed_forward + self_attention\n",
        "    encoder = layers.LayerNormalization()(encoder)\n",
        "    return encoder"
      ],
      "metadata": {
        "id": "lMh9Q5Lpq4iX"
      },
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def Transformer(dim_subsecventa, dim_embedding, dim_vocab, nr_encoders, dim_feed_forward):\n",
        "    tensor_intrare = Input(shape=(dim_subsecventa,))\n",
        "    layer_embedding = layers.Embedding(dim_vocab, dim_embedding)(tensor_intrare)\n",
        "    pos_encoding = Positional_Encoding(dim_subsecventa, dim_embedding)\n",
        "\n",
        "    tensor_pos_encoding = tf.convert_to_tensor(pos_encoding, dtype=tf.float32)\n",
        "    # adaugam inca o dimensiune la tensor pt a sti din ce batch face parte\n",
        "    tensor_pos_encoding = tf.expand_dims(tensor_pos_encoding, axis=0)\n",
        "    layer_pos_encoding = layer_embedding + tensor_pos_encoding\n",
        "\n",
        "    layers_encoder = layer_pos_encoding\n",
        "    for _ in range(nr_encoders):\n",
        "        layers_encoder = Encoder(layers_encoder, dim_embedding, dim_feed_forward)\n",
        "    # layers encoder are acum dim (batch, dim_subsecventa, dim_embedding)\n",
        "\n",
        "    # obtinem informatie despre fiecare subsecventa\n",
        "    layer_medie_pe_subsecvente = layers.GlobalAveragePooling1D()(layers_encoder)\n",
        "    # un strat Dense care produce prob\n",
        "    # activare softmax pt probabilitati\n",
        "    layer_final = layers.Dense(dim_vocab, activation='softmax')(layer_medie_pe_subsecvente)\n",
        "\n",
        "    return Model(tensor_intrare, layer_final)"
      ],
      "metadata": {
        "id": "Jr3MdYxTrA5o"
      },
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "csv_file = pd.read_csv('/content/sample_data/shortjokes.csv')\n",
        "\n",
        "NUM_JOKES = 50000\n",
        "\n",
        "jokes = csv_file['Body'].head(NUM_JOKES).astype(str).to_numpy()\n",
        "train_jokes, val_jokes = train_test_split(jokes, test_size=0.2, random_state=42)\n",
        "print(jokes)\n",
        "\n",
        "dim_subsecventa = 30\n",
        "nr_encoders = 2\n",
        "dim_embedding = 64\n",
        "dim_feed_forward = 256"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "h9Ai98kErlGw",
        "outputId": "ec6abe43-1ae2-44a4-cc5e-f6da5cdcf4d8"
      },
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['[me narrating a documentary about narrators] \"I can\\'t hear what they\\'re saying cuz I\\'m talking\"'\n",
            " 'Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men.'\n",
            " \"I've been going through a really rough period at work this week It's my own fault for swapping my tampax for sand paper.\"\n",
            " ...\n",
            " 'Why is faith greater than science? Science made buildings and planes but faith brought them together.'\n",
            " 'There is a new Barbie doll on the market -  Junkie Barbie ...complete with needle tracks'\n",
            " 'My Friend Told Me His Girlfriend Talks a lot in Her Sleep.. ..Apparently \"I Know\" wasn\\'t the right answer.']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "tokenizer = Tokenizer(oov_token=\"<OOV>\")\n",
        "tokenizer.fit_on_texts(jokes)\n",
        "\n",
        "secvente = tokenizer.texts_to_sequences(jokes)\n",
        "secvente = pad_sequences(secvente, maxlen=dim_subsecventa, padding='post', truncating='post')\n"
      ],
      "metadata": {
        "id": "4RqUEd5StnQx"
      },
      "execution_count": 50,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "X = []\n",
        "y = []\n",
        "\n",
        "for sec in secvente:\n",
        "    for i in range(1, len(sec)):\n",
        "        X.append(sec[:i])\n",
        "        y.append(sec[i])\n",
        "\n",
        "X = pad_sequences(X, maxlen=dim_subsecventa, padding='pre')\n",
        "y = np.array(y)"
      ],
      "metadata": {
        "id": "rxAFLAWhv_SA"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dim_vocab = tokenizer.num_words or len(tokenizer.word_index) + 1\n",
        "model = Transformer(dim_subsecventa, dim_embedding, dim_vocab, nr_encoders, dim_feed_forward)\n",
        "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
        "model.fit(X, y, batch_size=64, epochs=10, validation_split=0.1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dPX0iA5Zxs-e",
        "outputId": "ad53add1-1804-4be4-e319-86bdcbce5825"
      },
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m138s\u001b[0m 6ms/step - accuracy: 0.4481 - loss: 4.2749 - val_accuracy: 0.4758 - val_loss: 3.8243\n",
            "Epoch 2/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m129s\u001b[0m 6ms/step - accuracy: 0.4775 - loss: 3.7506 - val_accuracy: 0.4938 - val_loss: 3.6337\n",
            "Epoch 3/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m141s\u001b[0m 6ms/step - accuracy: 0.4961 - loss: 3.5474 - val_accuracy: 0.4969 - val_loss: 3.5962\n",
            "Epoch 4/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m117s\u001b[0m 6ms/step - accuracy: 0.5045 - loss: 3.4379 - val_accuracy: 0.5068 - val_loss: 3.5027\n",
            "Epoch 5/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m116s\u001b[0m 6ms/step - accuracy: 0.5065 - loss: 3.3871 - val_accuracy: 0.5093 - val_loss: 3.4894\n",
            "Epoch 6/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 6ms/step - accuracy: 0.5104 - loss: 3.3261 - val_accuracy: 0.5112 - val_loss: 3.4919\n",
            "Epoch 7/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 6ms/step - accuracy: 0.5145 - loss: 3.2701 - val_accuracy: 0.5116 - val_loss: 3.5032\n",
            "Epoch 8/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m121s\u001b[0m 6ms/step - accuracy: 0.5178 - loss: 3.2236 - val_accuracy: 0.5102 - val_loss: 3.5445\n",
            "Epoch 9/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m142s\u001b[0m 6ms/step - accuracy: 0.5167 - loss: 3.2283 - val_accuracy: 0.5149 - val_loss: 3.5167\n",
            "Epoch 10/10\n",
            "\u001b[1m20391/20391\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m138s\u001b[0m 6ms/step - accuracy: 0.5193 - loss: 3.1891 - val_accuracy: 0.5151 - val_loss: 3.5328\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.src.callbacks.history.History at 0x7892109479d0>"
            ]
          },
          "metadata": {},
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def genereaza_gluma(model, tokenizer, dim_subsecventa, prompt, dim_maxima_gluma, cuvinte_enervante=None):\n",
        "    if cuvinte_enervante is None:\n",
        "        cuvinte_enervante = set()\n",
        "\n",
        "    secventa = tokenizer.texts_to_sequences([prompt])\n",
        "    secventa = pad_sequences(secventa, maxlen=dim_subsecventa, padding='pre')\n",
        "\n",
        "    generated_text = prompt\n",
        "    cuvinte_generate = set(prompt.split())\n",
        "\n",
        "    for _ in range(dim_maxima_gluma):\n",
        "        predictie = model.predict(secventa, verbose=0)\n",
        "        sorted_indices = np.argsort(-predictie[0])\n",
        "\n",
        "        next_token = ''\n",
        "        for idx in sorted_indices:\n",
        "            candidate = tokenizer.index_word.get(idx, '')\n",
        "            if candidate not in cuvinte_enervante and candidate not in cuvinte_generate and candidate != '':\n",
        "                next_token = candidate\n",
        "                next_token_index = idx\n",
        "                break\n",
        "\n",
        "        if next_token == '':\n",
        "            break\n",
        "\n",
        "        generated_text += ' ' + next_token\n",
        "        cuvinte_generate.add(next_token)\n",
        "        secventa = tf.concat([secventa[:, 1:], tf.constant([[next_token_index]])], axis=1)\n",
        "\n",
        "    return generated_text\n"
      ],
      "metadata": {
        "id": "YIbX6vHC1LYw"
      },
      "execution_count": 105,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "cuvinte_de_evitat = {'joke', 'idea', 'of'}\n",
        "inceput = 'A'\n",
        "gluma_generata = genereaza_gluma(model, tokenizer, dim_subsecventa, inceput, 10, cuvinte_de_evitat)\n",
        "print(gluma_generata)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B_x-_Wjo1pSa",
        "outputId": "be62b59f-509b-4b02-92ac-d29f8e5ca0c4"
      },
      "execution_count": 113,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "A man walks into a bar and asks his wife says\n"
          ]
        }
      ]
    }
  ]
}