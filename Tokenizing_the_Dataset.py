{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "45ad4725-8fdb-4b1b-abc8-d21227aaf696",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers in c:\\users\\ghass\\anaconda3\\lib\\site-packages (4.44.2)\n",
      "Requirement already satisfied: datasets in c:\\users\\ghass\\anaconda3\\lib\\site-packages (2.21.0)\n",
      "Requirement already satisfied: torch in c:\\users\\ghass\\anaconda3\\lib\\site-packages (2.4.1)\n",
      "Requirement already satisfied: filelock in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (3.13.1)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (0.24.6)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (1.26.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (23.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (6.0.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (2023.10.3)\n",
      "Requirement already satisfied: requests in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (2.32.2)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (0.4.5)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (0.19.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers) (4.66.4)\n",
      "Requirement already satisfied: pyarrow>=15.0.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (17.0.0)\n",
      "Requirement already satisfied: dill<0.3.9,>=0.3.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (0.3.8)\n",
      "Requirement already satisfied: pandas in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (2.2.2)\n",
      "Requirement already satisfied: xxhash in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (3.5.0)\n",
      "Requirement already satisfied: multiprocess in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (0.70.16)\n",
      "Requirement already satisfied: fsspec<=2024.6.1,>=2023.1.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from fsspec[http]<=2024.6.1,>=2023.1.0->datasets) (2024.3.1)\n",
      "Requirement already satisfied: aiohttp in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from datasets) (3.9.5)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch) (4.11.0)\n",
      "Requirement already satisfied: sympy in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch) (3.2.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch) (3.1.4)\n",
      "Requirement already satisfied: setuptools in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch) (69.5.1)\n",
      "Requirement already satisfied: aiosignal>=1.1.2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.2.0)\n",
      "Requirement already satisfied: attrs>=17.3.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (23.1.0)\n",
      "Requirement already satisfied: frozenlist>=1.1.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.4.0)\n",
      "Requirement already satisfied: multidict<7.0,>=4.5 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (6.0.4)\n",
      "Requirement already satisfied: yarl<2.0,>=1.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from aiohttp->datasets) (1.9.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers) (2024.8.30)\n",
      "Requirement already satisfied: colorama in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from tqdm>=4.27->transformers) (0.4.6)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from jinja2->torch) (2.1.3)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from pandas->datasets) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from pandas->datasets) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from pandas->datasets) (2023.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from sympy->torch) (1.3.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "!pip install transformers datasets torch\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "127b024a-bffb-44d8-80aa-38225dd110ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'text': 'AI is transforming consumer behavior by predicting purchases and personalizing recommendations.'}"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "# Load the dataset\n",
    "dataset = load_dataset('text', data_files={'train': 'data/dataset.txt'})\n",
    "\n",
    "# Preview the dataset\n",
    "dataset['train'][0]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "92233783-bd27-4c2b-9bc0-585f36f02901",
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "bbdc01ab-6f3e-414a-8c11-ce04e6150240",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "dataset = load_dataset('text', data_files={'train': 'Data/dataset.txt'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "be7b1318-7bdd-47a3-b898-92011608a3b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the GPT-2 tokenizer and model\n",
    "tokenizer = GPT2Tokenizer.from_pretrained('gpt2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "23fabbb1-ab0d-4b54-b4d5-c98a601d1e95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Add a new pad token\n",
    "tokenizer.add_special_tokens({'pad_token': '[PAD]'})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "558a2c20-2f48-4a0e-afd9-73cb68f4a699",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the GPT-2 model\n",
    "model = GPT2LMHeadModel.from_pretrained('gpt2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ad676799-3534-4310-9e20-50d896c7fdd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embedding(50258, 768)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Resize the model embeddings to accommodate the new token\n",
    "model.resize_token_embeddings(len(tokenizer))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "d3cbcc93-a9c2-49cc-a88d-653350957d45",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenize the dataset\n",
    "def tokenize_function(examples):\n",
    "    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "8537ebf3-6a30-4818-b4a9-7b555a39938e",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenized_dataset = dataset.map(tokenize_function, batched=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0a771b13-8acc-4b5b-88fe-9ce139f869bb",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
