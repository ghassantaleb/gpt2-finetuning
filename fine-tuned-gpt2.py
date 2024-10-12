{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
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
   "execution_count": 2,
   "id": "127b024a-bffb-44d8-80aa-38225dd110ab",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'text': 'AI is transforming consumer behavior by predicting purchases and personalizing recommendations.'}"
      ]
     },
     "execution_count": 2,
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
   "execution_count": 3,
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
   "execution_count": 54,
   "id": "bbdc01ab-6f3e-414a-8c11-ce04e6150240",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Load the dataset\n",
    "dataset = load_dataset('text', data_files={'train': 'data/dataset.txt'})['train']\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
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
   "execution_count": 58,
   "id": "23fabbb1-ab0d-4b54-b4d5-c98a601d1e95",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 58,
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
   "execution_count": 60,
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
   "execution_count": 62,
   "id": "ad676799-3534-4310-9e20-50d896c7fdd5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Embedding(50258, 768)"
      ]
     },
     "execution_count": 62,
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
   "execution_count": 66,
   "id": "d3cbcc93-a9c2-49cc-a88d-653350957d45",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenize the dataset\n",
    "def tokenize_function(examples):\n",
    "    # Tokenize the input text\n",
    "    inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)\n",
    "    \n",
    "    # For causal language modeling, labels are the same as input_ids\n",
    "    inputs[\"labels\"] = inputs[\"input_ids\"].copy()\n",
    "    \n",
    "    return inputs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "8537ebf3-6a30-4818-b4a9-7b555a39938e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "799f00cb48304200a03e6585f1ce2492",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Map:   0%|          | 0/40 [00:00<?, ? examples/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Apply the tokenization function to the dataset\n",
    "tokenized_dataset = dataset.map(tokenize_function, batched=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "60dbbdef-fdae-4bf4-828f-e33eaa22678f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: transformers[torch] in c:\\users\\ghass\\anaconda3\\lib\\site-packages (4.44.2)\n",
      "Requirement already satisfied: filelock in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (3.13.1)\n",
      "Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (0.24.6)\n",
      "Requirement already satisfied: numpy>=1.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (1.26.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (23.2)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (6.0.1)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (2023.10.3)\n",
      "Requirement already satisfied: requests in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (2.32.2)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (0.4.5)\n",
      "Requirement already satisfied: tokenizers<0.20,>=0.19 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (0.19.1)\n",
      "Requirement already satisfied: tqdm>=4.27 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (4.66.4)\n",
      "Requirement already satisfied: accelerate>=0.21.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (1.0.1)\n",
      "Requirement already satisfied: torch in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from transformers[torch]) (2.4.1)\n",
      "Requirement already satisfied: psutil in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from accelerate>=0.21.0->transformers[torch]) (5.9.0)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0,>=0.23.2->transformers[torch]) (2024.3.1)\n",
      "Requirement already satisfied: typing-extensions>=3.7.4.3 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from huggingface-hub<1.0,>=0.23.2->transformers[torch]) (4.11.0)\n",
      "Requirement already satisfied: sympy in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch->transformers[torch]) (1.12)\n",
      "Requirement already satisfied: networkx in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch->transformers[torch]) (3.2.1)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch->transformers[torch]) (3.1.4)\n",
      "Requirement already satisfied: setuptools in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from torch->transformers[torch]) (69.5.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from tqdm>=4.27->transformers[torch]) (0.4.6)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers[torch]) (2.0.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers[torch]) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers[torch]) (2.2.2)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from requests->transformers[torch]) (2024.8.30)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from jinja2->torch->transformers[torch]) (2.1.3)\n",
      "Requirement already satisfied: mpmath>=0.19 in c:\\users\\ghass\\anaconda3\\lib\\site-packages (from sympy->torch->transformers[torch]) (1.3.0)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install transformers[torch]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "4d93d387-0080-41de-bd52-063d80f1d3d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up training arguments\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=\"./gpt2-finetuned\",\n",
    "    overwrite_output_dir=True,\n",
    "    num_train_epochs=3,\n",
    "    per_device_train_batch_size=4,\n",
    "    save_steps=10_000,\n",
    "    save_total_limit=2,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "d0a723e9-2c1c-4eae-bb92-3a1832ec24b3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set up the Trainer\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=tokenized_dataset,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "379839a4-53ef-453c-955c-f1618b27a957",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "    <div>\n",
       "      \n",
       "      <progress value='30' max='30' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
       "      [30/30 26:06, Epoch 3/3]\n",
       "    </div>\n",
       "    <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       " <tr style=\"text-align: left;\">\n",
       "      <th>Step</th>\n",
       "      <th>Training Loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table><p>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "TrainOutput(global_step=30, training_loss=0.9033639272054036, metrics={'train_runtime': 1708.3869, 'train_samples_per_second': 0.07, 'train_steps_per_second': 0.018, 'total_flos': 31355043840000.0, 'train_loss': 0.9033639272054036, 'epoch': 3.0})"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "trainer.train()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "65c98f51-6b38-49dc-9367-b18adbd8c7cc",
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
