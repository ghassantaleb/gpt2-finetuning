# gpt2-finetuning

## Large Files pushed

Due to GitHub's file size limitations, files are hosted externally:

- [Fine-tuned GPT-2 Model (Google Drive)](https://drive.google.com/drive/folders/1Snf4L11zeRFASTq6i1BApyRtmdpycQZH?usp=drive_link)

# Fine-Tuning GPT-2 for Sentiment Classification of Customer Reviews

## Project Overview
This project aims to fine-tune a pre-trained GPT-2 model to perform sentiment classification on customer reviews. We started with a custom dataset containing customer reviews labeled as either positive (1) or negative (0). The primary objective was to evaluate the model's ability to predict the sentiment of unseen reviews based on the training it received from the fine-tuning process.

## Objectives
1- Fine-tune the GPT-2 model to classify customer reviews as positive or negative.
2- Generate predictions for new, unseen customer reviews.
3- Evaluate the model using standard metrics such as accuracy, precision, recall, and F1 score.

## Project Structure 
Data:
The dataset used for this project consists of a collection of customer reviews, each accompanied by a sentiment label (1 for positive and 0 for negative). Here's a sample of the data:

Review	Label
"This product is fantastic! I love it and would recommend it to everyone."	1
"Terrible experience. It broke after one use. I will never buy this again."	0
Model:
The model used for this task is GPT-2, a pre-trained language model from OpenAI. We fine-tuned GPT-2 on the custom dataset to adapt its capabilities for the specific task of sentiment classification.

Training Process:
We fine-tuned the model over several epochs using the Hugging Face Transformers library, optimizing the model for the classification task. The model was trained with a small learning rate and evaluated using a validation set to prevent overfitting.

Evaluation:
After training, we evaluated the model's performance on a test set of customer reviews, using metrics such as accuracy, precision, recall, and F1 score. These metrics provide insight into how well the model classifies the reviews.

## Steps to Run the Project
Data Preprocessing:

The dataset was tokenized using GPT-2's tokenizer with padding and truncation to ensure uniform input lengths.
Model Training:

The model was trained using the customer reviews dataset with an 80-20 train-test split. The loss was minimized over multiple epochs, and the training loss was monitored.
Evaluation:

The fine-tuned model was evaluated on unseen data using accuracy and F1 score as the primary metrics. Predictions were added to the dataset for analysis.

## Installation and Requirements
Python 3.7+
Transformers library (Hugging Face)
PyTorch
Pandas
Scikit-learn

## Results
After fine-tuning and evaluating the model, we achieved the following metrics on the validation set:

Accuracy: 85%
F1 Score: 0.83
Precision: 0.85
Recall: 0.82

## Challenges Encountered
Logits Interpretation:
During evaluation, we discovered that the logits produced by GPT-2 required careful handling for binary classification. Initially, the model produced outputs that were not immediately interpretable as sentiment labels, which required post-processing of logits using a sigmoid function to classify the sentiment.

Imbalanced Predictions:
In some cases, the model tended to predict the same label (class imbalance). This was addressed by adjusting the logits and implementing token truncation and padding during preprocessing.

## Future Improvements
Model Enhancement:
I plan to further optimize the model by:

Experimenting with different learning rates and batch sizes.
Incorporating additional data to help the model generalize better.
Trying different pre-processing techniques, such as balancing the dataset or using data augmentation to create more variety.
Evaluation Improvement:
I will explore alternative evaluation strategies, including:

Hyperparameter tuning to improve classification performance.
Implementing alternative language models (such as BERT or RoBERTa) for comparison.

# Conclusion
This project demonstrates how a GPT-2 model can be adapted for sentiment analysis tasks, such as classifying customer reviews. While our initial results show that further refinement is necessary, this serves as a foundation for exploring how generative models like GPT-2 can be utilized in classification problems.

Feel free to explore, experiment, and contribute to improving the model's performance!
