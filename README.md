# BERT Fine tuning


Pre-trained language representations have been shown to improve many downstream NLP tasks such as question answering, and natural language inference. To apply pre-trained representations to these tasks, there are two main strategies:  

1 - The feature-based approach, which uses the pre-trained representations as additional features to the downstream task.  

2 - Or the fine-tuning-based approach, which trains the downstream tasks by fine-tuning pre-trained parameters.  

In this repo, I work through fine-tuning different BERT pretrained models on a downstream task - multi class text classification.  

These are the steps I am going to follow  

1 - Load the state-of-the-art pre-trained BERT models  
2 - Load the native multi class text classification dataset  
3 - Fine-tune the loaded pre-trained BERT model on the multi class text classification dataset  

## Pre-trained models using

Following are the huggingface pretrained language models I am using for this purpose.  

1 - bert-base  
2 - bert-large  
3 - albert  
4 - roberta  
5 - distilbert  
6 - xlnet  




