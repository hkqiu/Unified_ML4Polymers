import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from transformers import AutoTokenizer

def visualize_token_distribution(prompts, tokenizer):
    doc_lengths = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        doc_lengths.append(len(tokens))
    
    doc_lengths = np.array(doc_lengths)
    sns.distplot(doc_lengths)
    plt.title('Token Length Distribution')
    plt.show()