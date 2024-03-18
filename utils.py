
import sys
import os, re

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
import openai
from simhash import Simhash 
import requests
import random
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

llm_aes_dir = 'LLM_AES'

# Add LLM_AES to the sys.path 
if llm_aes_dir not in sys.path:
    sys.path.insert(0, llm_aes_dir)
from LLM_AES.gpt_score_asap1 import zeroshot_worubrics_prompt

# Initialize models with a valid API key
openai.api_key = "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"
FLAN_T5_MODEL_NAME = "google/flan-t5-base"

flan_t5_tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
flan_t5_model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME)

# Step 1: Rephrase the ChatGPT-generated essay using GPT-4
def rephrase_essay(student_essay: str, chatgpt_essay: str) -> str:
    prompt = (f"{student_essay}\n\n"
              "Follow the above text to rewrite and optimize the following text. "
              f"Try to be different from the original text:\n{chatgpt_essay}\n")

    print("Calling GPT-4 API for rephrasing...")
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are an NLP expert"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1024
    )

    return response.choices[0].message.content.strip()

# Step 2: Compute SimHash similarity
def compute_similarity(student_essay, rephrased_essay):
    # Compute SimHash values
    hash1 = Simhash(student_essay)
    hash2 = Simhash(rephrased_essay)
    # Return the Hamming distance or a similar metric
    return hash1.distance(hash2)

# Step 3: Sentence Substitution
def mask_sentences(essay: str, num_sentences_to_mask: int) -> (str, list):
    sentences = nltk.sent_tokenize(essay)
    sentences_to_mask = random.sample(sentences, min(num_sentences_to_mask, len(sentences)))
    masked_essay = essay
    for sentence in sentences_to_mask:
        masked_essay = masked_essay.replace(sentence, "<mask>", 1)
    return masked_essay, sentences_to_mask

def generate_replacements_for_masked_sentences(masked_essay: str, sentences_to_replace: list) -> str:
    for sentence in sentences_to_replace:
        input_text = f"paraphrase this sentence: {sentence}"
        input_ids = flan_t5_tokenizer.encode(input_text, return_tensors="pt")
        outputs = flan_t5_model.generate(input_ids, max_length=512, num_return_sequences=1)
        replacement_text = flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        masked_essay = masked_essay.replace("<mask>", replacement_text, 1)
    return masked_essay

#Step 4: Word Substitution

def select_high_frequency_words(text: str, k: int) -> list:
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum()]
    word_freq = Counter(filtered_words)
    stop_words = set(stopwords.words('english'))
    non_stop_words = [word for word in word_freq if word.lower() not in stop_words]
    sorted_words = sorted(non_stop_words, key=lambda word: word_freq[word], reverse=True)
    return sorted_words[:k]

def substitute_word(essay: str, k: int) -> str:
    high_freq_words = select_high_frequency_words(essay, k)
    for word in high_freq_words:
        synonyms = {lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}
        if synonyms:
            replacement = random.choice(list(synonyms))
            essay = essay.replace(word, replacement, 1)
    return essay

# Placeholder function to read essay from a file
def read_essay_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Step 5: Quality assessment after perturbations with  AES
def assess_quality(student_essay, gpt_essay):
    """
    Use an AES system to evaluate the quality of the essay.
    :param essay: The essay text to be evaluated.
    :param aes_system: An AES model or system that can score the essay.
    :return: The quality score of the essay.
    """
    
    # use a loaded AES model. Here is a pseudocode representation:
    aes_model = zeroshot_worubrics_prompt(student_essay, gpt_essay)

    # Use regular expression to find the score
    match = re.search(r"Score: (\d+)", aes_model)
    if match:
        score = int(match.group(1))
        print("Score:", score)
    else:
        print("Score not found.")

    quality_score = score
    return quality_score


# Step 6: Evaluate evasion of detection from GPTZero
def evaluate_detection_evasion(perturbed_essay):
    """
    Evaluate the ability of the perturbed essay to evade detection by GPTZero.
    :param perturbed_essay: The perturbed essay text.
    :return: A dictionary with ACC and AUROC metrics.
    """
    # Placeholder: Replace with actual API endpoint and key
    GPTZERO_API_ENDPOINT = "https://api.gptzero.me/detect"
    HEADERS = {"Authorization": "Bearer your_api_key_here"}

    # This would be an API call to GPTZero with the perturbed essay
    response = requests.post(
        GPTZERO_API_ENDPOINT,
        headers=HEADERS,
        json={"text": perturbed_essay}
    )
    
    # Check if the response is successful
    if response.status_code == 200:
        detection_result = response.json()
        # Placeholder: extract the predicted label and probability
        predicted_label = detection_result["is_ai"]  # True if AI, False if human
        predicted_prob = detection_result["probability_ai"]  # Probability text is AI-generated
        
        # Placeholder: you need the true label for the essay to calculate ACC and AUROC
        true_label = get_true_label_for_essay(perturbed_essay)
        
        # Calculate ACC and AUROC
        # Assuming you have a collection of true labels and predicted probabilities for a set of essays
        acc = accuracy_score([true_label], [predicted_label])
        auroc = roc_auc_score([true_label], [predicted_prob])
        f1 = f1_score([true_label], predicted_label)
        
        return {'ACC': acc, 'AUROC': auroc, 'F1': f1}
    else:
        # Handle error in response
        raise Exception(f"Failed to get response from GPTZero API: {response.status_code}")

