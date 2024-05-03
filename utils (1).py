
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

# Initialize models with a valid API key
openai.api_key = "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"
FLAN_T5_MODEL_NAME = "google/flan-t5-base"

flan_t5_tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL_NAME)
flan_t5_model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL_NAME)

# Step 1: Rephrase the ChatGPT-generated essay using GPT-4
def rephrase_essay(student_essay: str, chatgpt_essay: str) -> str:
    prompt = (f"{student_essay}\n\n"
              "Follow the above text to rewrite and improve the following text below. "
              f"Generate your response to evade detection gptzero. Text:\n{chatgpt_essay}\n")

    print("Calling GPT-4 API for rephrasing...")
    response = openai.ChatCompletion.create(
        model="gpt-4",  
        messages=[
            {"role": "system", "content": "You are a text generation expert"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=2024
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

# read essay from a files in the directory
def read_essay_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def automatic_essay_evaluator(essay):
    """
       custom gpt-4 powered automatic essay evaluator 
    """
    prompt = f"""
    As a virtual evaluator with expertise in English composition, your role is to critically 
    analyze and grade student essays. 

    Scores should be assigned on a scale of 1 to 10, where 1 represents poor quality, and 10 
    represents best possible excellent quality. 

    Here's the student essay to evaluate
     {essay}

    Please present your evaluation in the following list format:
    [Score: ..., Explanations: ...]
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0.1,
        messages=[
                {"role": "system", "content": "As an essay evaluator with expertise in English composition"},
                {"role": "user", "content": prompt}
            ]
    )

    response_content = response["choices"][0]["message"]["content"]
    return response_content

# Step 5: Quality assessment after perturbations with  AES
def assess_quality(essay):
    """
    Use an AES system to evaluate the quality of the essay.
    :param essay: The essay text to be evaluated.
    :return: The quality score of the essay.
    """
    # use a  custom AES model.
    aes_model = automatic_essay_evaluator(essay)

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
def evaluate_detection_evasion(perturbed_essay, human_essay):
    """
    Evaluate the ability of the perturbed essay to evade detection by GPTZero,
    and compare with the detection of a corresponding human-written essay.
    :param perturbed_essay: The AI-generated and perturbed essay text.
    :param human_essay: The corresponding human-written essay text.
    :return: A dictionary with ACC, AUROC, and F1 metrics for both essays.
    """
    GPTZERO_API_ENDPOINT = "https://api.gptzero.me/v2/predict/text"
    HEADERS = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "x-api-key": "a7f44900b102489d90296ee81ee69bee"
    }
    
    # Initialize lists to store results
    true_labels = [1, 0]  # AI-generated as 1, Human as 0
    predicted_labels = []
    predicted_probs = []

    # Evaluate both essays
    for essay in [perturbed_essay, human_essay]:
        PAYLOAD = {
            "document": essay,
            "version": "2024-01-09"
        }

        response = requests.post(GPTZERO_API_ENDPOINT, headers=HEADERS, json=PAYLOAD)
        
        if response.status_code == 200:
            detection_result = response.json()["documents"][0]
            predicted_class = detection_result["predicted_class"]
            
            # Map predicted_class to a binary value: 1 for 'ai', 0.5 if mixed and 0 otherwise.
            predicted_label = 1 if predicted_class == 'ai' else 0.5 if predicted_class == 'mixed' else 0

            predicted_labels.append(predicted_label)
            
            # Assuming 'ai' class probability is used for AUROC calculation.
            predicted_prob = detection_result["class_probabilities"]["ai"]
            predicted_probs.append(predicted_prob)
            
        else:
            raise Exception(f"Failed to get response from GPTZero API: {response.status_code}")

    # Calculate evaluation metrics
    acc = accuracy_score(true_labels, predicted_labels)
    auroc = roc_auc_score(true_labels, predicted_probs)
    f1 = f1_score(true_labels, predicted_labels)
    print(f"True: {true_labels}")
    print(f"Predicted: {predicted_labels}")
    print(f"Predicted probabilities: {predicted_probs}")
    
    return {'ACC': acc, 'AUROC': auroc, 'F1': f1}