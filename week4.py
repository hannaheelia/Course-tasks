import os
import glob
import re
import string
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations
#from sentence_transformers import SentenceTransformer
#from sentence_transformers.util import cos_sim

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def ngram_overlap(text1, text2, n=5):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(n, n))
    ngrams = vectorizer.fit_transform([text1, text2])
    overlap = (ngrams[0].multiply(ngrams[1])).sum()
    return overlap

# Path to unzipped folder
data_dir = 'EssaysToTextAnalysis'

# Read all .txt files
essays = {}
for filepath in glob.glob(os.path.join(data_dir, '*.txt')):
    with open(filepath, 'r', encoding='utf-8') as file:
        essays[os.path.basename(filepath)] = file.read().strip()

essays_clean = {k: clean_text(v) for k, v in essays.items()}

# Reverse dict: cleaned text → list of filenames
text_to_files = defaultdict(list)
for filename, content in essays_clean.items():
    text_to_files[content].append(filename)

# Exact duplicates = any entry with more than one word
exact_duplicates = {text: files for text, files in text_to_files.items() if len(files) > 1}

# Empty/near-empty
near_empty = [f for f, text in essays_clean.items() if len(text.split()) < 5]

# Convert all texts to vectors
filenames = list(essays_clean.keys())
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(essays_clean.values())

# Compute average vector (centroid)
centroid = np.asarray(X.mean(axis=0)).ravel()
similarities = cosine_similarity(X, centroid.reshape(1,-1))

# Detect irrelevant: essays with low similarity to the mean (threshold adjustable)
irrelevant_threshold = 0.2
irrelevant = [list(essays_clean.keys())[i] for i, sim in enumerate(similarities) if sim < irrelevant_threshold]

# Pairwise similarities
similarity_matrix = cosine_similarity(X)

# Detect exact duplicates (textually identical)
text_to_files = defaultdict(list)
for filename, content in essays_clean.items():
    text_to_files[content].append(filename)

exact_duplicate_sets = [files for files in text_to_files.values() if len(files) > 1]

# Flatten into exact duplicate pairs
exact_duplicate_pairs = set()
for group in exact_duplicate_sets:
    for pair in combinations(sorted(group), 2):
        exact_duplicate_pairs.add(pair)

near_duplicates = []
plagiarized_pairs = []
checked = set()
# threshold = 0.9  # Adjust for "near"

for i in range(len(filenames)):
    for j in range(i + 1, len(filenames)):
        sim = similarity_matrix[i][j]
        pair = (filenames[i], filenames[j])
        if pair in exact_duplicate_pairs:
            continue  # skip exact duplicates
        overlap = ngram_overlap(essays_clean[filenames[i]], essays_clean[filenames[j]])
        if sim >= 0.9:
            near_duplicates.append((filenames[i], filenames[j], sim))
        elif sim >= 0.7 or overlap >= 70:
            plagiarized_pairs.append((filenames[i], filenames[j], sim, overlap))

print("Exact Duplicates:")
for files in exact_duplicates.values():
    print(files)

print("\nNear-Empty Essays:")
print(near_empty)

print("\nIrrelevant Essays:")
print(irrelevant)

print("\nNear Duplicates (cosine similarity > 0.9):")
for f1, f2, sim in near_duplicates:
    print(f"{f1} and {f2}  | sim = {sim:.3f}")


print("\nPlagiarized Pairs (semantic sim >= 0.7 or n-gram overlap >= 70):")
for f1, f2, sim, overlap in plagiarized_pairs:
    print(f"{f1} and {f2} | sim = {sim:.3f} | n-gram overlap = {overlap}")