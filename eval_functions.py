import math
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
#from train import last_train_loss, last_val_loss

#perplexity
def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return math.exp(loss)

# Compute perplexity for both train and validation loss
train_ppl = compute_perplexity(1.0553)
val_ppl = compute_perplexity(1.4894)

print(f"Train Perplexity: {train_ppl:.2f}")
print(f"Validation Perplexity: {val_ppl:.2f}")

# --- Plot: Perplexity Bar Chart ---

plt.figure(figsize=(5, 4))
plt.bar(['Train Perplexity', 'Validation Perplexity'], [train_ppl, val_ppl], color=['blue', 'red'])
plt.title("Model Perplexity Comparison")
plt.ylabel("Perplexity")
plt.tight_layout()
plt.savefig("perplexity_comparison.png")
plt.show()

#---------------


#Repeated N-Grams
from collections import Counter

def check_ngram_repetition(text, n=3):
    words = text.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    
    most_common_ngrams = ngram_counts.most_common(10)  # Top 10 repeated sequences
    return most_common_ngrams

with open("output.txt", "r", encoding="utf-8") as f:
    generated_text = f.read()

print("Most Common 3-Gram Repetitions:", check_ngram_repetition(generated_text))

# --- Plot: 3-Gram Repetition Frequency ---

common_ngrams = check_ngram_repetition(' '.join(generated_text))
labels = [' '.join(gram) for gram, _ in common_ngrams]
counts = [count for _, count in common_ngrams]

plt.figure(figsize=(10, 5))
plt.barh(labels[::-1], counts[::-1])  # Reverse for top-down order
plt.xlabel("Frequency")
plt.title("Top 10 Repeated 3-Grams in Generated Text")
plt.tight_layout()
plt.savefig("ngram_repetition.png")
plt.show()

#-----------------

#Sentiment Analysis

#install dependency:
#python -m spacy download en_core_web_sm
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(text):
    """Perform sentiment analysis using TextBlob (polarity) and spaCy (named entity checks)."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Ranges from -1 (negative) to +1 (positive)
    
    # Named Entity Recognition (NER) to see if model hallucinates entities
    doc = nlp(text)
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return polarity, named_entities

# Read the generated text
with open("output.txt", "r", encoding="utf-8") as f:
    generated_text = f.readlines()

# Analyze sentiment per sentence
sentiments = []
for line in generated_text[:50]:  # Limit to first 50 lines for quick evaluation
    polarity, entities = analyze_sentiment(line)
    sentiments.append(polarity)

    print(f"Text: {line.strip()[:100]}")  # Print first 100 chars of the sentence
    print(f"Sentiment Polarity: {polarity:.3f}")
    print(f"Named Entities: {entities}\n")

# Calculate the overall sentiment of the generated text
overall_sentiment = sum(sentiments) / len(sentiments)
print(f"Overall Sentiment of Generated Text: {overall_sentiment:.3f}")

# --- Plot: Named Entity Frequency ---
all_entities = []
for line in generated_text[:50]:
    _, entities = analyze_sentiment(line)
    all_entities.extend([label for _, label in entities])

entity_counts = Counter(all_entities)

if entity_counts:
    plt.figure(figsize=(6, 4))
    plt.bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
    plt.title("Named Entity Frequency")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("entity_frequency.png")
    plt.show()
else:
    print("No named entities detected to plot.")

# --- Plot: Sentiment Score Histogram ---
plt.figure(figsize=(6, 4))
plt.hist(sentiments, bins=10, color='orchid', edgecolor='black')
plt.title("Distribution of Sentiment Polarity")
plt.xlabel("Polarity Score")
plt.ylabel("Number of Sentences")
plt.tight_layout()
plt.savefig("sentiment_histogram.png")
plt.show()

# --- Plot: Sentiment trend over lines ---

plt.figure(figsize=(10, 4))
plt.plot(sentiments, marker='o', linestyle='-', label="Sentiment Polarity")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel("Line Number")
plt.ylabel("Polarity")
plt.title("Sentiment Trend Over Generated Text")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sentiment_trend.png")
plt.show()
#----------------