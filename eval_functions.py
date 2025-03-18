import math
import spacy
from textblob import TextBlob

def compute_perplexity(loss):
    """Compute perplexity from loss"""
    return math.exp(loss)

# Compute perplexity for both train and validation loss
train_ppl = compute_perplexity(1.0553)
val_ppl = compute_perplexity(1.4894)

print(f"Train Perplexity: {train_ppl:.2f}")
print(f"Validation Perplexity: {val_ppl:.2f}")


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