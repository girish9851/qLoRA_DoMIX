# 📘 MCQA vs Answer Matching Evaluation Notebook Template

# --- Install dependencies ---
# Uncomment if running for the first time
# !pip install rapidfuzz sentence-transformers

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# Load sentence transformer model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- MCQA Evaluation ---
def evaluate_mcqa(correct_answers, model_answers):
    correct = sum(1 for ca, ma in zip(correct_answers, model_answers) if ca == ma)
    accuracy = correct / len(correct_answers)
    return accuracy

# --- Answer Matching: Fuzzy ---
def normalize(text: str) -> str:
    return text.strip().lower()

def fuzzy_match(model_answer, references, threshold=85):
    model_answer = normalize(model_answer)
    for ref in references:
        ref = normalize(ref)
        if fuzz.ratio(model_answer, ref) >= threshold:
            return True
    return False

# --- Answer Matching: Semantic ---
def semantic_match(model_answer, references, threshold=0.8):
    embeddings = semantic_model.encode([model_answer] + references, convert_to_tensor=True)
    sim_scores = util.cos_sim(embeddings[0], embeddings[1:])
    return sim_scores.max().item() >= threshold

# --- Unified CLI/Notebook Runner ---
def compare_evaluations():
    print("\n=== MCQA vs Answer Matching Evaluation ===\n")

    questions = [
        "Which of the following elements is a noble gas?",
        "Why does the moon shine?",
        "What makes a better insulator?"
    ]

    choices = [
        ["A. Oxygen", "B. Nitrogen", "C. Argon", "D. Carbon"],
        ["A. Makes own light", "B. Reflects sunlight", "C. Absorbs energy", "D. Refracts starlight"],
        ["A. Metal spoon", "B. Paper towel", "C. Steel rod", "D. Iron nail"]
    ]

    correct_mcqa = ["C", "B", "B"]
    model_mcqa   = ["C", "B", "A"]  # Simulating an error in last one

    reference_answers = [
        ["argon"],
        ["it reflects sunlight", "because it reflects light from the sun"],
        ["spoon"]
    ]

    model_answers = [
        "argon",
        "because it reflects light from the sun",
        "metal spoon"  # Close, but wrong
    ]

    print("\U0001F9EA MCQA Evaluation:")
    mcqa_acc = evaluate_mcqa(correct_mcqa, model_mcqa)
    print(f"MCQA Accuracy: {mcqa_acc:.2%}\n")

    print("\U0001F9EA Fuzzy Matching Evaluation:")
    fuzzy_correct = sum(fuzzy_match(m, r) for m, r in zip(model_answers, reference_answers))
    fuzzy_acc = fuzzy_correct / len(model_answers)
    print(f"Fuzzy Match Accuracy: {fuzzy_acc:.2%}\n")

    print("\U0001F9E0 Semantic Matching Evaluation:")
    semantic_correct = sum(semantic_match(m, r) for m, r in zip(model_answers, reference_answers))
    semantic_acc = semantic_correct / len(model_answers)
    print(f"Semantic Match Accuracy: {semantic_acc:.2%}")

# Run the evaluation
compare_evaluations()
