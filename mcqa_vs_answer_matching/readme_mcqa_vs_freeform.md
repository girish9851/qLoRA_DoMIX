# 🧪 README: MCQA vs Free-Form Answer Matching Evaluator

This project provides a simple, extensible framework to evaluate language models using both:

- ✅ **Multiple-Choice Question Answering (MCQA)**
- 🤖 **Free-form Answer Matching** (Fuzzy and Semantic)

---

## 📂 File Structure

```bash
├── mcqa_vs_freeform_eval.py  # Core script (MCQA + Answer Matching logic)
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and usage
```

---

## 🚀 Features

- Compare MCQA accuracy with flexible open-answer evaluations
- Fuzzy string matching using RapidFuzz
- Semantic similarity using Sentence Transformers (`all-MiniLM-L6-v2`)
- CLI/Notebook-style evaluation runner

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Dependencies:

```
rapidfuzz
sentence-transformers
torch
```

---

## 🧪 How to Run

```bash
python mcqa_vs_freeform_eval.py
```

Or copy the script into a Jupyter notebook cell to test interactively.

---

## 📝 Example Output

```
=== MCQA vs Answer Matching Evaluation ===

🧪 MCQA Accuracy: 66.67%
🧪 Fuzzy Match Accuracy: 66.67%
🧠 Semantic Match Accuracy: 100.00%
```

---

## 📊 Use Cases

- Evaluate generative models that don't produce labels
- Compare classical vs free-form QA benchmarks
- Test reasoning quality beyond forced choice

---

## 🔧 Extending the Project

### Add Your Own Questions

In `mcqa_vs_freeform_eval.py`, edit:

```python
questions = [ ... ]
choices = [ ... ]
correct_mcqa = [ ... ]
model_mcqa = [ ... ]
reference_answers = [ ... ]
model_answers = [ ... ]
```

### Add a Web UI (Optional)

To create a live demo:

```bash
pip install streamlit
```

Then create `app.py` with Streamlit or use Gradio.

Would you like us to generate a Streamlit or Gradio app next?

---

## 🧠 Citation

Based on the evaluation techniques discussed in:

> "Answer Matching Outperforms Multiple Choice for Language Model Evaluation" — arXiv:2507.02856

