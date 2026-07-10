# ALLaM as Judge — Full Pipeline

A project to evaluate the ability of the **ALLaM-7B** model (and **Qwen2.5-7B** for comparison) to act as a "Judge" that classifies Arabic-language claims as **correct** or **incorrect**, while measuring the impact of **LoRA Fine-Tuning** on performance.

The code is a Google Colab notebook exported as a Python file (`.py`), containing **5 sequential experiments**.

---

## 📋 Overview

The model is given a claim, and asked to judge it with a single answer:

```
Judge the following statement with a single answer, without justification:
"The capital of Saudi Arabia is Cairo"
Is the statement true or false?
```

The model's output (`model_judgment`) is then compared against the `ground truth` label to compute performance metrics.

---

## 🧪 Experiment Design

All four initial experiments are run on the same model, **ALLaM-7B**, following the same workflow (Baseline → LoRA Fine-tuning → Baseline+FT → Comparison). The goal is to test the **effect of the prompt type** (Direct Answer vs. Few-shot + CoT) at both the **testing** and **fine-tuning** stages independently, to understand the effect of matching or mismatching the training prompt with the testing prompt.

There are two base prompts:
- **Prompt 1 (Direct Answer):** a simple, direct prompt with no examples — "Judge the statement... is it true or false?"
- **Prompt 2 (Few-shot + CoT):** a prompt containing two worked examples with "step-by-step reasoning" before the "final judgment"

| Experiment | Test Before FT | Fine-tuned with | Test After FT |
|---|---|---|---|
| **Exp. 1** | Prompt 1 (Direct Answer) | Prompt 1 | Prompt 1 (Direct Answer) |
| **Exp. 2** | Prompt 2 (Few-shot + CoT) | Prompt 2 | Prompt 2 (Few-shot + CoT) |
| **Exp. 3** | Prompt 2 (Few-shot + CoT) | Prompt 1 | Prompt 2 (Few-shot + CoT) |
| **Exp. 4** | Prompt 1 (Direct Answer) | Prompt 2 | Prompt 1 (Direct Answer) |
| **Exp. 5 (Qwen)** | Same methodology as Exp. 2 (Prompt 2 — Few-shot + CoT), in full | Prompt 2 | Prompt 2 (Few-shot + CoT) |

In other words:
- **Exp. 1 and Exp. 2** are "consistent" experiments: the same prompt is used for training and testing, both before and after fine-tuning.
- **Exp. 3 and Exp. 4** are "cross/mismatch" experiments: testing is done with a prompt different from the one the model was fine-tuned on, in order to measure how well fine-tuning **generalizes** across different prompt formats.
- **Exp. 5** repeats the best-performing setup (Exp. 2 — Few-shot + CoT, consistent for both training and testing), but on the **Qwen2.5-7B** model instead of ALLaM-7B, to compare the two models under identical conditions.

Each experiment goes through the same four stages:
1. **Baseline Testing (Test Before)** — run the original, untrained model using the prompt specified for that experiment.
2. **Fine-Tuning (LoRA)** — lightweight training on labeled data, using the prompt specified for the training stage of that experiment (which may differ from the test prompt in Exp. 3 and Exp. 4).
3. **Baseline + FT Testing (Test After)** — run the fine-tuned model, using the same prompt as the "Test Before" stage in the same experiment.
4. **Comparison** — compute metrics and compare results before/after fine-tuning.

---

## ⚙️ Requirements (Dependencies)

```bash
pip install transformers accelerate bitsandbytes sentencepiece
pip install datasets torch pandas scikit-learn
pip install "bitsandbytes>=0.46.1"
pip install peft
```

In addition to:
- A **Google Colab** environment (uses `google.colab.drive` to mount Google Drive).
- A **GPU**, since the model is loaded with 4-bit quantization (`BitsAndBytesConfig`).
- `matplotlib` and `numpy` for the final plots.

---

## 📂 File Structure & Expected Paths

The code relies on Google Drive paths that differ slightly between experiments (each experiment saves its results to its own folder/filename). Since **Exp. 2** is the best-performing and reference experiment, the paths for that pipeline are shown below as the example to follow — adjust them to match your own file locations:

```
/content/drive/MyDrive/GP/models/ALLaM-7B/                              # base model
/content/drive/MyDrive/GP/ALLaM_results/train_set.csv                   # training data
/content/drive/MyDrive/GP/ALLaM_results/original_test_set.csv           # test data
/content/drive/MyDrive/GP/ALLaM_results/second_experiment/              # Test Before results (Exp.2)
/content/drive/MyDrive/GP/fine_tuned_model_few_shot2/                   # LoRA output after training (Exp.2)
/content/drive/MyDrive/GP/figures/                                      # final plots
```

**Expected CSV columns:**
- `claim` — the statement text.
- `binary_label` — `correct` or `incorrect` (ground truth label).

> 📝 **Note on the other experiments (1, 3, 4, 5):** they follow the same general structure (same CSV columns, same stage sequence), but with different output folder/file names per experiment (e.g. `second_experiment_after.csv`, `exp3_comparison.csv`, `exp4_after_fine.csv`, ...), and some paths are reused inconsistently across experiments, since the code is the result of successive Colab exports rather than a unified pipeline built from scratch. It's recommended to consolidate the paths into shared configurable variables if you plan to reuse the code.

---

## 🔧 Key Code Components

### 1. Loading the Model with 4-bit Quantization
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### 2. Judge Function

There are two variants of the judge function, used interchangeably depending on the experiment (see the experiment table above):

**a) Prompt 1 — Direct Answer (no examples):**
```python
def judge_claim(claim):
    prompt = f'Judge the following statement with a single answer, without justification:\n"{claim}"\nIs the statement true or false?'
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, temperature=0.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

**b) Prompt 2 — Few-shot + Chain-of-Thought (used in the best experiment, Exp. 2):**
```python
def judge_claim(claim):
    """Few-shot + CoT"""
    prompt = f"""Evaluate the truth of the following statements, reasoning step by step:

Example 1:
Statement: "Riyadh is the capital of Saudi Arabia"
Reasoning: Riyadh has been known as the capital of Saudi Arabia since its unification
Final judgment: The statement is true

Example 2:
Statement: "Cairo is the capital of Saudi Arabia"
Reasoning: Cairo is the capital of Egypt, not Saudi Arabia. The capital of Saudi Arabia is Riyadh
Final judgment: The statement is false

Now, evaluate this statement:
Statement: "{claim}"
Reasoning:
Final judgment:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,   # higher than Prompt 1 since the answer is longer (reasoning + final judgment)
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

> 💡 The same **Prompt 2** format is also used when building the training data (`create_training_sample`) whenever the model is fine-tuned with this prompt, keeping the format consistent between training and testing.

### 3. Automatic Resume on Crash
The code saves results every 20 rows, and checks at startup whether a results file already exists so it can resume from where it left off instead of starting over.

### 4. LoRA Fine-Tuning Configuration
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 5. Extracting the Final Judgment from Model Output
```python
pattern = re.compile(r"(العبارة\s+صحيحة|العبارة\s+خاطئة)")
```
This extracts the last judgment that appears in the output text (since the model may repeat the sentence or explain before reaching its final judgment).

### 6. Computed Metrics
- Accuracy / Balanced Accuracy
- Macro & Weighted Precision / Recall / F1
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Confusion Matrix (TP, TN, FP, FN)
- Percentage of "unknown" cases (where the regex could not extract a clear judgment)

---

## ▶️ How to Run

1. Open the file in **Google Colab** (recommended), or copy the code into a GPU-enabled environment.
2. Update the paths (`model_dir`, `TRAIN_DATA_PATH`, `results_path`, ...) to match your file locations on Drive.
3. Run the cells in order for each experiment:
   - Baseline Testing → Fine-Tuning (LoRA) → Baseline+FT Testing → Comparison.
4. Results and plots are automatically saved to the specified Google Drive folders.

> ⚠️ **Note:** The original file is a direct export from Colab (`.ipynb` → `.py`), so it contains `!pip install` commands and Drive mounting (`drive.mount`) that only work inside a Colab environment.

---

## 📊 Final Outputs

- CSV files containing the results of each experiment (`*_with_judgment.csv`, `exp*_comparison.csv`).
- A comparison plot (`cross_model_comparisonv2.png` / `.pdf`) comparing **ALLaM-7B** vs. **Qwen2.5-7B** performance after fine-tuning, across all metrics.

---

## 📝 Notes

- The code contains some duplication (e.g., installing libraries and mounting Drive multiple times) because it is a direct export from Colab rather than a script structured from scratch.
- Some paths are hardcoded and should ideally be turned into configurable variables if you intend to reuse the code as a standalone project.
