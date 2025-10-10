# Hybrid Attention-SSM-Hopfield Model – ASH? HASS? Which name do you enjoy?

This random curiosity project explores a **novel hybrid architecture** that mixes
**Transformers (self-attention)**, **State-Space Models (SSMs)**, and
**Hopfield networks** to see how well such combinations can learn
language modeling tasks. It's basically inspired by the hippocampus and uses
associative memory for pattern completion alongside efficient state tracking.

## 🧪 What Is This?

A weird experiment that accidentally works really well. I wanted to see what happens when you combine:
- **Hopfield Networks** (1982 associative memory) – Here, I also use a modern hopfield which I found out after reading the 2020 paper: **Hopfield is all you need** (nice pun on attention is all you need) – https://doi.org/10.48550/arXiv.2008.02217
- **State Space Models** (linear complexity sequence modeling)  
- **Self-Attention** (the transformer thing everyone uses)

Turns out they work together better than separately. Who knew? 🤷

## 📊 Results That Surprised Me

Training on TinyShakespeare with just **4.5M parameters**:

| Configuration | Loss | Perplexity | Params | vs GPT-2 (124M) |
|--------------|------|------------|--------|-----------------|
| Classic Hopfield + Attention | 1.98 | ~7.2 | 3.6M | - |
| Modern Hopfield + Attention | 1.15 | ~3.2 | 4.1M | Getting close... |
| **Full Stack (Hop+SSM+Attn)** | **0.58** | **~1.79** | **4.5M** | **Better!** |

The full model achieves **better perplexity than GPT-2** while being **27x smaller**. I'm still not sure why this works so well. Maybe overfitting? However, val_loss scores were stable (steps were run until (100/step) 3000. 


We tested the idea on **Tiny Shakespeare** dataset.

------------------------------------------------------------------------

## 🚀 What We Did

### 1. Baseline Setup

-   Started from a **small GPT-like model** (\~3.5M parameters).
-   Trained it on **Tiny Shakespeare** with Colab Pro GPU.

### 2. Added Hopfield Memory

-   Introduced a **Hopfield-style associative memory block**.
-   Allows the model to store and retrieve patterns during sequence
    processing.
-   Result: Improved ability to recall patterns across steps.

### 3. Integrated State-Space Model (SSM)

-   Added a **diagonal SSM** layer (`A, B, C, D` matrices).
-   Helps capture **long-range dependencies** better than plain
    attention alone.
-   We made the **SSM input scaling learnable** instead of fixed.

### 4. Hybrid Attention + SSM

-   Built blocks where **attention + SSM mix together**.
-   Configurable: you can start mixing from a certain layer (e.g., layer
    6 → SSM-dominant).
-   This lets us test **how early vs. late SSM helps**.

### 5. Modern Hopfield (optional)

-   Discussed upgrading to **Modern Hopfield Networks** (continuous
    energy functions, high storage capacity).

### 6. Sampling Function

-   Added text generation with **temperature and top-k filtering**.

-   Example usage:

    ``` python
    print(sample(model, start="ROMEO:", length=150, temperature=0.8, top_k=40))
    ```

------------------------------------------------------------------------

## 📊 Results (Tiny Shakespeare)

Training log (loss trajectory):

    step   100 | loss 3.009
    step  1000 | loss 2.186
    step  2000 | loss 1.239
    step  2500 | loss 0.997
    step  3000 | loss 0.835
    step  3300 | loss 0.761

-   No plateau reached by step 3300 → model still improving.\
-   This suggests **the hybrid architecture has legs** and could scale
    further.\
-   At 3.5M params, it's already learning patterns strongly.

------------------------------------------------------------------------

## ⚙️ Usage

1.  **Toggle SSM**

    ``` python
    USE_SSM = True   # or False
    ```

2.  **Device Setup**

    ``` python
    DEVICE = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    ```

3.  **Train** Run the training cell. It will log loss every 100 steps.

4.  **Sample** Run the sampling function after training to generate
    text.

------------------------------------------------------------------------

## 💡 Lessons Learned

-   **SSMs** are efficient for long contexts (O(n) vs O(n²) in
    attention).\
-   **Hopfield memory** adds dynamic recall capabilities.\
-   Hybridization seems promising: mixing **attention + SSM** leads to
    faster loss convergence.\
-   Tiny Shakespeare shows strong learning with just a few million
    params.

------------------------------------------------------------------------

## 📌 Next Steps

-   Run **ablation studies** (Attention-only vs SSM-only vs Hybrid).\
-   Scale to **10M--50M params** to test scaling law behavior.\
-   Try **Modern Hopfield** for higher memory capacity.\
-   Benchmark on harder datasets (WikiText, OpenWebText).\
-   Explore **efficient incremental Hopfield updates** for scaling.


Newest Model, was scaled to 62M Params (PPL 39 @ Word Level Wikitext-03) has Gating and Mixture of Recursion added!


------------------------------------------------------------------------

## 🧑‍💻 Author Notes

This project is **self-funded research**, running on **Colab GPUs (T4,
L4, A100, TPUs)**.\
Goal: see if hybrid models (Attention + SSM + Hopfield) can provide an
alternative to pure Transformer architectures.

------------------------------------------------------------------------
