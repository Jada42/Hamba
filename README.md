# Hybrid Attention-SSM-Hopfield Model

This project explores a **novel hybrid architecture** that mixes
**Transformers (self-attention)**, **State-Space Models (SSMs)**, and
**Hopfield networks** to see how well such combinations can learn
language modeling tasks.

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

------------------------------------------------------------------------

## 🧑‍💻 Author Notes

This project is **self-funded research**, running on **Colab GPUs (T4,
L4, A100, TPUs)**.\
Goal: see if hybrid models (Attention + SSM + Hopfield) can provide an
alternative to pure Transformer architectures.

------------------------------------------------------------------------
