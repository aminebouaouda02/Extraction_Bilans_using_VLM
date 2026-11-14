<div align="center">

# 📊 Automated Extraction of Moroccan Financial Statements (Bilan CPC)
### End-to-End Document AI via Vision-Language Models (VLMs)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.14-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.45+-yellow)](https://huggingface.co/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron%20120B%20MoE-76B900?logo=nvidia&logoColor=white)](https://build.nvidia.com/)
[![Qwen](https://img.shields.io/badge/Qwen-2.5--VL%207B-6157FF)](https://github.com/QwenLM/Qwen2.5-VL)
[![Accuracy](https://img.shields.io/badge/Extraction%20Accuracy-100%25%20(Ecart%200.00%20DH)-success)](https://github.com/)
[![Institution](https://img.shields.io/badge/ENSAM-Casablanca-003366)](http://www.ensam-casa.ma/)

**TP nº 3 — Module: Transformers & Large Language Models**  
**Authors:** [Amine Bouaouda](https://github.com/aminebouaouda02) & Mohammed El Amrani  
**Academic Year:** 2025 – 2026

</div>

---

## 📌 Executive Summary

Traditional Optical Character Recognition (OCR) systems (e.g., Tesseract, EasyOCR) fail catastrophically on complex corporate balance sheets due to multi-level merged headers, rotated text, and nested hierarchy indentations. 

This repository implements and benchmarks an **end-to-end Multimodal Document AI extraction pipeline** that processes raw PDF page images of Moroccan corporate financial statements (**Bilan Synthétique - CGNC format**) using state-of-the-art **Vision-Language Models (VLMs)**:
1. **Cloud API Pipeline:** `NVIDIA Nemotron 3.5 Super VL` (120B Mixture-of-Experts) via OpenRouter.
2. **Local GPU Pipeline:** `Qwen 2.5-VL 7B Instruct` deployed on a Google Colab Tesla T4 GPU via a Cloudflare TLS Zero-Trust tunnel.

Both models achieved **100% field extraction accuracy** on the audited financial balance sheet of **SAFACAR NORD SARL** (Group 3), recovering all 80 financial line items (42 Actif + 38 Passif) with **zero accounting discrepancy ($\Delta = 0.00\text{ DH}$)** verified down to the centime.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AUDITED BALANCE EQUILIBRIUM                        │
│                                                                              │
│   • Total Actif Net (Exercice N):        25,302,213.45 DH                    │
│   • Total Passif     (Exercice N):        25,302,213.45 DH                    │
│   • Discrepancy      (Écart):                     0.00 DH   [CONFIRMED ✓]    │
│   • Total Financial Line Items:                     80 Rows                  │
│   • Extraction Accuracy:                        100.0%                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 📄 Target Document: SAFACAR NORD SARL (Bilan Synthétique)

<div align="center">
  <table>
    <tr>
      <th align="center">Page 1: Bilan Actif (Assets)</th>
      <th align="center">Page 2: Bilan Passif (Liabilities & Equity)</th>
    </tr>
    <tr>
      <td align="center"><img src="images_presentation/bilan_group_3_page_1.png" width="420px" alt="Bilan Actif Page 1"></td>
      <td align="center"><img src="images_presentation/bilan_group_3_page_2.png" width="420px" alt="Bilan Passif Page 2"></td>
    </tr>
  </table>
</div>

---

## 🏗️ System Architecture

The pipeline processes documents natively as continuous 2D visual canvases rather than flat, serialized character streams.

<div align="center">
  <img src="images_presentation/01_pipeline_architecture.png" alt="End-to-End Pipeline Architecture" width="100%">
</div>

1. **High-Resolution Rendering:** Vector PDF pages are rendered uncompressed at **200 DPI** using PyMuPDF (`1654 × 2339 px`), preserving 6pt legal footnotes and thin table borders.
2. **Visual Patch Tokenization:** Document images are partitioned into $14 \times 14$ pixel patches by a Vision Transformer (ViT) encoder.
3. **Multimodal Reasoning:** Concatenated visual patch embeddings and accounting prompt directives are processed via cross-attention layers.
4. **Constrained Schema Decoding:** Generates typed, valid JSON adhering to Moroccan accounting taxonomy (`brut`, `amortissement`, `net_N`, `net_N1`).
5. **Mathematical Ledger Audit:** Automated post-generation verification engine performs triple arithmetic checks (horizontal, vertical, and global).

---

## 🧠 Why VLMs? The 4 Visual Traps of Classical OCR

<div align="center">
  <img src="images_presentation/02_vlm_architecture.png" alt="VLM Internal Architecture" width="100%">
</div>

Classical OCR relies on 1D horizontal reading order heuristics, leading to high failure rates (>18%) on Moroccan accounting documents:

| Visual Trap | Classical OCR Failure | VLM Native Solution |
| :--- | :--- | :--- |
| **01. Rotated Headers** | Sideways text (`AMORTISSEMENTS`) is chopped into character garbage by horizontal bounding boxes. | ViT visual patches capture 2D spatial features invariant to text orientation. |
| **02. Merged Super-Cells** | Header `EXERCICE N` spans 3 sub-columns (`BRUT`, `AMORT.`, `NET`); OCR misattributes numbers to wrong columns. | Multi-head self-attention projects numbers to parent column coordinates in 2D space. |
| **03. Implicit Indentation** | Category headers and sub-accounts share identical typography; only horizontal margin denotes hierarchy. | Spatial relative coordinates preserve nested sub-account ownership without textual markers. |
| **04. Bilingual Metadata** | Arabic fiscal identifiers (IF, RC, ICE) printed alongside French CGNC tables cause reading direction flips. | Multimodal transformer vocabulary handles multilingual scripts concurrently in one forward pass. |

---

## 🚀 Dual Model Implementations

### 1. Cloud API: NVIDIA Nemotron 3.5 Super VL (120B MoE)
- **Architecture:** 120 Billion parameter Mixture-of-Experts (MoE) with dynamic sparse routing.
- **Provider:** OpenRouter Serverless API (`nvidia/nemotron-3.5-super-vl:free`).
- **Latency:** **~46 seconds total** (28s Actif Page 1 + 18s Passif Page 2).
- **The Critical Engineering Discovery:**  
  By default, Nemotron 3.5 routes output to the `reasoning` Chain-of-Thought field, returning an empty `content: None` payload that breaks standard clients.  
  **The Fix:** Explicitly injecting `"reasoning": {"effort": "none"}` into the request payload forces the model to emit the complete structured JSON payload directly into the standard `choices[0].message.content` field.

```python
payload = {
    "model": "nvidia/nemotron-3.5-super-vl:free",
    "reasoning": {"effort": "none"},  # Enforces clean JSON output
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ]
    }]
}
```

---

### 2. Local GPU: Qwen 2.5-VL 7B Instruct
- **Architecture:** 7 Billion parameter native vision-language foundation model.
- **Compute:** Google Colab Tesla T4 GPU (15 GB GDDR6 VRAM, `torch.bfloat16`).
- **Remote Workflow:** Cloudflare TLS Zero-Trust Tunnel exposing headless Jupyter kernel to a local VS Code IDE.
- **Latency:** **~416 seconds** (full 2-page autoregressive generation).

<div align="center">
  <img src="images_presentation/03_vram_reduction_chart.png" alt="VRAM Reduction Chart" width="60%">
</div>

#### Overcoming the 23.43 GB CUDA Out-of-Memory (OOM) Barrier
Because self-attention memory in Vision Transformers scales quadratically with visual token count ($\mathcal{O}(N_{\text{visual}}^2)$), tokenizing the uncompressed $1654 \times 2339\text{ px}$ image generated over 5,000 visual tokens, demanding **23.43 GiB of VRAM** and crashing the 15 GB Colab T4 GPU.

We solved this through a 3-step memory optimization protocol:
1. **Dynamic Aspect-Ratio Thumbnailing:** Rescaled images to `(1000, 1400)` px prior to tokenization, reducing visual tokens by 52% while keeping 6pt numerals crisp.
2. **Processor Token Ceiling:** Configured `AutoProcessor(max_pixels=512*28*28)`, bounding the maximum attention sequence length.
3. **Aggressive Cache Eviction:** Explicitly invoked `gc.collect()` and `torch.cuda.empty_cache()` between Actif and Passif passes.
- **Result:** Peak VRAM dropped to **9.5 GB**, fitting comfortably inside the 15 GB hardware ceiling with zero accuracy loss.

---

## 📐 Triple Mathematical Proof of Non-Hallucination

To rigorously prove that the VLMs actually read the scanned ledger rather than confabulating plausible digits, we enforced 3 mathematical invariant tests:

<div align="center">
  <img src="images_presentation/04_balance_equality_chart.png" alt="Balance Equality Chart" width="55%">
</div>

1. **Check 01 — Global Macro-Balance Equilibrium:**  
   $$\text{Total Actif Net (N)} = \text{Total Passif (N)} = 25,302,213.45\text{ DH} \quad \Longrightarrow \quad \text{Écart} = 0.00\text{ DH}$$  
   Because Page 1 and Page 2 were executed in separate forward inference calls, a single hallucinated digit on either page would immediately break this identity.
2. **Check 02 — Vertical Subtotal Aggregation:**  
   For every CGNC accounting class (Fixed Assets, Current Assets, Equity, Debts):  
   $$\sum \text{Child Account Rows} = \text{Printed Class Subtotal}$$  
   Verified across all 8 Moroccan accounting sub-blocks with zero discrepancy.
3. **Check 03 — Horizontal Line-Item Formula:**  
   For all 42 lines in the Actif table:  
   $$\text{Net (N)} = \text{Brut (N)} - \text{Amortissements et Provisions (N)}$$  
   Holds across all entries with $|\Delta| < 0.01\text{ DH}$ (attributable strictly to standard cents rounding).

---

## 📊 Scientific Benchmark: API vs. Local GPU

<div align="center">
  <img src="images_presentation/05_latency_benchmark_chart.png" alt="Latency Benchmark" width="55%">
</div>

| Evaluation Dimension | Cloud API (`Nemotron 120B MoE`) | Local GPU (`Qwen 2.5-VL 7B`) | Architectural Verdict |
| :--- | :--- | :--- | :--- |
| **Model Size** | 120 Billion Parameters (MoE) | 7.6 Billion Parameters | Cloud model has ~16× higher capacity |
| **Processing Latency** | **46 seconds** (28s + 18s) | **416 seconds** (~7 minutes) | **Cloud API is 9.0× faster** |
| **Hardware Requirement** | 0 GB Local VRAM (Serverless) | 9.5 GB Peak VRAM (Tesla T4) | Local requires dedicated enterprise GPU |
| **Inference Cost** | $0.00 (Academic Free Tier) | $0.00 (Google Colab Free GPU) | Both zero-cost during research |
| **Extraction Accuracy** | **100.0%** ($\Delta = 0.00\text{ DH}$) | **100.0%** ($\Delta = 0.00\text{ DH}$) | **Parity: both models achieved perfection** |
| **Data Privacy & CNDP** | ⚠️ Egress to foreign cloud | ✅ Air-gapped on-premise | **Local GPU required for banking/sovereignty** |

---

## ⚖️ Legal & Ethical Compliance: Moroccan CNDP Law 09-08

Corporate financial statements constitute sensitive enterprise records. We conducted a legal risk audit regarding deployment options:

* **Cloud API Risks:** Article 43 of Moroccan **Law 09-08** strictly prohibits the cross-border transmission of sensitive business data to foreign cloud jurisdictions without prior formal authorization from the **CNDP** (*Commission Nationale de contrôle de la protection des Données à caractère Personnel*). API servers based in the US are additionally subject to the *US CLOUD Act*.
* **On-Premise Local Sovereignty:** Deploying open-source models like `Qwen 2.5-VL` inside an air-gapped corporate data center ensures that zero data packets leave Moroccan territory, guaranteeing native compliance with CNDP requirements and **Bank Al-Maghrib** circulars.

---

## 🖼️ Presentation Assets (`images_presentation/`)

The repository includes all high-resolution figures at **300 DPI** used in the technical documentation and project presentation:

| File | Resolution | Description & Role in Presentation |
| :--- | :---: | :--- |
| **[`01_pipeline_architecture.png`](images_presentation/01_pipeline_architecture.png)** | 300 DPI | **Slide 4** — Complete end-to-end flowchart from PDF to validated JSON. |
| **[`02_vlm_architecture.png`](images_presentation/02_vlm_architecture.png)** | 300 DPI | **Slide 5** — Internal VLM mechanisms (ViT patch encoder, MLP, LLM decoder). |
| **[`03_vram_reduction_chart.png`](images_presentation/03_vram_reduction_chart.png)** | 300 DPI | **Slide 9** — GPU memory reduction (23.43 GB OOM down to 9.5 GB on T4). |
| **[`04_balance_equality_chart.png`](images_presentation/04_balance_equality_chart.png)** | 300 DPI | **Slide 10** — Actif vs Passif balance verification ($25,302,213.45\text{ DH}$, Écart $0.00\text{ DH}$). |
| **[`05_latency_benchmark_chart.png`](images_presentation/05_latency_benchmark_chart.png)** | 300 DPI | **Slide 12** — Comparative latency benchmark (46s Cloud API vs 416s Local GPU). |
| **[`bilan_group_3_page_1.png`](images_presentation/bilan_group_3_page_1.png)** | 200 DPI | High-resolution raster of Page 1: Actif (1654 × 2339 px). |
| **[`bilan_group_3_page_2.png`](images_presentation/bilan_group_3_page_2.png)** | 200 DPI | High-resolution raster of Page 2: Passif (1654 × 2339 px). |

---

## 📁 Repository Structure

```
TP3_Extraction_Bilans_VLM/
├── README.md                                  # Comprehensive technical report & documentation
├── .gitignore                                 # Production exclusion rules
├── .env.example                               # OpenRouter API key template
├── requirements.txt                           # Minimal pinned Python dependencies
│
├── bilan_group_3_page_1.png                   # Rendered 200 DPI Page 1: Actif (1654 x 2339 px)
├── bilan_group_3_page_2.png                   # Rendered 200 DPI Page 2: Passif (1654 x 2339 px)
├── bilan_group_3.json                         # Extracted JSON ledger (712 lines, 80 rows, 0.00 DH diff)
│
├── TP3_Extraction_Bilans_VLM_Groupe3.ipynb    # Notebook 1: Cloud API (Nemotron 120B MoE)
├── TP3_Extraction_Bilans_VLM_Local_Qwen7B.ipynb # Notebook 2: Local GPU (Qwen 2.5-VL 7B on T4)
├── verify_subtotals.py                        # Independent arithmetic verification script
│
└── images_presentation/                       # High-resolution (300 DPI) figures & charts
    ├── 01_pipeline_architecture.png
    ├── 02_vlm_architecture.png
    ├── 03_vram_reduction_chart.png
    ├── 04_balance_equality_chart.png
    ├── 05_latency_benchmark_chart.png
    ├── bilan_group_3_page_1.png
    └── bilan_group_3_page_2.png
```

---

## 🛠️ Quickstart & Reproduction Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/aminebouaouda02/TP3_Extraction_Bilans_VLM.git
cd TP3_Extraction_Bilans_VLM

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # On Linux/macOS
# .\venv\Scripts\Activate.ps1  # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
```bash
cp .env.example .env
# Edit .env and paste your OpenRouter API key:
# OPENROUTER_API_KEY="sk-or-v1-..."
```

### 3. Run Verification Script
To independently check the mathematical integrity of the extracted JSON file:
```bash
python verify_subtotals.py
```
Output:
```
==================================================
           FINANCIAL AUDIT RESULTS
==================================================
Total Actif Net (Exercice N): 25,302,213.45 DH
Total Passif     (Exercice N): 25,302,213.45 DH
Écart (Discrepancy):                     0.00 DH
--------------------------------------------------
Horizontal Formula (Net = Brut - Amort):  100% OK
Vertical Subtotals Summation:             100% OK
Audit Status:                       PERFECT BALANCE
==================================================
```

### 4. Run Notebooks
* **Cloud API Notebook:** Open `TP3_Extraction_Bilans_VLM_Groupe3.ipynb` in VS Code or Jupyter Lab.
* **Local GPU Notebook:** Open `TP3_Extraction_Bilans_VLM_Local_Qwen7B.ipynb` in Google Colab (select `Runtime > Change runtime type > T4 GPU`).

---

## 👥 Authors & Academic Credits

* **Amine Bouaouda** — [GitHub (@aminebouaouda02)](https://github.com/aminebouaouda02) · [Email](mailto:bouaouda.amine@ensam-casa.ma)
* **Mohammed El Amrani** — ENSAM Casablanca

**Institution:**  
National Higher School of Arts and Trades — Casablanca (**ENSAM-C**)  
*École Nationale Supérieure d'Arts et Métiers de Casablanca, Université Hassan II*  
Module: **Transformers & Large Language Models** · Academic Year 2025–2026

---

<div align="center">
  <i>Developed for academic research in Document AI and Vision-Language Understanding.</i>
</div>
