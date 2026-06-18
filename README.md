# Multimodal Assistant for the Visually Impaired

## Demo Video

https://drive.google.com/file/d/1oXNv2oao5bxq3Pfw52WVLcV_b_vQSLIv/view?usp=sharing

> Architecture: **Offline-First** • **CPU-Optimized** • **Fine-Tuning Ready**

A multimodal virtual assistant designed to support visually impaired users by understanding their surroundings through speech recognition (STT), image analysis (VLM), and natural voice responses (TTS). The project has been restructured to support both model fine-tuning workflows and efficient CPU-based inference deployment.

---

## Project Structure

```text
CS431_multimodal_for_visually_impaired/
├── application/          # Source code for inference applications (Web App & CLI Pipeline)
├── finetuning/           # Training and fine-tuning scripts (VLM/STT)
├── models/               # Model weights (GGUF, CTranslate2)
├── test_data/            # Sample audio and image data for testing
└── README.md             # Project documentation
```

---

## Fine-Tuning Guide

The project includes dedicated scripts for fine-tuning models to improve performance on domain-specific tasks.

### 1. Dataset Preparation

Prepare your training dataset according to the target model requirements (e.g., JSONL files, instruction-response formatting for VLMs, custom speech datasets for STT fine-tuning).

### 2. Run Training

Update the following commands according to your training configuration:

```bash
# Example:
# cd finetuning
# python train.py --model_path ../models/Qwen3-VL --data_path data/train.jsonl
```

---

## Installation & Usage

The following instructions describe how to set up and run the inference pipeline located in the `application/` directory.

### 1. Environment Setup

Python 3.10 or newer is required. Using a virtual environment is highly recommended.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r application/requirements.txt
```

---

## 2. Model Preparation (IMPORTANT)

The inference pipeline is optimized for CPU execution and requires models to be converted into the appropriate formats before deployment.

### A. Speech Recognition Model (STT – PhoWhisper)

Convert PhoWhisper into the CTranslate2 format:

```bash
huggingface-cli download vinai/PhoWhisper-tiny --local-dir temp_model
ct2-transformers-converter --model temp_model --output_dir models/phowhisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8
```

### B. Vision-Language Model (VLM – Qwen3-VL)

Place the following GGUF files inside the `models/` directory:

* `mmproj-Qwen3-VL-2B-Finetuned-BF16.gguf`
* `Qwen3-VL-2B-Finetuned-BF16-Q4_K_M.gguf`

---

## 3. Running the System

### Launch the Web Application (Recommended)

```bash
cd application
python app_qwen_gguf.py
```

Open the web interface at:

```text
http://127.0.0.1:7860
```

### Run the CLI Pipeline

The system automatically loads data from the `test_data/` directory, performs speech recognition, analyzes images, and generates spoken responses.

```bash
cd application
python qwen_pipeline_gguf.py
```

---

## Notes

* **CPU-First Design:** The inference pipeline uses GGUF and INT8 quantization to maximize CPU efficiency. Image decoding issues (`stb_image`) have been resolved using PIL.
* **Fully Offline Operation:** Internet access is only required for the initial model download. Once installed, the entire system runs 100% offline.
* **Automatic GPU Detection (STT):** NVIDIA GPU available → CUDA (float16); otherwise → CPU fallback (int8).
* **Optional GPU Acceleration for VLM:** Open `application/qwen_pipeline_gguf.py`, locate `n_gpu_layers=0`, and change it to `n_gpu_layers=-1` to offload all supported layers to the GPU.
