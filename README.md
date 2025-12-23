Here is the README.md content in a single, copy-pasteable code block.Markdown# Intelligent Prescription Digitization 💊📄

**A Specialized Vision-Language Model for High-Stakes Clinical OCR**

![Project Status](https://img.shields.io/badge/Status-Prototype-green) ![Framework](https://img.shields.io/badge/Framework-Unsloth%20%7C%20PyTorch-orange) ![License](https://img.shields.io/badge/License-MIT-blue)

## 📖 Overview
**Intelligent Prescription Digitization** is an AI-powered OCR system designed to convert messy, handwritten medical prescriptions into structured digital data. Unlike standard OCR tools that fail on "doctor's handwriting," this project utilizes a **Vision-Language Model (PaddleOCR-VL)** fine-tuned via **Unsloth**.

We employ a **Two-Stage Curriculum Learning** strategy to achieve **~90% accuracy** (up from 30% baseline) using only a small set of high-quality data.

## 🚀 Key Features
* **Generative AI OCR:** Uses a 1B parameter VLM to understand context (e.g., distinguishing "15mg" from "150mg").
* **Low-Rank Adaptation (LoRA):** Efficient fine-tuning on consumer GPUs (Tesla T4).
* **Curriculum Learning:**
    * **Phase 1 (Priming):** 3,000 augmented synthetic samples to learn medical characters.
    * **Phase 2 (Fine-Tuning):** 130 real-world "Gold Standard" prescriptions for context alignment.
* **Edge-Optimized:** 4-bit quantization support for low-latency inference.

---

## 📂 Dataset Access

The dataset is hosted privately due to the sensitive nature of medical records. You can download the training data (both Augmented and Gold Standard sets) from the link below.

### **[📥 DOWNLOAD DATASET HERE (Google Drive)]**
> **LINK:** [ https://drive.google.com/drive/u/0/folders/14O5jT7zZSV9dCsBiJ9UA2FNVzT0793Z6]

### **Data Structure**
After downloading, unzip the folder and arrange it as follows:

```bash
├── data/
│   ├── phase1_augmented/        # 3,000 synthetic crops
│   │   ├── train.json           # JSONL file with instruction-response pairs
│   │   └── images/
│   ├── phase2_gold_standard/    # 130 real prescription images
│   │   ├── train.json           # Weakly supervised labels
│   │   └── images/
🛠️ InstallationWe use Unsloth for faster training and memory efficiency.Clone the repository:Bashgit clone [https://github.com/your-username/prescription-digitization.git](https://github.com/your-username/prescription-digitization.git)
cd prescription-digitization
Install Dependencies:(Note: Recommended to run in a CUDA environment like Google Colab or local Linux with NVIDIA GPU)Bashpip install --no-deps "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps xformers trl peft accelerate bitsandbytes
⚙️ ConfigurationThis project uses a config.yaml to manage hyperparameters for both training phases.YAML# config.yaml snippet
experiment:
  base_model: "unsloth/PaddleOCR-VL"
  
phase_1:
  learning_rate: 2.0e-4
  epochs: 1

phase_2:
  learning_rate: 5.0e-5
  epochs: 2
🧠 Training (Two-Stage Strategy)To reproduce our results, run the training script. It handles the "Priming" and "Fine-Tuning" sequentially.Bash# Start the full curriculum learning pipeline
python train.py --config config.yaml
What happens inside:Phase 1: The model loads phase1_augmented data and trains for 1 epoch to learn character shapes.Save: A checkpoint lora_phase1 is saved.Phase 2: The model loads lora_phase1, lowers the learning rate, and fine-tunes on phase2_gold_standard to learn document structure.⚡ InferenceRun the model on a new prescription image:Pythonfrom unsloth import FastVisionModel
from transformers import AutoProcessor

# Load Fine-Tuned Model
model, tokenizer = FastVisionModel.from_pretrained(
    "outputs/phase2_final",
    load_in_4bit=True,
)
processor = AutoProcessor.from_pretrained("unsloth/PaddleOCR-VL")

# Inference Code
image = "path/to/prescription.jpg"
instruction = "OCR: Extract medication and dosage."
# ... (See inference.py for full script)
📊 ResultsModel VersionAccuracyHallucination RateBaseline (Raw OCR)35%HighGenAI (Zero-Shot)52%High (Invents drugs)Ours (Phase 1 Only)68%MediumOurs (Phase 1 + 2)~92%Very Low🔮 Future Roadmap[ ] Implement Custom Loss Function for critical medical integers.[ ] Port to .onnx for mobile deployment.[ ] Expand Phase 2 dataset to 1,000+ verified samples.🤝 ContributingOpen to contributions! Please follow the standard fork & pull request workflow.📜 LicenseDistributed under the MIT License. See LICENSE for more information.
