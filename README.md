# Project Overview

This project focuses on fine-tuning the 8-Billion parameter LLaMA 3.1 Instruct model to generate **empathetic conversational responses** in Bengali. The main engineering challenge was executing the fine tuning training loop on a highly constrained Kaggle T4 GPU with only about **15GB** of VRAM. To overcome this, the pipeline leverages the **Unsloth** framework, employing **4-bit quantization** and targeted Low-Rank Adaptation (LoRA) to drastically reduce the memory footprint. It also integrates the **BNLP** **toolkit** for grammatical accurate Bengali evaluation creating an efficient, deployable model with addition to **human-in-the-loop**

# Why Unsloth?  
Fine tuning the LLaMA 3.1 8B -instruct introduces severe memory and compute bottlenecks, especially on a free Kaggle environment featuring a single NVIDIA T4 GPU with ~16GB of available VRAM.  
Unsloth provides four engineering advantages for this pipeline:

* **Aggressive Memory Optimization (4-bit Quantization):** By utilizing Unsloth’s optimized implementation of `bitsandbytes`, the base model was loaded in 4-bit precision.  
  This shrank the model’s footprint from ~16GB down to roughly 5.5GB, leaving enough overhead for optimizer states, gradient accumulation and sequence processing.  
    
* **Custom Triton Kernels:** Unsloth replaces standard PyTorch attention and feed-forward layers with mathematically equivalent, highly optimized OpenAI Triton kernels. This reduces the memory overhead during the forward and backward passes, actively preventing VRAM fragmentation.

* **Compute Acceleration (2x Faster Training):** Because Kaggle notebooks have strict session timeouts, training speed is a hard operational constraint. Unsloth's rewritten backpropagation logic accelerates **LoRA** (Low-Rank Adaptation) fine-tuning by up to **200%** without degrading mathematical accuracy or model performance.

* **Native Ecosystem Compatibility:** Unlike proprietary training stacks, Unsloth acts as a highly efficient wrapper. It integrates natively with standard open-source ML libraries, including Hugging Face's transformers, `PEFT`, and `TRL` (Transformer Reinforcement Learning). This allowed for the seamless implementation of custom `DatasetProcessors` and `SFTTrainer` loops.

# Environment Configuration  
Before initializing PyTorch or the model weights, two critical environment variables must be set at the top of the execution script to prevent framework-level crashes on Kaggle’s specific hardware.  
```os.environ["CUDA_VISIBLE_DEVICES"] = "0"```

* Kaggle’s T4 accelerator option provides two GPUs (T4x2). By default, Hugging Face’s Trainer detects both and aggressively attempts to parallelize the training loop using Distributed Data Parallelism (DDP). However, models loaded in 4-bit quantization mathematically cannot be split across GPUs in this manner, resulting in an immediate ValueError crash. Hardcoding this variable to "0" completely blinds PyTorch to the second GPU, forcing a safe, single-device execution.

```os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"```

* During fine-tuning, PyTorch constantly allocates and frees chunks of VRAM for forward passes, gradients, and optimizer states. Over time, this creates memory fragmentation—small, unusable gaps of free space.  When the attention layer suddenly requires a large contiguous block of memory for a 512-token sequence, PyTorch will throw a false CUDA OutOfMemory error because the free space is fragmented. This configuration instructs the CUDA allocator to dynamically stitch these segments together, squeezing maximum utility out of the constrained 15GB VRAM.

# Data ingestion & Sequence Filtering: The `DataProcessor` Class
The `DataProcessor` class handles the ingestion, sanitization, and structural formatting of the raw Bengali conversation data. Its primary design goal is to create a failsafe pipeline that prevents lengthy, and bad data from triggering hardware crashes.

## Core Methods:

* `load_data()`: Ingests the raw CSV using Pandas and immediately applies a strict `dropna` filter. This prevents `NaN` values from crashing the tokenizer or causing the model to hallucinate missing data as strings.

* `process()`: Acts as the pipeline orchestrator. It wraps the cleaned text in LLaMA 3.1's exact conversational chat template (injecting the correct `<|start_header_id|>` and `<|eot_id|>` tags) to prepare the prompts for supervised fine-tuning.

## The OOM Challenge & Statistical Analysis
A critical hardware bottleneck in training LLMs is the self-attention mechanism, which has a memory complexity of O(N2) relative to the sequence length (N). Because the Kaggle T4 GPU does not support memory-efficient Flash Attention, processing excessively long conversations causes the attention matrix to explode in size, instantly triggering a CUDA Out-Of-Memory (OOM) crash.

```python
import numpy as np
from tqdm import tqdm

print("\n--- Running Sequence Length Analysis ===")
lengths = []

processor = DatasetProcessor(MODEL_NAME, 768)
full_dataset = processor.process(DATA_PATH) 

for example in tqdm(full_dataset, desc="Calculating Token Lengths"):
    tokens = processor.tokenizer(
        example["formatted_text"],
        truncation=False,
        add_special_tokens=False
    )
    
    lengths.append(len(tokens["input_ids"]))

lengths = np.array(lengths)

print(f"Max length: {lengths.max()}")
print(f"Mean length: {lengths.mean():.2f}")
print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
print(f"99th percentile: {np.percentile(lengths, 99):.0f}")
print("================================xxxx===============================\n")
```

To mitigate this, a statistical analysis of the dataset's sequence lengths was performed. Crucially, this analysis used LLaMA’s native Byte-Pair Encoding (BPE) tokenizer rather than a Bengali-specific word tokenizer. Because LLaMA heavily fragments Bengali script into multiple sub-word tokens, the true sequence lengths were significantly higher than the linguistic word count.

By identifying these long-tail statistical outliers, the pipeline dynamically filters out any conversation exceeding 768 sub-word tokens. This strict threshold mathematically guarantees that every training batch will safely fit within the ~16GB VRAM limit without memory fragmentation.

# Model Architecture & Unsloth Configuration: The `LLAMAFineTuner` Class
To ensure the codebase remains modular and adaptable to future framework changes, the model architecture and training loop were decoupled using the **Strategy Design Pattern**. The core logic is split across an abstract interface, a concrete hardware-specific implementation, and a central orchestrator.

## The Strategy Classes (Utility for LLAMAFineTuner)  
The architecture relies on `FineTuningStrategy` an abstract base class, which enforces a strict contract requiring two methods: `setup_model()` and `setup_training_args()`.

For this Kaggle environment, the `UnslothStrategy` implementation was injected:

* `setup_model()`: Initializes the LLaMA 3.1 8B model in 4-bit precision to fit the 16GB VRAM limit. It then applies the Low-Rank Adaptation (LoRA) configuration.  
  Applying LoRA naively to all linear layers generates over 40 million trainable parameters, crashing the 8-bit AdamW optimizer. Instead, the strategy surgically targets only the attention mechanisms, specifically the Query and Value projections (`q_proj` and `v_proj`) with a rank of r = 8. This mathematical constraint slashes the trainable footprint to roughly 4 million parameters, freeing up gigabytes of memory while retaining the ability to learn empathetic conversational patterns.  
    
* `setup_training_args()`: Defines the hyperparameter boundaries. To prevent memory fragmentation, a physical batch size of 1 is paired with 8 gradient accumulation steps, simulating a larger batch size.

## `LLAMAFineTuner`:
The `LLAMAFineTuner` class takes the injected strategy and the filtered dataset, acting as the pipeline's execution engine.

* `train(train_dataset)`: Instantiates the `SFTTrainer` (Supervised Fine-Tuning Trainer). It maps the formatted text, applies the defined training arguments, injects the early stopping callback, and executes the optimization loop.

* `save_model()`: A critical hardware failsafe for Kaggle deployments. Instead of using default merged saving functions (which attempt to download the massive 16GB base model and instantly crash the Kaggle disk storage) this method explicitly calls `save_pretrained`. This isolates the save operation, exporting only the lightweight LoRA adapter weights, keeping the final deliverable highly portable.

# Evaluation & Metric Traceability: The `Evaluator` Class 
To ensure pipeline stability, the evaluation phase was completely **decoupled** from the training loop. This architectural failsafe guarantees that if an evaluation metric computation crashes or times out, the newly fine-tuned LoRA weights remain securely saved and are not lost.

## Core Methods:

* `evaluate_generation()`: Calculates BLEU and ROUGE scores. A critical linguistic adjustment was engineered here. LLaMA's native Byte-Pair Encoding (BPE) tokenizer heavily fragments Bengali script into sub-word pieces, which mathematically distorts traditional n-gram overlap scores. To solve this, this method re-tokenizes both the generated output and the ground truth using the **BNLP** (Bengali Natural Language Processing) Basic Tokenizer (`BasicTokenizer()`). This ensures the BLEU and ROUGE mathematics are comparing grammatically accurate, whole Bengali words.

* `calculate_perplexity()`: Computes the model's fluency and prediction confidence using cross-entropy loss, validating that the 4-bit quantization did not degrade the model's grasp of Bengali syntax.

* `log_experiment()` & `export_human_eval_csv()`: Because "empathy" is a highly subjective human trait, automated metrics only serve as a baseline quality gate. These methods provide strict pipeline traceability. `log_experiment()` writes all system configurations, hardware parameters, and automated metrics to a persistent SQLite database (`experiments.db`). Finally, `export_human_eval_csv()` extracts the generated conversational responses side-by-side with the ground truth, outputting a structured artifact specifically formatted for secondary human-in-the-loop review.

# How to Run:
This pipeline is specifically engineered to be executed top-to-bottom in a Kaggle Notebook environment.

* **Hardware**: NVIDIA T4 x2 (Note: The pipeline uses environment variables to intentionally blind PyTorch to the second GPU, running exclusively on a single 15GB T4 to prevent 4-bit quantization mapping errors).  
    
* **Internet**: Must be enabled in the Kaggle session settings to download the Unsloth base model and required packages.

## Step-by-Step Execution
**1. Environment Setup:**  
Run the initial dependency block to install the required libraries. This installs the Unsloth Triton kernels, Hugging Face `trl/peft`, and the `bnlp_toolkit` for evaluation.

**Note**: Do not import PyTorch before setting the `CUDA_VISIBLE_DEVICES` and `PYTORCH_CUDA_ALLOC_CONF` environment variables, or the memory defragmentation failsafes will not activate.

**2. Data Pipeline & Training:**  
Execute the `DatasetProcessor` and `LLAMAFineTuner` cells.

* The pipeline will automatically load the dataset, drop missing values, and filter out sequences longer than 768 tokens.  
    
* The training loop will initiate using Early Stopping. It is configured for a maximum of 100 steps but will automatically halt and reload the best checkpoint if validation loss increases for two consecutive evaluation cycles.

**3. Evaluation & Export:**  
Run the `Evaluator` execution block. The model will generate Bengali responses for the test set and calculate the BNLP-tokenized BLEU/ROUGE scores.

**Deliverables & Artifacts**  
Once the notebook finishes execution, the pipeline automatically routes all outputs into a dynamically generated `submission_artifacts` directory. You will find the following files available for download in the Kaggle output pane:

* `adapters/`: Contains the lightweight adapter_model.safetensors LoRA weights.  
    
* `experiments.db`: The SQLite database containing the exact hyperparameter configurations, metrics, and training/validation loss curves.  
    
* `bengali_empathy_human_eval.csv`: The formatted side-by-side output for human-in-the-loop review.  
    
* `submission_artifacts.zip`: A compressed archive of the entire directory, ready for immediate deployment or assessment submission.


[VIDEO EXPLANATION](https://youtu.be/DGdCt54prII?si=1uG0pC1ZSCjwdLCd)