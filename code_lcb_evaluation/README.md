### Requirements
You can install the required packages with the following command:

```bash
pip install -r requirements.txt 
pip install vllm==v0.6.3.post1
pip install transformers==4.45.2
```

### Dataset

```bash
huggingface-cli download --repo-type dataset livecodebench/code_generation_lite --local-dir code_generation_lite
```

### Evaluation

We use the evaluation code from [FuseO1-Preview](https://github.com/fanqiwan/FuseAI/tree/main/FuseO1-Preview) and make a slight modification to support evaluation for TinyR1-32B-Preview.

You can evaluate with the following command:

```bash
bash ./sh/evaluate_ds_tiny.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-32B ./lcb_results/DeepSeek-R1-Distill-Qwen-32B

bash ./sh/evaluate_ds_tiny.sh "<YOUR_MODEL_PATH>"
```