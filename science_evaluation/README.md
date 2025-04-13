





### Requirements
You can install the required packages with the following command:

```bash
pip install -r requirements.txt 
pip install vllm==v0.6.3.post1
pip install transformers==4.45.2
# pip install transformers==4.43.1
```

### Evaluation

We use the evaluation code from [FuseO1-Preview](https://github.com/fanqiwan/FuseAI/tree/main/FuseO1-Preview) and make a slight modification to support evaluation for TinyR1-32B-Preview.

You can evaluate with the following command:

```bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

python3 eval_tiny.py --model="<YOUR_MODEL_PATH>" --evals=GPQADiamond --tp=4 --temperatures 0.6 --max_tokens 32768 --repeat 4 --output_dir="<YOUR_RESULT_DIR>"

python3 eval_tiny.py --model  qihoo360/TinyR1-32B-Preview --evals=GPQADiamond --tp=4 --temperatures 0.6 --max_tokens 32768 --repeat 4 --output_dir="<YOUR_RESULT_DIR>"

python3 eval_tiny.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --evals=GPQADiamond --tp=4 --temperatures 0.6 --max_tokens 32768 --repeat 4 --output_dir="<YOUR_RESULT_DIR>"

python3 eval_tiny.py --model Qwen/QwQ-32B-Preview --evals=GPQADiamond --tp=4 --temperatures 0.6 --max_tokens 32768 --repeat 4 --output_dir="<YOUR_RESULT_DIR>"


```