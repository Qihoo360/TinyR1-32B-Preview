### Requirements
You can install the required packages with the following command:

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

### Evaluation

You can evaluate with the following command:

```bash
set -ex

export CUDA_VISIBLE_DEVICES="0,1,2,3"
prompt_type="deepseek-math-cot"
MODEL_NAME_OR_PATH="<YOUR_MODEL_PATH>"

timestamp=$(date +"%Y%m%d%H%M%S")
EVAL_DIR="${MODEL_NAME_OR_PATH}/eval_aime/eval_${timestamp}"
mkdir -p "${EVAL_DIR}"

LOG_PATH=${EVAL_DIR}/benchmark-aime.log
OUTPUT_DIR=${EVAL_DIR}

echo "EVAL_DIR: $EVAL_DIR"
for ((seed=0; seed<32; seed++)); do
    mkdir -p "${EVAL_DIR}/seed-${seed}"
    bash ./sh/eval_aime.sh $prompt_type ${MODEL_NAME_OR_PATH} "${EVAL_DIR}/seed-${seed}" $seed >> ${LOG_PATH} 2>&1
done

```

The commands mentioned above have been integrated into the run-benchmark.sh script. You can also execute the script directly by running:
```bash
sh ./sh/run_benchmark_aime.sh <MODEL_NAME_OR_PATH>

```
`<MODEL_NAME_OR_PATH>` is a mandatory parameter that must be replaced with the path to your model.