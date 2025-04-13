
set -ex

export CUDA_VISIBLE_DEVICES="0,1,2,3"
prompt_type="deepseek-math-cot"
MODEL_NAME_OR_PATH=$1
echo $MODEL_NAME_OR_PATH



timestamp=$(date +"%Y%m%d%H%M%S")
EVAL_DIR="${MODEL_NAME_OR_PATH}/eval_aime/eval_${timestamp}"
mkdir -p "${EVAL_DIR}"


LOG_PATH=${EVAL_DIR}/benchmark-aime.log
OUTPUT_DIR=${EVAL_DIR}


echo "EVAL_DIR: $EVAL_DIR"
for ((seed=0; seed<16; seed++)); do
    mkdir -p "${EVAL_DIR}/seed-${seed}"
    bash ./sh/eval_aime.sh $prompt_type ${MODEL_NAME_OR_PATH} "${EVAL_DIR}/seed-${seed}" $seed >> ${LOG_PATH} 2>&1
done

