

#!/bin/bash
techo() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $@"
}
export DS_SKIP_CUDA_CHECK=1
export DISABLE_VERSION_CHECK=1



model=""
data_id_path=""
output_dir=""
model_max_length=""
learning_rate=""
num_train_epochs="10"
save_steps=1000
gradient_accumulation_steps="1"
lr_scheduler_type="constant_with_warmup"  
packing_type="neat_packing"
template=""
hostfile=""

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift ;;
    --data-id-path) data_id_path="$2"; shift ;;
    --output-dir) output_dir="$2"; shift ;;
    --model-max-length) model_max_length="$2"; shift ;;
    --learning-rate) learning_rate="$2"; shift ;;
    --num-train-epochs) num_train_epochs="$2"; shift ;;
    --save-steps) save_steps="$2"; shift ;;
    --gradient-accumulation-steps) gradient_accumulation_steps="$2"; shift ;;
    --lr-scheduler-type) lr_scheduler_type="$2"; shift ;; 
    --weight_decay) weight_decay="$2"; shift ;;
    --warmup_ratio) warmup_ratio="$2"; shift ;;
    --packing_type) packing_type="$2"; shift ;;
    --template) template="$2"; shift ;;
    --hostfile) hostfile="$2"; shift ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

[ ! -d "$output_dir" ] && mkdir -p "$output_dir"

log_file="${output_dir}/train_$(date +'%Y%m%d_%H%M%S').log"


exec > >(tee -a "$log_file") 2>&1


if [ ! -f "$hostfile" ] || [ -z "$hostfile" ]; then
  hostfile_param=""
else
  hostfile_param="--hostfile=$hostfile"
fi

techo "开始运行"





if [ -z "$model" ] || [ -z "$data_id_path" ] || [ -z "$output_dir" ] || [ -z "$model_max_length" ] || [ -z "${template}" ]; then
  techo "Error: Missing required arguments"
  exit 1
fi








techo "dataset_dir: $(dirname "$data_id_path")"
techo "dataset: $(basename "$data_id_path" | sed 's/\.[^.]*$//')"

packing_params=""
if [[ "$packing_type" =~ ^(neat-packing|neatpacking|neat_packing)$ ]]; then
  packing_params="--neat_packing True --packing True"
elif [ "$packing_type" == "packing" ]; then
  packing_params="--packing True"
fi

techo "packing_params: $packing_params"


cmd="deepspeed $hostfile_param 360-LLaMA-Factory/src/train.py \
  --stage sft \
  --do_train \
  --model_name_or_path \"$model\" \
  --dataset_dir \"$(dirname "$data_id_path")\" \
  --dataset \"$(basename "$data_id_path" | sed 's/\.[^.]*$//')\" \
  --template \"$template\" \
  --finetuning_type full \
  --output_dir \"${output_dir}\" \
  --cache_dir .cache \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len \"$model_max_length\" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps \"$gradient_accumulation_steps\" \
  --lr_scheduler_type \"$lr_scheduler_type\" \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 1 \
  --save_strategy "steps" \
  --save_steps \"$save_steps\" \
  --learning_rate \"$learning_rate\" \
  --num_train_epochs \"$num_train_epochs\" \
  --plot_loss \
  --save_only_model True \
  --deepspeed 360-LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
  --bf16 True \
  --flash_attn fa2 \
  --gradient_checkpointing True \
  --seed 42 \
  --sequence_parallel_size 1 \
  --preprocessing_num_workers 64 \
  --enable_liger_kernel true \
  $packing_params "





  
set -o pipefail
eval "$cmd" 2>&1 | awk '{ print strftime("[%Y-%m-%d %H:%M:%S]") , $0; fflush(); }' | tee -a ${log_file}

# 捕获运行状态码
exit_status=$?
techo "运行状态码: $exit_status"
techo "训练完成，日志文件位于: $log_file"

