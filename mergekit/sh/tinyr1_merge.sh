#!/bin/bash





MODEL1=$1
MODEL2=$2
MODEL3=$3
OUTPUT_DIR=$4
MERGE_METHOD="arcee_fusion"
DTYPE="bfloat16"
OUT_DTYPE="bfloat16"
FUSE_NAME="tinyr1_32b_preview_fusion"

OUTPUT_DIR="$OUTPUT_DIR/$FUSE_NAME"
mkdir -p $OUTPUT_DIR

# 定义 YAML 文件路径
YAML_FILE1="./tinyr1_config/"$FUSE_NAME"_config1.yml"

# 写入内容到 YAML 文件
cat <<EOF > "$YAML_FILE1"
models:
  - model: $MODEL1
  - model: $MODEL2
merge_method: $MERGE_METHOD
base_model: $MODEL2
dtype: $DTYPE
out_dtype: $OUT_DTYPE
EOF

echo "YAML 文件1已生成：$YAML_FILE1"



MID_MODEL_DIR=""$OUTPUT_DIR"_Mid_Model"
mkdir -p $MID_MODEL_DIR

echo "Merging first model "
mergekit-yaml "$YAML_FILE1" "$MID_MODEL_DIR" 


# 定义第二个Fuse YAML 文件路径
YAML_FILE2="./tinyr1_config/"$FUSE_NAME"_config2.yml"

# 写入内容到 YAML 文件
cat <<EOF > "$YAML_FILE2"
models:
- model: $MID_MODEL_DIR
- model: $MODEL3
merge_method: $MERGE_METHOD
base_model: $MODEL3
dtype: $DTYPE
out_dtype: $OUT_DTYPE
EOF

echo "YAML 文件2已生成：$YAML_FILE2"


echo "Merging second model"
mergekit-yaml "$YAML_FILE2" "$OUTPUT_DIR" 

echo "Finish Merge!"


