# python3 evalv0220.py --model "${dir%/}" --evals=GPQADiamond --tp=4 --temperatures 0.6 --max_tokens 32768

import argparse
import subprocess
import os
import json
from datetime import datetime
import numpy as np

# Define eval to split mapping
eval_to_split = {
  "MATH500": "test", 
  "AIME": "train", 
  "GPQADiamond": "train", 
  "MMLU": "test",
  "MMLUPro": "test",
  "LiveCodeBench": "test",
  "GSM8K": "test",
  "ARC-C": "test",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process model path, prompt format, and evals to run.")
    parser.add_argument("--model", required=True, type=str, help="Path to the model.")
    parser.add_argument("--evals", required=True, type=str, help="Comma-separated list of evals to run (no spaces).")
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument("--filter-difficulty", action="store_true", help="Filter difficulty.")
    parser.add_argument("--source", type=str, help="Source for the dataset.")
    # parser.add_argument("--output_file", required=True, type=str, help="Output file to write results to.")
    parser.add_argument("--output_dir", required=False, type=str, help="Output dir to write results to.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0], help="Temperature for sampling.")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
    parser.add_argument("--startseed", required=False, type=int, default=0, help="seed.")
    parser.add_argument("--repeat", required=False, type=int, default=1, help="repeat times.")
    return parser.parse_args()

def extract_accuracy_from_output(output):
    # Iterate through all lines from the end to the beginning
    lines = output.splitlines()[::-1]
    for line in lines:
        try:
            # Attempt to parse a JSON object from the line
            data = json.loads(line.replace("'", '"'))
            if "acc" in data:
                return data["acc"]
        except json.JSONDecodeError:
            continue 
    return None

def write_logs_to_file(logs, output_file):
    try:
        with open(output_file, "w") as file:
            file.write(logs)
        print(f"Logs successfully written to {output_file}")
    except IOError as e:
        print(f"Failed to write logs to file {output_file}: {e}")

def main():
    args = parse_arguments()

    # Extract the arguments
    model_path = args.model
    evals = args.evals.split(",")

    # 获取当前时间戳
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # 拼接 output_dir
    if args.output_dir:
        # 合并 model 的 basename
        output_dir = args.output_dir + "/" + os.path.basename(args.model) + "/eval-" + timestamp
    else:
        output_dir = f"{args.model}/eval-{timestamp}"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    output_file = output_dir + "/log.txt"
    tp = args.tp
    temperatures = [str(t) for t in args.temperatures]

    script_path = "inference_and_checkv0220.py"

    # Hold all logs 
    all_logs = ""
    results = {}
        
    # Run the Python command for each eval and collect logs
    startseed = args.startseed
    for eval_name in evals:
        results[eval_name] = {}
        results[eval_name]["score_list"] = [-1] * args.repeat
        for r in range(args.repeat):
            command = [
                "python", script_path, 
                "--model", model_path, 
                "--dataset", eval_name, 
                "--split", eval_to_split[eval_name], 
                "--tp", str(tp),
                "--result_dir", output_dir,
                "--max_tokens", str(args.max_tokens),
                "--seed", str(startseed + r),
                "--temperatures",
            ]
            command.extend(temperatures)  # 将 temperatures 加入到参数中
            if args.filter_difficulty:
                assert args.source != "", "No source passed for filtering difficulty."
                command.append("--filter-difficulty")
                command.append("--source")
                command.append(args.source)
            print(f"Running eval {eval_name} with command {command}")
            all_logs += f"\nRunning eval: {eval_name} with command {command}\n"
            try:
                with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
                    output_lines = []
                    for line in proc.stdout:
                        print(line, end="")  # 实时输出到控制台
                        output_lines.append(line)
                        all_logs += line
                    proc.wait()
                    if proc.returncode != 0:
                        raise subprocess.CalledProcessError(proc.returncode, command)

                    # 捕获输出用于后续处理
                    output = "".join(output_lines)
                    accuracy = extract_accuracy_from_output(output)
                    results[eval_name]["score_list"][r] = accuracy

            except subprocess.CalledProcessError as e:
                error_message = f"Error occurred while running eval {eval_name}: {e}\n"
                print(error_message)
                all_logs += error_message

        # 计算各项指标，并转换 numpy 数据为 Python 内置类型
        avg_score = sum(results[eval_name]["score_list"]) / args.repeat
        std_score = float(np.std(results[eval_name]["score_list"]))
        min_score = min(results[eval_name]["score_list"])
        max_score = max(results[eval_name]["score_list"])

        results[eval_name]["Pass1@Ave" + str(args.repeat)] = avg_score
        results[eval_name]["Pass1@Ave"] = avg_score
        results[eval_name]["Pass1@Std" + str(args.repeat)] = std_score
        results[eval_name]["Pass1@Std"] = std_score
        results[eval_name]["Pass1@Min" + str(args.repeat)] = min_score
        results[eval_name]["Pass1@Min"] = min_score
        results[eval_name]["Pass1@Max" + str(args.repeat)] = max_score
        results[eval_name]["Pass1@Max"] = max_score

    # 将所有 stdout / stderr 的日志写入文件
    write_logs_to_file(all_logs, output_file)

    results_file = output_dir + "/result_summary_one_ckpt.json"
    print("results_file:", results_file)
    with open(results_file, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == "__main__":
    main()
