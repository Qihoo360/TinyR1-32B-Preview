import json
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from vllm import LLM, SamplingParams
from tqdm import tqdm
from util.task_handlers import *
from util.model_utils import *
from openai import OpenAI
import concurrent.futures
from functools import partial

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def fetch_response_openai(llm, model_name, max_tokens, temp, prompt):
    model_name = model_name.replace("openai/", "")
    if "o1" in model_name:
        # O1 doesn't support system prompt
        # NOTE: might want to implement this inside handler instead 
        for p in prompt:
            p["role"] = "user"
        
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=1, # has to be 1
            max_completion_tokens=max_tokens 
        )
    else:
        response = llm.chat.completions.create(
            model=model_name,
            messages=prompt,
            n=1,
            temperature=temp,
            max_tokens=max_tokens
        )
    return response

def perform_inference_and_check(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args):
    # results = handler.load_existing_results(result_file)
    results={}
    print(f"Loaded {len(results)} existing results.")
    
    train_data = handler.load_and_filter_dataset(
        args.start, 
        args.end, 
        split=args.split, 
        source=args.source,
        filter_difficulty=args.filter_difficulty, 
        args=args
    )
    remaining_data = handler.process_remaining_data(train_data, results)
    conversations = handler.make_conversations(remaining_data, system_prompt, args.model)

    for temp in temperatures:
        # 根据模型类型获得推理结果
        if args.model.startswith("openai"):
            fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                responses = list(e.map(fetch_partial, conversations))
        else:
            # 一些特定的 model 需要特殊的 stop_token_ids 或 sampling_params
            if any(substring in args.model for substring in [
                "DeepSeek-R1-Distill-Qwen-32B",
                "FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview",
                "FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview",
                "FuseO1-DeepSeekR1-QwQ-32B-Preview",
                "FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview",
                "R1-Distill-Qwen"
            ]):
                # print(["\nUsing vLLM model, n=", args.n])
                sampling_params = SamplingParams(
                    n=args.n, 
                    max_tokens=max_tokens, 
                    top_p=0.95, 
                    seed=args.seed, 
                    temperature=temp, 
                    stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], 
                    stop_token_ids=[151645, 151643]
                )
            else:
                sampling_params = SamplingParams(
                    n=args.n, 
                    max_tokens=max_tokens, 
                    top_p=0.95, 
                    seed=args.seed, 
                    temperature=temp, 
                    stop=["<|im_end|>", "<|endoftext|>"], 
                    stop_token_ids=[151645, 151643]
                )
            responses = llm.chat(
                messages=conversations, 
                sampling_params=sampling_params, 
                use_tqdm=True
            )
        
        # 用来统计所有生成的正确率（默认：把每个生成都视作一次判断）
        total_correct = 0 
        total_finish = 0

        # 用来并行处理的容器
        future_to_task = {}
        # 用来记录 token 用量：key=(idx, choice_idx)
        token_usages = {}

        # 开启进程池，异步处理所有生成
        with ProcessPoolExecutor(max_workers=32) as executor:
            for idx, response in enumerate(responses):
                # ----- 情况 1：OpenAI 返回多个 choices -----
                if args.model.startswith("openai"):
                    # 针对每个 choice，都提交一个任务进行评测
                    for c_idx, choice in enumerate(response.choices):
                        response_str = choice.message.content.strip()
                        f = executor.submit(handler.update_results, remaining_data[idx], response_str)
                        future_to_task[f] = (idx, c_idx)
                    
                    # OpenAI 的 token usage 通常对整个请求一次性统计（而不是每个 choice 单独）
                    # 如果你想把它平分/重复记录都可以，这里仅作简单记录
                    for c_idx in range(len(response.choices)):
                        token_usages[(idx, c_idx)] = {
                            "completion_tokens": response.usage.completion_tokens,
                            "prompt_tokens": response.usage.prompt_tokens
                        }

                # ----- 情况 2：vLLM 返回多个 outputs -----
                else:
                    for c_idx, output in enumerate(response.outputs):
                        response_str = output.text.strip()
                        f = executor.submit(handler.update_results, remaining_data[idx], response_str)
                        future_to_task[f] = (idx, c_idx)
                        
                        # vLLM 每个 output 通常可以得到单独的 token_ids
                        token_usages[(idx, c_idx)] = {
                            "completion_tokens": len(output.token_ids),
                            "prompt_tokens": len(response.prompt_token_ids)
                        }

            # 等待所有任务完成，并存储结果
            for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
                idx, c_idx = future_to_task[future]
                response_entry = future.result()
                
                # 统计正确率（把每个生成视作一次独立判断）
                total_correct += response_entry["correctness"]
                total_finish += 1

                # 将结果存进 results
                problem_key = remaining_data[idx][handler.get_question_key()]
                if problem_key not in results:
                    # 初始化此问题的结果结构
                    results[problem_key] = remaining_data[idx]
                    if isinstance(handler, NUMINATaskHandler):
                        results[problem_key]["messages"] = ""
                    results[problem_key]["responses"] = {}
                    results[problem_key]["token_usages"] = {}
                    # 存储 prompt 方便分析
                    prompt = conversations[idx][1]["content"]
                    # results[problem_key]["prompt"] = prompt
                    results[problem_key]["prompt"] = conversations[idx]
                    results[problem_key]["output_token_ids"] = output.token_ids

                # 把每个 temperature 下的所有回答存成一个 list
                if str(temp) not in results[problem_key]["responses"]:
                    results[problem_key]["responses"][str(temp)] = []
                results[problem_key]["responses"][str(temp)].append(response_entry)

                # 同理，token 用量也存为 list
                if str(temp) not in results[problem_key]["token_usages"]:
                    results[problem_key]["token_usages"][str(temp)] = []
                results[problem_key]["token_usages"][str(temp)].append(token_usages[(idx, c_idx)])
        
        # 最后输出本轮温度下的整体正确率
        print(f"Final acc: {total_correct}/{total_finish}")
        acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
        print(json.dumps({"acc": acc}))

    # --------------------------------------------------
    # 汇总所有 temperatures 的 token 用量并输出到文件
    completion_tokens = []
    prompt_tokens = []
    for key in results:
        for temp in temperatures:
            usage_list = results[key].get("token_usages", {}).get(str(temp), [])
            for usage in usage_list:
                completion_tokens.append(usage.get("completion_tokens", 0))
                prompt_tokens.append(usage.get("prompt_tokens", 0))

    # Token usage summary
    result_dir, result_name = os.path.split(result_file)
    token_usage_dir = os.path.join(result_dir, "token_usage")
    os.makedirs(token_usage_dir, exist_ok=True)

    token_usage_result_file = os.path.join(token_usage_dir, result_name)
    token_dict = {
        "completion_tokens": sum(completion_tokens),
        "prompt_tokens": sum(prompt_tokens),
        "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
        "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
    }

    with open(token_usage_result_file, "w") as f:
        json.dump(token_dict, f, indent=4)

    print(f"Token usage saved to {token_usage_result_file}")

    # 保存所有结果（包含多路回答）到原来的 result_file
    with open(result_file, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

# def perform_inference_and_check(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args):
#     results = handler.load_existing_results(result_file)
#     print(f"Loaded {len(results)} existing results.")
#     train_data = handler.load_and_filter_dataset(args.start, args.end, split=args.split, source=args.source, \
#                                                  filter_difficulty=args.filter_difficulty, args=args)
#     remaining_data = handler.process_remaining_data(train_data, results)
#     conversations = handler.make_conversations(remaining_data, system_prompt, args.model)

#     for temp in temperatures:
        
#         if args.model.startswith("openai"):
#             fetch_partial = partial(fetch_response_openai, llm, args.model, max_tokens, temp)

#             with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
#                 responses = list(e.map(fetch_partial, conversations))

#         else:
#             # if args.model in ["DeepSeek-R1-Distill-Qwen-32B", "FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview", "FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview", "FuseO1-DeepSeekR1-QwQ-32B-Preview", "FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview", "R1-Distill-Qwen"]:
#             if any(substring in args.model for substring in [
#                     "DeepSeek-R1-Distill-Qwen-32B",
#                     "FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview",
#                     "FuseO1-DeepSeekR1-QwQ-SkyT1-Flash-32B-Preview",
#                     "FuseO1-DeepSeekR1-QwQ-32B-Preview",
#                     "FuseO1-DeepSeekR1-Qwen2.5-Instruct-32B-Preview",
#                     "R1-Distill-Qwen"
#                 ]):
#                 print(["\nUsing vLLM model, n=", args.n]*10)
#                 sampling_params = SamplingParams(n=args.n, max_tokens=max_tokens, top_p=0.95, seed=0, temperature=temp, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643])
#             else:
#                 sampling_params = SamplingParams(n=args.n, max_tokens=max_tokens, top_p=0.95, seed=0, temperature=temp, stop=["<|im_end|>", "<|endoftext|>"], stop_token_ids=[151645, 151643])
#             responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
            
#         total_correct = 0 
#         total_finish = 0
#         with ProcessPoolExecutor(max_workers=32) as executor:
#             # future_to_task = {
#             #     executor.submit(handler.update_results, remaining_data[idx], response): idx
#             #     for idx, response in enumerate(responses)
#             # }
#             future_to_task = {}
#             token_usages = {}
#             for idx, response in enumerate(responses):
#                 if args.model.startswith("openai"):
#                     response_str = response.choices[0].message.content.strip()
#                 else:
#                     response_str = response.outputs[0].text.strip()
#                 future_to_task[executor.submit(handler.update_results, remaining_data[idx], response_str)] = idx
#                 # print(f"Request output: {response}")
                
#                 if args.model.startswith("openai"):
#                     token_usages[idx] = response.usage
#                 else:
#                     token_usages[idx] = {
#                         "completion_tokens": len(response.outputs[0].token_ids),
#                         "prompt_tokens": len(response.prompt_token_ids)
#                     }

#             for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Processing Generations"):
#                 idx = future_to_task[future]
#                 response_entry = future.result()
#                 total_correct += response_entry["correctness"]
#                 total_finish += 1

#                 problem_key = remaining_data[idx][handler.get_question_key()]
#                 if problem_key not in results:
#                     results[problem_key] = remaining_data[idx]
#                     if isinstance(handler, NUMINATaskHandler):
#                         results[problem_key]["messages"] = ""
#                     results[problem_key]["responses"] = {}
#                     results[problem_key]["token_usages"] = {}
#                     prompt = conversations[idx][1]["content"]
#                     results[problem_key]["prompt"] = prompt

#                 results[problem_key]["responses"][str(temp)] = response_entry
                
#                 if args.model.startswith("openai"):
#                     results[problem_key]["token_usages"][str(temp)] = {
#                         "completion_tokens": token_usages[idx].completion_tokens,
#                         "prompt_tokens": token_usages[idx].prompt_tokens,
#                     }
#                 else:
#                     # TODO: vLLM model, can it do the same thing
#                     results[problem_key]["token_usages"][str(temp)] = token_usages[idx] 
        
#         print(f"Final acc: {total_correct}/{total_finish}")
#         acc = round(total_correct / total_finish, 4) if total_finish > 0 else 0
#         print(json.dumps({"acc": acc}))

#     completion_tokens = [
#         results[key].get("token_usages", {}).get(str(temp), {}).get("completion_tokens", 0)
#         for key in results for temp in temperatures
#     ]
#     prompt_tokens = [
#         results[key].get("token_usages", {}).get(str(temp), {}).get("prompt_tokens", 0)
#         for key in results for temp in temperatures
#     ]

#     # Token usage summary
#     result_dir, result_name = os.path.split(result_file)
#     token_usage_dir = os.path.join(result_dir, "token_usage")
#     os.makedirs(token_usage_dir, exist_ok=True)

#     # Construct the token usage result file path
#     token_usage_result_file = os.path.join(token_usage_dir, result_name)

#     # Prepare the token usage dictionary
#     token_dict = {
#         "completion_tokens": sum(completion_tokens),
#         "prompt_tokens": sum(prompt_tokens),
#         "avg_completion_tokens": round(sum(completion_tokens) / len(completion_tokens), 3) if completion_tokens else 0,
#         "avg_prompt_tokens": round(sum(prompt_tokens) / len(prompt_tokens), 3) if prompt_tokens else 0,
#     }

#     # Save the token usage dictionary to the result file
#     with open(token_usage_result_file, "w") as f:
#         json.dump(token_dict, f, indent=4)

#     print(f"Token usage saved to {token_usage_result_file}")
    
#     with open(result_file, 'w', encoding='utf-8') as file:
#         json.dump(results, file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

def perform_check(handler: TaskHandler, temperatures, result_file, args):
    results = handler.load_existing_results(result_file)

def perform_inference_and_save(handler: TaskHandler, temperatures, max_tokens, result_file, llm, system_prompt, args):
    results = handler.load_existing_results(result_file)
    

def main():
    parser = argparse.ArgumentParser(description="Unified inference and checking for different datasets/tasks.")
    parser.add_argument("--dataset", type=str, required=True, choices=["NUMINA", "APPS", "TACO", "MATH500", "AIME", "GPQADiamond", "MMLU", "MMLUPro", "LiveCodeBench", "GSM8K", "ARC-C"], help="Dataset to process.")
    parser.add_argument("--model", type=str, required=True, default="Qwen/QwQ-32B-Preview", help="The model to run.")
    parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
    parser.add_argument("--split", type=str, default="train", help="Split to use for apps (e.g., train, test).")
    parser.add_argument("--source", type=str, help="Source for the dataset.")
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=-1, help="End index.")
    parser.add_argument("--filter-difficulty", action="store_true", help="Filter difficulty.")
    parser.add_argument("--result-dir", type=str, default="./", help="Result dir to save files.")
    parser.add_argument("--check", action="store_true", help="Perform evaluation checks on generated samples.")
    parser.add_argument("--inference", action="store_true", help="Perform inference.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0], help="Temperature for sampling.")
    parser.add_argument("--math-difficulty-lower-bound", type=int, default=None, help="Lowest difficulty level for math.")
    parser.add_argument("--math-difficulty-upper-bound", type=int, default=None, help="Highest difficulty level for math.")
    parser.add_argument("--n", type=int, default=1, help="Number of samples generated per problem.")
    parser.add_argument("--result_dir", required=True, type=str, help="Output dir to write results to.")
    parser.add_argument("--seed", required=False, type=int, default=0, help="seed.")
    args = parser.parse_args()
    
    handler: TaskHandler = TASK_HANDLERS[args.dataset]()
    temperatures = [1] if args.model.startswith("openai/o1") else args.temperatures 
    
    print(f"Temperature: {temperatures}")
    max_tokens = args.max_tokens
    if temperatures == [0] and args.n > 1:
        args.n = 1
        print("Warning: Temperature 0 does not support multiple samples. Setting n=1.")
    print(" args.n=", args.n)

    # create result dir if not exists
    if args.result_dir and not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    if args.math_difficulty_lower_bound is not None or  args.math_difficulty_upper_bound is not None:
        result_file = os.path.join(args.result_dir, f"{MODEL_TO_NAME.get(args.model, 'fuse')}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}_{args.math_difficulty_lower_bound}_{args.math_difficulty_upper_bound}_seed{args.seed}.json")
    else:
        result_file = os.path.join(args.result_dir, f"{MODEL_TO_NAME.get(args.model, 'fuse')}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}_seed{args.seed}.json")

    if args.check:
        # check if converted file exists
        if args.math_difficulty_lower_bound is not None or args.math_difficulty_upper_bound is not None:
            converted_file = f"{args.result_dir}/converted_{MODEL_TO_NAME.get(args.model, 'fuse')}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}_{args.math_difficulty_lower_bound}_{args.math_difficulty_upper_bound}.json"
        else:
            converted_file = f"{args.result_dir}/converted_{MODEL_TO_NAME.get(args.model, 'fuse')}_{args.dataset}_{args.split}_{args.source}_{args.start}_{args.end}.json"
        if os.path.exists(converted_file):
            result_file = converted_file
        perform_check(handler, temperatures, result_file, args)
        return
    elif args.inference:
        llm = OpenAI() if args.model.startswith("openai") else LLM(model=args.model, tensor_parallel_size=args.tp)
        system_prompt = SYSTEM_PROMPT.get(args.model, SYSTEM_PROMPT["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"])
        perform_inference_and_save(handler, temperatures, max_tokens, result_file, llm, system_prompt, args)
        return

    llm = OpenAI() if args.model.startswith("openai") else LLM(model=args.model, tensor_parallel_size=args.tp)
    system_prompt = SYSTEM_PROMPT.get(args.model, SYSTEM_PROMPT["deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"])
    perform_inference_and_check(handler, temperatures, max_tokens, result_file, llm, system_prompt, args)

if __name__ == "__main__":
    main()
