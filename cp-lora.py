# 这个文件需要在云上跑

from transformers import AutoModelForSequenceClassification,AutoTokenizer
import os
print("开始转换模型")
 
# 需要保存的lora路径
lora_path= os.path.abspath("./output/rola")
# 模型路径
model_path = os.path.abspath('./LLM-Research/Meta-Llama-3___1-8B-Instruct')
# 检查点路径
checkpoint_dir = os.path.abspath('./output/llama3_1_instruct_lora/checkpoint-699')
print("11")
# checkpoint = [file for file in os.listdir(checkpoint_dir) if 'checkpoint-' in file][-1] #选择更新日期最新的检查点
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
print("22")

# 保存模型
model.save_pretrained(lora_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
# 保存tokenizer
tokenizer.save_pretrained(lora_path)
print("转换完成")