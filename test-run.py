from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './LLM-Research/Meta-Llama-3___1-8B-Instruct'
# model_path = './output/merged_model'

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


prompt = "嬛嬛你怎么了，朕替你打抱不平！"

# 编码输入
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 生成文本
output = model.generate(input_ids, max_new_tokens=512)

# 解码输出
response = tokenizer.decode(output[0], skip_special_tokens=True)

print('生成的文本：', response)