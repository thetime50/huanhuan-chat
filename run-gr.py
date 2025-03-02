from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import gradio as gr

# 设置路径
model_path = './LLM-Research/Meta-Llama-3___1-8B-Instruct'
lora_path = './output/llama3_1_instruct_lora/checkpoint-699'  # 修改为你的 lora 输出对应 checkpoint 地址

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

# 加载 lora 权重
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 定义生成对话的函数
def generate_response(user_input):
    prompt = user_input
    messages = [
        {"role": "system", "content": "假设你是皇帝身边的女人--甄嬛。"},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

# 创建 Gradio 接口
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.inputs.Textbox(label="输入你的问题"),
    outputs=gr.outputs.Textbox(label="嬛嬛的回复"),
    title="甄嬛对话模型",
    description="与甄嬛进行对话，输入你的问题，看看她会怎么回答！"
)

# 启动 Gradio 应用
iface.launch()

