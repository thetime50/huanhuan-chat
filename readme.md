https://github.com/KMnO4-zx/huanhuan-chat
conda create --name huanhuan-chat python=3.12 pytorch cudatoolkit transformers accelerate peft datasets
pip install modelscope

conda create --name huanhuan-chat2 python=3.12 pytorch=2.3.0 cudatoolkit transformers=4.43.1 accelerate=0.32.1 peft=0.11.1 datasets=2.20.0 pytorch-cuda=12.1
conda install pytorch==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
// torchvision==0.18.0 torchaudio==2.3.0 
pip install modelscope==1.16.1
https://pytorch.org/get-started/previous-versions/


https://momodel.cn/ 
https://colab.research.google.com/drive/1jBes6EYQHXdO_7cLuf8FZ0oRUN0sPcgX
版本不太兼容

https://momodel.cn/workspace/67c456983a290e5e6c56b13f/app
显存不足

https://gpushare.com/
https://mistgpu.com/signup/
https://www.autodl.com/login
https://mistgpu.com/signup/


transformers Trainer 断点训练

https://blog.csdn.net/spiderwower/article/details/138755776
https://zhuanlan.zhihu.com/p/715841536


python convert_hf_to_gguf.py ../huanhuan-chat\output\merged_model --outtype q8_0  --outfile ../huanhuan-chat\output\result\huan.gguf

ollama create hua --file .\ModelFile

python convert_hf_to_gguf.py ../huanhuan-chat\LLM-Research\Meta-Llama-3___1-8B-Instruct --outtype q8_0  --outfile ../huanhuan-chat\output\result\llama.gguf


ollama create huan --file .\ModelFile
ollama run huan
如果出现了Error: invalid file magic的错误， 大概率是这个gguf文件中的某些操作ollama还不支持， 如有些特殊的量化操作等。

