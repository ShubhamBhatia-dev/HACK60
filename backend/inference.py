from llama_cpp import Llama
from config import phi_prompt


# test file 



llm = None

def get_llm():
    global llm
    if llm is None:
        llm = Llama(
	model_path="./models/phi_model.gguf",
    n_gpu_layers=-1,
    verbose=False ,
    n_threads = 8 ,
    n_batch=512)
    return llm



def chatWithMe(data) :
    llm = get_llm()
    prompt = phi_prompt(data)
    response = llm(prompt=prompt ,stop=["<|end|>", "<|user|>"] , echo=False , max_tokens=200,temperature=0.3)

    return response["choices"][0]["text"].strip()

