from vllm import LLM, SamplingParams

model = LLM(model_path= r"C:/Users/nboateng/OneDrive - Nice Systems Ltd/Documents/Research/LLM/llm_local_machine/llama-7b.Q2_K.gguf", 
            device="cpu", 
            max_input_length=512, 
            max_output_length=256)

output = model.generate(
    inputs=["This is an example input."],
    n=1,  # Number of generations
    stop=None,  # Stop criteria
    stream=False  # Stream output (optional)
)


print(output.texts[0]) 