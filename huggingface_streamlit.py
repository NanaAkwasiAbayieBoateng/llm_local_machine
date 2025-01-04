import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

#Create a Streamlit app object
app = st.sidebar.title("Lmstudio Demo")

#Define the model selection options
model_options =["HuggingFaceTB/SmolLM-1.7B-Instruct","HuggingFaceTB/SmolLM-360M-Instruct","HuggingFaceTB/SmolLM-135M-Instruct"]

#Define the default model (index 2)
default_model_option = model_options[1]

#Create a select box for selecting the model
model_selectbox = st.sidebar.selectbox("Language Model", 
                                       options=model_options, index=1)

#Define the default temperature value
temperature_value = 0.5

#Create a slider for setting the temperature of the text generation
temperature_slider = st.sidebar.slider("Temperature",
                                        min_value=0.0, max_value=1.0, 
                                        step=0.05, 
                                        value=temperature_value,
                                        help="Controls the randomness of the generated text. Lower values make the model more deterministic, while higher values make it more creative.")

#Define the default seed value
seed_value = 5238

#Create a slider for setting the seed of the text generation
seed_slider = st.sidebar.slider("Seed", min_value=0, max_value=99999,
                                 step=1, value=seed_value,
                                 help="Controls the randomness of how the model selects the next tokens during text generation.")

#Define the default top p value
top_p_value = 0.75

#Create a slider for setting the top p of the text generation
top_p_slider = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0,
                                  step=0.05,
                                    value=top_p_value,
                                    help= "Top-p value, also known as nucleus sampling acts as a control knob for the level of randomness and creativity in the LLM's output")

#Define the default length value
length_value = 256

#Create a slider for setting the length of the text generation
response_token_length_slider = st.sidebar.slider("Response Token", min_value= 1,
                                   max_value=99999, step=16, 
                                   value=length_value,
                                   help="Sets the maximum number of tokens the model can generate in response to a prompt.")

#Define the default device type value
device_type = "cpu"

#Create a select box for selecting the device type of the text generation
device_selectbox = st.sidebar.selectbox("Device Type", options=["cpu", "gpu"], index=0)

#Set up the LmStudio object with selected model and seed
#lmstudio = LmStudio(model_selectbox, seed=seed_slider, temperature=temperature_slider, top_p=top_p_slider,
#device=device_type)

#Define default prompt
#prompt = "write a short blog post about lmstudio by 



#Create a text input for entering the prompt of the text generation
#text_input = st.sidebar.text_area("Prompt", value=prompt, height=100)


# Function to generate text
def generate_response(prompt,device,model,top_p,temperature,max_new_tokens):
  tokenizer = AutoTokenizer.from_pretrained(model)
  # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
  model = AutoModelForCausalLM.from_pretrained(model).to(device)
  messages = [{"role": "user", "content": prompt}]
  input_text=tokenizer.apply_chat_template(messages, tokenize=False)
  inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
  #inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
  output = model.generate(inputs, temperature=temperature, top_p=top_p, max_new_tokens=max_new_tokens,do_sample=True)
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
  return generated_text



# checkpoint = models[0]

# device = "cpu" # for GPU usage or "cpu" for CPU usage
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# # for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
# model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# messages = [{"role": "user", "content": "What is the capital of the united states of america?"}]
# input_text=tokenizer.apply_chat_template(messages, tokenize=False)
# print(input_text)
# inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
# outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
# print(tokenizer.decode(outputs[0]))


# prompt = "What is the capital of the united states of america?" 
# device = "cpu"
# model = models[0]
# temperature=0.2
# top_p=0.9
# max_new_tokens= 50
# generate_text(prompt,device,model,temperature,top_p,max_new_tokens)

def main():
    st.title("Huggingface LLM App")

    # User input field
    user_input = st.text_input("Enter your prompt:")

    # Button to trigger generation
    if st.button("Generate"):
        if user_input:
            #response = generate_response(user_input)
            response = generate_response(prompt=user_input,
                                         device=device_selectbox,
                                         model=model_selectbox,
                                         top_p=top_p_slider,
                                         temperature=temperature_slider,
                                         max_new_tokens=response_token_length_slider)
            st.write("Model Response:")
            st.write(response)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()