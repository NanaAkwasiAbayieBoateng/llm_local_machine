import streamlit as st
from ollama import chat
from ollama import ChatResponse






def generate_response(prompt,model):
    try:
        response: ChatResponse = chat(model, messages=[
        {
         'role': 'user',
          'content': prompt},
          ])
    
    except Exception as e:
        print(f"An error occurred during response generation: {e}")
    
    return response['message']['content']


model_selected = st.selectbox('Select model', ['llama3.2:1b', 'gemma2:2b'])

#generate_response(prompt='why is the sky blue ?')

def main():
    st.title("Ollama LLM App")

    # User input field
    user_input = st.text_input("Enter your prompt:")

    # Button to trigger generation
    if st.button("Generate"):
        if user_input:
            #response = generate_response(user_input)
            response = generate_response(prompt=user_input,model=model_selected)
            st.write("Model Response:")
            st.write(response)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()