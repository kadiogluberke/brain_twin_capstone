from llama_cpp import Llama

llm = Llama(model_path="./qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            n_ctx = 512,
            n_threads = 8, 
            n_gpu_layers=5,
            verbose=False,
            chat_format="chatml"
            )

# chat_format="llama-2"

print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")



#### CHAT COMPLETION ##########################


# response = llm.create_chat_completion(
#       messages = [
#         {
#           "role": "system",
#           "content": "You are helpful chatbot"

#         },
#         {
#           "role": "user",
#           "content": " Name the planets in the solar system?"
#         }
#       ]
# )

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")

# print(response)

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")


messages = [
        {
          "role": "system",
          "content": "You are helpful chatbot"

        },
        {
          "role": "user",
          "content": " Name the planets in the solar system?"
        }
      ]


try:
    stream_iterator = llm.create_chat_completion( 
            messages=messages,
            max_tokens=32, 
            stream=True
            # stop=stop_sequences
        )
    
    # for chunk in stream_iterator:
    #         # ChatCompletion stream chunks have a different structure
    #         delta_content = chunk['choices'][0]['delta'].get('content', '')
    #         if not delta_content:
    #             continue # Skip empty deltas

    #         # If token usage is in the chunk (often in the last chunk for streaming)

    #         yield ChatMessageStreamDelta(
    #             content=delta_content,
    #             token_usage=SimpleNamespace(**token_usage_info)
    #         )



except Exception as e:
    print(f"Error during streaming LLM generation: {e}")
            

print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")

res = ""
for chunk in stream_iterator:
    print("#### CHUNK ####")
    print(chunk)
    text_chunk = chunk['choices'][0]['delta'].get('content', '')
    res += text_chunk

print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")

print(res)

print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")








#### TEXT COMPLETION ##########################


# output = llm(
#       "Q: Name the planets in the solar system? A: ", 
#       max_tokens=32,
#     #   stop=["Q:", "\n"], 
#       echo=True
# ) 

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")

# print(output)

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")




# try:
#     stream_iterator = llm(
#         "Q: Name the planets in the solar system? A: ", 
#         max_tokens=512,
#         #   stop=["Q:", "\n"], 
#         echo=True,
#         stream=True
#     ) 

    # for chunk in stream_iterator:
    #     text_chunk = chunk['choices'][0]['text']
    #     token_usage_info = {'output_tokens': 1}
    #     yield Resp(content_text=text_chunk, token_usage_info=token_usage_info)

# except Exception as e:
#     print(f"Error during streaming LLM generation: {e}")
            

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")

# res = ""
# for chunk in stream_iterator:
#     text_chunk = chunk['choices'][0]['text']
#     res += text_chunk


# print(res)

# print("\n\n\n\n\n ######################################################## \n\n\n\n\n\n")






