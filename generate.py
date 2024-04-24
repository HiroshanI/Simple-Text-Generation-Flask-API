def generate_text(model, tokenizer, input_message, n_tokens):
    input_encoded = tokenizer.encode(input_message, return_tensors='pt')
    MAX_LEN = n_tokens
    outputs = model.generate(
        input_ids = input_encoded, 
        max_length = MAX_LEN, 
        do_sample=True,
        temperature=.8,
        top_k=50,
        top_p=.85
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)