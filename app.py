from flask import Flask, request, render_template
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer
import jax.numpy as jnp

app = Flask(__name__)

# Path to the local tokenizer directory
LOCAL_TOKENIZER_PATH = r"C:\Users\Shruti Rathod\OneDrive\Desktop\smart-recipe"

# Path to the local model directory
LOCAL_MODEL_PATH = r"C:\Users\Shruti Rathod\OneDrive\Desktop\smart-recipe"

# Load tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH, use_fast=True)

# Load model from the local directory
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)

prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)
        for k, v in tokens_map.items():
            text = text.replace(k, v)
        new_texts.append(text)

    return new_texts

def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = tokenizer(
        _inputs, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="jax"
    )

    input_ids = jnp.array(inputs.input_ids)  # Ensure JAX tensor compatibility
    attention_mask = jnp.array(inputs.attention_mask)

    output = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        **generation_kwargs
    )

    if isinstance(output, tuple):  # Handling different return types
        output_ids = output[0]  # Take the first element if it's a tuple
    else:
        output_ids = output.sequences if hasattr(output, "sequences") else output

    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(output_ids, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    data = request.form
    items = [data.get('ingredients', '')]

    # Debugging prints
    print("Items received:", items)
    print("Type of items:", type(items))

    if hasattr(items, "shape"):
        print("Shape of items:", items.shape)
    else:
        print("Items does not have a shape attribute.")

    generated = generation_function(items)
    recipe_sections = []

    for text in generated:
        sections = text.split("\n")
        headline = None  # Initialize to prevent UnboundLocalError

        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                section = section.replace("title:", "").strip()
                headline = "TITLE"
            elif section.startswith("ingredients:"):
                section = section.replace("ingredients:", "").strip()
                headline = "INGREDIENTS"
            elif section.startswith("directions:"):
                section = section.replace("directions:", "").strip()
                headline = "DIRECTIONS"

            if headline == "TITLE":
                recipe_sections.append(f"[{headline}]: {section.capitalize()}")
            elif headline in ["INGREDIENTS", "DIRECTIONS"]:
                section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                recipe_sections.append(f"[{headline}]:")
                recipe_sections.extend(section_info)

    return render_template('index.html', ingredients=data.get('ingredients', ''), recipe_sections=recipe_sections)

if __name__ == '__main__':
    app.run(debug=True)
