import cohere
import openai
import replicate
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")


# def gptneo(prompt):
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids

#     gen_tokens = model.generate(
#         input_ids,
#         do_sample=True,
#         temperature=0.6,
#         max_length=200,
#     )
#     gen_text = tokenizer.batch_decode(gen_tokens)[0]

#     return gen_text


app = Flask(__name__)
CORS(app)

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/getres", methods=["POST"])
def call_api():
    try:
        data = request.get_json()
        prompt = data["prompt"]
        model = data["model"]

        match model:
            case "command-nightly":
                response = co.generate(
                    model=model,
                    prompt=prompt,
                    max_tokens=200,
                )

                answer = response.generations[0].text

            case "gpt-3.5-turbo":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                )

                answer = response.choices[0].message["content"]

            case "stable-diffusion":
                response = replicate.run(
                    "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                    input={"prompt": prompt},
                )

                answer = response[-1]

            # case "gptneo":
            #     answer = gptneo(prompt)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
