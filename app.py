from flask import Flask, render_template, request
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

app = Flask(__name__)

def get_llama_response(input_text, no_words, blog_style):
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 500,
                                'temperature': 0.7})

    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """

    prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                            template=template)

    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        no_words = request.form['no_words']
        blog_style = request.form['blog_style']
        response = get_llama_response(input_text, no_words, blog_style)
        return render_template('index.html', response=response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
