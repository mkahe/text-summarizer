from flask import Flask, render_template, request, jsonify
import text_miner

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    request_data = request.form
    text1 = request_data["text1"]
    text2 = request_data["text2"]
    option = request_data["option"]

    spliter = text_miner.split_html_text if option == "2" else text_miner.split_text

    sentences = spliter(text1)

    text_list = [i.text for i in sentences]
    text_list = text_miner.lemmatizer(text_list)
    sentence_tfidf = text_miner.get_score_based_on_tfidf(text_list)

    ne_list = [i.named_entities for i in sentences]
    ne_list = text_miner.lemmatizer(ne_list)
    ne_score_tfidf = text_miner.get_score_based_on_tfidf(ne_list)
    
    for i in range(len(sentences)):
        sentences[i].weight = sentence_tfidf[i] + 2 * ne_score_tfidf[i]

    summary = text_miner.get_summary(sentences)

    texts = [sum[0].text for sum in summary]
    texts = " ".join(texts)
    print(texts)
    rouge_score = text_miner.get_rouge_score(texts, text2)
    print(rouge_score)

    return render_template("summary.html", summary=summary, rouge_score = rouge_score[0])



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
