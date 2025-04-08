from flask import Flask, request, render_template_string
import os
import pandas as pd

from processor import process_csv_file, clean_dataframe_and_save
from embedder import add_chunks_to_chroma
from orchestrator import classify_query
from query import query_documents, generate_answer
from analyzer import analyze_dataframe

app = Flask(__name__)
UPLOAD_FOLDER = "csvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TEMPLATE = '''
<h1>CSV Upload</h1>
<form method="post" action="/upload" enctype="multipart/form-data">
  <input type="file" name="csv_file">
  <input type="submit" value="Upload CSV">
</form>

<h1>Ask a Question</h1>
<form method="post" action="/ask">
  <input type="text" name="query" style="width: 400px;" placeholder="質問を入力してください">
  <input type="submit" value="Ask">
</form>

{% if answer %}
<h2>Answer</h2>
<p>{{ answer }}</p>
{% endif %}
'''

@app.route('/')
def index():
    return render_template_string(TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('csv_file')
    if not file:
        return "⚠️ ファイルが見つかりませんでした"
    
    if file.filename == "":
        return "⚠️ ファイル名が空です"

    try:
        print("filename:", file.filename)
        df = pd.read_csv(file, header=None)
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        print("file_path:", file_path)
        clean_dataframe_and_save(df, file_path)

        # CSVを読み込んでチャンク分割＋ChromaDBへ登録
        chunks = process_csv_file(df)
        add_chunks_to_chroma(chunks, source_id=file.filename)

        return "✅ CSVアップロード & インデックス完了"
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"

@app.route('/ask', methods=['POST'])
def ask():
    query = request.form.get('query', '')
    if not query:
        return render_template_string(TEMPLATE, answer="⚠️ 質問が空です")

    classification = classify_query(query)
    print("分類結果:", classification)

    if classification == "search":
        docs = query_documents(query)
        answer = generate_answer(query, docs)
    elif classification == "analyze":
        from utils import select_best_csv_file_for_question
        file_name = select_best_csv_file_for_question(query)
        csv_path = os.path.join("csvs", file_name)
        print("csv_path:", csv_path)
        df = pd.read_csv(csv_path)
        answer = analyze_dataframe(df, query)
    else:
        answer = "⚠️ 質問の分類に失敗しました"

    return render_template_string(TEMPLATE, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
