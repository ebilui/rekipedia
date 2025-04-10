from flask import Flask, request, render_template_string
from flask_socketio import SocketIO, emit
import os
import pandas as pd

from processor import clean_dataframe_and_save
from embedder import add_chunks_to_chroma_streaming
from orchestrator import classify_query
from query import query_documents, generate_answer
from analyzer import analyze_dataframe

app = Flask(__name__)
socketio = SocketIO(app)
UPLOAD_FOLDER = "csvs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TEMPLATE = '''
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.0/socket.io.min.js"></script>
<script>
    var socket = io();

    socket.on('progress', function(msg) {
        document.getElementById('progress').innerText = '進行状況: ' + msg.data.toFixed(2) + '%';
    });

    // ファイルアップロード用の関数
    function uploadFile() {
        const formData = new FormData();
        const fileInput = document.getElementById('csv_file');
        formData.append('csv_file', fileInput.files[0]);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(result => {
            document.getElementById('upload_result').innerText = result;
        })
        .catch(error => {
            document.getElementById('upload_result').innerText = 'エラーが発生しました: ' + error;
        });
    }

    // 質問送信用の関数
    function askQuestion() {
        const query = document.getElementById('query').value;
        
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: 'query=' + encodeURIComponent(query)
        })
        .then(response => response.text())
        .then(result => {
            document.getElementById('answer').innerHTML = result;
        })
        .catch(error => {
            document.getElementById('answer').innerText = 'エラーが発生しました: ' + error;
        });
    }
</script>

<h1>CSV Upload</h1>
<input type="file" id="csv_file">
<button onclick="uploadFile()">Upload CSV</button>
<div id="upload_result"></div>
<div id="progress" style="margin-top: 20px; font-size: 20px; color: blue;"></div>

<h1>Ask a Question</h1>
<input type="text" autocomplete="on" id="query" style="width: 400px;" placeholder="質問を入力してください">
<button onclick="askQuestion()">Ask</button>

<div id="answer"></div>
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
        try:
            # まずファイルの内容を確認
            content = file.read()
            if not content.strip():
                return "⚠️ ファイルが空です"
            
            # ファイルポインタを先頭に戻す
            file.seek(0)
            
            # エンコーディングを試行
            try:
                df = pd.read_csv(file, header=None, encoding="utf-8")
            except UnicodeDecodeError:
                file.seek(0)
                try:
                    df = pd.read_csv(file, header=None, encoding="shift_jis")
                except UnicodeDecodeError:
                    return "⚠️ エンコーディングエラーが発生しました。ファイルのエンコーディングを確認してください。"
            
            # DataFrameが空でないことを確認
            if df.empty:
                return "⚠️ CSVファイルにデータが含まれていません"
            
            # 列が存在することを確認
            if len(df.columns) == 0:
                return "⚠️ CSVファイルに列が存在しません"

            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            print("file_path:", file_path)
            clean_dataframe_and_save(df, file_path)

            add_chunks_to_chroma_streaming(df, source_id=file.filename, socketio=socketio)

            return "✅ CSVアップロード & インデックス完了"
            
        except pd.errors.EmptyDataError:
            return "⚠️ CSVファイルが空です"
        except pd.errors.ParserError:
            return "⚠️ CSVファイルの形式が正しくありません"
            
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

	return answer

if __name__ == '__main__':
	socketio.run(app, debug=True)
