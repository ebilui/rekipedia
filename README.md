# プロジェクト名

このプロジェクトは、CSVデータを処理し、質問応答を行うアプリケーションです。

## 必要なパッケージのインストール

このプロジェクトを実行するには、以下のパッケージをインストールする必要があります。以下のコマンドを実行してください。

```bash
# 仮想環境を作成
python -m venv venv

# 仮想環境をアクティブ化
source venv/bin/activate

# 必要なパッケージをインストール
pip install flask pandas langchain langchain-experimental openai sentence-transformers chromadb flask_socketio
```

## 環境変数の設定

OpenAI APIを使用するために、`OPENAI_API_KEY`を環境変数として設定する必要があります。以下のコマンドを使用して設定できます。

```bash
export OPENAI_API_KEY='your_api_key_here'
```

## アプリケーションの起動

アプリケーションを起動するには、以下のコマンドを実行してください。

```bash
python app.py
```

その後、ブラウザで `http://127.0.0.1:5000` にアクセスすると、アプリケーションを使用できます。

## 使用方法

1. CSVファイルをアップロードします。
2. 質問を入力して、関連する情報を取得します。
