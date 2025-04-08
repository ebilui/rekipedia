from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import os

# OpenAI APIキーを環境変数から取得（設定済み前提）
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# 質問を分類するプロンプトテンプレート
prompt = ChatPromptTemplate.from_template("""
あなたはユーザーからの質問を次の2つのカテゴリのどちらかに分類するAIです：
- search: 似ている文書を検索する必要がある質問
- analyze: 表のデータをPandasを使って集計・分析する必要がある質問

質問: {question}

分類:（search または analyze のいずれかだけを出力）
""")

# LangChainのチェーンを構築
chain = prompt | llm | parser

def classify_query(question: str) -> str:
    """質問を search または analyze に分類"""
    result = chain.invoke({"question": question})
    result = result.strip().lower()
    if result not in ["search", "analyze"]:
        return "search"
    return result

# CLIテスト用
if __name__ == "__main__":
    q = input("質問を入力してください: ")
    print("分類結果:", classify_query(q))