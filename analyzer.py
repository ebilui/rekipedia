from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
import pandas as pd
import os

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    temperature=0,
    model="gpt-4o",
    openai_api_key=openai_api_key
)

def analyze_dataframe(df: pd.DataFrame, query: str) -> str:
    """
    LangChainのPandas Agentを使って自然言語でDataFrameを分析
    :param df: pandas DataFrame
    :param query: 自然言語の質問
    :return: 答え（文字列）
    """
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    result = agent.run(query)
    return result
