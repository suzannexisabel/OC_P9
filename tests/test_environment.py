import faiss

from fastapi import FastAPI
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from mistralai.client import Mistral

from ragas.llms import LangchainLLMWrapper


def test_imports():
    assert faiss is not None
    assert FastAPI is not None
    assert ChatMistralAI is not None
    assert ChatPromptTemplate is not None
    assert Mistral is not None
    assert LangchainLLMWrapper is not None