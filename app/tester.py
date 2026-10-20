from dotenv import load_dotenv
load_dotenv()

from state import BlogState
from agents import get_llm,duplicate_checker_agent
from graph import build_blog_graph
from langgraph.types import Command
import streamlit as stlt
import uuid

llm=get_llm()

final_blog="""
Intro Hook
Imagine a single neural network that can write poetry, debug code, translate languages, and even draft legal contracts—all from a prompt. That’s the promise of a Large Language Model (LLM). As an AI developer, you’ve likely heard the buzz, but how do these models actually work, and what does it mean for your next project? Let’s unpack the core concepts, architecture, training tricks, and practical take‑aways that make LLMs the powerhouse behind today’s AI breakthroughs.

1. The Rise of LLMs: From RNNs to Transformers
The journey began with recurrent neural networks (RNNs) and long short‑term memory (LSTM) units, which struggled with long‑range dependencies. The transformer architecture, introduced in 2017, shifted the paradigm by using self‑attention to capture relationships regardless of token distance. This change allowed models to scale dramatically—think GPT‑3 with 175 billion parameters or PaLM with 540 billion.

Why the scaling? Empirical studies show that performance on a wide range of language tasks improves roughly linearly with model size, up to a point. The “scaling laws” discovered by OpenAI and DeepMind provide a roadmap: double the data, double the compute, and you’ll see a predictable boost in accuracy. For developers, this means that investing in larger models (or distributed training) can yield tangible gains, but also comes with higher resource demands.

2. Core Architecture: The Transformer Backbone
At its heart, an LLM is a stack of transformer layers, each comprising:

Multi‑Head Self‑Attention – Learns which tokens to focus on when generating the next token. The “heads” allow the model to capture different relational patterns simultaneously.
Feed‑Forward Network (FFN) – A small MLP that adds non‑linearity after attention.
Layer Normalization & Residual Connections – Stabilize training and preserve gradient flow.
The input is tokenized into sub‑words (e.g., Byte‑Pair Encoding or SentencePiece). Positional embeddings inject order information, enabling the model to understand sequence structure. The final layer projects hidden states to a vocabulary distribution via a softmax, producing the next token probability.
"""
response=duplicate_checker_agent(llm,"What is LLM?",final_blog)
stlt.write(response)