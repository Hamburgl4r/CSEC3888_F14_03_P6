"""Retrieval modules for the September demo.

Search is hybrid: BM25 and the vector store run over the same chunks, then
their rankings are merged. Neither step is an input to the other.

The language model that writes answers is a later module. Do not add it here
until retrieval and click-through citations are working.
"""
