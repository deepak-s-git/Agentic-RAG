import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

def init_and_upload_vector_store(collection_name, all_chunks, doc_metadata, embeddings, vector_size=384):
    client = QdrantClient("http://localhost:6333")
    
    with st.spinner("Creating vector database..."):
        try:
            client.delete_collection(collection_name)
        except:
            pass

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [{"content": chunk, "metadata": metadata} for chunk, metadata in zip(all_chunks, doc_metadata)]

        client.upload_collection(
            collection_name=collection_name,
            vectors=embeddings,
            payload=payload,
            ids=ids,
            batch_size=256,
        )
    return client

def search_vector_store(client, collection_name, query_embedding, top_k=5):
    return client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k,
    )
