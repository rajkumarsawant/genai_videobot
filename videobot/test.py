import os
import streamlit as st
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from PIL import Image
from dotenv import load_dotenv
import re
from gtts import gTTS
import cv2
import subprocess

# Load environment variables from .env file
load_dotenv()

# Get the OpenAPI key from environment variables
open_api_key = os.getenv('OPEN_API_KEY')

# Initialize vector stores
text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)

# Load documents
output_folder = 'videobot/Rajkumar'
documents = SimpleDirectoryReader(output_folder).load_data()

# Create index
index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)
retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=7,image_similarity_top_p=0.01)

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

def remove_asterisks_numbers_special_chars(text):
    """Removes asterisks (*), numbers, and special characters from a string."""
    pattern = r"[a-zA-Z ]+"
    return ''.join(re.findall(pattern, text))

def display_images(image_paths):
    for img_path in image_paths:
        img = Image.open(img_path)
        st.image(img, use_column_width=True)

# Streamlit app layout
st.title("Generative AI videobot")

query_str = st.text_input("Enter your query:")

if st.button("Submit"):
    if query_str:
        img, text = retrieve(retriever_engine, query_str)
        context_str = "".join(text)
        image_documents = SimpleDirectoryReader(input_files=img).load_data()
        openai_mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=open_api_key, max_new_tokens=1500)

        qa_tmpl_str = (
            "Based on the provided information, including relevant images and retrieved text from the pdf, "
            "accurately and precisely answer the query without any additional prior knowledge.\n"
            "Give me detailed answer only in steps with Sr No and don't add any further information after steps."
            "Provide answer only in 7 steps.\n"

            "Query: {query_str}\n"
            "Answer: "
        )

        result = openai_mm_llm.complete(
            prompt=qa_tmpl_str.format(query_str=query_str),
            image_documents=image_documents,
        )

        st.markdown("### Answer")
        st.text(result.text)
        
        txt = result.text.split('.\n\n')
        cleaned_text = [remove_asterisks_numbers_special_chars(i) for i in txt]

        st.markdown("### Relevant Images")
        display_images(img)
        
        st.write("Generated Video")
        video_file = open('videobot\Video_to_show.avi', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
