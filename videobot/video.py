import streamlit as st
import os
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from PIL import Image
import matplotlib.pyplot as plt
import subprocess
from gtts import gTTS
import cv2
import re
from dotenv import load_dotenv

# Initialize Streamlit app
st.title("Generative Video Generation App")

# Load environment variables from .env file
load_dotenv()
open_api_key = os.getenv('OPEN_API_KEY')

# Define the input query field
query_str = st.text_input("Enter your query:")

if st.button("Generate Video"):
    if query_str:
        # Initialize vector stores
        text_store = LanceDBVectorStore(uri="lancedb", table_name="text_collection")
        image_store = LanceDBVectorStore(uri="lancedb", table_name="image_collection")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        
        # Load documents
        output_folder = 'videobot/Rajkumar'
        documents = SimpleDirectoryReader(output_folder).load_data()
        
        # Create multimodal index
        index = MultiModalVectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        # Create retriever engine
        retriever_engine = index.as_retriever(similarity_top_k=1, image_similarity_top_k=7)
        
        # Define retrieve function
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
        
        # Retrieve data based on query
        img, text = retrieve(retriever_engine, query_str)
        
        # Prepare the context string
        context_str = "".join(text)
        image_documents = SimpleDirectoryReader(input_files=img).load_data()
        openai_mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=open_api_key, max_new_tokens=1500)
        
        qa_tmpl_str = (
            "Based on the provided information, including relevant images and retrieved text from the pdf, "
            "accurately and precisely answer the query without any additional prior knowledge.\n"
            "Give me detailed answer only in steps with Sr No and don't add any further information after steps. "
            "Provide answer only in 7 steps. "
            "Query: {query_str}\n"
            "Answer: "
        )
        
        # Generate answer using OpenAI model
        result = openai_mm_llm.complete(
            prompt=qa_tmpl_str.format(query_str=query_str),
            image_documents=image_documents,
        )
        
        # Display result text
        st.subheader("Generated Answer:")
        st.write(result.text)
        
        # Display images
        st.subheader("Retrieved Images:")
        for image_path in img:
            st.image(image_path)
        
        # Text-to-speech conversion
        mytext = result.text
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save("videobot/welcome.mp3")
        
        # Define the list of images (assuming they are in the same folder)
        image_list = img
        
        # Define the video name and frame rate
        video_name = "videobot/output_video.avi"
        fps = 1 / 9.2  # Adjust FPS for desired speed
        
        # Get the first image to determine frame size
        frame = cv2.imread(image_list[0])
        height, width, channels = frame.shape
        
        # Create a video writer object
        video = cv2.VideoWriter(video_name, 0, fps, (width, height))
        
        # Loop through the image list and write to video
        for image in image_list:
            image_path = os.path.join(os.getcwd(), image)  # Get full path for image
            frame = cv2.imread(image_path)  # Read the image
            video.write(frame)
        
        # Release resources
        video.release()
        cv2.destroy ```python
        # Release resources
        video.release()
        cv2.destroyAllWindows()
        
        # Display video
        st.subheader("Generated Video:")
        st.video(video_name)
        
        st.success("Video created successfully!")

# Define a function to clean text
def remove_asterisks_numbers_special_chars(text):
    """Removes asterisks (*), numbers, and special characters from a string.

    Args:
        text: The string to process.

    Returns:
        A string containing only alphabets and spaces.
    """
    # Use a regular expression to match only alphabets and spaces
    pattern = r"[a-zA-Z ]+"
    return ''.join(re.findall(pattern, text))

else:
    st.warning("Please enter a query to generate a video.")
