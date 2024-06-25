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







text_store=LanceDBVectorStore(uri="lancedb",table_name="text_collection")
image_store=LanceDBVectorStore(uri="lancedb",table_name="image_collection")


storage_context=StorageContext.from_defaults(vector_store=text_store,image_store=image_store)

output_folder='videobot\Rajkumar'

documents=SimpleDirectoryReader(output_folder).load_data()



from dotenv import load_dotenv
 
# Load environment variables from .env file
load_dotenv()
 
# Get the OpenAPI key from environment variables
open_api_key = os.getenv('OPEN_API_KEY')
 



index = MultiModalVectorStoreIndex.from_documents(documents,storage_context=storage_context)

retriever_engine=index.as_retriever(similarity_top_k=1, image_similarity_top_k=7)

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)

    retrieved_image = []
    retrieved_text = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
        else:
            display_source_node(res_node, source_length=200)
            retrieved_text.append(res_node.text)

    return retrieved_image, retrieved_text

query="How to rebuild engine"

img,text=retrieve(retriever_engine,query)

context_str = "".join(text)
image_documents = SimpleDirectoryReader( input_files=img).load_data()
openai_mm_llm = OpenAIMultiModal(model="gpt-4o", api_key=open_api_key, max_new_tokens=1500)

qa_tmpl_str=(
    "Based on the provided information, including relevant images and retrieved text from the pdf, \
    accurately and precisely answer the query without any additional prior knowledge.\n"
    "Give me detailed answer only in steps with Sr No and dont add any further information after steps"
    "Provide answer only in 7 steps"
    "Query: {query_str}\n"
    "Answer: "
)

query_str="How to rebuild engine"

result=openai_mm_llm.complete(
    prompt=qa_tmpl_str.format(
        query_str=query_str
    ),
    image_documents=image_documents,
)

print(result.text)

txt=result.text.split('.\n\n')


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


cleaned_text=[]
for i in txt:
  x=remove_asterisks_numbers_special_chars(i)
  cleaned_text.append(x)



# Import the required module for text
# to speech conversion

# This module is imported so that we can
# play the converted audio

# The text that you want to convert to audio
mytext = result.text

# Language in which you want to convert
language = 'en'

# Passing the text and language to the engine,
# here we have marked slow=False. Which tells
# the module that the converted audio should
# have a high speed
myobj = gTTS(text=mytext, lang=language, slow=False)

# Saving the converted audio in a mp3 file named
# welcome
myobj.save("videobot\welcome.mp3")

# Playing the converted file
os.system("start welcome.mp3")




# Define the list of images (assuming they are in the same folder)
image_list = img

# Define the video name and frame rate
video_name = "videobot\output_video.avi"
fps = 1/9.2 # Adjust FPS for desired speed

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
cv2.destroyAllWindows()

print("Video created successfully!")
print(len(img))

