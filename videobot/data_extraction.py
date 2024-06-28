from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from pikepdf import Pdf, PdfImage


filename = "videobot\Rajkumar\doc.pdf"


example = Pdf.open(filename)

for i, page in enumerate(example.pages):
    for j, (name, raw_image) in enumerate(page.images.items()):
        image = PdfImage(raw_image)
        out = image.extract_to(fileprefix=f"{filename}-page{i:03}-img{j:03}")





import PyPDF2

def get_text_from_pdf(pdf_path):
    """Extracts text from a PDF file, handling potential errors.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A string containing the extracted text, or an empty string if errors occur.
    """

    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'.")
    except PermissionError:
        print(f"Error: Permission denied accessing '{pdf_path}'.")
    except Exception as e:  # Catch other potential exceptions
        print(f"Error: An unexpected error occurred: {e}")
    return text

def save_text_to_file(text, output_filename):
    """Saves text to a .txt file.

    Args:
        text: The text to save.
        output_filename: The desired filename for the .txt file.
    """

    with open(output_filename, 'w', encoding='utf-8') as txt_file:
        txt_file.write(text)

# Example usage with error handling
pdf_file = "videobot\Rajkumar\doc.pdf"
output_txt = "videobot\Rajkumar\extracted_text.txt"

try:
    extracted_text = get_text_from_pdf(pdf_file)
    save_text_to_file(extracted_text, output_txt)
    print(f"Text extracted from PDF and saved to '{output_txt}'.")
except Exception as e:  # Catch any errors during execution
    print(f"An error occurred: {e}")