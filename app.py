import os
import json
import time
import glob
import logging
import concurrent.futures
from uuid import uuid4
# from dotenv import load_dotenv
import tempfile

import streamlit as st
# import fitz  # PyMuPDF
from PIL import Image
from ultralytics import YOLO
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from nltk.tokenize import word_tokenize
import torch
import nltk
try:
    import fitz
except ImportError:
    os.system('pip install PyMuPDF')
    import fitz

# Download required NLTK resource if needed.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]

# Create temporary directories for this session (these persist while the app runs)
if "base_dir" not in st.session_state:
    st.session_state.base_dir = tempfile.TemporaryDirectory()
BASE_DIR = st.session_state.base_dir.name
LOW_RES_DIR = os.path.join(BASE_DIR, "40_dpi")
HIGH_RES_DIR = os.path.join(BASE_DIR, "500_dpi")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Create these directories if they don't exist
for d in [LOW_RES_DIR, HIGH_RES_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize session state for processed flag and results
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.gemini_documents = None
    st.session_state.vector_store = None
    st.session_state.compression_retriever = None
if "logs" not in st.session_state:
    st.session_state.logs = []

# Use columns for layout: left for pipeline & logs, right for chat.
col1, col2 = st.columns([1, 3])

def log_message(msg):
    st.session_state.logs.append(msg)
    # Clear the container and reprint the logs
    col1.markdown("\n".join(st.session_state.logs))

# -------------------------
# Pipeline Functions
# -------------------------

def pdf_to_images(pdf_path, output_dir, fixed_length=1080):
    log_message(f"Converting PDF to images at fixed length {fixed_length}px...")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log_message(f"Created directory: {output_dir}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log_message(f"Error opening PDF: {e}")
        raise
    def process_page(page_number):
        page = doc[page_number]
        scale = fixed_length / page.rect.width
        matrix = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=matrix)
        image_filename = f"{base_name}_page_{page_number + 1}.jpg"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)
        log_message(f"Saved image: {image_path}")
        return image_path
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, i) for i in range(len(doc))]
        file_paths = [future.result() for future in concurrent.futures.as_completed(futures)]
    doc.close()
    log_message("PDF conversion completed.")
    return file_paths

class BlockDetectionModel:
    def __init__(self, weight, device=None):
        self.device = "cuda" if (device is None and torch.cuda.is_available()) else "cpu"
        self.model = YOLO(weight).to(self.device)
        log_message(f"YOLO model loaded on {self.device}.")

    def predict_batch(self, images_dir):
        if not os.path.exists(images_dir) or not os.listdir(images_dir):
            raise ValueError(f"Directory {images_dir} is empty or does not exist.")
        images = glob.glob(os.path.join(images_dir, "*.jpg"))
        log_message(f"Found {len(images)} low-res images for detection.")
        results = self.model(images)
        output = {}
        for result in results:
            image_name = os.path.basename(result.path)
            labels = result.boxes.cls.tolist()
            boxes = result.boxes.xywh.tolist()
            output[image_name] = [{"label": label, "bbox": box} for label, box in zip(labels, boxes)]
        log_message("Block detection completed.")
        return output

def scale_bboxes(bbox, src_size=(662, 468), dst_size=(4000, 3000)):
    scale_x = dst_size[0] / src_size[0]
    scale_y = scale_x
    return bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y

def crop_and_save(detection_output, output_dir):
    log_message("Cropping detected regions using high-res images...")
    output_data = {}
    for image_name, detections in detection_output.items():
        image_resource_path = os.path.join(output_dir, image_name.replace(".jpg", ""))
        image_path = os.path.join(HIGH_RES_DIR, image_name)
        os.makedirs(image_resource_path, exist_ok=True)
        if not os.path.exists(image_path):
            log_message(f"High-res image missing: {image_path}")
            continue
        try:
            with Image.open(image_path) as image:
                image_data = {}
                for det in detections:
                    label = det["label"]
                    bbox = det["bbox"]
                    label_dir = os.path.join(image_resource_path, str(label))
                    os.makedirs(label_dir, exist_ok=True)
                    x, y, w, h = scale_bboxes(bbox)
                    cropped_img = image.crop((x - w / 2, y - h / 2, x + w / 2, y + h / 2))
                    cropped_name = f"{label}_{len(os.listdir(label_dir)) + 1}.jpg"
                    cropped_path = os.path.join(label_dir, cropped_name)
                    cropped_img.save(cropped_path)
                    image_data.setdefault(label, []).append(cropped_path)
                image_data["Image_Path"] = image_path
                output_data[image_name] = image_data
                log_message(f"Cropped images saved for {image_name}")
        except Exception as e:
            log_message(f"Error cropping {image_name}: {e}")
    log_message("Cropping completed.")
    return output_data

def process_with_gemini(image_paths, prompt):
    log_message(f"Asynchronously processing {len(image_paths)} images with Gemini OCR in bulk...")
    contents = [prompt]
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img_resized = img.resize((int(img.width / 2), int(img.height / 2)))
                contents.append(img_resized)
        except Exception as e:
            log_message(f"Error opening {path}: {e}")
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    time.sleep(4)  # Simple rate-limiting
    response = client.models.generate_content(model="gemini-2.0-flash", contents=contents)
    log_message("Gemini OCR bulk response received.")
    resp_text = response.text.strip()
    if resp_text.startswith("```"):
        resp_text = resp_text.replace("```", "").strip()
        if resp_text.lower().startswith("json"):
            resp_text = resp_text[4:].strip()
    try:
        return json.loads(resp_text)
    except json.JSONDecodeError:
        log_message(f"Failed to parse JSON: {resp_text}")
        return None

def process_page_with_metadata(page_key, blocks, prompt):
    log_message(f"Processing page: {page_key}")
    all_imgs = []
    for block_type, paths in blocks.items():
        if block_type != "Image_Path":
            all_imgs.extend(paths)
    if not all_imgs:
        log_message(f"No cropped images for {page_key}")
        return None
    raw_metadata = process_with_gemini(all_imgs, prompt)
    if raw_metadata:
        doc = Document(
            page_content=json.dumps(raw_metadata),
            metadata={"drawing_path": blocks["Image_Path"], "drawing_name": page_key, "content": "everything"}
        )
        log_message(f"Document created for {page_key}")
        return doc
    else:
        log_message(f"No metadata extracted for {page_key}")
        return None

def process_all_pages(data, prompt):
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_page_with_metadata, key, blocks, prompt): key for key, blocks in data.items()}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                else:
                    log_message(f"No document returned for {key}")
            except Exception as e:
                log_message(f"Error processing {key}: {e}")
    log_message(f"Total {len(documents)} documents processed asynchronously.")
    return documents

# -------------------------
# UI Layout
# -------------------------
with col1:
    st.header("Processing Pipeline")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        # Save PDF to temporary directory (BASE_DIR)
        pdf_path = os.path.join(BASE_DIR, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success("PDF uploaded successfully.")
    
    if uploaded_pdf and not st.session_state.processed:
        if st.button("Run Processing Pipeline"):
            log_message("PDF uploaded successfully.")
            log_message("Converting PDF to images concurrently...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_low = executor.submit(pdf_to_images, pdf_path, LOW_RES_DIR, 662)
                future_high = executor.submit(pdf_to_images, pdf_path, HIGH_RES_DIR, 4000)
                low_res_paths = future_low.result()
                high_res_paths = future_high.result()
            log_message("PDF conversion completed.")
            log_message("Running YOLO detection on low-res images...")
            yolo_model = BlockDetectionModel("best_small_yolo11_block_etraction.pt")
            detection_results = yolo_model.predict_batch(LOW_RES_DIR)
            log_message("Block detection completed.")
            log_message("Cropping detected regions using high-res images...")
            cropped_data = crop_and_save(detection_results, OUTPUT_DIR)
            log_message("Cropping completed.")
            ocr_prompt = """
                You are an advanced system specialized in extracting standardized metadata from construction drawing texts.
                Within the images you receive, there will be details pertaining to a single construction drawing.
                Your job is to identify and extract exactly below fields from this text:
                - 1st image has details about the drawing_title and scale
                - 2nd Image has details about the client or project
                - 4th Images has Notes
                - 3rd Images has rest of the informations
                - last image is the full image from which the above image are cropped
                1. Purpose_Type_of_Drawing (examples: 'Architectural', 'Structural', 'Fire Protection')
                2. Client_Name
                3. Project_Title
                4. Drawing_Title
                5. Floor
                6. Drawing_Number
                7. Project_Number
                8. Revision_Number (must be a numeric value, or 'N/A' if it cannot be determined)
                9. Scale
                10. Architects (list of names; use ['Unknown'] if no names are identified)
                11. Notes_on_Drawing (any remarks or additional details related to the drawing)

                Key Requirements:
                - If any field is missing, return an empty string ('') or 'N/A' for that field.
                - Return only a valid JSON object containing these nine fields in the order listed, with no extra text.
                - Preserve all text in its original language (no translation), apart from minimal cleaning (e.g., removing stray punctuation) if truly necessary.
                - Do not wrap the final JSON in code fences.
                - Return ONLY the final JSON object with these fields and no additional commentary.
                Below is an example json format:
                {{
                    "Purpose_Type_of_Drawing": "Architectural",
                    "Client_Name": "문촌주공아파트주택  재건축정비사업조합",
                    "Project_Title": "문촌주공아파트  주택재건축정비사업",
                    "Drawing_Title": "분산 상가-7  단면도-3  (근린생활시설-3)",
                    "Floor": "주단면도-3",
                    "Drawing_Number": "A51-2023",
                    "Project_Number": "EP-201
                    "Revision_Number": 0,
                    "Scale": "A1 : 1/100, A3 : 1/200",
                    "Architects": ["Unknown"],
                    "Notes_on_Drawing": "• 욕상 줄눈의 간격 등은 실시공 시 변경될 수 있음.\\n• 욕상 출눈 틈에는 실란트가 시공되지 않음.\\n• 지붕의 재료, 형태, 구조는 실시공 시 변경될 수 있음.\\n• 지붕층 난간의 형태와 설치 위치는 안전성, 입면, 디자인을 고려하여 변경 가능함.\\n• 단열재의 종류는 단열성능 관계 내역을 참조.\\n• 도면상 표기된 욕상 및 지하의 무근 콘크리트 두께는 평균 두께를 의미하며, 본 시공 시 구배를 고려하여 두께가 증감될 수 있음.\\n• 외벽 단열 부분과 환기 덕트가 연결되는 부위는 기밀하게 마감해야 함."
                }}
                """
            
            log_message("Extracting metadata using Gemini OCR asynchronously...")
            gemini_documents = process_all_pages(cropped_data, ocr_prompt)
            log_message("Metadata extraction completed.")
            gemini_json_path = os.path.join(BASE_DIR, "gemini_documents.json")
            with open(gemini_json_path, "w") as f:
                json.dump([doc.dict() for doc in gemini_documents], f, indent=4)
            log_message("Gemini documents saved.")
            log_message("Building vector store for semantic search...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            example_embedding = embeddings.embed_query("sample text")
            d = len(example_embedding)
            index = faiss.IndexFlatL2(d)
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            uuids = [str(uuid4()) for _ in range(len(gemini_documents))]
            vector_store.add_documents(documents=gemini_documents, ids=uuids)
            log_message("Vector store built and documents indexed.")
            log_message("Setting up retrievers...")
            bm25_retriever = BM25Retriever.from_documents(gemini_documents, k=10, preprocess_func=word_tokenize)
            retriever_ss = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever_ss],
                weights=[0.6, 0.4]
            )
            log_message("Setting up RAG pipeline...")
            compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=ensemble_retriever
            )
            log_message("RAG pipeline set up.")
            st.session_state.processed = True
            st.session_state.gemini_documents = gemini_documents
            st.session_state.vector_store = vector_store
            st.session_state.compression_retriever = compression_retriever
            log_message("Processing pipeline completed.")

with col2:
    st.header("Chat Interface")
    st.info("Enter your query below to search the processed PDF data.")
    query = st.text_input("Query:")
    if query and st.session_state.processed:
        st.write("Searching...")
        try:
            results = st.session_state.compression_retriever.invoke(query)
            st.markdown("### Retrieved Documents:")
            for doc in results:
                drawing = doc.metadata.get("drawing_name", "Unknown")
                st.write(f"**Drawing:** {drawing}")
                try:
                    st.json(json.loads(doc.page_content))
                except Exception:
                    st.write(doc.page_content)
                img_path = doc.metadata.get("drawing_path", "")
                if img_path and os.path.exists(img_path):
                    st.image(Image.open(img_path), width=400)
        except Exception as e:
            st.error(f"Search failed: {e}")

st.write("Streamlit app finished processing.")
