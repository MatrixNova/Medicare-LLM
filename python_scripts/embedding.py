import os
import json
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models # <-- Make sure 'models' is imported
from sentence_transformers import SentenceTransformer
# Import the Qdrant config variables
from config import HUGGINGFACETOKEN, QDRANT_ENDPOINT_URL, QDRANT_API_KEY, logger 
from typing import Dict, Any, List

# --- Configuration ---
METADATA_DIR = Path(r"D:\Medicare\Medicare-LLM\text_output_chunked")
RAW_TEXT_DIR = Path(r"D:\Medicare\Medicare-LLM\text_output")
COLLECTION_NAME = "main-rag"
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
EMBEDDING_DIMENSION = 768 
MIN_CHUNK_LENGTH = 50 

def get_embedding_model(model_name: str, token: str) -> SentenceTransformer:
    """
    Initializes and returns the SentenceTransformer model.
    """
    logger.info(f"Loading embedding model: {model_name}")
    try:
        model = SentenceTransformer(model_name, use_auth_token=token)
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        logger.error("Make sure your HUGGINGFACETOKEN is set correctly in .env")
        raise

def get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    """
    Initializes and returns the Qdrant client for Qdrant Cloud.
    """
    if not url or not api_key:
        logger.error("QDRANT_ENDPOINT_URL or QDRANT_API_KEY not set in .env file.")
        raise ValueError("Qdrant URL and API Key must be set.")
        
    logger.info(f"Connecting to Qdrant Cloud at {url}")
    try:
        client = QdrantClient(
            url=url,
            api_key=api_key,
        )
        logger.info("Qdrant Cloud connection successful.")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant Cloud: {e}")
        raise

def create_qdrant_collection(client: QdrantClient, collection_name: str, embedding_dim: int):
    """
    *** MODIFIED ***
    Creates the Qdrant collection AND the payload index for filtering.
    """
    try:
        # This will delete and recreate the collection
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE
            )
        )
        logger.info(f"Collection '{collection_name}' created/recreated successfully.")
        
        # Create a payload index for "disease_name_short" to enable filtering.
        logger.info("Creating payload index for 'disease_name_short'...")
        client.create_payload_index(
            collection_name=collection_name,
            field_name="disease_name_short",
            # The error message specifically requested a 'keyword' type index
            field_schema=models.PayloadSchemaType.KEYWORD 
        )
        logger.info("Payload index created successfully.")

    except Exception as e:
        logger.error(f"Failed to create collection or index: {e}")
        raise

def main():
    """
    Main function to run the embedding and upsert process.
    """
    logger.info("--- Starting Hybrid Embedding and Indexing Process ---")
    
    # 1. Initialize models and clients
    try:
        model = get_embedding_model(EMBEDDING_MODEL_NAME, HUGGINGFACETOKEN)
        client = get_qdrant_client(QDRANT_ENDPOINT_URL, QDRANT_API_KEY)
    except Exception:
        logger.error("Failed to initialize models or clients. Exiting.")
        return

    # 2. Create collection AND THE INDEX
    create_qdrant_collection(client, COLLECTION_NAME, EMBEDDING_DIMENSION)

    metadata_files = list(METADATA_DIR.glob("*.json"))
    if not metadata_files:
        logger.warning(f"No structured .json files found in {METADATA_DIR}. Exiting.")
        return
    
    logger.info(f"Found {len(metadata_files)} structured JSON files to process.")

    points_batch = []
    batch_size = 32
    
    for metadata_file_path in metadata_files:
        try:
            
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                document_payload = json.load(f) 
            
            
            raw_text_file_path = RAW_TEXT_DIR / metadata_file_path.name
            if not raw_text_file_path.exists():
                logger.warning(f"No matching raw text file at {raw_text_file_path} for {metadata_file_path.name}. Skipping.")
                continue

            with open(raw_text_file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            page_texts = raw_data.get("pages", [])
            if not page_texts:
                logger.warning(f"No 'pages' found in {raw_text_file_path.name}. Skipping.")
                continue

            # 3. Create a vector for EACH page (chunk)
            for page_num, page_item in enumerate(page_texts):
                
                chunk_text = ""
                try:
                    if isinstance(page_item, str):
                        chunk_text = page_item.strip()
                    elif isinstance(page_item, dict):
                        if "text" in page_item and isinstance(page_item["text"], str):
                            chunk_text = page_item["text"].strip()
                        elif "content" in page_item and isinstance(page_item["content"], str):
                            chunk_text = page_item["content"].strip()
                        elif "page_content" in page_item and isinstance(page_item["page_content"], str):
                            chunk_text = page_item["page_content"].strip()
                        else:
                            chunk_text = json.dumps(page_item)
                    elif page_item is not None:
                        chunk_text = str(page_item).strip()
                
                except Exception as e:
                    logger.warning(f"Could not extract text from page item {page_num} in {metadata_file_path.name}: {e}. Skipping item.")
                    continue

                if len(chunk_text) < MIN_CHUNK_LENGTH:
                    continue
                
                # 4. Create vector for the raw text chunk
                vector = model.encode(chunk_text).tolist()
                
                # 5. Create the payload
                point_payload = document_payload.copy()
                point_payload['raw_text_chunk'] = chunk_text
                point_payload['source_filename'] = metadata_file_path.name
                point_payload['page_number'] = page_num + 1
                
                point = models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=point_payload 
                )
                points_batch.append(point)

                # 6. Upload batch if full
                if len(points_batch) >= batch_size:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points_batch,
                        wait=True 
                    )
                    logger.info(f"Upserted batch of {len(points_batch)} points.")
                    points_batch = []

        except Exception as e:
            logger.error(f"Error processing file {metadata_file_path.name}: {e}")

    # 7. Upload any remaining points
    if points_batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch,
            wait=True 
        )
        logger.info(f"Upserted final batch of {len(points_batch)} points.")

    logger.info("--- Hybrid Embedding and Indexing Process Complete ---")

if __name__ == "__main__":
    main()