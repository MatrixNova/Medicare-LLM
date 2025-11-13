import os
from google import genai
from qdrant_client import QdrantClient, models # <-- Import models
from sentence_transformers import SentenceTransformer
import json # <-- Import json
from config import (
    QDRANT_ENDPOINT_URL, 
    QDRANT_API_KEY, 
    HUGGINGFACETOKEN, 
    GEMINI_API_KEY, 
    logger
)
from typing import List, Dict, Any, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_LIBS_AVAILABLE = True
except ImportError:
    LOCAL_LIBS_AVAILABLE = False

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
# Using 2.5 Pro for high-quality generation
GENERATOR_MODEL_NAME = 'gemini-2.5-pro' 
# Using 2.5 Flash for fast, cheap filter extraction
FILTER_MODEL_NAME = 'gemini-2.5-flash' 
COLLECTION_NAME = "main-rag"

# --- !! NEW: MedGemma Configuration !! ---
# Set this flag to True to use MedGemma, False to use Gemini 2.5 Pro API*****************************************
USE_LOCAL_MEDGEMMA = False
# Instruction-tuned MedGemma 4B model from Hugging Face
MEDGEMMA_MODEL_NAME = 'google/medgemma-4b-it' 
# --- !! END NEW !! ---

class RAGAgent:
    def __init__(self):
        logger.info("Initializing RAGAgent")

        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.GENERATOR_MODEL_NAME = GENERATOR_MODEL_NAME
        self.COLLECTION_NAME = COLLECTION_NAME

        # --- !!Add MedGemma config to self !! ---
        self.use_local_medgemma = USE_LOCAL_MEDGEMMA
        self.medgemma_model_name = MEDGEMMA_MODEL_NAME
        self.medgemma_model = None
        self.medgemma_tokenizer = None
        
        
        # 1. Initialize Embedding Model (for querying)
        logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL_NAME}")
        self.embed_model = SentenceTransformer(
            self.EMBEDDING_MODEL_NAME, 
            use_auth_token=HUGGINGFACETOKEN
        )
        logger.info("Embedding model loaded.")
        
        # 2. Initialize Qdrant Client (for retrieving)
        logger.info(f"Connecting to Qdrant Cloud...")
        if not QDRANT_ENDPOINT_URL or not QDRANT_API_KEY:
            raise ValueError("QDRANT_ENDPOINT_URL or QDRANT_API_KEY not found in .env file.")
        
        self.qdrant_client = QdrantClient(
            url=QDRANT_ENDPOINT_URL, 
            api_key=QDRANT_API_KEY
        )
        logger.info("Qdrant Cloud connection successful.")
        
        # 3. Initialize Generative Model Client (THE FIX IS HERE)
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
            
        logger.info(f"Initializing generative client...")
        # Use the client pattern from your other scripts.
        # This one client will be used for all genAI calls.
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        
        logger.info(f"Using generator model: {GENERATOR_MODEL_NAME}")
        logger.info(f"Using filter model: {FILTER_MODEL_NAME}")
        
        # --- !! NEW: Load local model if requested !! ---
        if self.use_local_medgemma:
            if LOCAL_LIBS_AVAILABLE:
                # Short CUDA Check
                self.gpu_available = torch.cuda.is_available()
                if self.gpu_available:
                    logger.info(f"CUDA GPU detected ({torch.cuda.get_device_name(0)}). MedGemma will run on GPU.")
                else:
                    logger.warning("--- NO GPU DETECTED ---")
                    logger.warning("MedGemma will run on CPU. Generation will be EXTREMELY SLOW.")
                    logger.warning("Set USE_LOCAL_MEDGEMMA = False to use the fast Gemini API.")
                
                # Load the local model
                self._load_local_medgemma()
            else:
                logger.error("USE_LOCAL_MEDGEMMA is True, but 'transformers' or 'torch' are not installed.")
                raise ImportError("Missing required libraries for local MedGemma inference.")
        # --- !! END NEW !! ---

        logger.info("RAGAgent initialized successfully.")

    # --- !! NEW: Helper function to load MedGemma !! ---
    def _load_local_medgemma(self):
        """
        Loads the MedGemma model and tokenizer into memory.
        Uses 4-bit quantization for efficiency.
        """
        logger.info(f"Loading local MedGemma model: {self.medgemma_model_name}...")
        logger.warning("This may take several minutes and require significant VRAM.")
        
        try:
            self.medgemma_tokenizer = AutoTokenizer.from_pretrained(
                self.medgemma_model_name, 
                token=HUGGINGFACETOKEN # Use token if it's a gated model
            )
            self.medgemma_model = AutoModelForCausalLM.from_pretrained(
                self.medgemma_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto", # Automatically use GPU if available
                quantization_config=None, # Or add bitsandbytes 4-bit config
                token=HUGGINGFACETOKEN
            )
            logger.info("Local MedGemma model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local MedGemma model: {e}")
            raise

    # --- !!Helper function to generate with MedGemma !! ---
    def _generate_with_medgemma(self, prompt: str) -> str:
        """
        Generates a response using the loaded local MedGemma model.
        """
        # MedGemma uses a specific chat template
        # We apply it to the prompt created by build_prompt
        chat_prompt = [
            {"role": "user", "content": prompt}
        ]
        inputs = self.medgemma_tokenizer.apply_chat_template(
            chat_prompt, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.medgemma_model.device)

        # Generate the output
        output_ids = self.medgemma_model.generate(
            inputs,
            max_new_tokens=1500, # Max length of the answer
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Decode the response, skipping the prompt part
        response_text = self.medgemma_tokenizer.batch_decode(
            output_ids[:, inputs.shape[1]:], # Only decode the new tokens
            skip_special_tokens=True
        )[0]
        
        return response_text.strip()
    

    def extract_filters_from_query(self, query: str) -> Optional[models.Filter]:
       
        logger.info("Extracting filters from query...")
        prompt = f"""
        You are a query analysis assistant. Your job is to extract *only* the name of a disease from the user's query.
        Respond with *only* a JSON object: {{"disease_name_short": "Name of Disease"}}
        If no disease is mentioned, return: {{"disease_name_short": null}}

        User Query: "What are the lab findings for scrub typhus?"
        Assistant: {{"disease_name_short": "Scrub typhus"}}

        User Query: "Tell me about treatments for dengue in diabetic patients."
        Assistant: {{"disease_name_short": "Dengue fever"}}

        User Query: "What is the typical fever pattern?"
        Assistant: {{"disease_name_short": null}}

        User Query: "{query}"
        Assistant:
        """
        
        try:
            
            # Use the config dictionary pattern from your text_chunk.py
            config_dict = {
                "response_mime_type": "application/json"
            }
            
            response = self.client.models.generate_content(
                model=FILTER_MODEL_NAME, # No "models/" prefix
                contents=[prompt],       # contents must be a list
                config=config_dict
            )
            
            filter_data = json.loads(response.text)
            disease_name = filter_data.get("disease_name_short")
            
            if disease_name:
                logger.info(f"Found filter: disease_name_short = {disease_name}")
                # Create and return a Qdrant Filter object
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key="disease_name_short",
                            match=models.MatchValue(value=disease_name)
                        )
                    ]
                )
            else:
                logger.info("No disease filter found.")
                return None
        except Exception as e:
            logger.error(f"Error during filter extraction: {e}")
            return None

    def search_knowledge_base(self, query: str, filters: models.Filter = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the query and searches Qdrant for the top_k most relevant documents,
        applying metadata filters if provided.
        """
        logger.info(f"Embedding query: '{query[:50]}...'")
        query_vector = self.embed_model.encode(query).tolist()
        
        search_type = "Hybrid" if filters else "Vector"
        logger.info(f"Performing {search_type} search in collection '{COLLECTION_NAME}'...")
        
        search_results = self.qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=filters,  # <-- THIS ENABLES HYBRID SEARCH
            limit=top_k,
            with_payload=True 
        )
        
        retrieved_contexts = [result.payload for result in search_results]
        logger.info(f"Retrieved {len(retrieved_contexts)} contexts.")
        return retrieved_contexts

    def build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """
        Builds a prompt using the 'raw_text_chunk' as the main context,
        and other metadata as supplementary information.
        """
        context_str = "\n\n---\n\n".join(
            [f"Source File: {doc.get('source_filename', 'N/A')}\n"
             f"Page Number: {doc.get('page_number', 'N/A')}\n"
             f"Case Diagnosis: {doc.get('disease_name_short', 'N/A')}\n"
             f"Detailed Diagnosis: {doc.get('final_diagnosis', 'N/A')}\n"
             f"Chief Complaints: {doc.get('chief_complaint', 'N/A')}\n"
             f"History of Present Illnesses: {doc.get('history_of_present_illness', 'N/A')}\n"
             f"Physical Examinations: {doc.get('physical_exam', 'N/A')}\n"
             f"Lab Diagnostics Results: {doc.get('labs_and_diagnostics', 'N/A')}\n"
             f"Differential Diagnosis: {doc.get('differential_diagnosis', 'N/A')}\n"
             f"Relevant Text Snippet: {doc.get('raw_text_chunk', 'N/A')}\n"
             for doc in context_docs]
        )
        
        prompt = f"""
        You are an expert clinical diagnostic assistant. Your task is to answer the user's question based *only* on the provided clinical case snippets.
        
        Pay close attention to the "Relevant Text Snippet" for each source, as this is the text that most closely matched the user's query.
        Use the "Case Diagnosis" to understand the high-level context of the snippet. "Detailed Diagnosis", "Chief Complaints", "History of Present Illnesses", "Physical Examinations", "Lab Diagnostics Results", and "Differential Diagnosis" provide additional clinical context that may help in formulating your answer; with "Detailed Diagnosis" being particularly elaborative on the reasoning as to why a certain diagnosis was reached.
        
        Do not use any external knowledge. If the answer is not in the provided context, state that clearly: "I could not find an answer in the provided case reports."
        
        **PROVIDED CONTEXT:**
        {context_str}
        
        **USER QUESTION:**
        {query}
        
        **ASSISTANT ANSWER:**
        """
        return prompt

    def ask(self, query: str) -> tuple[str, list[str]]:
       
        # 1. Extract filters from query
        qdrant_filter = self.extract_filters_from_query(query)
        
        # 2. Retrieve context with hybrid search
        retrieved_contexts = self.search_knowledge_base(
            query, 
            filters=qdrant_filter
        )

        if not retrieved_contexts and qdrant_filter is not None:
            logger.warning(f"Hybrid search for '{query}' returned no results. Retrying with pure vector search.")
            # SECOND ATTEMPT: Pure vector search
            retrieved_contexts = self.search_knowledge_base(
                query, 
                filters=None
            )
        
        context_strings = [doc.get('raw_text_chunk', '') for doc in retrieved_contexts]

        if not retrieved_contexts:
            logger.warning("No relevant context found in the database.")
            return "I'm sorry, I could not find any relevant information in the clinical cases to answer your question."
            
        # 3. Build the new, improved prompt
        prompt = self.build_prompt(query, retrieved_contexts)
        

        # --- !! Step 4 - Choose generator !! ---
        logger.info("Generating final answer from context...")
        answer_text = ""
        try:
            if self.use_local_medgemma and self.medgemma_model:
                # --- Generate with Local MedGemma ---
                logger.info(f"Using local generator: {self.medgemma_model_name}")
                answer_text = self._generate_with_medgemma(prompt)
            
            else:
                # --- Generate with Gemini 2.5 Pro API ---
                logger.info(f"Using API generator: {self.GENERATOR_MODEL_NAME}")
                safety_settings_list = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
                config = {"safety_settings": safety_settings_list}
                
                response = self.client.models.generate_content(
                    model=GENERATOR_MODEL_NAME, 
                    contents=[prompt],
                    config=config
                )
                answer_text = response.text

            return answer_text, context_strings
        
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            logger.debug(f"Failed prompt:\n{prompt}") 
            return f"An error occurred while generating the response: {e}", context_strings
       

# --- Main execution block ---
if __name__ == "__main__":
    try:
        agent = RAGAgent()
        
        query = "What are the lab tests needed to accurately identify dengue fever?"
        
        print(f"\nTesting agent with query: '{query}'\n")
        answer, contexts = agent.ask(query)
        
        print("\n--- GENERATED ANSWER ---")
        print(answer)
        print("--------------------------")
        
    except Exception as e:
        logger.error(f"Failed to run RAGAgent: {e}")