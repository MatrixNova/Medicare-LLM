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
from abc import ABC, abstractmethod  # Import Abstract Base Class

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    LOCAL_LIBS_AVAILABLE = True
except ImportError:
    LOCAL_LIBS_AVAILABLE = False

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
COLLECTION_NAME = "main-rag"

# --- !! MODIFIED: Model selection is now just a string !! ---
# This generator is used for the final answer.
# Change this string to "google/medgemma-4b-it" to use MedGemma.
# Change this string to "gemini-2.5-pro" to use that model.
GENERATOR_MODEL_NAME = 'google/medgemma-4b-it' 

# This generator is used for filter extraction.
FILTER_MODEL_NAME = 'gemini-2.5-flash'
# --- !! END MODIFIED !! --- 

# ===================================================================
# --- Step 1: Define the Generator Strategy (The "Interface") ---
# ===================================================================
class GeneratorStrategy(ABC):
    """
    Abstract Base Class for all generator models.
    It guarantees that every generator has a .generate() method.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Takes a full prompt and returns the model's text response."""
        pass

# ===================================================================
# --- Step 2: Create Concrete Strategies (The "Implementations") ---
# ===================================================================

class GeminiAPIStrategy(GeneratorStrategy):
    """
    Generator strategy for all Google Gemini API models.
    It uses the genai.Client to make API calls.
    """
    def __init__(self, model_name: str, client: genai.Client, is_json_output: bool = False):
        super().__init__(model_name)
        self.client = client
        self.is_json_output = is_json_output
        logger.info(f"[Strategy] Initialized GeminiAPIStrategy for {model_name}")

    def generate(self, prompt: str) -> str:
        safety_settings_list = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
        
        config = {"safety_settings": safety_settings_list}
        if self.is_json_output:
            config["response_mime_type"] = "application/json"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=config
            )
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemini API call for {self.model_name}: {e}")
            return f"Error generating response: {e}"


class LocalHuggingFaceStrategy(GeneratorStrategy):
    """
    Generator strategy for any local Hugging Face CausalLM model
    (like MedGemma, Llama, Mistral, etc.).
    """
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if not LOCAL_LIBS_AVAILABLE:
            raise ImportError("LocalHuggingFaceStrategy requires 'transformers' and 'torch'.")
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        logger.info(f"[Strategy] Loading local model: {self.model_name}...")
        if self.device == "cpu":
            logger.warning("--- NO GPU DETECTED ---")
            logger.warning(f"Loading {self.model_name} on CPU. Generation will be EXTREMELY SLOW.")
        else:

            # --- !! THIS IS THE MEMORY FIX !! ---
            # We are forcing 4-bit quantization to fit the model on your 4GB card.
            # This requires the `bitsandbytes` library.
            try:
                from transformers import BitsAndBytesConfig
                
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                
                logger.info(f"CUDA GPU detected. Loading {self.model_name} with 4-bit quantization.")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, 
                    token=os.environ.get("HF_TOKEN")
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    quantization_config=quant_config, # Apply the 4-bit config
                    token=os.environ.get("HF_TOKEN")
                )
                
            except ImportError:
                logger.error("`bitsandbytes` library not found. Cannot apply 4-bit quantization.")
                logger.error("Please run: pip install bitsandbytes")
                raise
            except Exception as e:
                logger.error(f"Error during model quantization: {e}")
                logger.warning("Falling back to non-quantized loading. This will likely fail.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map=self.device)
            # --- !! END OF FIX !! ---

            logger.info(f"CUDA GPU detected. Loading {self.model_name} to GPU.")

        # Load model with quantization for efficiency
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            token=os.environ.get("HF_TOKEN") # Use env var for token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            quantization_config=None, # Add bitsandbytes config here if needed
            token=os.environ.get("HF_TOKEN")
        )
        logger.info(f"[Strategy] Local model {self.model_name} loaded successfully.")

    def generate(self, prompt: str) -> str:
        # Apply the chat template for the specific model
        chat_prompt = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            chat_prompt, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.device)

        # Generate output
        output_ids = self.model.generate(
            inputs,
            max_new_tokens=1500,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # Decode and return
        response_text = self.tokenizer.batch_decode(
            output_ids[:, inputs.shape[1]:],
            skip_special_tokens=True
        )[0]
        return response_text.strip()


# ===================================================================
# --- Step 3: Create the Factory (The "Model Switch") ---
# ===================================================================
class GeneratorFactory:
    """
    Builds the correct generator strategy object based on the model name.
    """
    def __init__(self, genai_client: genai.Client):
        self.genai_client = genai_client
        self.loaded_local_models = {} # Cache for local models

    def create_generator(self, model_name: str) -> GeneratorStrategy:
        if model_name.startswith("gemini-"):
            # It's a Google API model
            is_json = "json" in model_name # Simple check
            return GeminiAPIStrategy(model_name, self.genai_client, is_json_output=is_json)
        
        elif model_name.startswith("google/medgemma") or "/" in model_name:
            # It's a Hugging Face model (e.g., "google/medgemma-4b-it" or "meta-llama/Llama-3-8B")
            # We cache it so we don't reload the 4B model every time
            if model_name not in self.loaded_local_models:
                self.loaded_local_models[model_name] = LocalHuggingFaceStrategy(model_name)
            return self.loaded_local_models[model_name]

        else:
            raise ValueError(f"Unknown model type or name: {model_name}")
# ===================================================================

class RAGAgent:
    def __init__(self):
        logger.info("Initializing RAGAgent")
        
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.COLLECTION_NAME = COLLECTION_NAME
        # The "main" generator is the one selected at the top
        self.GENERATOR_MODEL_NAME = GENERATOR_MODEL_NAME

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
        
        # 3. Initialize Generative Client and Factory (REFACTORED)
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        self.genai_client = genai.Client(api_key=GEMINI_API_KEY)
        
        # --- !! THIS IS THE FIX !! ---
        # The factory is ONLY for the swappable answer generator.
        self.generator_factory = GeneratorFactory(self.genai_client)

        # Filter extraction should ALWAYS use the fast/cheap API model.
        # This prevents loading two local models.
        self.filter_generator = GeminiAPIStrategy(
            model_name=FILTER_MODEL_NAME, 
            client=self.genai_client, 
            is_json_output=True
        )
        
        # Now, create ONLY the answer generator using the factory.
        self.answer_generator = self.generator_factory.create_generator(self.GENERATOR_MODEL_NAME)
        
        logger.info(f"Using generator for answers: {self.GENERATOR_MODEL_NAME}")
        logger.info(f"Using generator for filters: {FILTER_MODEL_NAME} (API)")
        # --- !! END OF FIX !! ---
        logger.info("RAGAgent initialized successfully.")

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
            # --- !! REFACTORED !! ---
            # Just call .generate() on the strategy object.
            # No more API-specific logic here.
            response_text = self.filter_generator.generate(prompt)
            # --- !! END REFACTORED !! ---
            
            filter_data = json.loads(response_text)
            disease_name = filter_data.get("disease_name_short")
            
            if disease_name:
                logger.info(f"Found filter: disease_name_short = {disease_name}")
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
        logger.info(f"Performing {search_type} search in collection '{self.COLLECTION_NAME}'...")
        
        search_results = self.qdrant_client.search(
            collection_name=self.COLLECTION_NAME, # Use self.
            query_vector=query_vector,
            query_filter=filters, # THis enables hybrid search
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
        

        # --- !! REFACTORED !! ---
        # 4. Generate the answer
        logger.info(f"Generating final answer using {self.answer_generator.model_name}...")
        try:
            # The RAGAgent just calls .generate().
            # It doesn't know or care if it's a local model or an API.
            answer_text = self.answer_generator.generate(prompt)
            return answer_text, context_strings
        
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            logger.debug(f"Failed prompt:\n{prompt}") 
            return f"An error occurred while generating the response: {e}", context_strings
        # --- !! END REFACTORED !! ---

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