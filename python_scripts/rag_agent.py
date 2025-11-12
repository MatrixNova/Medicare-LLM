# from google import genai
# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from config import (
#     QDRANT_ENDPOINT_URL, 
#     QDRANT_API_KEY, 
#     HUGGINGFACETOKEN, 
#     GEMINI_API_KEY, 
#     logger
# )
# from typing import List, Dict, Any
# 
# # --- Configuration ---
# # This is YOUR chosen embedding model
# EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
# 
# # This is your generative model (the "chat agent")
# GENERATOR_MODEL_NAME = 'gemini-2.5-pro'
# 
# # This is the Qdrant collection you uploaded your data to
# COLLECTION_NAME = "main-rag"
# 
# class RAGAgent:
#     def __init__(self):
#         """
#         Initializes the RAG agent by loading all necessary models and clients.
#         """
#         logger.info("Initializing RAGAgent...")
#         
#         # 1. Initialize Embedding Model (for querying)
#         logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
#         self.embed_model = SentenceTransformer(
#             EMBEDDING_MODEL_NAME, 
#             use_auth_token=HUGGINGFACETOKEN
#         )
#         logger.info("Embedding model loaded.")
#         
#         # 2. Initialize Qdrant Client (for retrieving)
#         logger.info(f"Connecting to Qdrant Cloud...")
#         if not QDRANT_ENDPOINT_URL or not QDRANT_API_KEY:
#             raise ValueError("QDRANT_ENDPOINT_URL or QDRANT_API_KEY not found in .env file.")
#         
#         self.qdrant_client = QdrantClient(
#             url=QDRANT_ENDPOINT_URL, 
#             api_key=QDRANT_API_KEY
#         )
#         logger.info("Qdrant Cloud connection successful.")
#         
#         # 3. Initialize Generative Model (for generating)
#         if not GEMINI_API_KEY:
#             raise ValueError("GEMINI_API_KEY not found in .env file.")
#             
#         logger.info(f"Initializing generative model: {GENERATOR_MODEL_NAME}")
#         #genai.configure(api_key=GEMINI_API_KEY)
#         #self.gen_model = genai.GenerativeModel(GENERATOR_MODEL_NAME)
#         self.client = genai.Client(api_key=GEMINI_API_KEY)
#         
#         logger.info("RAGAgent initialized successfully.")
# 
#     def search_knowledge_base(self, query: str, top_k: int = 40) -> List[Dict[str, Any]]:
#         """
#         Embeds the query and searches Qdrant for the top_k most relevant documents.
#         """
#         logger.info(f"Embedding query: '{query[:50]}...'")
#         # Use your S-BioBert model to create the query vector
#         query_vector = self.embed_model.encode(query).tolist()
#         
#         logger.info(f"Searching collection '{COLLECTION_NAME}' in Qdrant...")
#         search_results = self.qdrant_client.search(
#             collection_name=COLLECTION_NAME,
#             query_vector=query_vector,
#             limit=top_k,
#             with_payload=True  # This is crucial! It returns your JSON data
#         )
#         
#         # Extract just the payloads (your original JSON) from the search results
#         retrieved_contexts = [result.payload for result in search_results]
#         logger.info(f"Retrieved {len(retrieved_contexts)} contexts.")
#         return retrieved_contexts
# 
#     def build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
#         """
#         Builds a comprehensive prompt for the generative model,
#         using the structured JSON context you created.
#         """
#         # Convert the list of JSON payloads into a readable context string
#         context_str = "\n\n---\n\n".join(
#             [f"Source Document: {doc.get('source_filename', 'N/A')}\n"
#              f"Diagnosis: {doc.get('final_diagnosis', 'N/A')}\n"
#              f"Disease Name: {doc.get('disease_name_short', 'N/A')}\n"
#              f"History: {doc.get('history_of_present_illness', 'N/A')}\n"
#              f"Labs: {doc.get('labs_and_diagnostics', 'N/A')}" 
#              for doc in context_docs]
#         )
#         
#         prompt = f"""
#         You are an expert clinical diagnostic assistant. Your task is to answer the user's question based *only* on the provided clinical case # summaries.
#         
#         Do not use any external knowledge. If the answer is not in the provided context, state that clearly: "I could not find an answer in the # provided case reports."
#         
#         **PROVIDED CONTEXT:**
#         {context_str}
#         
#         **USER QUESTION:**
#         {query}
#         
#         **ASSISTANT ANSWER:**
#         """
#         return prompt
# 
#     def ask(self, query: str) -> str:
#         """
#         Main method to run the full RAG pipeline.
#         Query -> Embed -> Search -> Augment -> Generate
#         """
#         # retrieve context
#         retrieved_contexts = self.search_knowledge_base(query)
#         
#         if not retrieved_contexts:
#             logger.warning("No relevant context found in the database.")
#             return "I'm sorry, I could not find any relevant information in the clinical cases to answer your question."
#             
#         # build  prompt
#         prompt = self.build_prompt(query, retrieved_contexts)
#         
#         # generate the answer
#         logger.info("Generating final answer from context...")
#         try:
#             # Set safety settings to be less restrictive (medical data can be sensitive)
#             safety_settings_list = [
#                 {
#                     "category": "HARM_CATEGORY_HARASSMENT",
#                     "threshold": "BLOCK_NONE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_HATE_SPEECH",
#                     "threshold": "BLOCK_NONE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#                     "threshold": "BLOCK_NONE"
#                 },
#                 {
#                     "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#                     "threshold": "BLOCK_NONE"
#                 }
#             ]
#             config = {
#                 "safety_settings": safety_settings_list
#             }
#             response = self.client.models.generate_content(
#                 model=GENERATOR_MODEL_NAME,
#                 contents=[prompt], # 'contents' expects a list
#                 config=config
#             )
#             return response.text
#         except Exception as e:
#             logger.error(f"Error during answer generation: {e}")
#             return f"An error occurred while generating the response: {e}"
# 
# # --- Main execution block ---
# if __name__ == "__main__":
#     try:
#         agent = RAGAgent()
#         
#         # --- Test Query ---
#         test_query = "What is a definitive diagnostic method for Marburg virus disease?"
#         
#         print(f"\nTesting agent with query: '{test_query}'\n")
#         answer = agent.ask(test_query)
#         
#         print("\n--- GENERATED ANSWER ---")
#         print(answer)
#         print("--------------------------")
#         
#     except Exception as e:
#         logger.error(f"Failed to run RAGAgent: {e}")

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

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
# Using 2.5 Pro for high-quality generation
GENERATOR_MODEL_NAME = 'gemini-2.5-pro' 
# Using 2.5 Flash for fast, cheap filter extraction
FILTER_MODEL_NAME = 'gemini-2.5-flash' 
COLLECTION_NAME = "main-rag"

class RAGAgent:
    def __init__(self):
        logger.info("Initializing RAGAgent")

        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME
        self.GENERATOR_MODEL_NAME = GENERATOR_MODEL_NAME
        self.COLLECTION_NAME = COLLECTION_NAME
        
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
            # (THE FIX IS HERE)
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
             f"Relevant Text Snippet: {doc.get('raw_text_chunk', 'N/A')}"
             for doc in context_docs]
        )
        
        prompt = f"""
        You are an expert clinical diagnostic assistant. Your task is to answer the user's question based *only* on the provided clinical case snippets.
        
        Pay close attention to the "Relevant Text Snippet" for each source, as this is the text that most closely matched the user's query.
        Use the "Case Diagnosis" to understand the high-level context of the snippet.
        
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
        
        # 4. Generate the answer
        logger.info("Generating final answer from context...")
        try:
            # Set safety settings as a list of dictionaries
            safety_settings_list = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Create the config dictionary
            config = {
                "safety_settings": safety_settings_list
            }
            
            response = self.client.models.generate_content(
                model=GENERATOR_MODEL_NAME, # No "models/" prefix
                contents=[prompt],          # contents must be a list
                config=config
            )
            return response.text, context_strings
        
        except Exception as e:
            logger.error(f"Error during answer generation: {e}")
            logger.debug(f"Failed prompt:\n{prompt}") 
            return f"An error occurred while generating the response: {e}", context_strings

# --- Main execution block ---
if __name__ == "__main__":
    try:
        agent = RAGAgent()
        
        query = "A patient has 5-day history of fever, generalized abdominal pain, and frontal headache. What is their most likely diagnosis?"
        
        print(f"\nTesting agent with query: '{query}'\n")
        answer, contexts = agent.ask(query)
        
        print("\n--- GENERATED ANSWER ---")
        print(answer)
        print("--------------------------")
        
    except Exception as e:
        logger.error(f"Failed to run RAGAgent: {e}")