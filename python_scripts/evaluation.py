import os
from rag_agent import RAGAgent
from evaluation_set import EVAL_SET 
from config import (
    GEMINI_API_KEY, 
    HUGGINGFACETOKEN, 
    logger
) 

from ragas import evaluate
from ragas.metrics import (
    faithfulness,                # How factual is the answer (based on context)?
    answer_relevancy,            # How relevant is the answer to the question?
    context_recall,              # Did you retrieve all the *needed* info?
    context_precision,           # Are the retrieved chunks *actually relevant*?
)
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_community.embeddings import HuggingFaceEmbeddings

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Instantiate the Embedding Model (for judging similarity)
EMBEDDING_MODEL_NAME = 'pritamdeka/S-BioBert-snli-multinli-stsb'
hf_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'token': HUGGINGFACETOKEN}
)

logger.info("RAGAS, Gemini LLM, and HF Embeddings configured.")

logger.info("Initializing RAGAgent...")
agent = RAGAgent()

results = []
logger.info(f"Running agent over {len(EVAL_SET)} evaluation questions...")
for item in EVAL_SET:
    query = item["question"]
    logger.info(f"Processing question: {query[:50]}...")
    
   
    answer, context_list = agent.ask(query)
    
    results.append({
        "question": query,
        "ground_truth": item["ground_truth"],
        "answer": answer,
        "contexts": context_list  # RAGAS expects 'contexts' as a list of strings
    })

logger.info("All questions processed. Results generated.")


eval_dataset = Dataset.from_list(results)


logger.info("Starting RAGAS evaluation...")
metrics_to_run = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]



result = evaluate(
    eval_dataset,
    metrics=metrics_to_run,
    llm=gemini_llm,
    embeddings=hf_embeddings
)


logger.info("Evaluation complete.")


print("\n--- RAGAS Evaluation Results ---")
print(result)
print("---------------------------------")


df = result.to_pandas()
print(df.head())

output_filename = "ragas_evaluation_results.csv"
df.to_csv(output_filename, index=False, encoding='utf-8')
logger.info(f"Evaluation results saved to {output_filename}")