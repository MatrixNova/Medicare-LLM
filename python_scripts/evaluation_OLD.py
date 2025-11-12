import os
import json
import polars as pl
import wandb
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision
)
from python_scripts.config import (
    logger,
    GEMINI_API_KEY,
    WANDB_API_KEY,
    WANDB_PROJECT,
    WANDB_ENTITY,
    HUGGINGFACETOKEN
)
from .rag_agent import RAGAgent

# We need numpy to calculate the average (mean) of the scores
import numpy as np

# --- RAGAs Configuration ---
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
# Use the correct LangChain wrapper for Sentence Transformers
from langchain_community.embeddings import HuggingFaceEmbeddings


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# This LLM is used by RAGAs to "judge" the results.
ragas_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0))

# This embedding model is used by RAGAs for its internal metric calculations.
logger.info("Loading S-BioBert via LangChain for RAGAs evaluation...")
model_kwargs = {'use_auth_token': HUGGINGFACETOKEN}
ragas_embeddings = HuggingFaceEmbeddings(
    model_name="pritamdeka/S-BioBert-snli-multinli-stsb",
    model_kwargs=model_kwargs
)
logger.info("RAGAs embedding model loaded.")
# --- End of RAGAs Configuration ---


EVAL_FILE = "evaluation_set.jsonl"


class RAGEvaluator:
    def __init__(self, agent: RAGAgent):
        self.rag_agent = agent
        logger.info("RAG Evaluator initialized with RAGAs metrics.")

    def load_evaluation_set(self, filepath: str):
        """
        Loads the evaluation set from a JSONL file.
        Returns:
          questions: list[str]
          ground_truth_answers_list: list[Any]
        """
        questions = []
        ground_truth_answers_list = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {filepath} line {lineno}: {e}")
                        continue

                    q = data.get("question")
                    if q is None:
                        logger.warning(f"No 'question' field in line {lineno}. Skipping.")
                        continue
                    questions.append(q)

                    gt_answer = data.get("ground_truth_answer")
                    if gt_answer is None:
                        logger.warning(f"No ground_truth_answer for question (line {lineno}): {q}")
                        ground_truth_answers_list.append(None)
                    else:
                        ground_truth_answers_list.append(gt_answer)

        except FileNotFoundError:
            logger.error(f"Evaluation file not found: {filepath}")
            return [], []
        except Exception as e:
            logger.error(f"Error loading evaluation set: {e}", exc_info=True)
            return [], []

        return questions, ground_truth_answers_list

    # Helper functions to robustly coerce ground-truth formats:
    @staticmethod
    def extract_first_string(x):
        """
        Recursively extract the first string from nested lists/tuples.
        Returns '' if nothing string-like is found.
        """
        if isinstance(x, str):
            return x
        if isinstance(x, (list, tuple)) and x:
            return RAGEvaluator.extract_first_string(x[0])
        return ""

    def run_rag_for_evaluation(self, questions: list) -> dict:
        """
        Runs the RAG agent for all questions and collects data in the format
        needed for RAGAs (question, answer, contexts).
        """
        logger.info(f"Running RAG agent for {len(questions)} questions...")
        results = {"question": [], "answer": [], "contexts": []}

        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}: {question[:80]}...")
            answer = "Error during generation"
            contexts_str_list = []
            try:
                retrieved_contexts_dicts = self.rag_agent.search_knowledge_base(question)

                if not retrieved_contexts_dicts:
                    logger.warning("No context found by agent.")
                    answer = "No Context Found"
                else:
                    # Save contexts as JSON strings (RAGAs expects contexts list[str])
                    contexts_str_list = [json.dumps(doc) for doc in retrieved_contexts_dicts]
                    prompt = self.rag_agent.build_prompt(question, retrieved_contexts_dicts)
                    safety_settings = {
                        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
                        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
                        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
                    }
                    answer = self.rag_agent.gen_model.generate_content(
                        prompt,
                        safety_settings=safety_settings
                    ).text

            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}", exc_info=True)
                if "quota" in str(e).lower():
                    logger.error("Quota exceeded. Stopping generation.")
                    break

            results["question"].append(question)
            results["answer"].append(answer)
            results["contexts"].append(contexts_str_list)

        return results

    def run_evaluation(self):
        """Loads dataset, runs agent, evaluates with RAGAs, and logs to W&B."""

        try:
            wandb.login(key=WANDB_API_KEY)
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                job_type="evaluation",
                config={
                    "embedding_model": self.rag_agent.EMBEDDING_MODEL_NAME,
                    "generator_model": self.rag_agent.GENERATOR_MODEL_NAME,
                    "judge_model": "gemini-2.5-flash",
                    "collection_name": self.rag_agent.COLLECTION_NAME
                }
            )
            logger.info(f"W&B Run initialized. View at: {wandb_run.url}")
        except Exception as e:
            logger.error(f"Failed to initialize W&B. Check API key/project settings. {e}")
            wandb_run = None

        questions, ground_truth_answers_list = self.load_evaluation_set(EVAL_FILE)
        if not questions:
            if wandb_run:
                wandb_run.finish()
            return

        rag_results = self.run_rag_for_evaluation(questions)

        min_len = min(len(rag_results["question"]), len(ground_truth_answers_list))
        if min_len == 0:
            logger.error("No RAG results generated. Cannot evaluate.")
            if wandb_run:
                wandb_run.finish()
            return

        if min_len < len(questions):
            logger.warning(f"Only evaluating {min_len}/{len(questions)} due to errors.")

        # --- [NEW SIMPLIFIED DATASET PREP] ---
        dataset_list = []
        for i in range(min_len):
            raw_gt_answer = ground_truth_answers_list[i]
            gt_answer_str = self.extract_first_string(raw_gt_answer)

            dataset_list.append({
                "question": rag_results["question"][i],
                "answer": rag_results["answer"][i],
                "contexts": rag_results["contexts"][i],
                "ground_truth": gt_answer_str
            })

        logger.info("Verifying dataset for RAGAs...")
        for i, s in enumerate(dataset_list[:5]):
            gt_val = s.get("ground_truth")
            if not isinstance(gt_val, str):
                logger.error(f"Sample {i} has invalid ground_truth type: {type(gt_val)}")
                raise TypeError(f"Dataset preparation failed: ground_truth must be a string. Got: {type(gt_val)}")
            logger.info(f"Sample {i} verified. ground_truth_type={type(gt_val)}")

        dataset = Dataset.from_list(dataset_list)
        logger.info("HF Dataset created successfully.")
        # --- [END OF SIMPLIFIED BLOCK] ---


        logger.info("Running RAGAs evaluation (This may take several minutes)")
        metrics_to_run = [
            faithfulness,
            answer_relevancy,
            context_precision,
        ]

        try:
            run_config = RunConfig(max_workers=1)

            result = evaluate(
                dataset=dataset,
                metrics=metrics_to_run,
                llm=ragas_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=True, 
                run_config=run_config
            )
            logger.info("--- RAGAs Evaluation Complete ---")

            print("\n--- RAGAs Evaluation Summary ---")
            print(result)
            print("------------------------------")

            # --- [FINAL FIX BLOCK v3] ---
            # 'result' contains LISTS of scores. We must handle this.
            
            # 1. Create a dict with the LISTS for W&B histograms
            list_summary_dict = {
                "faithfulness_scores": result['faithfulness'],
                "answer_relevancy_scores": result['answer_relevancy'],
                "context_precision_scores": result['context_precision']
            }
            logger.info(f"Logging score lists to W&B: {list_summary_dict}")

            # 2. Create a dict with the AVERAGES for the CSV and W&B summary
            avg_summary_dict = {
                "faithfulness": np.mean(result['faithfulness']),
                "answer_relevancy": np.mean(result['answer_relevancy']),
                "context_precision": np.mean(result['context_precision'])
            }
            logger.info(f"Logging summary averages: {avg_summary_dict}")

            if wandb_run:
                # Log both! Histograms and the final average numbers
                wandb.log(list_summary_dict)
                wandb.log(avg_summary_dict)
            # --- [END FINAL FIX BLOCK] ---

            # Save summary results
            try:
                # [FIX 2] Save the AVERAGE scores to the CSV
                df = pl.DataFrame([avg_summary_dict])
                df.write_csv("ragas_evaluation_summary_results.csv")
                logger.info("Summary results saved to ragas_evaluation_summary_results.csv")
            except Exception as e:
                logger.error(f"Failed to save summary CSV: {e}")

            # Save the detailed per-question scores
            try:
                # [FIX 3] To save CSV/W&B Table, we must "flatten" the 'contexts' list
                dataset_for_csv = dataset.map(
                    lambda x: {"contexts_str": str(x["contexts"])}
                )
                
                detailed_df = dataset_for_csv.to_pandas()
                
                # Drop the original nested column to avoid errors
                if "contexts" in detailed_df.columns:
                     detailed_df = detailed_df.drop(columns=["contexts"])
                
                detailed_pl_df = pl.from_pandas(detailed_df)
                detailed_pl_df.write_csv("ragas_evaluation_detailed_results.csv")
                logger.info("Detailed results saved to ragas_evaluation_detailed_results.csv")
                
                if wandb_run:
                    # Log the "safe" dataframe to W&B
                    wandb_table = wandb.Table(dataframe=detailed_df)
                    wandb_run.log({"evaluation_details": wandb_table})
            except Exception as e:
                logger.warning(f"Could not save detailed per-question results: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"RAGAs evaluation failed: {e}", exc_info=True)

        finally:
            if wandb_run:
                wandb_run.finish()
                logger.info("W&B run finished.")


if __name__ == "__main__":
    try:
        agent = RAGAgent()
        evaluator = RAGEvaluator(agent=agent)
        evaluator.run_evaluation()
    except Exception as e:
        logger.error(f"Failed to run evaluation: {e}", exc_info=True)