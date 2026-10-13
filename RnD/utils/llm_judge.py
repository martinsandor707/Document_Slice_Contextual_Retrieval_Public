import pandas as pd
import instructor
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI
from devtools import debug
from tqdm import tqdm

# 1. Define the Schema for Scientific Rigor
# By using Pydantic, we force the LLM to 'think' in structured data.
class RelevanceEvaluation(BaseModel):
    chain_of_thought: str = Field(
        ..., 
        description="A brief reasoning step explaining why the score was given."
    )
    score: Literal[0,1,2,3] = Field(
        ..., 
        description="The relevance score (0, 1, 2, or 3) based on the grading rubric."
    )

# 2. Configure the Client
# We point the OpenAI client to the local Ollama endpoint.
MODEL_TAG = "qwen3-vl:8b-instruct-q4_K_M"

client = instructor.from_provider(
    f"ollama/{MODEL_TAG}",
    base_url="http://localhost:11434/v1",
    mode=instructor.Mode.JSON,
)

def grade_chunk_relevance(question: str, chunk_text: str, model_name: str) -> RelevanceEvaluation:
    """
    Uses Qwen to grade a single chunk against a question.
    Returns the integer score (0-3).
    """
    
    # Precise rubric for the system prompt
    system_prompt = """
    You are an impartial expert judge evaluating retrieval quality for a RAG system.
    Evaluate the relevance of the PASSAGE to the QUESTION using this strict scale:
    
    0: Irrelevant. The passage is on a different topic or does not help.
    1: Tangential. Mentions related entities but does not answer the question.
    2: Relevant/Partial. Provides useful context or a partial answer.
    3: Highly Relevant. Contains the direct answer or core evidence required.
    """

    try:
        resp = client.create(
            model=model_name,
            response_model=RelevanceEvaluation,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION: {question}\nPASSAGE: {chunk_text}"}
            ],
            temperature = 0
        )
        return resp
    except Exception as e:
        print(f"Error grading chunk: {e}")
        return 0 # Fail-safe: assume irrelevant if model crashes

# --- Example Usage with your DataFrame ---

# Mock Data (Replace 'df' with your actual dataframe)
data = {
    'question': ["What is the capital of France?", "How does photosynthesis work?"],
    'retrieved_texts': [
        "Paris is the capital of France.", 
        "The Eiffel Tower is a tall structure."
    ]
}
df = pd.DataFrame(data)

print(f"Starting evaluation using {MODEL_TAG}...\n")

# Apply the grading function
# Note: For large datasets, consider using a loop with tqdm for progress bars


for idx, row in tqdm(df.iterrows(), total=len(df), desc="Grading Chunks"):
    resp = grade_chunk_relevance(row['question'], row['retrieved_texts'], MODEL_TAG)
    df.at[idx, 'llm_grade'] = resp.score
    df.at[idx, 'reasoning'] = resp.chain_of_thought


# Display results
debug(df)