import json
import random
import time
from pathlib import Path

import requests  # HTTP calls to Ollama API
from tqdm import tqdm #Progress bar

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "llama3.2:3B"
OUTPUT_FILE = Path(__file__).resolve().parent / "dataset.jsonl" #ensures the dataset is saved in the same folder as your script
NUM_EXAMPLES = 300
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2

resumes = [
    "Python developer with 2 years experience in Flask and REST APIs.",
    "Java backend engineer with 4 years experience in Spring Boot and MySQL.",
    "Data analyst with 3 years experience in Python, SQL, and Tableau.",
    "Frontend developer with 2 years experience in React and JavaScript.",
    "ML engineer with 1 year experience in scikit-learn and pandas.",
    "DevOps engineer with 5 years experience in Docker, Kubernetes, and AWS.",
    "Full stack developer with 3 years experience in Node.js and React.",
    "Android developer with 2 years experience in Java and Kotlin.",
    "Data scientist with 4 years experience in Python, TensorFlow, and NLP.",
    "Cloud engineer with 3 years experience in AWS, Terraform, and CI/CD.",
]

job_descriptions = [
    "Backend Python engineer needed. Must know Django or Flask, REST APIs, and PostgreSQL.",
    "Java developer required. Spring Boot experience mandatory. AWS knowledge is a plus.",
    "Data analyst role. SQL and Python required. Tableau or Power BI experience preferred.",
    "Frontend engineer needed. Strong React and JavaScript skills. TypeScript is a plus.",
    "Machine learning engineer. Must know Python, scikit-learn, and model deployment.",
    "DevOps role. Docker and Kubernetes required. CI/CD pipeline experience needed.",
    "Full stack developer. Node.js backend and React frontend experience required.",
    "Android developer needed. Kotlin preferred. Experience with REST APIs required.",
    "Data scientist position. NLP and deep learning experience required. Python mandatory.",
    "Cloud engineer role. AWS certified preferred. Terraform and infrastructure-as-code experience needed.",
]

PROMPT_TEMPLATE = """You are an expert HR recruiter and ATS system.

Analyze the following resume against the job description and respond in this EXACT format:

Match Score: [number]/100
Strengths: [2-3 specific strengths from the resume that match the JD]
Gaps: [2-3 specific skills or experiences missing from the resume]
Top Interview Questions:
1. [question]
2. [question]
3. [question]

Resume : {resume}

Job Description : {jd}

Respond only in the format above. No extra text."""

# Sanity Check 
def check_ollama_connection():
    """Fail fast with a clear message if Ollama is not reachable."""
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10) #get which model it have 
        response.raise_for_status()
    except requests.RequestException as err:
        raise RuntimeError(
            "Cannot connect to Ollama at http://localhost:11434. "
            "Start Ollama first, then run this script again."
        ) from err

# Engine 
def generate_example(resume, jd):
    prompt = PROMPT_TEMPLATE.format(resume=resume, jd=jd)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False, # Rather sending single text word-to-word, it waits until the entire response is finished...send one single package
    }

    # Retry loop helps keep long dataset runs stable if one request fails.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            # .get v/s .post = GET used to 'check' ; = POST request is used to 'send data' (payload to ollama address)
            response.raise_for_status()
            result = response.json()
            return result["response"].strip()
            # when ollama replies it gives extra information (like how long it took) . we want only text stored under the key "response"
        except (requests.RequestException, KeyError, ValueError) as err:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to generate example after {MAX_RETRIES} attempts: {err}"
                ) from err
            time.sleep(RETRY_DELAY_SECONDS)

# Orchestrator
def main():
    check_ollama_connection()
    print(f"Generating {NUM_EXAMPLES} examples...")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file_obj:
        # Write each entry immediately so partial progress is not lost on failures.
        for _ in tqdm(range(NUM_EXAMPLES), desc="Creating dataset", unit="example"):
            resume = random.choice(resumes)
            jd = random.choice(job_descriptions)
            output = generate_example(resume, jd)

            entry = {
                "instruction": "Analyze this resume against the job description.",
                "input": f"Resume: {resume}\n\nJob Description: {jd}",
                "output": output,
            }
            file_obj.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Done. Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
