from unsloth import FastLanguageModel
import json

MODEL_DIR = "training/output"
MAX_SEQ_LENGTH = 2048

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model) #Locks model from learning and rather generating text quickly

# Test pairs — unseen during training
test_pairs = [
    {
        "resume": "Python developer with 3 years experience in Django, REST APIs, and PostgreSQL.",
        "jd": "Backend engineer needed. Django or Flask required. PostgreSQL and Docker experience preferred."
    },
    {
        "resume": "Frontend developer with 2 years experience in React and TypeScript.",
        "jd": "Full stack developer needed. React frontend and Node.js backend experience required."
    },
    {
        "resume": "Data scientist with 4 years experience in Python, TensorFlow, and NLP.",
        "jd": "ML engineer role. Must know PyTorch and model deployment. NLP experience is a plus."
    },
    {
        "resume": "Java developer with 1 year experience in Spring Boot.",
        "jd": "Senior Java engineer. 5+ years Spring Boot required. Microservices and Kafka experience needed."
    },
    {
        "resume": "DevOps engineer with 3 years experience in Docker, Kubernetes, and AWS.",
        "jd": "Cloud engineer needed. AWS certified preferred. Terraform and CI/CD pipeline experience required."
    },
]

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert HR recruiter and ATS system.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Analyze this resume against the job description.

Resume: {resume}

Job Description: {jd}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

def evaluate(resume, jd):
    prompt = PROMPT_TEMPLATE.format(resume=resume, jd=jd)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only assistant response
    response = decoded.split("assistant")[-1].strip()
    return response

print("=" * 60)
print("EVALUATION — Fine-tuned RecruitSheriff Model")
print("=" * 60)

for i, pair in enumerate(test_pairs):
    print(f"\nTest {i+1}")
    print(f"Resume: {pair['resume']}")
    print(f"JD: {pair['jd']}")
    print("Output:")
    output = evaluate(pair["resume"], pair["jd"])
    print(output)
    print("-" * 60)