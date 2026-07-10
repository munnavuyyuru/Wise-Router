import json
import random
import os

random.seed(42)

TASKS_PER_TYPE = 70
TASK_TYPES = [
    "math", "sentiment", "code_debug", "code_gen",
    "summarization", "ner", "logic", "factual",
]

MATH_TEMPLATES = [
    "What is {} + {}?",
    "Calculate {} - {}.",
    "What is {}% of {}?",
    "Average of {}, {}, {}.",
    "Solve x + {} = {}.",
    "What is {} x {}?",
    "If {} items cost ${}, how much do {} cost?",
    "What is {} / {}?",
]

SENTIMENT_TEMPLATES = [
    "What is the sentiment of: '{}'?",
    "Classify the sentiment: '{}'",
    "Is this positive or negative: '{}'",
    "Sentiment analysis: '{}'",
    "How does this feel: '{}'",
]

CODE_DEBUG_TEMPLATES = [
    "Fix this code:\ndef {}():\n    pass\n",
    "Debug this:\n{}",
    "What's wrong with:\n{}",
    "Fix the bug in:\n{}",
]

CODE_GEN_TEMPLATES = [
    "Write a function to {}.",
    "Implement {} in Python.",
    "Create a {} algorithm.",
    "Write code for {}.",
]

SUMMARIZATION_TEMPLATES = [
    "Summarize in {} sentences: '{}'",
    "Give a brief summary of: '{}'",
    "Summarize: '{}'",
]

NER_TEMPLATES = [
    "Extract names from: '{}'",
    "Find all emails in: '{}'",
    "Extract entities: '{}'",
    "NER from text: '{}'",
    "Identify the named entities: '{}'",
]

LOGIC_TEMPLATES = [
    "{} is taller than {}. {} is taller than {}. Who is tallest?",
    "{} is before {}. {} is after {}. What is the order?",
    "If {} > {} and {} > {}, which is greatest?",
]

FACTUAL_TEMPLATES = [
    "What is the capital of {}?",
    "Who invented {}?",
    "What is {}?",
]


POSITIVE_PHRASES = [
    "The product was absolutely amazing and exceeded all expectations",
    "Great service, friendly staff, wonderful experience overall",
    "I loved it! Beautiful design and fantastic quality",
    "Excellent customer support, very helpful and efficient",
    "The best purchase I've made all year, highly recommend",
]

NEGATIVE_PHRASES = [
    "Terrible experience, worst customer service ever",
    "The product was broken and the staff was rude",
    "Disappointed with the quality, it's a waste of money",
    "Awful, wouldn't recommend to anyone",
    "Horrible. Slow, expensive, and poorly made",
]

NEUTRAL_PHRASES = [
    "The meeting is scheduled for 3 PM tomorrow",
    "Please find attached the document you requested",
    "The package was delivered on Tuesday",
    "Here is the quarterly report for review",
    "The system will be down for maintenance from 2-4 AM",
]


def generate() -> list[dict]:
    data = []

    for ttype in TASK_TYPES:
        for i in range(TASKS_PER_TYPE):
            if ttype == "math":
                a, b = random.randint(1, 100), random.randint(1, 100)
                c, d = random.randint(1, 20), random.randint(1, 10)
                template = random.choice(MATH_TEMPLATES)
                if "{} + {}" in template or "{} - {}" in template or "{} x {}" in template or "{} / {}" in template:
                    prompt = template.format(a, b)
                elif "% of" in template:
                    prompt = template.format(random.randint(5, 50), random.randint(50, 500))
                elif "Average" in template:
                    prompt = template.format(a, b, a + b)
                elif "Solve" in template:
                    prompt = template.format(b, a + b)
                elif "items cost" in template:
                    q = random.randint(2, 10)
                    prompt = template.format(q, q * random.randint(2, 20), random.randint(1, 5) * q)
                else:
                    prompt = template.format(a, b)

            elif ttype == "sentiment":
                phrase = random.choice(POSITIVE_PHRASES + NEGATIVE_PHRASES + NEUTRAL_PHRASES)
                prompt = random.choice(SENTIMENT_TEMPLATES).format(phrase)

            elif ttype == "code_debug":
                func_name = random.choice(["process_data", "calculate", "validate", "parse_input", "transform"])
                prompt = random.choice(CODE_DEBUG_TEMPLATES).format(func_name)

            elif ttype == "code_gen":
                task_desc = random.choice([
                    "sort a list of numbers", "find the maximum element",
                    "reverse a string", "check if a number is prime",
                    "calculate fibonacci", "validate an email address",
                ])
                prompt = random.choice(CODE_GEN_TEMPLATES).format(task_desc)

            elif ttype == "summarization":
                sentences = random.randint(1, 3)
                text = " ".join(random.sample([
                    "The quick brown fox jumps over the lazy dog.",
                    "Python is a high-level programming language.",
                    "Machine learning is transforming many industries.",
                    "The capital of France is Paris, known for its art and culture.",
                    "Renewable energy sources like solar and wind are growing rapidly.",
                ], 3))
                prompt = random.choice(SUMMARIZATION_TEMPLATES).format(sentences, text)

            elif ttype == "ner":
                sample = random.choice([
                    "John Smith (john@example.com) from New York",
                    "Contact support@company.com or call 555-123-4567",
                    "Meeting on January 15, 2024 at $500 per person",
                    "Alice and Bob work at Google in Mountain View",
                ])
                prompt = random.choice(NER_TEMPLATES).format(sample)

            elif ttype == "logic":
                names = random.sample(["Alice", "Bob", "Charlie", "Diana", "Eve"], 4)
                prompt = random.choice(LOGIC_TEMPLATES).format(*names)

            elif ttype == "factual":
                factual_items = [
                    "France", "Germany", "Japan", "United Kingdom",
                    "Python", "light bulb", "water", "gold",
                ]
                entity = random.choice(factual_items)
                prompt = random.choice(FACTUAL_TEMPLATES).format(entity)

            data.append({"prompt": prompt, "task_type": ttype})

    random.shuffle(data)
    return data


if __name__ == "__main__":
    data = generate()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    out = out.replace("\\", "/")
    print(f"Generated {len(data)} training samples -> {out}")
