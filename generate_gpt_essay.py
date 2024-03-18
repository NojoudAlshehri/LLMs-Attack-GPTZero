import os
import openai

# Initialize models with a valid API key
openai.api_key = "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"

# Define the paths
student_essay_path = "human"
gpt_essay_path = "gpt_essays"
gpt_prompt_path = "gpt_prompts"

def generate_gpt_essay(essay_prompt: str) -> str:   
    prompt = generate_gpt_prompt(student_essay)
    
    print("Calling GPT API for essay generation...")
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",  
        messages=[
            {"role": "system", "content": "You are a  writer"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=4090
    )
    return response.choices[0].message.content.strip()


def generate_gpt_prompt(student_essay: str) -> str:   
    prompt = (f"Generate a prompt corresponding to the student essay here:\n{student_essay}")

    print("Calling GPT API for prompt generation...")
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",  
        messages=[
            {"role": "system", "content": "You are a good essay writer"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=1_000  
    )
    return response.choices[0].message.content.strip()

# Create the gpt_essay_path if it doesn't exist
os.makedirs(gpt_essay_path, exist_ok=True)

# Loop through the essay files
for i in range(1, 1001):
    print(i)
    file_name = f"{i}.txt"
    student_essay_file_path = os.path.join(student_essay_path, file_name)
    gpt_essay_file_path = os.path.join(gpt_essay_path, file_name)
    gpt_prompt_file_path = os.path.join(gpt_prompt_path, file_name)
    
    # Read the student essay
    with open(student_essay_file_path, 'r', encoding='utf-8') as file:
        student_essay = file.read()
    
    # Generate essay with GPT
    gpt_essay = generate_gpt_essay(student_essay)
    
    # Save the GPT-generated essay
    with open(gpt_essay_file_path, 'w', encoding='utf-8') as file:
        file.write(gpt_essay)

print("Generation completed.")