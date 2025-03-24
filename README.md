# LLM Workflow Automation
This project automates content repurposing using Large Language Models (LLMs). It converts blog posts into summaries, social media posts, and email newsletters using different workflow approaches:

Sequential Workflow: Processes tasks step by step.

Self-Improving Workflow: Enhances content using feedback loops.

Agent-Orchestrated Workflow: An AI agent decides the best task order.

Workflow Comparison: Evaluates quality and speed.

Prerequisites
Ensure you have:

Python 3.10+

OpenAI API Key

Required Libraries (install using):

sh
Copy
Edit
pip install openai python-dotenv
Setup
Clone the repository

sh
Copy
Edit
git clone https://github.com/your-username/llm-workflow-project.git
cd llm-workflow-project
Create a .env file and add:

ini
Copy
Edit
API_KEY=your_openai_api_key
LLM_MODEL=gpt-4
Add a sample blog post in sample-blog.json:

json
Copy
Edit
{
  "title": "How AI is Changing Healthcare",
  "content": "AI is revolutionizing the healthcare industry...",
  "author": "Dr. Jane Doe"
}
Running the Code
Run the script:

sh
Copy
Edit
python llm_workflow.py
What Happens?
The script reads the blog post.

It applies different workflows to generate content.

Outputs are displayed in JSON format.

Workflow Overview
Sequential Workflow: Generates content in a fixed order.

Self-Improving Workflow: Refines outputs with self-correction.

Agent-Orchestrated Workflow: Dynamically decides the task sequence.

Comparing Workflows
To compare performance, run:

sh
Copy
Edit
python llm_workflow.py
Results include:

Quality scores

Execution time

Best workflow recommendations
