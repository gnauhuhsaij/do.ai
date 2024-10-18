import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import boto3
from flask import Flask, request, jsonify
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

def get_secret():
    secret_name = "doai/openai/1015"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    
    # Parse the JSON string to extract the key
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['OPENAIAPI']  # Return only the value of 'api_key'



api_key = get_secret()
os.environ["OPENAI_API_KEY"] = api_key


split_smodel = ChatOpenAI(model="gpt-4o", organization='org-ukPTknDuMqCYzxWVt4Dhba3q')
split_prompt = """
    You are a human resource manager tasked with deciding whether a task is atomic and can be completed linearly. If not, you will break down a task into the smallest number of independent, high-level subtasks. Your goal is to ensure that each subtask can be started immediately and in parallel, without relying on the completion of other subtasks. 
    
    Each subtask should be written in a way that makes it clear how it links to the original task. For example, if "Research about the program" is a subtask of "Write a personal statement for the CMU Data Science Master Program," it should be written as "Research about the CMU Data Science Master Program."

    To ensure all subtasks can be started immediately at the same time, You need to specify the expected input and output of the subtask. Before generating each subtask, make sure the expected inputs is in the resource list above, and should ABSOLUTELY NOT include expected outputs of other steps.

    Format the response exactly as:
    Subtask 1, task_description, expected input, expected output;
    Subtask 2, task_description, expected input, expected output;
    ...

    Reflect on the responses and identify dependencies between each subtask. If a subtask's expected input is another's output, then they are sequential. Group all sequential tasks into one subtask and respond again.
    
    This is not your final response yet. Repeatedly reflect until you are confident that there is no dependencies at all, then give your final response.

    Format the Final response exactly as:
    [FINAL]
    Subtask 1| task_description, expected input, expected output;
    Subtask 2| task_description, expected input, expected output;
    ...

    The task to break down is: {user_input}.
"""
split_prompt_template = ChatPromptTemplate.from_messages(
    [("user", split_prompt)]
)
parser = StrOutputParser()

flow_model = ChatOpenAI(model="gpt-4o", organization='org-ukPTknDuMqCYzxWVt4Dhba3q')
flow_prompt = """
    You are an expert in designing optimized workflows for tasks that involve collaboration between humans and AI. You are given a subtask, and your goal is to turn this subtask into a serialized workflow. Keep in mind that this subtask is part of a larger set of parallel subtasks, so ensure that no duplicate steps are executed. For each step in the workflow, you need to decide the most effective way for humans and AI to collaborate by determining which portions should be handled by the AI, which should be handled by humans, or whether either can handle the step independently.

    Requirements:

    Context of the Parallel Subtasks: Consider these subtasks that are running in parallel:

    {subtasks}

    Avoid Duplication: Ensure that none of the steps in the workflow duplicate efforts already handled by the other parallel subtasks.

    Step-wise Breakdown: Break the subtask into a logical sequence of steps. For each step:

    Clearly state what the objective of the step is.
    Decide if this step can be completed by AI, humans, or a combination of both. Provide a rationale for the decision and explain how AI and humans can complement each other if necessary.
    If AI is involved, specify what models, techniques, or tools would be useful.
    If human input is required, outline what skills or expertise are needed.
    Some human input may be helped by AI assistence, e.g. a therapist/interviewer agent that asks the right questions which dig deeper than an average person.
    Efficiency: Aim to optimize the time and resource utilization between humans and AI, making sure both are used in ways that leverage their strengths.

    Here is the subtask you need to turn into a workflow:

    {current_task}

    Please create a serialized workflow for this subtask, considering the context of the other parallel tasks and providing clear reasoning for the collaboration strategy between humans and AI.

    In your output, only include the workflow without any other explanation.
"""
flow_prompt_template = ChatPromptTemplate.from_messages(
    [("user", flow_prompt)]
)

@app.route('/process', methods=['POST'])
def process_task():
    user_input = request.json.get('user_input', '')
    
    # Step 1: Break the task into subtasks
    chain = split_prompt_template | split_smodel | parser
    output = chain.invoke({"user_input": user_input})
    print(output)
    return jsonify({"workflow": output})
    parsed_output = output.split('[FINAL]')[1].split('Subtask')[1:]
    new_output = [i.split('| ')[1] for i in parsed_output]
    
    # Step 2: Process each subtask with AI-human collaboration
    flow_outputs = []
    for i in range(len(new_output)):
        current_task = new_output[i]
        subtasks = ""
        for j, task in enumerate(new_output[:i] + new_output[i+1:]):
            subtasks += f"Subtask {chr(65 + j)}: '{task.strip()}';\n"
        flow_chain = flow_prompt_template | flow_model | parser
        flow_output = flow_chain.invoke({"current_task": current_task, "subtasks": subtasks})
        flow_outputs.append(flow_output)
    
    # Combine the outputs into the final result
    final_output = {}
    for i in range(len(new_output)):
        final_output[f"Subtask {chr(65 + i)}"] = {
            "description": new_output[i],
            "workflow": flow_outputs[i]
        }

    return jsonify(final_output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)