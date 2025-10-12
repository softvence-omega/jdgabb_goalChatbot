# api/v1/endpoints/project_task_question.py
import openai
import asyncio
from fastapi import APIRouter, HTTPException
from app.schemas.project import UserInput, ProjectGoal, Answer, SubTask, TaskDetails
from app.services.openai_service import generate_text
from app.utils.task_utils import extract_tasks

router = APIRouter()

# In-memory storage for projects
projects = {}

# Route to start a new project
@router.post("/start_project/")
async def start_project(user_input: UserInput):
    # Generate a simple project title based on the user's input
    project_id = len(projects)
    user_message = user_input.user_message
    
    response = await generate_text(f"Make a simple project title like the message in one line: {user_message}. No need to change the line. Chack only grammar. Do not include any additional text. Only provide the project title.")
    projects[project_id] = {
        "goal": response,
        "tasks": [],
        "answered_questions": [],
    }
    return {"project_id": project_id, "project_goal": response}

# Route to add a task to a project
@router.post("/add_task/{project_id}/")
async def add_task(project_id: int, project_goal: ProjectGoal):
    # Check if the project exists
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    
    # Extract tasks from the task paragraph using OpenAI
    tasks = await extract_tasks(project_goal.add_task)
    
    # Track the existing task descriptions to avoid duplication
    existing_task_descriptions = [task["task"] for task in project["tasks"]]
    
    # Add each task to the project under the goal with sequential numbering
    for i, task in enumerate(tasks, start=1):
        task_description = f"{task.strip()} for project goal: {project['goal']}"
        
        # Only add the task if it doesn't already exist
        if task_description not in existing_task_descriptions:
            project["tasks"].append({
                "task": task_description,
                "subtasks": [],
            })
        else:
            # If the task already exists, we skip it to avoid duplication
            continue

    return {"project_id": project_id, "tasks": project["tasks"]}


# Route to add details to a task
@router.post("/add_task_details/{project_id}/{task_index}/")
async def add_task_details(project_id: int, task_index: int, task_details: TaskDetails):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    if task_index >= len(project["tasks"]):
        raise HTTPException(status_code=404, detail="Task not found")

    # Add the task details to the selected task
    project["tasks"][task_index]["details"] = task_details.details
    
    return {"project_id": project_id, "tasks": project["tasks"]}

# Route to add a subtask to a task
@router.post("/add_subtask/{project_id}/{task_index}/")
async def add_subtask(project_id: int, task_index: int, subtask: SubTask):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    if task_index >= len(project["tasks"]):
        raise HTTPException(status_code=404, detail="Task not found")

    # Add the subtask to the task's subtask list
    project["tasks"][task_index]["subtasks"].append(subtask.subtask)
    
    return {"project_id": project_id, "tasks": project["tasks"]}



# Route to ask a question about the project goal
@router.post("/ask/{project_id}/")
async def ask_question(project_id: int):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    
    # Check if no question has been asked yet
    if not project.get("answered_questions"):
        # Generate the first question for the project goal
        question = await generate_text(f"Based on this project goal, what is the first question to ask: {project['goal']}")
        project["answered_questions"] = [{"question": question, "answer": None}]
    else:
        # If there are questions, get the next question based on the previous answers
        question = await generate_text(f"Based on the previous questions, what is the next question to ask: {project['goal']}")
        project["answered_questions"].append({"question": question, "answer": None})
    
    return {"question": question}

# Route to answer a question for the project
@router.post("/answer_question/{project_id}/")
async def answer_question(project_id: int, answer: Answer):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")

    project = projects[project_id]
    
    # Check if there is a question to answer
    if not project["answered_questions"]:
        raise HTTPException(status_code=400, detail="No question to answer")
    
    # Store the answer in the most recent question
    project["answered_questions"][-1]["answer"] = answer.answer
    
    return {"message": "Answer stored successfully"}


# New route for chat with project assistant
@router.post("/chat/{project_id}/")
async def chat_with_project_assistant(project_id: int, user_message: str):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    
    project = projects[project_id]
    
    # Gather all the information about the project
    project_goal = project["goal"]
    tasks = project["tasks"]
    answered_questions = project.get("answered_questions", [])
    
    # Format project context for OpenAI
    context = f"Project Goal: {project_goal}\n\n"
    
    context += "Tasks:\n"
    for task in tasks:
        context += f"- {task['task']}\n"
        for subtask in task['subtasks']:
            context += f"  - Subtask: {subtask}\n"
        if 'details' in task and task['details']:
            context += f"  - Details: {task['details']}\n"
    
    context += "\nAnswered Questions:\n"
    for q_a in answered_questions:
        context += f"Q: {q_a['question']}\n"
        context += f"A: {q_a['answer']}\n"

    # Add user's message to the context
    context += f"\nUser's message: {user_message}\n\nAssistant, based on this information, answer the user's query."

    # Get response from OpenAI with the context
    response = await generate_text_with_context(context)

    return {"project_id": project_id, "response": response}

# Function to get OpenAI response with context
async def generate_text_with_context(context: str):
    try:
        response = await asyncio.to_thread(openai.ChatCompletion.create,
                                            model="gpt-3.5-turbo",  # You can change this to "gpt-4" or any model you prefer
                                            messages=[
                                                {"role": "system", "content": "You are a helpful assistant named OLLIE that helps users with project details."},
                                                {"role": "user", "content": context}
                                            ],
                                            max_tokens=300,
                                            temperature=0.7,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

# Route to get project details
@router.get("/get_project/{project_id}/")
async def get_project(project_id: int):
    if project_id not in projects:
        raise HTTPException(status_code=404, detail="Project not found")
    return projects[project_id]
