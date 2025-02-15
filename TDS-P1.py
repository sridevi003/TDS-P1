import os
import requests
import json
import shutil
import pandas as pd
import datetime
import sqlite3
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

class TaskRequest(BaseModel):
    task: str

def is_path_safe(file_path):
    data_dir = "/data"
    abs_path = os.path.abspath(file_path)
    return abs_path.startswith(data_dir)

@app.post("/run")
async def run_task(request: TaskRequest):
    task = request.task

    try:
        # 1. Interact with the LLM
        llm_prompt = f"""
        Parse the following task description and extract the relevant information in JSON format:

        ```
        {task}
        ```

        Provide a JSON object with the following keys where applicable:
        - operation: The main operation to perform (e.g., "format", "count", "extract", "install").
        - file_path: The path to the file to operate on.
        - tool: The tool to use (e.g., "prettier").
        - version: The version of the tool (if applicable).
        - other_details: Any other relevant details (e.g., date format, email field).

        If the task is not understandable, return: {{"error": "Invalid task description"}}
        """

        headers = {
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
            "Content-Type": "application/json"
        }
        llm_response = requests.post(
            "https://api.proxy.ai/v1/complete",
            headers=headers,
            json={"model": "gpt-4o-mini", "prompt": llm_prompt}
        )

        llm_data = llm_response.json()

        if llm_response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"LLM API error: {llm_data}")

        try:
            task_info = json.loads(llm_data["choices"][0]["text"])
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON from LLM")

        if "error" in task_info:
            raise HTTPException(status_code=400, detail=task_info["error"])

        operation = task_info.get("operation")
        file_path = task_info.get("file_path")
        tool = task_info.get("tool")
        version = task_info.get("version")
        other_details = task_info.get("other_details")

        if file_path and not is_path_safe(file_path):
            raise HTTPException(status_code=400, detail="Access outside /data directory is not allowed")


        # --- Phase A ---
        if operation == "install" and tool == "uv":  # A1
            try:
                subprocess.run(["pip", "install", "uv"], check=True, capture_output=True, text=True)
                return {"status": "success", "message": "uv installed"}
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"uv installation error: {e.stderr}")

        elif operation == "run" and other_details == "datagen.py":  # A1
            user_email = "user@example.com"  # Extract from task if needed
            try:
                subprocess.run(["python3", "datagen.py", user_email], check=True, capture_output=True, text=True)
                return {"status": "success", "message": "datagen.py executed"}
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"datagen.py execution error: {e.stderr}")

        elif operation == "format" and tool == "prettier":  # A2
            try:
                subprocess.run(["npx", "prettier", "--write", file_path], check=True, capture_output=True, text=True)
                return {"status": "success", "message": "File formatted"}
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"Prettier error: {e.stderr}")

        elif operation == "count" and other_details == "wednesdays":  # A3
            try:
                with open(file_path, "r") as f:
                    dates = f.readlines()
                    wednesday_count = sum(1 for date in dates if "Wed" in date)
                output_file = file_path.replace(".txt", "-wednesdays.txt")
                with open(output_file, "w") as f:
                    f.write(str(wednesday_count))
                return {"status": "success", "message": f"Wednesdays counted: {wednesday_count}"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"{file_path} not found")

        elif operation == "sort" and other_details == "contacts":  # A4
            try:
                with open(file_path, "r") as f:
                    contacts = json.load(f)
                sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
                output_file = file_path.replace(".json", "-sorted.json")
                with open(output_file, "w") as f:
                    json.dump(sorted_contacts, f, indent=4)
                return {"status": "success", "message": "Contacts sorted"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"{file_path} not found")

        elif operation == "extract" and other_details == "logs":  # A5
            try:
                log_files = sorted([f for f in os.listdir("/data/logs") if f.endswith(".log")], key=lambda x: os.path.getmtime(os.path.join("/data/logs", x)), reverse=True)
                recent_logs = log_files[:10]
                first_lines = []
                for log_file in recent_logs:
                    with open(os.path.join("/data/logs", log_file), "r") as f:
                        first_line = f.readline().strip()
                        first_lines.append(first_line)
                output_file = "/data/logs-recent.txt"
                with open(output_file, "w") as f:
                    f.write("\n".join(first_lines))
                return {"status": "success", "message": "Recent logs extracted"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"/data/logs directory not found")

        elif operation == "extract" and other_details == "markdown_titles":  # A6
            try:
                index = {}
                for filename in os.listdir("/data/docs"):
                    if filename.endswith(".md"):
                        filepath = os.path.join("/data/docs", filename)
                        with open(filepath, "r") as f:
                            content = f.read()
                            soup = BeautifulSoup(content, "html.parser")
                            h1_tags = soup.find_all("h1")
                            if h1_tags:
                                index[filename] = h1_tags[0].text.strip()
                            else:
                                index[filename] = None  # Or handle no H1 as you prefer

                with open("/data/docs/index.json", "w") as f:
                    json.dump(index, f, indent=4)
                return {"status": "success", "message": "Markdown titles extracted"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f)

        if operation == "extract" and other_details == "email_sender":  # A7
            try:
                with open(file_path, "r") as f:
                    email_content = f.read()

                llm_prompt = f"""Extract the sender's email address from the following email content:
                ```
                {email_content}
                ```
                Return only the email address as a single line of text. If no sender email is found, return "No sender email found".
                """
                headers = {
                    "Authorization": f"Bearer {AIPROXY_TOKEN}",
                    "Content-Type": "application/json"
                }
                llm_response = requests.post(
                    "https://api.proxy.ai/v1/complete",
                    headers=headers,
                    json={"model": "gpt-4o-mini", "prompt": llm_prompt}
                )
                llm_data = llm_response.json()
                if llm_response.status_code != 200:
                    raise HTTPException(status_code=500, detail=f"LLM API error: {llm_data}")

                sender_email = llm_data["choices"][0]["text"].strip()

                output_file = file_path.replace(".txt", "-sender.txt")
                with open(output_file, "w") as f:
                    f.write(sender_email)
                return {"status": "success", "message": "Sender email extracted"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"{file_path} not found")

        elif operation == "extract" and other_details == "credit_card":  # A8
            raise HTTPException(status_code=501, detail="Credit card extraction requires OCR, which is not implemented.")

        elif operation == "find" and other_details == "similar_comments":  # A9
            try:
                with open(file_path, "r") as f:
                    comments = f.readlines()
                model = SentenceTransformer('all-mpnet-base-v2')
                embeddings = model.encode(comments)

                most_similar_pair = None
                max_similarity = -1

                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity = util.cos_sim(embeddings[i], embeddings[j])
                        if similarity > max_similarity:
                            max_similarity = similarity
                            most_similar_pair = (comments[i].strip(), comments[j].strip())

                output_file = file_path.replace(".txt", "-similar.txt")
                with open(output_file, "w") as f:
                    if most_similar_pair:
                        f.write(f"{most_similar_pair[0]}\n{most_similar_pair[1]}")
                    else:
                        f.write("")  # Or handle no similar pair as you prefer
                return {"status": "success", "message": "Similar comments found"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"{file_path} not found")

        elif operation == "fetch" and other_details == "image":  # A10
            try:
                img_url = "https://via.placeholder.com/150"  # Example, extract from task if needed
                img_data = requests.get(img_url).content
                output_file = "/data/image.jpg"  # Extract name from task if needed
                with open(output_file, "wb") as f:
                    f.write(img_data)
                return {"status": "success", "message": "Image downloaded"}
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"Image download error: {e}")

        # --- Phase B ---
        elif operation == "fetch" and other_details == "api_data":  # B1
            try:
                api_url = "https://jsonplaceholder.typicode.com/posts/1"  # Extract URL from task if needed
                response = requests.get(api_url)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                output_file = "/data/api_response.json"
                with open(output_file, "w") as f:
                    json.dump(response.json(), f, indent=4)  # Save as JSON
                return {"status": "success", "message": "API data fetched"}
            except requests.exceptions.RequestException as e:
                raise HTTPException(status_code=500, detail=f"API request error: {e}")

        elif operation == "clone" and other_details == "git_repo":  # B2
            raise HTTPException(status_code=501, detail="Git cloning is not implemented.")

        elif operation == "query" and other_details == "sqlite_db":  # B3
            try:
                db_path = "/data/ticket-sales.db"
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
                result = cursor.fetchone()[0]
                conn.close()
                output_file = "/data/ticket-sales-gold.txt"
                with open(output_file, "w") as f:
                    f.write(str(result or 0))  # Write 0 if result is None
                return {"status": "success", "message": "SQLite query executed"}
            except sqlite3.Error as e:
                raise HTTPException(status_code=500, detail=f"SQLite error: {e}")
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail=f"{db_path} not found")

        elif operation == "scrape" and other_details == "website":  # B4
            raise HTTPException(status_code=501, detail="Web scraping is not implemented.")

        elif operation == "compress" and other_details == "image":  # B5
            raise HTTPException(status_code=501, detail="Image compression is not implemented.")

        elif operation == "transcribe" and other_details == "audio":  # B6
            raise HTTPException(status_code=501, detail="Audio transcription is not implemented.")

        elif operation == "convert" and other_details == "markdown_to_html":  # B7
            raise HTTPException(status_code=501, detail="Markdown to HTML conversion is not implemented.")

        elif operation == "create" and other_details == "api_endpoint":  # B8
            raise HTTPException(status_code=501, detail="API endpoint creation is not implemented.")

        elif operation == "merge" and other_details == "csv_files":  # B9
            try:
                df1 = pd.read_csv("/data/file1.csv")
                df2 = pd.read_csv("/data/file2.csv")
                merged_df = pd.concat([df1, df2], ignore_index=True)
                merged_df.to_csv("/data/merged.csv", index=False)
                return {"status": "success", "message": "CSV files merged"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="CSV files not found")
            except pd.errors.EmptyDataError:  # Handle empty files
                raise HTTPException(status_code=400, detail="One or both CSV files are empty.")

        elif operation == "check" and other_details == "missing_values":  # B10
            try:
                df = pd.read_csv("/data/data.csv")
                missing = df.isnull().sum()
                with open("/data/missing_values.txt", "w") as f:
                    f.write(missing.to_string())
                return {"status": "success", "message": "Missing values checked"}
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="data.csv not found")

        else:
            raise HTTPException(status_code=400, detail="Unsupported operation or invalid task description")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
