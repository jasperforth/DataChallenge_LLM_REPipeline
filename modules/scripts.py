import pandas as pd
import tiktoken
import json
import re

from pathlib import Path
from datetime import datetime, timezone, timedelta
from openai import OpenAI

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define CEST timezone (UTC+2 during daylight saving time)
CEST = timezone(timedelta(hours=2))


# TODO - refactor as classes


def create_tasks(prompts: list, 
                 model: str="gpt-4o", 
                 temperature: int=0):
    """Create a list of tasks for processing prompts."""
    tasks = []
    for index, prompt in enumerate(prompts):
        task = {
            "custom_id": f"task-{index}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
            }
        }
        tasks.append(task)
    print(f"Created {len(tasks)} tasks")
    return tasks


def save_tasks_to_file(tasks: list, 
                       file_path: Path):
    """Save tasks to a JSONL file."""
    with open(file_path, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')
    print(f"Tasks saved to {file_path}")


def upload_batch_file(client: OpenAI, 
                      file_path: Path, 
                      purpose: str="batch"):
    """Upload the batch file to the client."""
    with open(file_path, "rb") as file:
        batch_file = client.files.create(file=file, purpose=purpose)
    print(f"Batch file uploaded: {batch_file}")
    return batch_file


def create_tasks_batch(prompts: list, 
                       client: OpenAI, 
                       tmp_dir: Path, 
                       step : str,
                       model: str="gpt-4o", 
                       temperature=0):
    if not prompts:
        raise ValueError("The prompts list is empty. Please provide valid prompts.")

    tasks = create_tasks(prompts, model, temperature)
    file_name = Path(f"batchinput_{step}.jsonl")
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    batch_file_path = tmp_dir / file_name
    save_tasks_to_file(tasks, batch_file_path)
    batch_file = upload_batch_file(client, batch_file_path)
    return batch_file


def load_newest_job_id(file_path: Path, step: str):
    if not file_path.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with file_path.open('r', encoding='utf-8') as file:
        data = json.load(file)
        
        if not data:
            raise ValueError("The file is empty.")
        
        # Filter data by task_nr
        filtered_data = [entry for entry in data if entry['step'] == step]
        
        if not filtered_data:
            raise ValueError(f"No job ID found for task number {step}")
        
        # Sort the data by timestamp in descending order
        data_sorted = sorted(data, key=lambda x: x['timestamp'], reverse=True)
        
        logging.info(f"The newest job ID for task {data_sorted[0]['step']} is: {data_sorted[0]['job_id']}")
        logging.info(f"Timestamp: {data_sorted[0]['timestamp']}")
        # Return the newest job_id
        return data_sorted[0]['job_id']
    

def add_job_to_file(file_path: Path, job_id: str, step: str):
    """Add job ID to a file with a timestamp."""
    # Ensure the directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    # Initialize an empty list if the file doesn't exist
    if not file_path.exists():
        data = []
    else:
        # Read existing data
        with file_path.open('r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []

    # Check if job_id already exists
    if not any(job['job_id'] == job_id for job in data):
        # Add new job with a timestamp
        new_entry = {
            "step": step, 
            "job_id": job_id,
            "timestamp": datetime.now(CEST).isoformat()
        }
        data.append(new_entry)
        
        # Write the updated data back to the file
        with file_path.open('w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)     
        logging.info(f"Job ID {job_id} added to the file.")
    else:
        logging.info(f"Job ID {job_id} already exists in the file.")
    

def retrieve_batch_job_status(client, job_id):
    """Retrieve the status of the batch job."""
    batch_job = client.batches.retrieve(job_id)
    status = batch_job.status
    batch_job_id = batch_job.id

    # Access the input_file_id directly
    input_file_id = batch_job.input_file_id
    
    completed = batch_job.request_counts.completed
    failed = batch_job.request_counts.failed
    total = batch_job.request_counts.total

    logging.info(f"Batch Job Status: {status}")
    logging.info(f"Batch Job ID: {batch_job_id}")
    logging.info(f"Input File ID: {input_file_id}")
    logging.info(f"Request Counts: Completed: {completed}, Failed: {failed}, Total: {total}")

    return {
        "status": status,
        "batch_job_id": batch_job_id,
        "input_file_id": input_file_id,
        "completed": completed,
        "failed": failed,
        "total": total
    }


def parse_and_clean_batch_responses(result):
    """Parse and clean JSON responses from a result string."""
    results = []
    for line in result.splitlines():
        try:
            # Parsing the JSON string into a dict and appending to the list of results
            json_object = json.loads(line.strip())
            results.append(json_object)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding line: {e}")
            continue

    logging.info(f"Parsed {len(results)} JSON objects.")

    cleaned_responses = []
    for res in results:
        try:
            raw_response = res['response']['body']['choices'][0]['message']['content']
            cleaned_response = clean_json_response(raw_response)
            # Load the cleaned JSON and append each design's data
            cleaned_json = json.loads(cleaned_response)
            # cleaned_responses.extend(cleaned_json)
            for item in cleaned_json:
                cleaned_responses.append(item)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding cleaned response: {e}")
            continue
        except KeyError as e:
            logging.error(f"Key error: {e}")
            continue

    logging.info(f"Cleaned {len(cleaned_responses)} responses.")
    
    df_responses = pd.DataFrame(cleaned_responses, index=None)
    df_responses["design_id"] = df_responses["design_id"].astype(int)
    logging.info("DataFrame created from cleaned responses.")
    return df_responses

############
# -------------------

def get_chat_completion(prompt, client, model="gpt-4o"):
    stream = client.chat.completions.create(
        model=model,
        # response_format={ 
        #     "type": "json_object"
        # },
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ],
        stream=True, 
        temperature=0  # Controls randomness, set to 0 for deterministic output
    )

    # Initialize an empty response string
    response = ""
    # Iterate over each chunk received from the stream
    for chunk in stream:
        # Check if the chunk contains text content and append it to the response
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content

    return response

def process_prompts(prompts, client, batch_start, batch_stop):
    responses_list = []
    total_output_tokens = 0
    total_output_price = 0

    for idx, prompt in enumerate(prompts[batch_start:batch_stop]):
        logging.debug(f"Processing prompt {idx + batch_start}: {prompt}")
        completion = get_chat_completion(prompt, client)
        logging.debug(f"Received completion: {completion}")

        if completion.strip() == "":
            raise ValueError("Received an empty response from the model.")
        
        # Calculate and log token count and price for the completion
        completion_token_count = count_tokens_prompt(completion)
        completion_price = calculate_output_price(completion_token_count)
        logging.info(f"Token count for completion: {completion_token_count}, Price: ${completion_price:.5f}")

        cleaned_completion = clean_json_response(completion)
        logging.debug(f"Cleaned completion: {cleaned_completion}")

        try:
            entries = json.loads(cleaned_completion)
            for entry in entries:
                responses_list.append(entry)
        except json.JSONDecodeError as e:
            logging.error("Final JSON format error:", e)
            logging.debug("Debug - Final cleaned response:\n%s", cleaned_completion)
            raise e

    return responses_list


def filter_source_dataframe(source_df: pd.DataFrame, 
                            json_dir: Path, 
                            json_filename:str ="enhanced_designs.json"):
    json_filepath = Path(json_dir) / json_filename
    if not json_filepath.exists():
        Path(json_dir).mkdir(parents=True, exist_ok=True)
        print(f"JSON file {json_filepath} does not exist.")
        return source_df 

    target_df = pd.read_json(json_filepath)
    design_ids = set(target_df['design_id'])
    filtered_source_df = source_df[~source_df['id'].isin(design_ids)].copy()

    return filtered_source_df


def filter_enhanced_designs(source_df: pd.DataFrame, 
                            json_dir: Path, 
                            json_filename: str="subject_object_pairs.json"):
    json_filepath = Path(json_dir) / json_filename
    if not json_filepath.exists():
        print(f"JSON file {json_filepath} does not exist.")
        return source_df 

    target_df = pd.read_json(json_filepath)
    design_ids = set(target_df['design_id'])
    filtered_source_df = source_df[~source_df['design_id'].isin(design_ids)].copy()

    return filtered_source_df



def filter_sop_dataframe(source_df, json_dir, json_filename="subject_object_pairs_with_predicates.json"):
    json_filepath = Path(json_dir) / json_filename
    if not json_filepath.exists():
        print(f"JSON file {json_filepath} does not exist.")
        return source_df 

    target_df = pd.read_json(json_filepath)
    target_df_not_nan = target_df[~target_df['predicate'].isna()]
    pairs_to_drop = set(zip(target_df_not_nan['design_id'], target_df_not_nan['s_o_id']))
    filtered_source_df = source_df[~source_df.apply(lambda row: (row['design_id'], row['s_o_id']) in pairs_to_drop, axis=1)]
    
    return filtered_source_df


def update_json_with_merged_df(merged_df: pd.DataFrame, 
                               columns: list, 
                               json_dir: Path, 
                               json_filename: str):
    json_filepath = Path(json_dir) / json_filename
    if json_filepath.exists():
        existing_df = pd.read_json(json_filepath)
    else:
        existing_df = pd.DataFrame(columns=columns)
        
    merged_df = pd.concat([existing_df, merged_df], ignore_index=True)

    merged_df.to_json(json_filepath, orient='records', indent=4)


# def clean_json_response(response):
#     cleaned_response = response.strip().strip("```").strip("json").strip()
#     cleaned_response = re.sub(r"#.*", "", cleaned_response)
#     cleaned_response = re.sub(r"//.*", "", cleaned_response)
#     cleaned_response = re.sub(r'(\w+):', r'"\1":', cleaned_response)
#     cleaned_response = re.sub(r": '([^']*)'", r': "\1"', cleaned_response)
#     if not cleaned_response.startswith("["):
#         cleaned_response = "[" + cleaned_response + "]"
#     cleaned_response = re.sub(r'"design_id":\s*"(\d+)"', r'"design_id": \1', cleaned_response)
#     cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
#     cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
#     cleaned_response = re.sub(r'\(\s*"(.*?)"\s*,\s*"(.*?)"\s*\)', r'["\1", "\2"]', cleaned_response)

#     try:
#         json.loads(cleaned_response)
#     except json.JSONDecodeError as e:
#         logging.error("JSON format error: %s", e)
#         logging.debug("Debug - Cleaned response:\n%s", cleaned_response[:200])
#         raise ValueError(f"Invalid JSON format: {e}")

#     return cleaned_response


def clean_json_response(response):
    # Initial cleaning to remove extraneous characters or code block delimiters
    cleaned_response = response.strip().strip("```").strip("json").strip()
    cleaned_response = re.sub(r"#.*", "", cleaned_response)
    cleaned_response = re.sub(r"//.*", "", cleaned_response)
    cleaned_response = re.sub(r'(\w+):', r'"\1":', cleaned_response)  # Enclose keys in quotes
    cleaned_response = re.sub(r": '([^']*)'", r': "\1"', cleaned_response)  # Enclose single-quoted values in double quotes
    if not cleaned_response.startswith("["):
        cleaned_response = "[" + cleaned_response + "]"  # Enclose in brackets if not already
    cleaned_response = re.sub(r'"design_id":\s*"(\d+)"', r'"design_id": \1', cleaned_response)
    cleaned_response = re.sub(r',\s*}', '}', cleaned_response)
    cleaned_response = re.sub(r',\s*]', ']', cleaned_response)
    cleaned_response = re.sub(r'\(\s*"(.*?)"\s*,\s*"(.*?)"\s*\)', r'["\1", "\2"]', cleaned_response)

    try:
        json.loads(cleaned_response)
    except json.JSONDecodeError as e:
        logging.error("JSON format error: %s", e)
        logging.debug("Debug - Cleaned response:\n%s", cleaned_response[:200])  # Log first 200 characters for debugging
        fixed_response = re.sub(r'(\w+):', r'"\1":', cleaned_response)
        fixed_response = re.sub(r'": (\w+)', r'": "\1"', fixed_response)
        fixed_response = re.sub(r'\}\s*\{', r'}, {', fixed_response)
        try:
            json.loads(fixed_response)
            logging.info("JSON format error fixed.")
            cleaned_response = fixed_response
        except json.JSONDecodeError as e:
            logging.error("Invalid JSON format after recovery attempt: %s", e)
            logging.debug("Debug - Fixed response with error:\n%s", fixed_response[:200])
            raise ValueError(f"Invalid JSON format after recovery attempt: {e}")
    
    return cleaned_response


def calculate_total_tokens_and_price(prompts: list, 
                                     batch_start: int, 
                                     batch_stop: int, 
                                     batch: bool=False):
    total_tokens = 0
    
    for idx, prompt in enumerate(prompts):
        if idx >= (batch_stop - batch_start):
            continue
        token_count = count_tokens_prompt(prompt)
        total_tokens += token_count
        price = calculate_input_price(token_count)
        if batch:
            price *= 0.5
        print(f"Token count for prompt {idx}: {token_count}, Price: ${price:.5f}")

    total_price = calculate_input_price(total_tokens)
    if batch:
        total_price *= 0.5
    print(f"Total token count: {total_tokens}")
    print(f"Total input price: ${total_price:.5f}")
    
    return total_tokens, total_price


def count_tokens_prompt(prompt, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt)
    return len(tokens)


def count_tokens_dict(data_dict, model="gpt-4o"):
    json_str = json.dumps(data_dict)
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(json_str)

    return len(tokens)


def calculate_input_price(token_count):
    return token_count / 1000000 * 5


def calculate_output_price(token_count):
    return token_count / 1000000 * 15


def generate_list_of_strings(row):
    list_of_strings = [(row["design_en"][start:stop], obj) for start, stop, obj in row["annotations"]]
    return list_of_strings


def query_design_by_id(df: pd.DataFrame, specific_id: int):
    # Filter the DataFrame for the specific ID
    result_row = df[df['id'] == specific_id]
    if result_row.empty:
        return f"No design found for ID: {specific_id}"

    result_row = result_row.iloc[0]
    full_design = result_row["design_en"]
    objects = [obj for str, obj in result_row["list_of_strings"]]
    strings = [str for str, obj in result_row["list_of_strings"]]

    return {
        "id": specific_id,
        "full_design": full_design,
        "strings": strings,
        "objects": objects,
    }





