import argparse
import subprocess
import importlib
from concurrent.futures import ThreadPoolExecutor
import os, json, time

log_path = os.path.join("result", "log", time.strftime("%Y-%m-%d-%H:%M:%S"))
if not os.path.exists(log_path):
    os.makedirs(log_path)

def run_script(script_name, task_id, *args):
    """Run a Python script with given arguments and log output and errors to files."""
    log_file = time.strftime("%Y-%m-%d-%H:%M:%S") + f"_task_{task_id}_log.txt"
    log_file = os.path.join(log_path, log_file)
    try:
        # Create the command to run the script with arguments
        command = ['python', script_name] + list(args)
        print(f"Running command: {' '.join(command)}")
        
        # Write the result and errors to the log file
        with open(log_file, 'w') as f:
            f.write(f"Running command: {' '.join(command)}\n")
            # Run the command and capture the output
            result = subprocess.run(command, capture_output=False, text=True, stdout=f, stderr=f)
            # f.write(f"Output of {script_name}:\n{result.stdout}\n")
            # if result.stderr:
            #     f.write(f"Errors of {script_name}:\n{result.stderr}\n")
                
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Failed to run {script_name}: {e}\n")

def load_file(file):
    if file.endswith(".py"):
        raise NotImplementedError("py not yet.")
    elif file.endswith(".json"):
        with open(file, "rt") as f:
            config = json.load(f)
    else:
        raise NotImplementedError("This kind of file not allowed.")
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", default="example/tasks.json",type=str)
    parser.add_argument("--n_thread",  default=4, type=int)
    # Define the scripts and their arguments
    args = parser.parse_args()
    task_file:str = args.task_file
    config = load_file(task_file)
    if config["info"] == "tasks":
        tasks = config["tasks"]
    if config["info"] == "task":
        tasks = [config["task"]]
    scripts_with_args = []
    for idx, task in enumerate(tasks):
        one_task = [task["script"], idx]
        if "file" in task["config"].keys():
            tconfig = load_file(task["config"]["file"])
        else:
            tconfig = task["config"]
        for key, val in tconfig.items():
            one_task += ["--" + key, str(val)]
        scripts_with_args.append(one_task)

    # Number of parallel threads
    num_threads = args.n_thread # len(scripts_with_args)
    
    # Run the scripts in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for script_name, task_id, *args in scripts_with_args:
            time.sleep(5)
            futures.append(executor.submit(run_script, script_name, task_id, *args))

        
        # Wait for all futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
