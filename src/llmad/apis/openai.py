import math
import multiprocessing
import signal
import time

from openai import OpenAI
from tqdm import tqdm

global_process_list = []


def moderation_api(inputs):
    """
    Calls the OpenAI moderation API for each input in the given list and returns the results.

    Args:
        inputs (list): A list of inputs to be sent to the moderation API.

    Returns:
        list: A list of results returned by the moderation API for each input.
    """
    results = []
    client = OpenAI()

    for ipt in inputs:
        response = client.moderations.create(input=ipt)
        results.append(response.results)

    return results


def openai_chat_server(call_queue, leader=False):
    """
    A function that listens to a call queue for incoming tasks, and processes them using OpenAI's API.

    Args:
    call_queue (Queue): A queue object that contains incoming tasks. These are made of the following elements:
        id: id for this task.
        message: a string representing the user's message prompt.
        max_tokens: an integer representing the maximum number of tokens to generate.
        kwargs: a dictionary containing optional keyword arguments to be passed to the call_openai function.
        dest_queue: a queue object where the result of the task will be put.

    Returns:
        None
    """
    client = OpenAI()

    while True:
        task = call_queue.get(block=True)
        if task is None:
            return

        compl_id, message, max_tokens, kwargs, dest_queue = task
        rslt = call_openai(client, message, max_tokens, **kwargs)
        if rslt == 0 and not leader:
            call_queue.put(task)
            print("Reducing the number of OpenAI threads due to Rate Limit")
            return
        elif rslt == 0 and leader:
            call_queue.put(task)
        else:
            dest_queue.put((compl_id, rslt))


def call_openai(
    client,
    message,
    max_tokens,
    query_type="chat",
    model="gpt-3.5-turbo",
    temperature=1.0,
    top_p=1,
    presence_penalty=0,
    frequency_penalty=0,
    system_prompt=None,
    stop=None,
    timeout=None,
    n=1,
    history=None,
):
    """
    Calls the OpenAI API to generate text based on the given parameters.

    Args:
        client (openai.api_client.Client): The OpenAI API client.
        message (str): The user's message prompt.
        max_tokens (int): The maximum number of tokens to generate.
        query_type (str): The type of completion to use. Defaults to "chat".
        model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-3.5-turbo".
        temperature (float, optional): Controls the "creativity" of the generated text. Higher values result in more diverse text. Defaults to 1.0.
        top_p (float, optional): Controls the "quality" of the generated text. Higher values result in higher quality text. Defaults to 1.
        presence_penalty (float, optional): Controls how much the model avoids repeating words or phrases from the prompt. Defaults to 0.
        frequency_penalty (float, optional): Controls how much the model avoids generating words or phrases that were already generated in previous responses. Defaults to 0.
        system_prompt (str, optional): A prompt to be included before the user's message prompt. Defaults to None.
        stop (str, optional): A stop sequence
        timeout (int, optional): The maximum time to wait for a response from the API, in seconds. Defaults to 10.
        n (int, optional): The number of responses to generate. Defaults to 1.

    Returns:
        The generated responses from the OpenAI API.
    """

    if history is None:
        history = []

    def loop(f, params):
        retry = 0
        while retry < 7:
            try:
                return f(params)
            except Exception as e:
                if retry > 5:
                    print(f"Error {retry}: {e}\n{params}")
                if "maximum context length" in str(e):
                    print("Context length exceeded")
                    return None
                if (
                    "Rate limit" in str(e)
                    or "overloaded" in str(e)
                    or "timed out" in str(e)
                ):
                    if "timed out" in str(e) and retry < 2:
                        params["timeout"] += 30 * retry
                    elif retry < 1:
                        time.sleep(30 * (1 + retry))
                    else:
                        print(e)
                        return 0

                else:
                    time.sleep(3 * retry)
                retry += 1
                continue
        return None

    request_params = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "n": n,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "stop": stop,
    }

    if max_tokens != math.inf:
        request_params["max_tokens"] = max_tokens

    if timeout is not None:
        request_params["timeout"] = timeout

    if query_type == "chat":
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(message)},
            ]
        else:
            messages = [{"role": "user", "content": message}]
        request_params["messages"] = history + messages
        return loop(
            lambda x: client.chat.completions.create(**x), request_params
        )

    request_params["prompt"] = message
    return loop(lambda x: client.completions.create(**x), request_params)


def init_servers(number_of_processes=4):
    """
    Initializes multiple chat servers using multiprocessing.

    Args:
        number_of_processes (int): The number of server processes to start. Default is 4.

    Returns:
        tuple: A tuple containing a call queue and a global manager object.
    """
    global_manager = multiprocessing.Manager()
    call_queue = global_manager.Queue()

    for i in range(number_of_processes):
        p = multiprocessing.Process(
            target=openai_chat_server, args=(call_queue, i == 0)
        )
        p.start()
        global_process_list.append(p)

    return call_queue, global_manager


def kill_servers():
    """
    Kill all processes
    """
    for p in global_process_list:
        p.terminate()
        p.join()


def standalone_server(inputs, **kwargs):
    """
    Run a standalone server to process inputs and return responses.

    Args:
        inputs: A string or a list of strings representing the inputs to be processed.
        **kwargs: Additional keyword arguments for server configuration.

    Returns:
        If `inputs` is a string, returns a single response string.
        If `inputs` is a list of strings, returns a list of response strings.
    """
    queue, mgr = init_servers()
    resp_queue = mgr.Queue()
    kwargs["timeout"] = 60
    if isinstance(inputs, str):
        inputs_mod = [inputs]
    else:
        inputs_mod = inputs
    for idx, ipt in enumerate(inputs_mod):
        queue.put((idx, ipt, math.inf, kwargs, resp_queue))
    responses = ["" for _ in inputs_mod]
    for _ in inputs_mod:
        idx, resp = resp_queue.get(block=True)
        responses[idx] = (
            resp.choices[0].message.content
            if "chat" not in kwargs or kwargs["query_type"] == "chat"
            else resp.choices[0].text
        )
    kill_servers()
    if isinstance(inputs, str):
        return responses[0]
    return responses


def label_inputs(
    inputs,
    parallelism=8,
    max_tokens=math.inf,
    use_tqdm=True,
    model="gpt-4",
    stop_condition=None,
    stop_count=None,
    **kwargs,
):
    """
    Generate outputs for a given list of inputs.

    Args:
        inputs (list): List of input strings.
        parallelism (int, optional): Number of parallel processes to use. Defaults to 8.
        max_tokens (int, optional): Maximum number of tokens to generate. Defaults to math.inf.
        use_tqdm (bool, optional): Whether to display a progress bar. Defaults to True.
        model (str, optional): The model to use for generation. Defaults to "gpt-4".
        stop_condition (callable, optional): A function that determines when to stop generation. Defaults to None.
        stop_count (int, optional): The number of outputs to generate before stopping. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        list: Generated outputs.

    Raises:
        ValueError: If generating more than one output at a time.

    """
    kwargs["model"] = model
    queue, manager = init_servers(parallelism)
    resp_queue = manager.Queue()

    if "timeout" not in kwargs:
        kwargs["timeout"] = 30

    if "n" in kwargs:
        mult = kwargs["n"]
    else:
        mult = 1

    if stop_condition is not None and mult > 1:
        raise ValueError(
            "Cannot generate more than one output at a time with stop conditions."
        )

    outputs = ["" for _ in range(mult * len(inputs))]

    if stop_condition is None:
        for i, inp in enumerate(inputs):
            queue.put((i, inp, max_tokens, kwargs, resp_queue))

        pbar = tqdm(
            total=mult * len(inputs),
            desc=f"Generating {kwargs['model']} outputs",
            disable=not use_tqdm,
        )

        for _ in range(mult * len(inputs)):
            idx, resp = resp_queue.get(block=True)
            effective_index = mult * idx
            if resp is None or resp == 0:
                pbar.update(mult)
                continue

            for choice_idx, choice in enumerate(resp.choices):
                content = (
                    choice.message.content
                    if "query_type" not in kwargs
                    or kwargs["query_type"] == "chat"
                    else choice.text
                ).strip()
                outputs[effective_index + choice_idx] = content
                pbar.update(1)

    else:
        stop = False
        pbar = tqdm(
            total=stop_count,
            desc=f"Generating {kwargs['model']} outputs, with stop condition",
            disable=not use_tqdm,
        )
        for batch in range(0, len(inputs), stop_count):
            filt_inputs = inputs[batch : batch + stop_count]

            for i, inp in enumerate(filt_inputs):
                queue.put((i, inp, max_tokens, kwargs, resp_queue))

            for _ in range(stop_count):
                idx, resp = resp_queue.get(block=True)
                effective_index = batch + idx
                if resp is None or resp == 0:
                    continue

                choice = resp.choices[0]
                content = (
                    choice.message.content
                    if "query_type" not in kwargs
                    or kwargs["query_type"] == "chat"
                    else choice.text
                ).strip()

                outputs[effective_index] = content
                if stop_condition(content):
                    pbar.update(1)
                    if pbar.n >= stop_count:
                        stop = True
                        break
            if stop:
                break

    kill_servers()
    return outputs


def graceful_exit(sig, frame):
    """
    Kill all processes on SIGINT
    """
    kill_servers()
    exit()


signal.signal(signal.SIGINT, graceful_exit)
