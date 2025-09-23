import builtins
import multiprocessing
import queue
import time
import traceback

queue = multiprocessing.Queue()
is_dead = multiprocessing.Event()
is_done = multiprocessing.Event()


class MockS3:
    def download_file(self, bucket, remote_filename, local_filename):
        pass


class MockSQS:
    def send_message(
        QueueUrl: str = None,
        MessageBody: str = None
    ):
        queue.put(MessageBody)

    def receive_message(
        QueueUrl: str = None,
        MaxNumberOfMessages: int = None,
        WaitTimeSeconds: float = None,
        VisibilityTimeout: float = None,
    ):
        out = []
        for _ in range(MaxNumberOfMessages):
            try:
                out.append({"Body": queue.get(WaitTimeSeconds), "ReceiptHandle": None})
            except queue.Empty:
                pass
        return {"Messages": out}

    def delete_message(
        QueueUrl: str = None,
        ReceiptHandle: str = None,
    ):
        pass


def modify_print(identifier):
    def modified_print(*args, **kwargs):
        args = (time.strftime("%H:%M"), identifier + ":") + args
        kwargs["flush"] = True
        return builtins.print(*args, **kwargs)

    return modified_print


def dispatch(function, is_dead, is_done):
    function.__globals__["print"] = modified_print = modify_print("DISPATCHER")

    modified_print("BEGIN")

    try:
        function()

    except Exception:
        for line in traceback.format_exc().split("\n"):
            modified_print(line)
        modified_print("END WITH ERROR 💀")

        is_dead.set()

    else:
        modified_print("END")

        is_done.set()


def work(identifier, function, queue, is_dead, is_done):
    function.__globals__["print"] = modified_print = modify_print(identifier)

    modified_print("BEGIN")

    while True:
        if is_dead.is_set():
            modified_print("END BECAUSE DEAD 💀")
            return

        if is_done.is_set() and queue.empty():
            modified_print("END BECAUSE DONE")
            return

        try:
            function()

        except Exception:
            for line in traceback.format_exc().split("\n"):
                modified_print(line)
            modified_print("END WITH ERROR 💀")

            is_dead.set()
            return

        modified_print(f"{queue.qsize()} remaining")


def run(dispatcher, worker, num_workers):
    dispatcher_process = multiprocessing.Process(
        target=dispatch, args=(dispatcher, is_dead, is_done)
    )

    fmt = f"WORKER-{{i:0{len(str(num_workers))}d}}"
    worker_processes = [
        multiprocessing.Process(
            target=work, args=(fmt.format(i=i), work, queue, is_dead, is_done)
        )
        for i in range(num_workers)
    ]

    dispatcher_process.start()
    for worker in worker_processes:
        worker.start()

    dispatcher_process.join()
    for worker in worker_processes:
        worker.join()
