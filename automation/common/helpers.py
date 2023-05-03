import subprocess
import sys
import typing


def run_command(
    command: str,
    args: list = None,
    workdir: str = None,
    stdin: str = None,
    live: bool = True,
    log_file_handler: typing.IO[str] = None,
) -> (str, str, int):
    stdout, stderr, exit_status = "", "", 0
    if workdir:
        command = f"cd {workdir}; " + command
    if args:
        command += " " + " ".join(args)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        shell=True,
    )

    if stdin:
        process.stdin.write(bytes(stdin, "ascii"))
        process.stdin.close()

    if live:
        for line in iter(process.stdout.readline, b""):
            stdout += str(line)
            sys.stdout.write(line.decode(sys.stdout.encoding))
            if log_file_handler:
                log_file_handler.write(line.decode(sys.stdout.encoding))
    else:
        stdout = process.stdout.read()

    stderr = process.stderr.read()

    exit_status = process.wait()

    return stdout, stderr, exit_status
