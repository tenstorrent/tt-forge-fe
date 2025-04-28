# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Helper functions for capturing output in tests


from loguru import logger
import sys
import io

from contextlib import redirect_stdout, redirect_stderr
from _pytest.capture import CaptureFixture

from ..utils import TestSweepsFeatures


class StringBuffer(io.StringIO):
    """
    String buffer stores the output of print statements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enabled = TestSweepsFeatures.params.capture_output

    def writeln(self, message):
        self.write(message)
        self.write("\n")

    def flush(self):
        self.truncate(0)
        self.seek(0)
        super().flush()


class GlobalStringBuffer:
    """
    Singleton class that contains a global StringBuffer.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalStringBuffer, cls).__new__(cls)
            cls._instance.buffer = StringBuffer()

        return cls._instance

    def capture_output(self, capfd: CaptureFixture[str] = None):
        if self.buffer.enabled:
            if capfd is None:
                return CaptureOutputDual()
            else:
                return CaptureOutputPytest(capfd)
        else:
            return CaptureOutputNone()

    @property
    def enabled(self):
        return self.buffer.enabled

    def flush(self):
        self.buffer.flush()

    @property
    def value(self):
        return self.buffer.getvalue()


global_string_buffer = GlobalStringBuffer()
string_buffer: StringBuffer = global_string_buffer.buffer


class CaptureOutputNone:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        pass


class CaptureOutputDual:
    def __init__(self):
        self.original_stdout = sys.stdout  # Save the original stdout
        self.original_stderr = sys.stderr  # Save the original stderr

    def __enter__(self):
        self.buffer = io.StringIO()  # Create an in-memory buffer for capturing output
        self.loguru_sink_id = logger.add(
            self.buffer,
            # format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            #        "<level>{level: <8}</level> | "
            #        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            #        "<level>{message}</level>\n",
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>",
            colorize=True,  # Disable colors because StringIO does not support ANSI codes
        )

        # # Redirecting sys.stdout to our class
        # sys.stdout = self

        # Using redirect_stdout and redirect_stderr for redirection
        self.stdout_redirector = redirect_stdout(self.buffer)
        self.stderr_redirector = redirect_stderr(self.buffer)

        # Activating redirection
        self.stdout_redirector.__enter__()
        self.stderr_redirector.__enter__()

    def __exit__(self, exctype, excinst, exctb):
        self.original_stdout.flush()  # Ensuring the buffer is flushed to stdout
        self.original_stderr.flush()  # Ensuring the buffer is flushed to stderr
        captured_output = self.buffer.getvalue()
        self.stdout_redirector.__exit__(None, None, None)  # Ending stdout redirection
        self.stderr_redirector.__exit__(None, None, None)  # Ending stderr redirection

        logger.remove(self.loguru_sink_id)  # Removing our loguru sink

        string_buffer.writeln(captured_output)  # Writing the captured output to the global buffer


class CaptureOutputPytest:
    def __init__(self, capfd: CaptureFixture[str]):
        self.capfd = capfd

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        out, err = self.capfd.readouterr()
        string_buffer.writeln(
            f"\n-------------------------------------------- stderr --------------------------------------------\n------------------------------------------------------------------------------------------------\n{err}\n------------------------------------------------------------------------------------------------\n-------------------------------------------- stderr --------------------------------------------\n \n\n"
        )
        string_buffer.writeln(
            f"\n-------------------------------------------- stdout --------------------------------------------\n------------------------------------------------------------------------------------------------\n{out}\n------------------------------------------------------------------------------------------------\n-------------------------------------------- stdout --------------------------------------------\n \n\n"
        )
