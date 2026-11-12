"""Cooperative terminal input for the HumanPlayer CLI, using the standard library.

Console reads poll for available input instead of leaving a blocking input()
call in a worker thread after cancellation. One CLI session owns the terminal
throughout its display, retries, and selection. Callers must use console_session
around read_line calls. Stream input is also supported for scripted CLI tests.
"""

import asyncio
from contextlib import contextmanager
from io import StringIO
import os
import select
import stat
import sys
from threading import Lock


_console_lock = Lock()
_POLL_INTERVAL = 0.02


@contextmanager
def console_session():
    """Reserve the shared terminal, rejecting simultaneous human prompts.

    Release on completion or any exception, including cancellation. The lock
    is acquired without blocking the event loop and is shared across players.
    """
    if not _console_lock.acquire(blocking=False):
        raise RuntimeError("Another human decision is already using the console")
    try:
        yield
    finally:
        _console_lock.release()


async def _read_windows_console() -> str:
    """Read and echo a Windows terminal line without blocking on getwch.

    Handle Enter, Backspace, Ctrl+C, and Ctrl+Z/Ctrl+D. Ignore function and
    arrow keys. On cancellation discard partial input and queued keystrokes.
    """
    import msvcrt

    characters = []
    extended_key = False
    try:
        while True:
            if not msvcrt.kbhit():
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            character = msvcrt.getwch()
            if extended_key:
                extended_key = False
            elif character in ("\x00", "\xe0"):
                extended_key = True
            elif character in ("\r", "\n"):
                print(flush=True)
                return "".join(characters)
            elif character == "\x03":
                raise asyncio.CancelledError("Human input interrupted")
            elif character in ("\x1a", "\x04"):
                raise EOFError("Human input ended")
            elif character in ("\b", "\x7f"):
                if characters:
                    characters.pop()
                    print("\b \b", end="", flush=True)
            elif character.isprintable():
                characters.append(character)
                print(character, end="", flush=True)
            await asyncio.sleep(0)
    except (asyncio.CancelledError, EOFError):
        while msvcrt.kbhit():
            msvcrt.getwch()
        print(flush=True)
        raise


def _windows_pipe_ready(fd: int) -> bool:
    """Check a redirected Windows stdin pipe without waiting for a whole line."""
    import ctypes
    from ctypes import wintypes
    import msvcrt

    peek = ctypes.WinDLL("kernel32", use_last_error=True).PeekNamedPipe
    peek.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    peek.restype = wintypes.BOOL
    available = wintypes.DWORD()
    handle = msvcrt.get_osfhandle(fd)
    if not peek(handle, None, 0, None, ctypes.byref(available), None):
        error = ctypes.get_last_error()
        if error in (109, 232):  # Broken pipe / pipe being closed: read EOF.
            return True
        raise ctypes.WinError(error)
    return available.value > 0


async def _read_stream_line(stream) -> str:
    """Read ready bytes from a pipe, regular file, or POSIX terminal.

    Reading one byte at a time avoids consuming the next decision's line.
    Return a final unterminated line at EOF, or raise EOFError if no data remains.
    Cancellation drops the partial line and flushes a POSIX terminal's input.
    """
    fd = stream.fileno()
    is_file = stat.S_ISREG(os.fstat(fd).st_mode)
    characters = bytearray()
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        while True:
            ready = is_file or (
                _windows_pipe_ready(fd) if os.name == "nt"
                else bool(select.select([fd], [], [], 0)[0])
            )
            if not ready:
                await asyncio.sleep(_POLL_INTERVAL)
                continue
            character = os.read(fd, 1)
            if not character:
                if not characters:
                    raise EOFError("Human input ended")
                return characters.decode(encoding, errors="replace")
            if character == b"\n":
                return characters.decode(encoding, errors="replace").rstrip("\r")
            characters.extend(character)
            await asyncio.sleep(0)
    except asyncio.CancelledError:
        if os.name != "nt" and stream.isatty():
            import termios

            termios.tcflush(fd, termios.TCIFLUSH)
            print(flush=True)
        raise


async def read_line(prompt: str) -> str:
    """Print a prompt and await one line while the caller owns console_session.

    Interactive input waits for Enter; typing a digit alone does not submit it.
    Support native Windows and POSIX terminals, redirected stdin, and StringIO
    for tests. There is no worker thread that can outlive a cancelled request.
    Ctrl+C cancels the decision; EOF propagates as EOFError. Do not mix these
    reads with input() or buffered reads from the same stdin stream.
    """
    print(prompt, end="", flush=True)
    stream = sys.stdin
    if isinstance(stream, StringIO):
        await asyncio.sleep(0)
        line = stream.readline()
        if line == "":
            raise EOFError("Human input ended")
        return line.rstrip("\r\n")
    if os.name == "nt" and stream.isatty():
        return await _read_windows_console()
    return await _read_stream_line(stream)
