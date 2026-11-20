"""Small HTTP helpers for the recipe's literal loopback endpoint."""

from contextlib import contextmanager
import http.client
import json
import socket
import threading
import time

HOST = "127.0.0.1"
DEFAULT_PORT = 18889
MODEL = "Qwen3.8-Flash-Next"
CONTEXT = 262144
PARALLEL = 5
MAX_RESPONSE = 8 * 1024 * 1024


def port_number(value):
    port = int(value)
    if not 1024 <= port <= 65535:
        raise ValueError("port must be between 1024 and 65535")
    return port


class ApiError(RuntimeError):
    pass


@contextmanager
def timed_connection(port, timeout):
    """Bound the full HTTP exchange, including a server that keeps sending bytes."""
    connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
    start = time.monotonic()
    watchdog = None
    deadline_socket = None
    try:
        connection.connect()
        deadline_socket = connection.sock.dup()

        def interrupt():
            try:
                deadline_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        remaining = timeout - (time.monotonic() - start)
        if remaining <= 0:
            raise ApiError(f"HTTP connection to {HOST}:{port} timed out")
        watchdog = threading.Timer(remaining, interrupt)
        watchdog.daemon = True
        watchdog.start()
        try:
            yield connection
            if time.monotonic() - start >= timeout:
                raise ApiError("HTTP exchange exceeded its deadline")
        except (ApiError, OSError, http.client.HTTPException) as error:
            if time.monotonic() - start >= timeout:
                raise ApiError(f"HTTP exchange at {HOST}:{port} timed out after {timeout} seconds") from error
            raise
    finally:
        if watchdog:
            watchdog.cancel()
        if deadline_socket:
            deadline_socket.close()
        connection.close()


def request(connection, path, payload=None, *, stream=False):
    if payload is None:
        connection.request("GET", path)
    else:
        connection.request(
            "POST", path, json.dumps(payload).encode(),
            {"Content-Type": "application/json",
             "Accept": "text/event-stream" if stream else "application/json"},
        )
    response = connection.getresponse()
    if response.status != 200:
        detail = response.read(512).decode(errors="replace").replace("\n", " ")
        raise ApiError(f"{path} returned HTTP {response.status}: {detail}")
    return response


def json_request(port, path, payload=None, *, timeout=5):
    try:
        with timed_connection(port, timeout) as connection:
            with request(connection, path, payload) as response:
                raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ApiError(f"{path} returned more than {MAX_RESPONSE} bytes")
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ApiError(f"{path} returned a non-object JSON response")
        return result
    except (OSError, http.client.HTTPException, ValueError) as error:
        raise ApiError(f"{path} at {HOST}:{port}: {error}") from error


def models_at(port):
    listing = json_request(port, "/v1/models").get("data")
    if not isinstance(listing, list):
        raise ApiError("/v1/models returned no model list")
    return [entry.get("id") for entry in listing if isinstance(entry, dict)]


def require_model(port):
    models = models_at(port)
    if MODEL not in models:
        raise ApiError(f"{MODEL} is absent from /v1/models at {HOST}:{port}")
    return models
