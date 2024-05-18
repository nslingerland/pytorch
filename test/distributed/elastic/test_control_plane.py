#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

import os
import pickle
import socket
import tempfile
from contextlib import contextmanager

import requests
from requests.adapters import HTTPAdapter

from torch.distributed.elastic.control_plane import (
    TORCH_WORKER_SERVER_SOCKET,
    worker_main,
)
from torch.testing._internal.common_utils import requires_cuda, run_tests, TestCase
from urllib3.connection import HTTPConnection
from urllib3.connectionpool import HTTPConnectionPool


class UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)


class UnixHTTPConnectionPool(HTTPConnectionPool):
    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")

        self.socket_path = socket_path

    def _new_conn(self):
        return UnixHTTPConnection(self.socket_path)


class UnixHTTPAdapter(HTTPAdapter):
    def __init__(self, socket_path: str) -> None:
        super().__init__()

        self.socket_path = socket_path

    def get_connection(self, url, proxies=None):
        return UnixHTTPConnectionPool(self.socket_path)

    def request_url(self, request, proxies):
        return request.path_url


@contextmanager
def local_worker_server() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        socket_path = os.path.join(tmpdir, "socket.sock")
        os.environ[TORCH_WORKER_SERVER_SOCKET] = socket_path

        with worker_main():
            session = requests.Session()
            session.mount("http://", UnixHTTPAdapter(socket_path))
            yield session


class WorkerServerTest(TestCase):
    def test_worker_server(self) -> None:
        with local_worker_server() as session:
            resp = session.get("http://worker/")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(
                resp.text,
                """<h1>torch.distributed.WorkerServer</h1>
<a href="/handler/">Handler names</a>
""",
            )

            resp = session.post("http://worker/handler/ping")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.text, "pong")

            resp = session.get("http://worker/handler/")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("ping", resp.json())

    @requires_cuda
    def test_dump_nccl_trace_pickle(self) -> None:
        with local_worker_server() as session:
            resp = session.post("http://worker/handler/dump_nccl_trace_pickle")
            self.assertEqual(resp.status_code, 200)
            out = pickle.loads(resp.content)


if __name__ == "__main__":
    run_tests()
