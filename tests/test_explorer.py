from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from threading import Thread

import numpy as np
import pytest
import SimpleITK as sitk

from explorer import CaseStore, make_handler


@pytest.fixture
def server(tmp_path):
    case = tmp_path / "subject001"
    case.mkdir()
    z, y, x = np.indices((30, 25, 25))
    parent = ((x - 12) ** 2 + (y - 12) ** 2 < 6**2) & (z > 3) & (z < 26)
    sitk.WriteImage(sitk.GetImageFromArray(np.where(parent, 330, 20).astype(np.int16)), str(case / "orig.nii"))
    sitk.WriteImage(sitk.GetImageFromArray(parent.astype(np.uint8)), str(case / "mask.nii"))
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Explorer</h1>")
    store = CaseStore(tmp_path)
    http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, static))
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    connection = HTTPConnection(*http.server_address[:2], timeout=10)
    yield connection, store
    connection.close()
    http.shutdown()
    http.server_close()
    store.executor.shutdown(wait=True)
    thread.join(timeout=5)


def request(connection, method, path, headers=None):
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    return response.status, response.read()


def test_analysis_api_serves_real_mesh_prediction_and_matching_buffers(server):
    connection, store = server
    status, body = request(connection, "GET", "/api/cases")
    assert status == 200
    assert json.loads(body) == [{"id": "subject001", "available": True}]
    assert request(connection, "POST", "/api/cases/subject001/analyze")[0] == 202
    store.executor.shutdown(wait=True)
    status, body = request(connection, "GET", "/api/cases/subject001")
    assert status == 200
    case = json.loads(body)
    assert len(case["mesh"]["vertices"]) > 0
    assert case["prediction"]["daughters"] == []
    count = int(np.prod(case["size_xyz"]))
    assert len(request(connection, "GET", "/api/cases/subject001/ct")[1]) == count * 2
    assert len(request(connection, "GET", "/api/cases/subject001/mask")[1]) == count
    assert json.loads(request(connection, "GET", "/api/cases/subject001/status")[1]) == {"status": "ready"}


def test_cross_origin_and_traversal_requests_are_rejected(server):
    connection, store = server
    assert request(
        connection, "POST", "/api/cases/subject001/analyze", {"Origin": "https://unrelated.invalid"}
    )[0] == 403
    assert not store.jobs
    assert request(connection, "GET", "/../subject001/orig.nii")[0] == 404
    assert request(connection, "GET", "/api/unknown")[0] == 404
    with pytest.raises(ValueError):
        store.paths("../subject001")


def test_unresolved_lfs_files_are_marked_unavailable(server):
    _, store = server
    (store.data_root / "subject001" / "orig.nii").write_text("version https://git-lfs.github.com/spec/v1\n")
    assert store.list_cases()[0]["available"] is False
