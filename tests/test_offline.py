import errno
import os
import socket

import pytest


@pytest.mark.skipif(os.environ.get("BRANCHSEED_NETWORK_BLOCKED") != "1", reason="Run under network syscall injection.")
def test_network_sandbox_really_blocks_sockets():
    with pytest.raises(OSError) as failure:
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    assert failure.value.errno == errno.ENETUNREACH
