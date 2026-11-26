"""Real file-RPC authority and same-key remote cell serialization."""

import json


def test_remote_concurrent_cells_keep_one_authority_window(tmp_path):
    import os
    import subprocess
    import threading
    from pm.shell import bash
    from tools.code_kernel_remote import (
        _REMOTE_KERNELS, execute_in_remote_kernel, shutdown_remote_kernels_for_owner,
    )
    from tools.thread_context import exact_tool_scope

    shell = bash()
    assert shell
    first_result_waiting, release_first_result = threading.Event(), threading.Event()
    second_request_published = threading.Event()

    class LocalFileEnvironment:
        def __init__(self):
            self.lock = threading.Lock()
            self.first_result_paused = False

        def get_temp_dir(self):
            return str(tmp_path)

        def execute(self, command, cwd="/", timeout=20, stdin_data=None):
            if "cell_req_000002" in command:
                second_request_published.set()
            if "cat" in command and "cell_res_000001.json" in command:
                with self.lock:
                    pause = not self.first_result_paused
                    self.first_result_paused = True
                if pause:
                    first_result_waiting.set()
                    assert release_first_result.wait(timeout=10)
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True,
                                    text=True, check=False)
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}

    env = LocalFileEnvironment()
    source_a, source_b = tmp_path / "parallel-a.txt", tmp_path / "parallel-b.txt"
    source_a.write_text("first remote cell", encoding="utf-8")
    source_b.write_text("second remote cell", encoding="utf-8")
    results, failures = {}, []
    second_call_entered = threading.Event()

    def run_cell(label, source):
        try:
            if label == "second":
                second_call_entered.set()
            with exact_tool_scope({"execute_code", "read_file"}):
                results[label] = execute_in_remote_kernel(
                    f"import hermes_tools; print(hermes_tools.read_file({str(source)!r})['content'])",
                    env=env, env_type="local-file-concurrency", task_env_id="remote-cell-window",
                    sandbox_tools=frozenset({"read_file"}), timeout=15,
                    max_tool_calls=5, reset=False, idle_exit=60,
                )
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=run_cell, args=("first", source_a))
    second = threading.Thread(target=run_cell, args=("second", source_b))
    first.start()
    try:
        assert first_result_waiting.wait(timeout=10)
        kernel = next(kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "remote-cell-window")
        first_authority = kernel.cell_authority
        assert first_authority is not None and first_authority.active
        second.start()
        # The first cell is held after execution but before its result is read. A
        # second same-key request must remain outside the RPC/cell authority window.
        assert second_call_entered.wait(timeout=3)
        assert not second_request_published.wait(timeout=0.5)
        assert kernel.cell_authority is first_authority
    finally:
        release_first_result.set()
        first.join(timeout=10)
        if second.ident is not None:
            second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert failures == [], failures
    assert results["first"] is not None and results["first"]["status"] == "success", results
    assert results["second"] is not None and results["second"]["status"] == "success", results
    assert "first remote cell" in results["first"]["stdout"]
    assert "second remote cell" in results["second"]["stdout"]
    assert second_request_published.is_set()
    kernel = next(kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "remote-cell-window")
    assert kernel.attached == 0 and kernel.cell_authority is None
    shutdown_remote_kernels_for_owner("remote-cell-window")
    assert not any(kernel.owner == "remote-cell-window" for kernel in _REMOTE_KERNELS.values())

def test_remote_persistent_kernel_runs_real_cell_and_rpc_over_file_transport(tmp_path):
    import os
    import subprocess
    from pm.shell import bash
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS
    from tools.code_kernel_remote import (
        _REMOTE_KERNELS, execute_in_remote_kernel, shutdown_remote_kernels_for_owner,
    )
    from tools.thread_context import exact_tool_scope

    shell = bash()
    assert shell
    class LocalFileEnvironment:
        def get_temp_dir(self): return str(tmp_path)
        def execute(self, command, cwd="/", timeout=20, stdin_data=None):
            result = subprocess.run([shell, "-c", command], cwd=cwd, timeout=timeout,
                                    env=dict(os.environ), input=stdin_data or "", capture_output=True,
                                    text=True, check=False)
            return {"output": result.stdout + result.stderr, "returncode": result.returncode}
    env = LocalFileEnvironment()
    source = tmp_path / "persistent-source.txt"
    source.write_text("remote persistent file fixture\n", encoding="utf-8")
    forbidden = tmp_path / "persistent-forbidden.txt"
    code = f'''\
import hermes_tools, json
read = hermes_tools.read_file({str(source)!r})
denied = hermes_tools._call("write_file", {{"path": {str(forbidden)!r}, "content": "forbidden"}})
print(json.dumps({{"read": read["content"], "denied": denied,
                  "has_write_stub": hasattr(hermes_tools, "write_file")}}))
'''
    with exact_tool_scope({"execute_code", "read_file"}):
        first = execute_in_remote_kernel(
            code, env=env, env_type="local-file-fixture", task_env_id="persistent-file-rpc",
            sandbox_tools=frozenset(SANDBOX_ALLOWED_TOOLS), timeout=8,
            max_tool_calls=5, reset=False, idle_exit=60,
        )
    assert first is not None and first["status"] == "success", first
    assert "remote persistent file fixture" in first["stdout"]
    assert "outside this execute_code cell's exact tool grant" in first["stdout"]
    assert '"has_write_stub": false' in first["stdout"]
    assert not forbidden.exists()
    kernels = [kernel for kernel in _REMOTE_KERNELS.values() if kernel.owner == "persistent-file-rpc"]
    assert len(kernels) == 1
    shutdown_remote_kernels_for_owner("persistent-file-rpc")
    assert not any(kernel.owner == "persistent-file-rpc" for kernel in _REMOTE_KERNELS.values())
    assert not kernels[0].is_alive()
