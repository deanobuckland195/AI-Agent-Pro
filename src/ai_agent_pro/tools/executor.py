"""Deny-by-default tool execution with isolated workers and bounded output."""
import asyncio
from dataclasses import asdict
import os
import signal
import sys
import threading
from time import monotonic

import psutil

from ai_agent_pro.tools.builtins import registry
from ai_agent_pro.tools.contracts import MESSAGES, ToolFailure, ToolResult, json_bytes, parse_json, redact_values, validate
from ai_agent_pro.tools.permissions import ToolBudget, ToolPermissions


def _failure_result(definition, code, started):
    """Build the same safe failure envelope for each exception category."""
    return ToolResult(False, definition.name if definition else 'unavailable',
                      definition.version if definition else 'unknown', None,
                      code, MESSAGES[code], monotonic() - started)


class ToolExecutor:
    def __init__(self, permissions=None, tool_registry=None):
        self.permissions = permissions if permissions is not None else ToolPermissions()
        self.registry = tool_registry if tool_registry is not None else registry()
        self._slots = threading.BoundedSemaphore(2)

    async def execute(self, agent_id, name, version, arguments, *, budget=None, cancellation=None):
        """agent_id and grants come from trusted host code, never a tool payload.

        A cancellation event returns a safe cancelled result. Cancelling the
        coroutine itself propagates CancelledError after terminating the worker.
        """
        started, definition, acquired = monotonic(), None, False
        try:
            definition = self.registry.get(name, version)
            grant = self.permissions.authorize(agent_id, definition)
            if type(arguments) is not dict:
                raise ToolFailure('invalid_input')
            validate(definition.input_schema, arguments, 'invalid_input')
            if len(json_bytes(arguments)) > 65536:
                raise ToolFailure('invalid_input')
            if cancellation is not None and cancellation.is_set():
                raise ToolFailure('cancelled')
            acquired = self._slots.acquire(blocking=False)
            if not acquired:
                raise ToolFailure('busy')
            active_budget = budget if budget is not None else ToolBudget()
            remaining, _ = active_budget.reserve()
            context = self.permissions.context(grant)
            deadline = min(started + definition.timeout_seconds, monotonic() + remaining)
            request = json_bytes(dict(definition=asdict(definition), arguments=arguments, context=context))
            if len(request) > 131072:
                raise ToolFailure('invalid_input')
            if monotonic() >= deadline:
                raise ToolFailure('timeout')
            response = await self._process(request, deadline - monotonic(),
                                           definition.result_limit_bytes + 1024, cancellation)
            if type(response) is not dict or type(response.get('ok')) is not bool:
                raise ToolFailure('invalid_output')
            if not response['ok']:
                if set(response) != {'ok', 'code'} or type(response['code']) is not str or response['code'] not in MESSAGES:
                    raise ToolFailure('invalid_output')
                raise ToolFailure(response['code'])
            if set(response) != {'ok', 'data'}:
                raise ToolFailure('invalid_output')
            data = response['data']
            validate(definition.output_schema, data, 'invalid_output')
            data = redact_values(data)
            validate(definition.output_schema, data, 'invalid_output')
            size = len(json_bytes(data))
            if size > definition.result_limit_bytes:
                raise ToolFailure('result_too_large')
            self.permissions.recheck(grant, definition)
            if definition.permission == 'knowledge.read':
                from ai_agent_pro.tools.knowledge import revalidate
                revalidate(data, context)
                self.permissions.recheck(grant, definition)
            if cancellation is not None and cancellation.is_set():
                raise ToolFailure('cancelled')
            if monotonic() >= deadline:
                raise ToolFailure('timeout')
            active_budget.charge(size)
            return ToolResult(True, definition.name, definition.version, data, None, None, monotonic() - started)
        except ToolFailure as error:
            return _failure_result(definition, error.code, started)
        except (TypeError, ValueError, OverflowError, RecursionError):
            return _failure_result(definition, 'invalid_input', started)
        except Exception:
            return _failure_result(definition, 'internal_error', started)
        finally:
            if acquired:
                self._slots.release()

    async def _process(self, request, seconds, maximum, cancellation):
        process, tasks = None, []
        try:
            async with asyncio.timeout(seconds):
                process = await asyncio.create_subprocess_exec(
                    sys.executable, '-I', '-B', '-m', 'ai_agent_pro.tools.worker',
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    start_new_session=True,
                    cwd=str(self.permissions.root),
                    env={'PATH': os.defpath, 'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1'})

                async def exchange():
                    process.stdin.write(request)
                    await process.stdin.drain()
                    process.stdin.close()
                    result = bytearray()
                    while block := await process.stdout.read(8192):
                        result.extend(block)
                        if len(result) > maximum:
                            raise ToolFailure('result_too_large')
                    await process.wait()
                    if process.returncode != 0:
                        raise ToolFailure('resource_limit' if process.returncode < 0 else 'internal_error')
                    try:
                        return parse_json(result)
                    except (ValueError, UnicodeError, RecursionError):
                        raise ToolFailure('invalid_output') from None

                async def watch():
                    monitor = psutil.Process(process.pid)
                    while process.returncode is None:
                        if cancellation is not None and cancellation.is_set():
                            raise ToolFailure('cancelled')
                        try:
                            if monitor.memory_info().rss > 256 * 1024**2:
                                raise ToolFailure('resource_limit')
                        except psutil.NoSuchProcess:
                            return
                        await asyncio.sleep(.02)

                tasks = [asyncio.create_task(exchange()), asyncio.create_task(watch())]
                output, _ = await asyncio.gather(*tasks)
                return output
        except TimeoutError:
            raise ToolFailure('timeout') from None
        finally:
            for task in tasks:
                task.cancel()
            if process:
                try:
                    # Kill the isolated process group, including a snapshot
                    # reader if cancellation interrupted its synchronous wait.
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process:
                await process.wait()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
