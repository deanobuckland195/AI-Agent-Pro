"""Fixed private worker entry point. Receives only host-selected tool definitions."""
import importlib
import math
import resource
import sys

from ai_agent_pro.tools.contracts import ToolDefinition, ToolFailure, json_bytes, parse_json, redact_values, validate


def main():
    try:
        wire = sys.stdin.buffer.read(131073)
        if len(wire) > 131072:
            raise ToolFailure('invalid_input')
        request = parse_json(wire)
        definition = ToolDefinition(**request['definition']).validate()
        # CPU protection is independent of the parent's wall-clock deadline.
        seconds = math.ceil(definition.timeout_seconds) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds))
        if sys.platform.startswith('linux'):
            resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
        validate(definition.input_schema, request['arguments'], 'invalid_input')
        module, name = definition.handler.split(':')
        handler = getattr(importlib.import_module(module), name)
        data = redact_values(handler(request['arguments'], request['context']))
        validate(definition.output_schema, data, 'invalid_output')
        if len(json_bytes(data)) > definition.result_limit_bytes:
            raise ToolFailure('result_too_large')
        response = {'ok': True, 'data': data}
    except ToolFailure as error:
        response = {'ok': False, 'code': error.code}
    except (MemoryError, RecursionError):
        response = {'ok': False, 'code': 'resource_limit'}
    except Exception:
        # Never serialize exception text, file paths, input, or traceback.
        response = {'ok': False, 'code': 'internal_error'}
    sys.stdout.buffer.write(json_bytes(response))


if __name__ == '__main__':
    main()
