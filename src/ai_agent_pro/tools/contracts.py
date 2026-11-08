"""Versioned, bounded JSON tool contracts. Trusted code registers implementations."""
from copy import deepcopy
from dataclasses import dataclass
import json
import re
import threading

from jsonschema import Draft202012Validator
from jsonschema.validators import extend
from ai_agent_pro.safety import redact

StrictValidator = extend(Draft202012Validator, type_checker=Draft202012Validator.TYPE_CHECKER.redefine_many({
    'integer': lambda checker, value: type(value) is int,
    'number': lambda checker, value: type(value) in (int, float),
    'object': lambda checker, value: type(value) is dict,
    'array': lambda checker, value: type(value) is list,
}))

MESSAGES = {
    'permission_denied': 'This tool or resource is not approved for this agent.',
    'unavailable': 'The requested tool version is not available.',
    'invalid_input': 'Tool input does not match its bounded schema.',
    'invalid_output': 'The tool returned an invalid result.',
    'invalid_expression': 'Use a bounded arithmetic expression with supported operators.',
    'resource_limit': 'The operation exceeded its resource limits.',
    'result_too_large': 'The tool result exceeded its size limit.',
    'timeout': 'The tool execution deadline was reached.',
    'cancelled': 'Tool execution was cancelled.',
    'budget_exhausted': 'The tool execution budget is exhausted.',
    'busy': 'The local tool execution capacity is busy.',
    'internal_error': 'The tool could not complete safely.',
}


class ToolFailure(Exception):
    def __init__(self, code):
        self.code = code if code in MESSAGES else 'internal_error'
        super().__init__(MESSAGES[self.code])


def parse_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = item
        return result
    def invalid_constant(value):
        raise ValueError('Non-finite JSON value')
    return json.loads(value, object_pairs_hook=pairs, parse_constant=invalid_constant)


def redact_values(value):
    if type(value) is str:
        return redact(value)
    if type(value) is list:
        return [redact_values(v) for v in value]
    if type(value) is dict:
        return {k: redact_values(v) for k, v in value.items()}
    return value


def json_bytes(value):
    def check(item, depth=0):
        if depth > 16:
            raise ValueError('JSON nesting limit')
        if type(item) is dict:
            if len(item) > 128 or any(type(k) is not str or len(k) > 128 for k in item):
                raise ValueError('JSON object limit')
            for v in item.values():
                check(v, depth + 1)
        elif type(item) is list:
            if len(item) > 256:
                raise ValueError('JSON array limit')
            for v in item:
                check(v, depth + 1)
        elif type(item) is str:
            if len(item) > 32768:
                raise ValueError('JSON string limit')
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError('JSON type')
    check(value)
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode()


def validate(schema, value, code):
    try:
        json_bytes(value)
        if not StrictValidator(schema).is_valid(value):
            raise ValueError('schema')
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ToolFailure(code) from None


def strict_schema(schema):
    allowed = {'type', 'properties', 'required', 'additionalProperties', 'items',
               'minItems', 'maxItems', 'minLength', 'maxLength', 'minimum', 'maximum',
               'enum', 'description', 'title'}
    # Exclude references and conditional/regex schemas at every level. Validation
    # must never resolve remote schemas or evaluate unbounded extension keywords.
    if type(schema) is not dict or not set(schema) <= allowed:
        raise ValueError('Only explicit bounded schema keywords are supported')
    json_bytes(schema)
    Draft202012Validator.check_schema(schema)
    kind = schema.get('type')
    kinds = kind if isinstance(kind, list) else [kind]
    if not kinds or not set(kinds) <= {'object', 'array', 'string', 'integer', 'number', 'boolean', 'null'}:
        raise ValueError('Every field must declare its JSON type')
    if 'object' in kinds:
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is not False or set(schema.get('required', [])) != set(props):
            raise ValueError('Objects must reject extra fields and require declared fields')
        for child in props.values():
            strict_schema(child)
    if 'array' in kinds:
        if not 0 <= schema.get('maxItems', -1) <= 256:
            raise ValueError('Arrays must be bounded')
        strict_schema(schema['items'])
    if 'string' in kinds and not 0 <= schema.get('maxLength', -1) <= 32768:
        raise ValueError('Strings must be bounded')


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    version: str
    description: str
    permission: str
    input_schema: dict
    output_schema: dict
    handler: str
    timeout_seconds: float = 2
    result_limit_bytes: int = 16384

    def validate(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,47}', self.name):
            raise ValueError('Invalid tool name')
        if not re.fullmatch(r'\d{1,3}\.\d{1,3}\.\d{1,3}', self.version):
            raise ValueError('Use a semantic tool version')
        if self.permission not in ('compute', 'source.inspect', 'project.read', 'knowledge.read'):
            raise ValueError('Only read-only Stage 4A capabilities are supported')
        if not re.fullmatch(r'ai_agent_pro\.tools\.[a-z_]+:[a-z_]+', self.handler):
            raise ValueError('Handlers must be vetted functions within the tools package')
        if not self.description or len(self.description) > 500:
            raise ValueError('Invalid description')
        if type(self.timeout_seconds) not in (int, float) or not .05 <= self.timeout_seconds <= 10:
            raise ValueError('Invalid timeout')
        if type(self.result_limit_bytes) is not int or not 256 <= self.result_limit_bytes <= 65536:
            raise ValueError('Invalid result limit')
        for schema in (self.input_schema, self.output_schema):
            if schema.get('type') != 'object':
                raise ValueError('Tool contracts must be objects')
            strict_schema(schema)
        return self


class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._lock = threading.RLock()

    def register(self, definition):
        value = deepcopy(definition).validate()
        key = (value.name, value.version)
        with self._lock:
            if key in self._tools:
                raise ValueError('Tool versions cannot be overwritten')
            self._tools[key] = value

    def get(self, name, version):
        if type(name) is not str or type(version) is not str:
            raise ToolFailure('unavailable')
        try:
            with self._lock:
                return deepcopy(self._tools[(name, version)])
        except KeyError:
            raise ToolFailure('unavailable') from None

    def catalog(self):
        with self._lock:
            tools = deepcopy(list(self._tools.values()))
        return [dict(name=t.name, version=t.version, description=t.description,
                     permission=t.permission, read_only=True, timeout_seconds=t.timeout_seconds,
                     result_limit_bytes=t.result_limit_bytes,
                     input_schema=deepcopy(t.input_schema), output_schema=deepcopy(t.output_schema))
                for t in tools]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    version: str
    data: dict | None
    error_code: str | None
    error: str | None
    elapsed_seconds: float
