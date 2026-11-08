"""Explicit registry construction. Importing this module grants no permissions."""
from ai_agent_pro.tools.contracts import ToolDefinition, ToolRegistry


def text(maximum, minimum=0):
    return dict(type='string', minLength=minimum, maxLength=maximum)


def integer(low, high):
    return dict(type='integer', minimum=low, maximum=high)


def array(items, maximum):
    return dict(type='array', items=items, maxItems=maximum)


def obj(**properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def registry():
    result = ToolRegistry()
    boolean = {'type': 'boolean'}
    path = text(240, 1)
    definitions = [
        ToolDefinition('calculator', '1.0.0', 'Bounded arithmetic, without executing code.', 'compute',
            obj(expression=text(256, 1)), obj(value=dict(type='number', minimum=-1e100, maximum=1e100)),
            'ai_agent_pro.tools.calculator:run', timeout_seconds=2, result_limit_bytes=1024),
        ToolDefinition('python_inspect', '1.0.0', 'Inspect Python syntax, imports and declarations without execution.',
            'source.inspect', obj(source=text(32768)),
            obj(syntax_valid=boolean, line_count=integer(0, 2000), imports=array(text(256), 64),
                functions=array(obj(name=text(256), line=integer(1, 2000), asynchronous=boolean), 64),
                classes=array(obj(name=text(256), line=integer(1, 2000)), 64), truncated=boolean,
                syntax_error=dict(type=['object', 'null'], properties={
                    'line': dict(type=['integer', 'null'], minimum=0),
                    'offset': dict(type=['integer', 'null'], minimum=0)},
                    required=['line', 'offset'], additionalProperties=False)),
            'ai_agent_pro.tools.source:run', timeout_seconds=3, result_limit_bytes=32768),
        ToolDefinition('project_read', '1.0.0', 'Read bounded lines from an explicitly approved project file.',
            'project.read', obj(path=path, start_line=integer(1, 10000), line_count=integer(1, 50)),
            obj(path=path, start_line=integer(1, 10000), lines=array(text(1000), 50), truncated=boolean),
            'ai_agent_pro.tools.filesystem:read', timeout_seconds=3, result_limit_bytes=32768),
        ToolDefinition('project_search', '1.0.0', 'Literal, case-insensitive search of explicitly approved files.',
            'project.read', obj(query=text(160, 1), paths=array(path, 32), max_results=integer(1, 20)),
            obj(matches=array(obj(path=path, line=integer(1, 262144), text=text(1000)), 20),
                files_searched=integer(0, 32), truncated=boolean),
            'ai_agent_pro.tools.filesystem:search', timeout_seconds=3, result_limit_bytes=32768),
        ToolDefinition('knowledge_search', '1.0.0', 'Offline lexical search of this agent\'s assigned indexed documents.',
            'knowledge.read', obj(query=text(200, 1), max_results=integer(1, 5)),
            obj(method=dict(type='string', enum=['local_lexical'], maxLength=20),
                sources=array(obj(label=text(3, 1), document_id=text(80, 1), generation_id=text(80, 1),
                    filename=text(256), location=text(256), excerpt=text(4000)), 5)),
            'ai_agent_pro.tools.knowledge:run', timeout_seconds=3, result_limit_bytes=32768),
    ]
    for definition in definitions:
        result.register(definition)
    return result
