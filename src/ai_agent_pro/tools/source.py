"""Python syntax and declaration inspection. Submitted source is never executed."""
import ast

from ai_agent_pro.tools.contracts import ToolFailure


def run(arguments, context):
    source = arguments['source']
    if len(source.encode('utf-8')) > 32768 or len(source.splitlines()) > 2000:
        raise ToolFailure('resource_limit')
    result = dict(syntax_valid=True, line_count=len(source.splitlines()), imports=[], functions=[],
                  classes=[], truncated=False, syntax_error=None)
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        result.update(syntax_valid=False, syntax_error={'line': error.lineno, 'offset': error.offset})
        return result
    except (ValueError, RecursionError):
        raise ToolFailure('invalid_input') from None
    nodes = list(ast.walk(tree))
    if len(nodes) > 4000:
        raise ToolFailure('resource_limit')
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            result['imports'].extend(s[:256] for s in names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result['functions'].append(dict(name=node.name[:256], line=node.lineno,
                                            asynchronous=isinstance(node, ast.AsyncFunctionDef)))
        elif isinstance(node, ast.ClassDef):
            result['classes'].append(dict(name=node.name[:256], line=node.lineno))
    for key in ('imports', 'functions', 'classes'):
        result['truncated'] |= len(result[key]) > 64
        result[key] = result[key][:64]
    return result
