import sys, importlib, threading, os, tempfile, inspect, os
from dataclasses import dataclass

@dataclass
class Update:
    path: str
    line: int
    new_content: str
    
@dataclass
class Ctx:
    current_consumed_offset: int
    updates: list[Update]

_current_ctx = threading.local()

def _get_ctx():
    ctx = getattr(_current_ctx, 'ctx', None)
    
    if ctx is None:
        raise Exception('not in expect test')
    return ctx

def get_output():
    ctx = _get_ctx()
    stdout = 1
    max_offset = os.lseek(stdout, 0, os.SEEK_CUR)
    data = os.pread(stdout, max_offset - ctx.current_consumed_offset, ctx.current_consumed_offset)

    ctx.current_consumed_offset = max_offset
    return data.decode('utf8', 'replace')

def test_case(f):
    module = sys.modules[f.__module__]
    if not hasattr(module, '__expect_test_cases'):
        module.__expect_test_cases = [] # type: ignore
    module.__expect_test_cases.append(f)
    return f

def _run(f):
    sys.stdout.flush()
    tmp_file = tempfile.TemporaryFile()
    initial_stdout = os.dup(1)
    try:
        os.dup2(tmp_file.fileno(), 1)
        ctx = _current_ctx.ctx = Ctx(current_consumed_offset=0, updates=[])
        f()
    finally:
        _current_ctx.ctx = None
        sys.stdout.flush()
        tmp_file.close()
        os.dup2(initial_stdout, 1)

    return ctx.updates

def _change_lines(lines, by_line):
    result = []
    i = 0
    while i < len(lines):
        text = lines[i]
        if i in by_line:
            quoted = by_line[i]

            s = text.split('expect(', 1)
            if len(s) == 1:
                raise Exception('unexpectedly missing expect( (%r)' % text)

            if s[1].count("'''") > 1:
                remaining_of_line = s[1].split("'''", 2)[2]
                i += 1
            else:
                i += 1
            
                while i < len(lines) and "'''" not in lines[i]:
                    i += 1

                remaining_of_line = lines[i].split("'''", 1)[1]
                
                if i == len(lines):
                    raise Exception('unclosed expect string')

                i += 1
                
            if "'''" in quoted:
                raise Exception("expect test output should not contain '''")

            result.append(s[0]
                          + "expect(r'''"
                          + quoted
                          + " '''"
                          + remaining_of_line)
        else:
            result.append(text)
            i += 1

    return '\n'.join(result)
            
def _apply(updates, path):
    path = os.path.realpath(path)
    by_line = {}
    for update in updates:
        if os.path.realpath(update.path) == path:
            if by_line.get(update.line) not in (None, update.new_content):
                raise Exception('multiple updates at %r' % update)
            by_line[update.line - 1] = update.new_content
    
    lines = open(path, 'r').read().splitlines()
    result = _change_lines(lines, by_line)
    with open(path + '.tmp', 'w') as f:
        f.write(result)
    os.rename(path + '.tmp', path)
    
def expect(expected_output):
    sys.stdout.flush()

    actual_output = get_output().strip()
    if actual_output != expected_output.strip():
        caller = inspect.getframeinfo(inspect.stack()[1][0])
        _get_ctx().updates.append(Update(caller.filename, caller.lineno, actual_output))
        
def test_modules(name):
    m = importlib.import_module(name)
    cases = getattr(m, '__expect_test_cases', [])
    
    if not cases:
        print('no test cases in', name)

    updates = []
    for test_case in cases:
        updates += _run(test_case)

    if updates:
        print('updating', m.__file__)
        _apply(updates, m.__file__)
        
if __name__ == '__main__':
    import expect_test
    for module in sys.argv[1:]:
        expect_test.test_modules(module)
