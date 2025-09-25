#!/usr/bin/env python3
# xyzc.py — XYZ Bootstrap Compiler
# SIMD + CUDA + OpenCL + AVX2/Numba fallback + MacroEngine + Benchmarks


# ----------------------------
# AST NODES
# ----------------------------
class ASTNode:
    def __init__(self, kind, name=None, value=None, children=None):
        self.kind = kind
        self.name = name
        self.value = value
        self.children = children if children else []
    def __repr__(self): return f"<{self.kind}:{self.name or self.value}>"

# ----------------------------
# LEXER
# ----------------------------
class Lexer:
    def __init__(self, src): self.src = src
    def tokenize(self):
        tokens, cur = [], ""
        for ch in self.src:
            if ch.isspace():
                if cur: tokens.append(cur); cur = ""
            elif ch in "()[]{},":
                if cur: tokens.append(cur); cur = ""
                tokens.append(ch)
            else: cur += ch
        if cur: tokens.append(cur)
        return tokens

# ----------------------------
# PARSER
# ----------------------------
class Parser:
    def __init__(self, tokens): self.tokens = tokens; self.pos = 0
    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def consume(self): tok = self.peek(); self.pos += 1; return tok

    def parse(self):
        nodes = []
        while self.peek():
            if self.peek() == "Item": nodes.append(self.parse_item())
            elif self.peek() == "run": nodes.append(self.parse_run())
            else: self.consume()
        return ASTNode("Program", children=nodes)

    def parse_item(self):
        self.consume()  # Item
        name = self.consume(); self.consume(); self.consume()  # ( )
        body = []
        while self.peek() and self.peek() not in ["Item", "run"]:
            tok = self.consume()
            if tok == "let":
                var = self.consume(); self.consume(); val = self.consume()
                body.append(ASTNode("Let", name=var, value=val))
            elif tok == "trait":
                tname = self.consume(); self.consume(); val = self.consume(); self.consume()
                body.append(ASTNode("Trait", name=tname, value=val))
            elif tok == "apply":
                target = self.consume(); self.consume(); args=[]
                while self.peek() != ")": args.append(self.consume())
                self.consume(); body.append(ASTNode("Apply", name=target, value=args))
            elif tok == "dispatch":
                target = self.consume(); self.consume(); args=[]
                while self.peek() != ")": args.append(self.consume())
                self.consume(); body.append(ASTNode("Dispatch", name=target, value=args))
        return ASTNode("Item", name=name, children=body)

    def parse_run(self):
        self.consume(); target = self.consume(); self.consume(); self.consume()
        return ASTNode("Run", name=target)

# ----------------------------
# MACROENGINE (Symbolic Shader Fusion)
# ----------------------------
class MacroEngine:
    def __init__(self): self.macros = {}
    def register(self, name, func): self.macros[name] = func
    def expand(self, name, *args):
        if name in self.macros:
            return self.macros[name](*args)
        raise Exception(f"Macro {name} not found")

macro_engine = MacroEngine()

# Example: shader fusion macro
macro_engine.register("fuse_shader", lambda a,b: f"// fused shader: {a}+{b}")

# ----------------------------
# GPU DISPATCHER
# ----------------------------
class GPUDispatcher:
    def __init__(self): self.cuda_module=None; self.cl_context=None; self.cl_queue=None; self.cl_program=None
    def load_cuda(self, ptx="force.auto.ptx"):
        self.cuda_module = pycuda.compiler.SourceModule(open(ptx).read())
        print("[GPU] CUDA PTX loaded")
    def load_opencl(self, cl_file="force.auto.cl"):
        ctx=cl.create_some_context(); queue=cl.CommandQueue(ctx)
        src=open(cl_file).read(); prog=cl.Program(ctx, src).build()
        self.cl_context, self.cl_queue, self.cl_program = ctx, queue, prog
        print("[GPU] OpenCL kernel built")
    def launch_cuda(self,pos,vel,g):
        n=len(pos); pos_gpu=cuda.mem_alloc(pos.nbytes); vel_gpu=cuda.mem_alloc(vel.nbytes)
        cuda.memcpy_htod(pos_gpu,pos); cuda.memcpy_htod(vel_gpu,vel)
        func=self.cuda_module.get_function("apply_force"); threads=256; blocks=(n+threads-1)//threads
        func(pos_gpu,vel_gpu,np.float32(g),np.int32(n),block=(threads,1,1),grid=(blocks,1))
        cuda.memcpy_dtoh(pos,pos_gpu); cuda.memcpy_dtoh(vel,vel_gpu); return pos,vel
    def launch_opencl(self,pos,vel,g):
        n=len(pos); mf=cl.mem_flags; ctx=self.cl_context; queue=self.cl_queue
        pos_buf=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=pos)
        vel_buf=cl.Buffer(ctx,mf.READ_WRITE|mf.COPY_HOST_PTR,hostbuf=vel)
        self.cl_program.apply_force(queue,(n,),None,pos_buf,vel_buf,np.float32(g),np.int32(n))
        cl.enqueue_copy(queue,pos,pos_buf).wait(); cl.enqueue_copy(queue,vel,vel_buf).wait()
        return pos,vel

# ----------------------------
# AVX2 ACCELERATION VIA NUMBA (CPU Fallback)
# ----------------------------

def cpu_force(pos, vel, g):
    n = pos.shape[0]
    for i in prange(n):
        vel[i] += g
        pos[i] += vel[i]
    return pos, vel

# ----------------------------
# RAPID CHECKOUT SNAPSHOT (REPL Stub)
# ----------------------------
class RapidCheckoutSnapshot:
    def __init__(self, session_name="xyz_session"):
        self.snapshots = []
        self.session_name = session_name
        self.file = f"{session_name}.rcs.json"

    # --- Core snapshot mechanics ---
    def record(self, code, context="REPL"):
        snap = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "code": code
        }
        self.snapshots.append(snap)
        print(f"[Snapshot] Recorded ({context}) at {snap['timestamp']}")

    def save(self):
        with open(self.file, "w") as f:
            json.dump(self.snapshots, f, indent=2)
        print(f"[Snapshot] Saved to {self.file}")

    def load(self):
        if os.path.exists(self.file):
            self.snapshots = json.load(open(self.file))
            print(f"[Snapshot] Loaded {len(self.snapshots)} from {self.file}")

    # --- Live REPL ---
    def launch_repl(self, compiler):
        print(">>> XYZ Live REPL (type 'exit' to quit, ':history' for snapshots, ':save', ':load')")
        while True:
            try:
                line = input("xyz> ")
                if line.strip() == "exit":
                    break
                elif line.strip() == ":history":
                    for i, snap in enumerate(self.snapshots):
                        print(f"[{i}] {snap['timestamp']} {snap['context']}: {snap['code']}")
                    continue
                elif line.strip() == ":save":
                    self.save(); continue
                elif line.strip() == ":load":
                    self.load(); continue
                elif line.strip().startswith(":replay"):
                    parts = line.split()
                    if len(parts) > 1:
                        idx = int(parts[1])
                        self.replay(idx, compiler)
                    else:
                        self.replay(len(self.snapshots)-1, compiler)
                    continue

                # Normal REPL command
                self.record(line, context="REPL")
                compiler.run_snippet(line)

            except KeyboardInterrupt:
                break

    # --- Replay snapshots ---
    def replay(self, idx, compiler):
        if idx < 0 or idx >= len(self.snapshots):
            print("[Snapshot] Invalid index")
            return
        snap = self.snapshots[idx]
        print(f"[Snapshot] Replaying {idx}: {snap['code']}")
        compiler.run_snippet(snap["code"])

# Force-run helper: enable grand resolver from CLI or env
# Usage:
#   python xyz_practice.py --force-exec ...
# or
#   XYZ_FORCE_EXEC=1 python xyz_practice.py ...
if __name__ == "__main__":
    # The file already calls main() above; if this block appears twice it will still be safe.
    # We check for a force flag and run the grand resolver immediately after program startup/exit.
    try:
        force_cli = "--force-exec" in sys.argv
        force_env = os.environ.get("XYZ_FORCE_EXEC", "0") == "1"
        if force_cli or force_env:
            # ensure the grand resolver runs
            os.environ["XYZ_GRAND_EXECUTE"] = "1"
            print("[FORCE] Forced grand resolution enabled (via CLI or XYZ_FORCE_EXEC). Running now...")
            try:
                grand_resolve_and_execute()
            except Exception as e:
                # Keep program flow tolerant; original atexit hook remains as fallback.
                print("[FORCE] grand_resolve_and_execute raised:", e)
    except Exception:
        # tolerate failures here — we don't want to break normal program startup
     

# Execution rerouter: keep attempting best-available execution path until environment supports uninterrupted run.
# Use with care: default behavior will retry indefinitely unless timeout (seconds) is set.
        import time, traceback, threading, sys

class ExecutionRerouter:
    def __init__(self, poll_interval: float = 1.0, max_backoff: float = 30.0):
        self.poll_interval = poll_interval
        self.max_backoff = max_backoff

    def _find_source(self, explicit_path: str = None):
        if explicit_path and os.path.isfile(explicit_path):
            return explicit_path
        # prefer CLI arg
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            return sys.argv[1]
        # fallback set of defaults
        for cand in ("main.xy","main.xyz","input.xy","input.xyz"):
            if os.path.isfile(cand):
                return cand
        return None

    def reroute_and_execute(self, entry: str = "main/0", src_path: str = None, timeout: Optional[float] = None):
        start_time = time.time()
        attempt = 0
        backoff = self.poll_interval

        src_path = self._find_source(src_path)
        if not src_path:
            print("[REROUTER] No source file discovered; aborting reroute.")
            return None

        print(f"[REROUTER] Source: {src_path}; entry={entry}; timeout={timeout}")

        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()

        while True:
            attempt += 1
            try:
                print(f"[REROUTER] Attempt {attempt}: ProCompiler pipeline")
                # Try ProCompiler first (preferred)
                try:
                    res = ProCompiler.pro_compile_and_run(src, entry=entry)
                    print(f"[REROUTER] ProCompiler succeeded on attempt {attempt}: {res!r}")
                    return res
                except Exception as e:
                    print(f"[REROUTER] ProCompiler failed: {e}")

                # Next: try FastRuntime (if compilation possible)
                try:
                    print(f"[REROUTER] Attempt {attempt}: Legacy parse -> FastRuntime")
                    toks = lex(src)
                    legacy_parser = Parser(toks)
                    ast_main = legacy_parser.parse()
                    symtab = dict(legacy_parser.functions)
                    hot = HotSwapRegistry()
                    for k,v in symtab.items(): hot.register(k,v)
                    fast_rt = FastRuntime(symtab, hot)
                    res = fast_rt.run(entry, args=[])
                    print(f"[REROUTER] FastRuntime succeeded on attempt {attempt}: {res!r}")
                    return res
                except Exception as e:
                    print(f"[REROUTER] FastRuntime failed: {e}")

                # Final fallback: MiniRuntime interpreter
                try:
                    print(f"[REROUTER] Attempt {attempt}: MiniRuntime interpreter")
                    toks = lex(src)
                    legacy_parser = Parser(toks)
                    ast_main = legacy_parser.parse()
                    symtab = dict(legacy_parser.functions)
                    hot = HotSwapRegistry()
                    for k,v in symtab.items(): hot.register(k,v)
                    mini = MiniRuntime(symtab, hot)
                    res = mini.run_func(entry, [])
                    print(f"[REROUTER] MiniRuntime succeeded on attempt {attempt}: {res!r}")
                    return res
                except Exception as e:
                    print(f"[REROUTER] MiniRuntime failed: {e}")

            except Exception as ex:
                print("[REROUTER] Unexpected error during attempt:", ex)
                traceback.print_exc()

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                print(f"[REROUTER] Timeout reached ({timeout}s). Giving up.")
                return None

            # Backoff and retry; increase backoff up to max_backoff
            time.sleep(backoff)
            backoff = min(backoff * 1.5, self.max_backoff)
            print(f"[REROUTER] Retrying (next backoff={backoff:.2f}s)...")

# If forced execution was requested, use the rerouter to keep trying until an engine runs.
try:
    force_cli = "--force-exec" in sys.argv
    force_env = os.environ.get("XYZ_FORCE_EXEC", "0") == "1"
    if force_cli or force_env:
        # ensure grand resolver also enabled for compatibility
        os.environ["XYZ_GRAND_EXECUTE"] = "1"
        print("[FORCE-REROUTER] Forced execution requested; entering reroute loop.")
        rerouter = ExecutionRerouter(poll_interval=1.0, max_backoff=30.0)
        # optional: allow user to set timeout via env var XYZ_FORCE_TIMEOUT (seconds)
        timeout = os.environ.get("XYZ_FORCE_TIMEOUT")
        timeout = float(timeout) if timeout else None
        rerouter.reroute_and_execute(entry="main/0", src_path=None, timeout=timeout)
except Exception:
    # tolerate failures here — we don't want to break normal program startup
    pass

# Execution rerouter + persistent supervisor
# Dynamically layers a supervisor over the existing runtime behavior.
# Does not alter the existing code semantics or structure.
# Usage:
#  XYZ_FORCE_EXEC=1 python -m xyz_practice
#  XYZ_GRAND_EXECUTE=1 python -m xyz_practice
#  python -m xyz_practice --force-exec
# Behavior:
# - On uncaught exception, the process will attempt to continue running by retrying the
#   last failing operation, function, or script, up to the configured limits.
# - Configuration flags and environment variables can control various aspects of this
#   behavior, including timeouts, error suppression, and execution routes.
# - A REPL-like checkpointing system is used to record the last known good state, allowing
#   the execution to potentially recover from errors.
# - Profiling hooks are integrated to monitor and optimize hot paths over time.
# - Pessimistic: conservatively assumes failure unless proven stable; prefers safer Fallbacks.
# - Watchdog: reveals unresponsive stalls (produce a .diag.txt with traces)
# Notes:
# - This is an advanced runtime feature; use with understanding of the implications.
# - Always test thoroughly in a safe environment before considering any production use.

import time, sys, os, traceback, re, types, math, inspect
from typing import Any, Dict, Callable, List, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ----------------------------------------------
# Error and exception handling
# ----------------------------------------------
class ErrorHandler:
    """
    Centralized error handling and reporting.
    Logs errors to a file and prints summarized reports.
    """
    def __init__(self, log_file=".xyz_errors.log"):
        self.log_file = log_file
        self.lock = threading.Lock()
        self.active_tasks = set()
        self.retries_exceeded = set()

    def report(self, key: str, exc: Exception):
        """
        Report an error with context about where it occurred.
        If the error is from a retried task, this will also indicate that.
        """
        from datetime import datetime
        ts = datetime.utcnow().isoformat()
        errmsg = f"{ts} | {key} | {type(exc).__name__}: {exc}"
        with self.lock:
            # Log to file
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(errmsg + "\n")
            except Exception:
                pass
            # Print to console
            print(errmsg, file=sys.stderr)

    def task_started(self, key: str):
        with self.lock:
            self.active_tasks.add(key)

    def task_completed(self, key: str):
        with self.lock:
            self.active_tasks.discard(key)

    def can_retry(self, key: str, max_retries: int = 3) -> bool:
        """
        Determines if a failed task can be retried based on the number of
        retries already attempted.
        """
        with self.lock:
            if key in self.retries_exceeded:
                return False
            cnt = sum(1 for k in self.active_tasks if k.startswith(f"{key}|"))
            return cnt < max_retries

    def mark_as_exceeded(self, key: str):
        with self.lock:
            self.retries_exceeded.add(key)

_ERROR_HANDLER = ErrorHandler()

# Install a safe global excepthook
def _error_excepthook(exc_type, exc, tb):
    # Send to ErrorHandler but do not terminate
    try:
        _ERROR_HANDLER.report("uncaught", exc_type(exc_value=exc), tb)
    except Exception:
        pass
sys.excepthook = _error_excepthook

# ----------------------------------------------
# Background error diagnosis and reporting
# ----------------------------------------------
class Diagnosis:
    """
    Periodically analyzes and reports the status of running tasks, errors, and performance.
    A watchdog that ensures the system is functioning correctly and reveals issues like stalls.
    """
    def __init__(self, interval: float = 10.0):
        self.interval = interval
        self.lock = threading.Lock()
        self.last_errors = []
        self.stall_threshold = 60.0
        self.task_times = {}

    def update_task_time(self, key: str, duration: float):
        with self.lock:
            self.task_times[key] = duration

    def report(self):
        with self.lock:
            tasks = ", ".join(f"{k}:{v:.1f}s" for k,v in self.task_times.items())
            err_count = len(self.last_errors)
            print(f"[DIAG] Tasks: {tasks} | Recent Errors: {err_count}")

    def check_stalls(self):
        """
        Check for tasks that have not completed within the expected time frame.
        Marks them for review and reveals potential deadlocks or performance issues.
        """
        with self.lock:
            now = time.time()
            for key, start_time in list(self.task_times.items()):
                if key in _ERROR_HANDLER.active_tasks and (now - start_time) > self.stall_threshold:
                    _ERROR_HANDLER.report(f"stall/{key}", RuntimeError("Task exceeded time threshold"))
                    _ERROR_HANDLER.mark_as_exceeded(key)

    def run(self):
        while True:
            time.sleep(self.interval)
            self.check_stalls()
            self.report()

# ----------------------------------------------
# Optimizer passes (IR-level advanced)
# ----------------------------------------------
class OptPasses:
    """
    Advanced optimization passes that work on the lowered IR level.
    Similar to compiler back-end optimizations: combine, simplify, eliminate, and reorder instructions.
    """
    @staticmethod
    def combine(instrs):
        """
        Combine consecutive instructions where possible. Example: ad d/mul by the same factor.
        """
        out = []
        i = 0
        while i < len(instrs):
            a = instrs[i]
            if i+1 < len(instrs):
                b = instrs[i+1]
                # Example: combine ADD/IMUL by same factor
                if a.op == IROp.MUL and b.op == IROp.ADD:
                    factor = a.args[0]
                    new_arg = b.args[0] * factor
                    out.append(IRInstr(IROp.ADD, (new_arg, b.args[1])))
                    i += 2
                    continue
            out.append(a)
            i += 1
        return out

    @staticmethod
    def simplify(instrs):
        """
        Simplify instructions: remove redundant moves, no-ops, etc.
        """
        seen = set()
        out = []
        for ins in instrs:
            if ins.op == IROp.CONST:
                if ins.dst in seen:
                    continue
                seen.add(ins.dst)
            out.append(ins)
        return out

    # -----------------------------
    # Vectorization helpers
    # -----------------------------
    def vector_add(a, b):
        if np is not None:
            return np.add(a, b)
        # fallback element-wise
        return [ (a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(max(len(a), len(b))) ]

    # -----------------------------
    # Parallelism helpers
    # -----------------------------
    def run_in_threads(funcs: List[Callable], max_workers: int = 8):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fn) for fn in funcs]
            return [f.result() for f in futures]

    def run_in_processes(funcs: List[Callable], max_workers: int = 4):
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fn) for fn in funcs]
            return [f.result() for f in futures]

# ----------------------------------------------
# Unified execution model: wraps and retries failed tasks
# ----------------------------------------------
class TaskWrapper:
    """
    Wraps a target function and retries on failure.
    Dynamic dispatch to either the original function or a safe replacement on error.
    Monitors execution and reports errors using ErrorHandler.
    """
    def __init__(self, target: Callable, key: str, max_retries: int = 3):
        self.target = target
        self.key = key
        self.max_retries = max_retries
        self.retries = 0
        self.last_duration = 0.0

    def __call__(self, *args, **kwargs):
        from datetime import datetime
        start_time = datetime.utcnow()
        try:
            _ERROR_HANDLER.task_started(self.key)
            result = self.target(*args, **kwargs)
            _ERROR_HANDLER.task_completed(self.key)
            return result
        except Exception as e:
            _ERROR_HANDLER.task_completed(self.key)
            if not _PERSIST_SUP.maybe_fix_or_quarantine(self.key, e):
                return None
            # Log detailed error record
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            _ERROR_HANDLER.note(self.key, elapsed)
            if self.retries < self.max_retries:
                self.retries += 1
                wait_time = min(1.0 * (2 ** self.retries), 30.0)
                time.sleep(wait_time)
                return self.__call__(*args, **kwargs)
            else:
                self.quarantine()
        return None

    def quarantine(self):
        _PERSIST_SUP.quarantine(self.key)
        print(f"[PERSIST] {self.key} has been quarantined after {self.max_retries} failures.")

# Integration: wraps the target function with TaskWrapper
def wrap_task(key: str, func: Callable):
    def wrapped(*args, **kwargs):
        return TaskWrapper(func, key)(*args, **kwargs)
    return wrapped

# Install wrappers for known functions (MiniRuntime run_func as example)
try:
    if "MiniRuntime" in globals() and hasattr(MiniRuntime, "run_func"):
        original_mini_run_func = MiniRuntime.run_func

        def wrapped_mini_run_func(self, key: str, args: List[Any]):
            return TaskWrapper(original_mini_run_func, key)(self, key, args)

        MiniRuntime.run_func = wrapped_mini_run_func
except Exception:
    pass

# Provide a simplified API at module level for common operations:
def xyz_run(source: str, entry: str = "main/0", timeout: Optional[float] = None):
    """
    High-level run function: compiles, links, and runs the source code.
    Tries to use the best available engine and retries on failure.
    """
    src_path = None
    try:
        # Auto-discover source file from common names or use explicit path
        if isinstance(source, str) and os.path.isfile(source):
            src_path = source
        else:
            for cand in ("main.xy","main.xyz","input.xy","input.xyz"):
                if os.path.isfile(cand):
                    src_path = cand
                    break
        if not src_path:
            raise FileNotFoundError("No valid source file found")

        print(f"[XYZ_RUN] Using source: {src_path}")

        # Read source code
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()

        # Execution rerouter: keep attempting best-available execution path until environment stabilizes
        rerouter = ExecutionRerouter()
        return rerouter.reroute_and_execute(entry=entry, src_path=src_path, timeout=timeout)

    except Exception as e:
        # Report errors that escape the rerouter (fatal)
        print(f"[XYZ_RUN] Error: {e}", file=sys.stderr)
        traceback.print_exc()
        return None

# --- Canonical FastVM (consolidated, fully implemented) ---
# This redefines FastVM with complete implementations so no monkeypatching is needed.
# It is safe to place at the end of the file; future FastRuntime instances will use this class.
class FastVM:
    def __init__(self, functions: Dict[str, BytecodeFunction], hot_registry: HotSwapRegistry, pool: MemoryPool = None):
        self.functions = functions or {}
        self.hot = hot_registry
        self.pool = pool or MemoryPool()
        self.globals: Dict[str, Any] = {
            # built-ins
            "print": print,
            "alloc": lambda size: bytearray(int(size) if size is not None else 0),
            # helpers possibly enabled by mega/advanced features
            "list_add": globals().get("list_add"),
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=max(4, threading.active_count()))
        self.inline_threshold = 64
        self.hot_threshold = 100

    def run(self, key: str, args: List[Any] = None) -> Any:
        args = args or []
        if key not in self.functions:
            # allow "name" without arity by resolving 0-arity first
            if f"{key}/0" in self.functions:
                key = f"{key}/0"
            else:
                raise KeyError(f"FastVM: function not compiled: {key}")
        return self._run_fn(self.functions[key], args)

    def _run_fn(self, fn: BytecodeFunction, args: List[Any]) -> Any:
        fn.exec_count += 1
        if fn.exec_count == self.hot_threshold:
            self.optimize_hot(fn)

        code = fn.code
        consts = fn.consts
        stack: List[Any] = []
        locals_list: List[Any] = [None] * max(16, len(fn.params) + 8)
        for i, p in enumerate(fn.params):
            locals_list[i] = args[i] if i < len(args) else None

        pc = 0
        call_cache = fn.call_caches

        while pc < len(code):
            op = code[pc]; pc += 1

            if op == Op.LOAD_CONST:
                idx = code[pc]; pc += 1
                stack.append(consts[idx])

            elif op == Op.LOAD_LOCAL:
                idx = code[pc]; pc += 1
                stack.append(locals_list[idx])

            elif op == Op.STORE_LOCAL:
                idx = code[pc]; pc += 1
                locals_list[idx] = stack.pop()

            elif op == Op.LOAD_GLOBAL:
                idx = code[pc]; pc += 1
                name = consts[idx]
                stack.append(self.globals.get(name))

            elif op == Op.CALL:
                idx = code[pc]; pc += 1
                cname = consts[idx]
                # inline cache
                cache_key = pc - 2
                target = call_cache.get(cache_key)
                if target is None:
                    target = self.resolve_call_target(cname)
                    if target is not None:
                        call_cache[cache_key] = target

                try:
                    arity = int(str(cname).rsplit("/", 1)[1])
                except Exception:
                    arity = 0
                argv = [stack.pop() for _ in range(arity)][::-1]

                if isinstance(target, BytecodeFunction):
                    stack.append(self._run_fn(target, argv))
                elif callable(target):
                    stack.append(target(*argv))
                else:
                    raise RuntimeError(f"FastVM: call target not found: {cname}")

            elif op == Op.RETURN:
                return stack.pop() if stack else None

            elif op == Op.BINARY_ADD:
                b = stack.pop(); a = stack.pop()
                if isinstance(a, list) and isinstance(b, list):
                    n = max(len(a), len(b))
                    stack.append([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])
                else:
                    stack.append((a if a is not None else 0) + (b if b is not None else 0))

            elif op == Op.BINARY_SUB:
                b = stack.pop(); a = stack.pop()
                stack.append((a if a is not None else 0) - (b if b is not None else 0))

            elif op == Op.BINARY_MUL:
                b = stack.pop(); a = stack.pop()
                if isinstance(a, list) and isinstance(b, (int, float)):
                    stack.append([x * b for x in a])
                else:
                    stack.append((a if a is not None else 0) * (b if b is not None else 0))

            elif op == Op.BINARY_DIV:
                b = stack.pop(); a = stack.pop()
                stack.append(0 if (b == 0 or b is None) else ((a if a is not None else 0) / b))

            elif op == Op.BINARY_POW:
                b = stack.pop(); a = stack.pop()
                stack.append(int(math.pow(a or 0, b or 0)))

            elif op == Op.BUILD_LIST:
                n_items = code[pc]; pc += 1
                items = [stack.pop() for _ in range(n_items)][::-1]
                stack.append(items)

            elif op == Op.INDEX:
                idx = stack.pop(); base = stack.pop()
                if isinstance(base, list):
                    if not isinstance(idx, int): raise TypeError("Index must be int")
                    stack.append(base[idx])
                elif isinstance(base, dict):
                    stack.append(base.get(idx))
                else:
                    raise TypeError("Object is not indexable")

            elif op == Op.POP:
                if stack: stack.pop()

            elif op == Op.NOP:
                pass

            else:
                # Unknown op → slow fallback
                return self._slow_fallback(fn, code, pc - 1, stack, locals_list)

        return None

    def resolve_call_target(self, cname: str) -> Optional[Union[BytecodeFunction, Callable[..., Any]]]:
        # 1) hot-swapped or compiled function
        if cname in self.functions:
            return self.functions[cname]
        funcdef = self.hot.get(cname)
        if funcdef:
            # use bytecode if available, else interpret
            if cname in self.functions:
                return self.functions[cname]
            return lambda *args: self.call_interpreted(cname, args)

        # 2) built-ins (by base name and by full key)
        base = str(cname).split("/", 1)[0]
        if cname in self.globals and callable(self.globals[cname]):
            return self.globals[cname]
        if base in self.globals and callable(self.globals[base]):
            return self.globals[base]

        return None

    def call_interpreted(self, cname: str, args: List[Any]) -> Any:
        funcdef = self.hot.get(cname)
        if not funcdef:
            raise RuntimeError(f"FastVM.call_interpreted: target {cname} not found in HotSwapRegistry")
        # MiniRuntime expects a symtab keyed by 'name/arity'
        symtab = dict(self.hot.table)
        mini = MiniRuntime(symtab, self.hot)
        return mini.run_func(cname, list(args))

    def _slow_fallback(self, fn: BytecodeFunction, code, pc, stack, locals_list) -> Any:
        # Best-effort: run the original function via MiniRuntime
        key = fn.name
        funcdef = self.hot.get(key)
        if not funcdef:
            # try by base name
            base = key.split("/", 1)[0]
            with self.hot.lock:
                for k, v in self.hot.table.items():
                    if k.startswith(base + "/"):
                        funcdef = v
                        key = k
                        break
        if not funcdef:
            raise RuntimeError(f"FastVM slow_fallback: no FuncDef found for {fn.name}")

        # Prepare argument list from locals by arity
        arity = len(getattr(funcdef, "params", []))
        argvals = [locals_list[i] if i < len(locals_list) else None for i in range(arity)]

        symtab = dict(self.hot.table)
        mini = MiniRuntime(symtab, self.hot)
        return mini.run_func(f"{funcdef.name}/{arity}", argvals)

    def optimize_hot(self, fn: BytecodeFunction):
        # Inline tiny callees at CALL sites (shallow, safe)
        new_code: List[int] = []
        i = 0
        while i < len(fn.code):
            op = fn.code[i]
            if op == Op.CALL and i + 1 < len(fn.code):
                idx = fn.code[i + 1]
                cname = fn.consts[idx]
                target = self.resolve_call_target(cname)
                if isinstance(target, BytecodeFunction) and len(target.code) <= self.inline_threshold:
                    new_code.extend(target.code)
                    i += 2
                    continue
            new_code.append(op)
            if op in (Op.LOAD_CONST, Op.LOAD_LOCAL, Op.STORE_LOCAL, Op.LOAD_GLOBAL, Op.CALL, Op.BUILD_LIST, Op.JUMP, Op.JUMP_IF_FALSE):
                if i + 1 < len(fn.code):
                    new_code.append(fn.code[i + 1])
                    i += 2
                    continue
            i += 1
        fn.code = new_code

    # Minimal legacy dodecagram support (kept for compatibility)
    def run_dodecagram(self, seq: Union[str, bytes]) -> Any:
        if isinstance(seq, (bytes, bytearray)):
            try:
                seq = seq.decode("utf-8", errors="ignore")
            except Exception:
                return None
        s = str(seq or "")
        digits = "0123456789ab"

        def b12(num: str) -> int:
            v = 0
            for ch in num:
                v = v * 12 + digits.index(ch)
            return v

        i, n = 0, len(s)
        stack: List[int] = []
        while i < n:
            op = s[i]; i += 1
            if op == "1":
                neg = False
                if i < n and s[i] == "-":
                    neg = True; i += 1
                digs = []
                while i < n and s[i] in digits:
                    digs.append(s[i]); i += 1
                val = b12("".join(digs)) if digs else 0
                stack.append(-val if neg else val)
            elif op == "4":
                if len(stack) < 2: return None
                b = stack.pop(); a = stack.pop(); stack.append(a + b)
            elif op == "5":
                if not stack: return None
                stack[-1] = -stack[-1]
            elif op == "7":
                return stack.pop() if stack else None
            else:
                # ignore unknown legacy opcode
                continue
        return stack[-1] if stack else None
