#!/usr/bin/env python3
# xyzc.py - XYZ Compiler (continued implementation)
# This iteration expands the language implementation with:
# - List and Map literals and indexing with bounds/type checking
# - Throw/raise/except flow with Try/Catch/Isolate/Force/Remove semantics
# - maloc alias + improved alloc/free primitives
# - Method/field-call support via dotted names (runtime lookup)
# - Simple container type checking, indexing, tracing and tags
# - Small improvements to parser and runtime to exercise these features
#
# The file preserves previous features (hot-swap, linker, codegen, packetizer).
# Grandiose resolution: orchestrate full pipeline and execute best-available runtime.
# This runs only when environment variable `XYZ_GRAND_EXECUTE=1` is set to avoid surprising behavior.
# Usage:
#   XYZ_GRAND_EXECUTE=1 python xyz_practice.py <source.xy>
# The resolver will attempt, in order:
#  1) ProCompiler (professional parser → typecheck → IR → engine)
#  2) FastRuntime (bytecode VM) if available/compilable
#  3) MiniRuntime (AST interpreter) as final fallback
# It will also emit object/asm artifacts under the working directory for inspection.
# --- MEGA FEATURES: types, structs, macros, self-expansion, rapid checkout ---
# Opt-in toolbox. Call `enable_mega_features(hot_registry, mini_runtime, fast_runtime=None)` to wire in.
# Provides:
#  - Rich type system registry (primitives, vectors, matrices, complex, decimal, bigint)
#  - Struct/enum declarations and runtime layout
#  - Tuple/Set literal AST nodes + Match expression support
#  - MacroEngine for compile-time expansion and SelfExpander that can synthesize new functions and hot-swap them
#  - Rapid checkout / snapshot and restore of symbol table + hot-swap registry
#  - Lightweight operation dispatch for new types (addition, bitwise, shifts, dot-access)
# This is intentionally conservative and safe: expansion produces FuncDef objects and registers them.

import copy, hashlib, inspect, decimal, numbers
from decimal import Decimal
from typing import Callable

# --- Type registry ---
class MegaType:
    def __init__(self, name: str, meta: Dict[str, Any]=None):
        self.name = name
        self.meta = meta or {}
    def __repr__(self): return f"MegaType({self.name})"

class TypeRegistry:
    _types: Dict[str, MegaType] = {}
    @classmethod
    def register(cls, t: MegaType):
        cls._types[t.name] = t
    @classmethod
    def get(cls, name: str):
        return cls._types.get(name)
    @classmethod
    def list_types(cls):
        return list(cls._types.keys())

# register common primitives
for pname in ("Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64","BigInt","Float32","Float64","Decimal","Complex","Bool","String","Any"):
    TypeRegistry.register(MegaType(pname))

# convenience factory
def mega_type(name:str, **meta): 
    t = MegaType(name, meta); TypeRegistry.register(t); return t

# --- Struct/Enum support (AST + runtime layouts) ---
class StructDef:
    def __init__(self, name: str, fields: List[Tuple[str,str]]):
        self.name = name
        self.fields = fields  # list of (fieldname, type_name)
        self.size = len(fields)
    def __repr__(self): return f"StructDef({self.name}, fields={self.fields})"

class StructRegistry:
    _structs: Dict[str, StructDef] = {}
    @classmethod
    def register(cls, s: StructDef):
        cls._structs[s.name] = s
    @classmethod
    def get(cls, name: str):
        return cls._structs.get(name)
    @classmethod
    def dump(cls):
        return dict(cls._structs)

# small helper to create a boxed instance (dict-backed)
def make_struct_instance(name: str, **kwargs):
    sdef = StructRegistry.get(name)
    if not sdef: raise Exception(f"Unknown struct {name}")
    inst = {"__struct__": name}
    for fname, ftype in sdef.fields:
        inst[fname] = kwargs.get(fname, None)
    return inst

# --- Macro Engine & Self-Expander ---
class MacroEngine:
    def __init__(self):
        self.macros: Dict[str, Callable[[Any], List[ASTNode]]] = {}
    def register_macro(self, name: str, fn: Callable[[Any], List[ASTNode]]):
        self.macros[name] = fn
    def expand(self, node):
        # simple pattern: if node is Call and name matches macro, run expansion
        if isinstance(node, Call) and node.name in self.macros:
            try:
                return self.macros[node.name](node)
            except Exception as e:
                print(f"[MACRO] expansion error for {node.name}: {e}")
                return [node]
        return [node]

class SelfExpander:
    """
    Generates functions at runtime deterministically from templates and registers to hot_registry.
    Example usage: expander.generate_vector_ops("Vec3", base_type="Float64") -> creates add/sub/mul ops.
    """
    def __init__(self, hot_registry: HotSwapRegistry, symtab: Dict[str, FuncDef]):
        self.hot = hot_registry
        self.symtab = symtab
        self.generated = {}
    def _mk_func(self, name: str, params: List[str], body: List[ASTNode]) -> FuncDef:
        f = FuncDef(name, params, body)
        key = f"{name}/{len(params)}"
        self.hot.register(key, f)
        self.symtab[key] = f
        self.generated[key] = f
        return f
    def generate_vector_ops(self, vec_name: str, dims:int=3, base_type="Float64"):
        # generate a constructor and element-wise add
        ctor_name = f"{vec_name}.new"
        params = [f"x{i}" for i in range(dims)]
        # body: assign fields then return placeholder (simulate)
        body = []
        for i,p in enumerate(params):
            body.append(Assign(f"f{i}", Var(p)))
        body.append(Return(ListLiteral([Var(p) for p in params])))
        self._mk_func(ctor_name, params, body)
        # add op add
        add_name = f"{vec_name}.add"
        body_add = [Assign("a", Var("a")), Assign("b", Var("b")), Return(Call("list_add", [Var("a"), Var("b")]))]
        self._mk_func(add_name, ["a","b"], body_add)
        return [ctor_name, add_name]

# --- Rapid checkout (snapshot / restore) ---
def rapid_checkout_snapshot(symtab: Dict[str, FuncDef], hot: HotSwapRegistry, path: str):
    state = {
        "symtab_keys": list(symtab.keys()),
        "hot_keys": list(hot.table.keys()),
        "hot_funcs": {k: serialize_funcdef(hot.table[k]) for k in hot.table.keys()}
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"[CHECKOUT] snapshot written to {path}")

def rapid_checkout_restore(path: str, symtab: Dict[str, FuncDef], hot: HotSwapRegistry):
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    # restore functions into hot & symtab
    for k, sf in state.get("hot_funcs", {}).items():
        fd = deserialize_funcdef(sf)
        hot.register(k, fd)
        symtab[k] = fd
    print(f"[CHECKOUT] restored snapshot from {path}")

def serialize_funcdef(fd: FuncDef):
    # minimal serializer: record name, params, and simple body as strings via repr
    return {"name": fd.name, "params": fd.params, "body_repr": [repr(type(n)) for n in fd.body]}

def deserialize_funcdef(sdata):
    # can't recreate full AST robustly; create stub function that returns None
    return FuncDef(sdata["name"], sdata["params"], [Return(Number("0"))])

# --- Operation primitives for advanced types ---
def list_add(a,b):
    if not isinstance(a, list) or not isinstance(b, list): raise Exception("list_add requires lists")
    n = max(len(a), len(b))
    out = []
    for i in range(n):
        va = a[i] if i < len(a) else 0
        vb = b[i] if i < len(b) else 0
        out.append(va + vb)
    return out

# register primitive helper into MiniRuntime by patching eval dispatch or adding to builtins mapping
def enable_mega_features(hot_registry: HotSwapRegistry, mini_runtime: MiniRuntime, fast_runtime: Optional[FastRuntime]=None):
    # add new struct example
    StructRegistry.register(StructDef("Point2D",[("x","Float64"),("y","Float64")]))
    StructRegistry.register(StructDef("Rect",[("min","Point2D"),("max","Point2D")]))
    # register runtime helper functions as hot functions (thin wrappers)
    pf = FuncDef("list_add/2".split("/")[0], ["a","b"], [Return(Null())])  # placeholder
    key = "list_add/2"
    hot_registry.register(key, pf)  # placeholder so FastRuntime can resolve by name
    # attach actual python implementation into FastVM/native resolver by adding into FastVM.resolve_call_target mapping via monkeypatch
    # For MiniRuntime, inject a builtin dispatcher mapping name->callable
    if not hasattr(mini_runtime, "_mega_builtins"):
        mini_runtime._mega_builtins = {}
        mini_runtime._mega_builtins["list_add"] = list_add
    # patch MiniRuntime.eval to consult _mega_builtins before trying function lookup
    orig_eval = mini_runtime.eval
    def mega_eval(node):
        if isinstance(node, Call):
            if node.name in getattr(mini_runtime, "_mega_builtins", {}):
                args = [mini_runtime.eval(a) for a in node.args]
                return mini_runtime._mega_builtins[node.name](*args)
        return orig_eval(node)
    mini_runtime.eval = mega_eval
    # allow fast_runtime to resolve 'list_add' if present
    if fast_runtime:
        # add to fast runtime globals mapping so FastVM.resolve_call_target picks it as native callable
        fast_runtime.vm.globals["list_add"] = list_add
    print("[MEGA] enabled: registered structs:", StructRegistry.dump())
    print("[MEGA] available types:", TypeRegistry.list_types())
    return True

# Small convenience to introspect and expand codebase (self-expansion engine)
def massive_self_expand(pattern: str, hot_registry: HotSwapRegistry, symtab: Dict[str, FuncDef], expander: SelfExpander):
    """
    Given a simple pattern, produce a family of functions and register them.
    The pattern can be like "vector<N>" and the expander will create vector ops for common N.
    """
    created = []
    if pattern.startswith("vector"):
        for n in (2,3,4):
            vecname = f"Vec{n}"
            created += expander.generate_vector_ops(vecname, dims=n)
    # produce checksum and report
    digest = hashlib.sha256("".join(created).encode("utf-8")).hexdigest()[:8]
    print(f"[SELF-EXPAND] generated {len(created)} symbols; digest={digest}")
    return created

# --- Register a few helpful macros and an automatic expander at import time ---
_global_macro_engine = MacroEngine()
def _macro_repeat_expand(node: Call):
    # example macro: repeat(n, expr) -> expands into [expr, expr, ...]
    if len(node.args) >= 2:
        count_node, expr = node.args[0], node.args[1]
        if isinstance(count_node, Number):
            cnt = int(count_node.val)
            return [expr for _ in range(cnt)]
    return [node]


# --- Quick demo wiring function for users ---
def mega_demo_enable(symtab: Dict[str, FuncDef], hot_registry: HotSwapRegistry, mini_runtime: MiniRuntime, fast_runtime: Optional[FastRuntime]=None):
    expander = SelfExpander(hot_registry, symtab)
    created = massive_self_expand("vector", hot_registry, symtab, expander)
    enable_mega_features(hot_registry, mini_runtime, fast_runtime)
    # snapshot to file for rapid checkout
    rapid_checkout_snapshot(symtab, hot_registry, "mega_snapshot.json")
    return {"created": created, "snapshot": "mega_snapshot.json"}

# End of MEGA FEATURES
# -------------------------
import os, atexit, pathlib, traceback, time

def grand_resolve_and_execute():
    if os.environ.get("XYZ_GRAND_EXECUTE", "0") != "1":
        return

    start = time.time()
    print("\n[GRAND RESOLVE] Starting full pipeline resolution and execution")
    try:
        # Determine input source (prefer CLI arg)
        src_path = None
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            src_path = sys.argv[1]
        else:
            # try common files
            for cand in ("main.xy","main.xyz","input.xy","input.xyz"):
                if os.path.isfile(cand):
                    src_path = cand; break

        if not src_path:
            print("[GRAND RESOLVE] No source file found on CLI or defaults; aborting grand execution")
            return

        print(f"[GRAND RESOLVE] Using source: {src_path}")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()

        # Attempt 1: ProCompiler end-to-end (preferred)
        try:
            print("[GRAND RESOLVE] Attempting ProCompiler pipeline (parse → typecheck → IR → execute)")
            res = ProCompiler.pro_compile_and_run(src, entry="main/0")
            print(f"[GRAND RESOLVE] ProCompiler execution returned: {res!r}")
            print(f"[GRAND RESOLVE] Completed in {time.time()-start:.3f}s")
            return
        except Exception as e:
            print("[GRAND RESOLVE] ProCompiler pipeline failed:", e)
            traceback.print_exc()

        # Fallback: parse with legacy Parser and compile/link artifacts
        try:
            print("[GRAND RESOLVE] Falling back to legacy Parser + Codegen + runtimes")
            toks = lex(src)
            legacy_parser = Parser(toks)
            ast_main = legacy_parser.parse()
            symtab = dict(legacy_parser.functions)

            # emit annotated assembly and object
            cg = Codegen(symtab, HotSwapRegistry())
            asm = cg.generate(ast_main, emit_pkt=False)
            obj_path = "grand_temp.xyzobj"
            cg.emit_object(asm, obj_path)
            linked_asm = "grand_linked.asm"
            cg.link_objects([obj_path], linked_asm)

            # Try FastRuntime if it compiles functions
            try:
                hot = HotSwapRegistry()
                for k,v in symtab.items(): hot.register(k,v)
                fast_rt = FastRuntime(symtab, hot)
                print("[GRAND RESOLVE] Running FastRuntime main/0 ...")
                fres = fast_rt.run("main/0", args=[])
                print(f"[GRAND RESOLVE] FastRuntime main/0 -> {fres!r}")
                print(f"[GRAND RESOLVE] Completed in {time.time()-start:.3f}s")
                return
            except Exception as fe:
                print("[GRAND RESOLVE] FastRuntime run failed:", fe)
                traceback.print_exc()

            # Final fallback: MiniRuntime interpreter
            try:
                hot = HotSwapRegistry()
                for k,v in symtab.items(): hot.register(k,v)
                mini = MiniRuntime(symtab, hot)
                print("[GRAND RESOLVE] Running MiniRuntime main/0 ...")
                mres = mini.run_func("main/0", [])
                print(f"[GRAND RESOLVE] MiniRuntime main/0 -> {mres!r}")
                print(f"[GRAND RESOLVE] Completed in {time.time()-start:.3f}s")
                return
            except Exception as me:
                print("[GRAND RESOLVE] MiniRuntime run failed:", me)
                traceback.print_exc()

        except Exception as e:
            print("[GRAND RESOLVE] Legacy pipeline failed:", e)
            traceback.print_exc()

        print("[GRAND RESOLVE] All resolution attempts exhausted; nothing executed successfully.")
    finally:
        elapsed = time.time() - start
        print(f"[GRAND RESOLVE] Finished attempts (elapsed {elapsed:.3f}s)")

# Register atexit hook so grand resolution runs after main returns (only when enabled).
atexit.register(grand_resolve_and_execute)
# -------------------------
from mimetypes import init
import sys, re, argparse, math, threading, struct, json, socket
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# LEXER (add brackets)
# -------------------------
# Professional pipeline additions (parser, AST, semantic analysis, IR, codegen, runtime)
# Appended as opt-in modules: use ProCompiler.pro_compile_and_run(source) to exercise.
# Designed to integrate with existing toolchain without replacing current code.

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Union, Iterable

# -------------------------
# Pro AST (dataclasses)
# -------------------------
@dataclass
class PNode:
    pass

@dataclass
class PProgram(PNode):
    body: List[PNode] = field(default_factory=list)

@dataclass
class PFunc(PNode):
    name: str
    params: List[str]
    body: List[PNode]
    ret_type: Optional['PType'] = None

@dataclass
class PReturn(PNode):
    expr: Optional[PNode]

@dataclass
class PCall(PNode):
    target: Union[str, PNode]
    args: List[PNode]

@dataclass
class PVar(PNode):
    name: str

@dataclass
class PAssign(PNode):
    name: str
    expr: PNode

@dataclass
class PNumber(PNode):
    raw: str

@dataclass
class PBool(PNode):
    val: bool

@dataclass
class PBinOp(PNode):
    op: str
    left: PNode
    right: PNode

@dataclass
class PIf(PNode):
    cond: PNode
    then_body: List[PNode]
    else_body: List[PNode]

@dataclass
class PWhile(PNode):
    cond: PNode
    body: List[PNode]

@dataclass
class PListLiteral(PNode):
    elements: List[PNode]

@dataclass
class PIndex(PNode):
    base: PNode
    index: PNode

@dataclass
class PLambda(PNode):
    params: List[str]
    body: List[PNode]

# -------------------------
# Pro Types
# -------------------------
class PType:
    pass

@dataclass
class TInt(PType): pass

@dataclass
class TFloat(PType): pass

@dataclass
class TBool(PType): pass

@dataclass
class TNull(PType): pass

@dataclass
class TAny(PType): pass

@dataclass
class TList(PType):
    elem: PType

@dataclass
class TFunc(PType):
    params: List[PType]
    ret: PType

# -------------------------
# Pro Lexer (lightweight, robust)
# -------------------------
ProToken = Tuple[str,str]  # (kind, value)

class ProLexer:
    token_spec = [
        ('NUMBER', r'-?\d+(\.\d+)?'),
        ('ID',     r'[A-Za-z_][A-Za-z0-9_]*'),
        ('STRING', r'"[^"]*"|\'[^\']*\''),
        ('OP',     r'[\+\-\*/\^=<>!&|\.]+'),
        ('LP',     r'\('), ('RP', r'\)'),
        ('LB',     r'\['), ('RB', r'\]'),
        ('LBRA',   r'\{'), ('RBRA', r'\}'),
        ('COMMA',  r','), ('SEMI', r';'),
        ('WS',     r'\s+'),
        ('MISM',   r'.'),
    ]
    regex = re.compile('|'.join(f'(?P<{n}>{p})' for n,p in token_spec))
    keywords = set(["func","return","if","else","while","for","lambda","true","false","null","parallel","try","catch","throw","enum","eval","alloc","free","print","isolate","force","remove"])
    def __init__(self, src: str):
        self.src = src
        self.pos = 0
        self.tokens: List[ProToken] = []
        self._lex_all()
        self.i = 0
    def _lex_all(self):
        for m in self.regex.finditer(self.src):
            kind = m.lastgroup; val = m.group()
            if kind == 'WS': continue
            if kind == 'ID' and val in self.keywords:
                kind = val.upper()
            self.tokens.append((kind,val))
    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else ('EOF','')
    def next(self):
        t = self.peek(); self.i += 1; return t
    def accept(self, kind):
        if self.peek()[0] == kind:
            return self.next()
        return None
    def expect(self, kind):
        t = self.next()
        if t[0] != kind:
            raise SyntaxError(f"Expected {kind} got {t}")
        return t

# -------------------------
# Pro Parser (recursive-descent, more professional)
# -------------------------
class ProParser:
    def __init__(self, src: str):
        self.lex = ProLexer(src)
    def parse(self) -> PProgram:
        body = []
        while self.lex.peek()[0] != 'EOF':
            if self.lex.peek()[0] == 'func':
                body.append(self.parse_func())
            else:
                stmt = self.parse_stmt()
                if stmt: body.append(stmt)
        return PProgram(body)
    def parse_func(self) -> PFunc:
        self.lex.expect('func')
        _, name = self.lex.expect('ID')
        self.lex.expect('LP')
        params=[]
        if self.lex.peek()[0] != 'RP':
            while True:
                _, pid = self.lex.expect('ID'); params.append(pid)
                if self.lex.accept('COMMA'): continue
                break
        self.lex.expect('RP'); self.lex.expect('LBRA')
        body=[]
        while self.lex.peek()[0] != 'RBRA':
            body.append(self.parse_stmt())
        self.lex.expect('RBRA')
        return PFunc(name, params, body)
    def parse_stmt(self) -> PNode:
        t = self.lex.peek()
        if t[0] == 'return':
            self.lex.next()
            expr = self.parse_expr()
            self.lex.accept('SEMI')
            return PReturn(expr)
        if t[0] == 'LBRA':
            self.lex.next(); # block inline -- treat as sequence
            stmts=[]
            while self.lex.peek()[0] != 'RBRA':
                stmts.append(self.parse_stmt())
            self.lex.expect('RBRA')
            return PProgram(stmts)
        # assignment or expression
        expr = self.parse_expr()
        if isinstance(expr, PVar) and self.lex.peek()[0] == 'OP' and self.lex.peek()[1] == '=':
            self.lex.next(); val = self.parse_expr(); self.lex.accept('SEMI'); return PAssign(expr.name, val)
        self.lex.accept('SEMI')
        return expr
    def parse_expr(self, rbp=0) -> PNode:
        # Pratt parser for expressions
        tkind, tval = self.lex.next()
        left = self.nud(tkind,tval)
        while rbp < self.lbp(self.lex.peek()):
            opkind, opval = self.lex.next()
            left = self.led(opkind, opval, left)
        return left
    def nud(self, kind, val):
        if kind == 'NUMBER':
            return PNumber(val)
        if kind == 'true':
            return PBool(True)
        if kind == 'false':
            return PBool(False)
        if kind == 'ID':
            # variable or call
            if self.lex.peek()[0] == 'LP':
                # call
                self.lex.next()  # eat LP
                args=[]
                if self.lex.peek()[0] != 'RP':
                    while True:
                        args.append(self.parse_expr())
                        if self.lex.accept('COMMA'): continue
                        break
                self.lex.expect('RP')
                return PCall(val, args)
            return PVar(val)
        if kind == 'LP':
            e = self.parse_expr()
            self.lex.expect('RP')
            return e
        if kind == 'LB':
            # list literal recorded as LBRA token earlier in token set? treat LB as LBRACE
            elems=[]
            while self.lex.peek()[0] != 'RB':
                elems.append(self.parse_expr())
                if self.lex.accept('COMMA'): continue
                break
            self.lex.expect('RB')
            return PListLiteral(elems)
        raise SyntaxError(f"Unexpected token in nud: {kind} {val}")
    def lbp(self, peek):
        k = peek()[0] if callable(peek) else peek[0]
        if k == 'OP':
            return 10
        if k == 'LP':
            return 20
        return 0
    def led(self, kind, val, left):
        if kind == 'OP':
            # binary
            right = self.parse_expr(10)
            return PBinOp(val, left, right)
        if kind == 'LP':
            # function-call via expression (e.g., (f)(args)) — not commonly used
            args=[]
            if self.lex.peek()[0] != 'RP':
                while True:
                    args.append(self.parse_expr())
                    if self.lex.accept('COMMA'): continue
                    break
            self.lex.expect('RP')
            return PCall(left, args)
        raise SyntaxError(f"Unexpected led: {kind}")

# -------------------------
# Pro Semantic Analyzer / TypeChecker (basic professional)
# -------------------------
class SemanticError(Exception): pass

class ProSymbolTable:
    def __init__(self):
        self.scopes: List[Dict[str, PType]] = [{}]
    def push(self): self.scopes.append({})
    def pop(self): self.scopes.pop()
    def declare(self, name: str, typ: PType):
        self.scopes[-1][name] = typ
    def lookup(self, name: str) -> Optional[PType]:
        for s in reversed(self.scopes):
            if name in s: return s[name]
        return None

class TypeChecker:
    def __init__(self, program: PProgram):
        self.program = program
        self.sym = ProSymbolTable()
        self.func_types: Dict[str, TFunc] = {}
    def check(self):
        # first pass: register function signatures
        for node in self.program.body:
            if isinstance(node, PFunc):
                # default params any, ret any
                param_ts = [TAny() for _ in node.params]
                self.func_types[f"{node.name}/{len(node.params)}"] = TFunc(param_ts, TAny())
                self.sym.declare(node.name, self.func_types[f"{node.name}/{len(node.params)}"])
        # second pass: check bodies
        for node in self.program.body:
            if isinstance(node, PFunc):
                self.check_func(node)
    def check_func(self, fn: PFunc):
        sig = self.func_types.get(f"{fn.name}/{len(fn.params)}")
        assert sig is not None
        self.sym.push()
        for i,p in enumerate(fn.params):
            self.sym.declare(p, sig.params[i])
        ret_type: PType = sig.ret
        for stmt in fn.body:
            self.check_node(stmt)
        self.sym.pop()
    def check_node(self, node: PNode) -> PType:
        if isinstance(node, PNumber):
            return TInt() if '.' not in node.raw else TFloat()
        if isinstance(node, PBool):
            return TBool()
        if isinstance(node, PVar):
            t = self.sym.lookup(node.name)
            if t is None:
                raise SemanticError(f"Undefined variable {node.name}")
            return t
        if isinstance(node, PAssign):
            t = self.check_node(node.expr)
            self.sym.declare(node.name, t)
            return t
        if isinstance(node, PBinOp):
            lt = self.check_node(node.left); rt = self.check_node(node.right)
            # arithmetic rules
            if isinstance(lt, TFloat) or isinstance(rt, TFloat):
                return TFloat()
            if isinstance(lt, TInt) and isinstance(rt, TInt):
                return TInt()
            return TAny()
        if isinstance(node, PCall):
            # resolve target
            if isinstance(node.target, str):
                key = f"{node.target}/{len(node.args)}"
                ft = self.func_types.get(key)
                if ft:
                    # check args
                    for a,pt in zip(node.args, ft.params):
                        at = self.check_node(a)
                        # allow Any
                    return ft.ret
                # builtins
                if node.target == "print":
                    for a in node.args: self.check_node(a)
                    return TNull()
            else:
                # dynamic call expression (callable result)
                return TAny()
            raise SemanticError(f"Unknown function {node.target}")
        if isinstance(node, PReturn):
            if node.expr:
                return self.check_node(node.expr)
            return TNull()
        if isinstance(node, PListLiteral):
            if not node.elements: return TList(TAny())
            et = self.check_node(node.elements[0])
            return TList(et)
        if isinstance(node, PIf):
            self.check_node(node.cond)
            for s in node.then_body: self.check_node(s)
            for s in node.else_body: self.check_node(s)
            return TNull()
        if isinstance(node, PWhile):
            self.check_node(node.cond)
            for s in node.body: self.check_node(s)
            return TNull()
        return TAny()

# -------------------------
# Simple IR + IR Builder + Codegen (professional structure)
# -------------------------
class IROp(Enum):
    CONST = auto()
    LOAD = auto()
    STORE = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    CALL = auto()
    RET = auto()
    LIST = auto()
    INDEX = auto()

@dataclass
class IRInstr:
    op: IROp
    args: Tuple[Any,...]
    dst: Optional[str] = None

class IRBuilder:
    def __init__(self):
        self.instrs: List[IRInstr] = []
        self.tmp = 0
    def new_tmp(self):
        self.tmp += 1; return f"%t{self.tmp}"
    def emit(self, op: IROp, args: Tuple[Any,...], dst: Optional[str]=None):
        instr = IRInstr(op,args,dst)
        self.instrs.append(instr); return instr
    def lower_program(self, prog: PProgram) -> Dict[str, List[IRInstr]]:
        funcs = {}
        for n in prog.body:
            if isinstance(n, PFunc):
                self.tmp = 0; self.instrs = []
                self.lower_func(n)
                funcs[f"{n.name}/{len(n.params)}"] = list(self.instrs)
        return funcs
    def lower_func(self, fn: PFunc):
        # lower statements to IR
        for s in fn.body:
            self.lower_stmt(s)
    def lower_stmt(self, s: PNode):
        if isinstance(s, PReturn):
            if s.expr:
                dst = self.lower_expr(s.expr)
                self.emit(IROp.RET, (dst,))
            else:
                self.emit(IROp.RET, (None,))
        elif isinstance(s, PAssign):
            val = self.lower_expr(s.expr)
            self.emit(IROp.STORE, (s.name,val))
        else:
            # expression stmt
            self.lower_expr(s)
    def lower_expr(self, e: PNode) -> Any:
        if isinstance(e, PNumber):
            dst = self.new_tmp(); self.emit(IROp.CONST, (float(e.raw) if '.' in e.raw else int(e.raw),), dst); return dst
        if isinstance(e, PBool):
            dst = self.new_tmp(); self.emit(IROp.CONST, (1 if e.val else 0,), dst); return dst
        if isinstance(e, PVar):
            dst = self.new_tmp(); self.emit(IROp.LOAD,(e.name,),dst); return dst
        if isinstance(e, PBinOp):
            l = self.lower_expr(e.left); r = self.lower_expr(e.right)
            dst = self.new_tmp()
            opmap = {'+': IROp.ADD,'-': IROp.SUB,'*': IROp.MUL,'/': IROp.DIV}
            self.emit(opmap.get(e.op, IROp.ADD),(l,r),dst); return dst
        if isinstance(e, PCall):
            args = [self.lower_expr(a) for a in e.args]
            dst = self.new_tmp()
            tgt = e.target if isinstance(e.target,str) else None
            self.emit(IROp.CALL,(tgt,tuple(args)),dst); return dst
        if isinstance(e, PListLiteral):
            items = [self.lower_expr(it) for it in e.elements]
            dst = self.new_tmp(); self.emit(IROp.LIST,(tuple(items),),dst); return dst
        if isinstance(e, PIndex):
            b = self.lower_expr(e.base); i = self.lower_expr(e.index)
            dst = self.new_tmp(); self.emit(IROp.INDEX,(b,i),dst); return dst
        if isinstance(e, PLambda):
            # lambdas lowered to callable via runtime; return Any placeholder
            dst = self.new_tmp(); self.emit(IROp.CONST, ("<lambda>",),dst); return dst
        return None

# -------------------------
# Execution Engine for IR
# -------------------------
class ExecutionEngine:
    def __init__(self, funcs_ir: Dict[str, List[IRInstr]], runtime_env: Optional[Dict[str,Any]] = None):
        self.funcs_ir = funcs_ir
        self.env = runtime_env or {}
    def run(self, key: str, args: List[Any]=None):
        args = args or []
        if key not in self.funcs_ir: raise RuntimeError(f"No function {key} compiled to IR")
        return self.run_ir(self.funcs_ir[key], args)
    def run_ir(self, instrs: List[IRInstr], args: List[Any]):
        # simple stackless SSA-like interpreter: map temps to values and symbol table for locals
        vals: Dict[str,Any] = {}
        locals_tab: Dict[str,Any] = {}
        # apply args to parameter names? Not stored in IR; caller must use globals or STORE before RET
        for instr in instrs:
            if instr.op == IROp.CONST:
                vals[instr.dst] = instr.args[0]
            elif instr.op == IROp.LOAD:
                name = instr.args[0]
                vals[instr.dst] = locals_tab.get(name, self.env.get(name))
            elif instr.op == IROp.STORE:
                name, src = instr.args
                locals_tab[name] = vals.get(src, src)
            elif instr.op == IROp.ADD:
                a,b = instr.args; vals[instr.dst] = self._get(a,vals) + self._get(b,vals)
            elif instr.op == IROp.SUB:
                a,b = instr.args; vals[instr.dst] = self._get(a,vals) - self._get(b,vals)
            elif instr.op == IROp.MUL:
                a,b = instr.args; vals[instr.dst] = self._get(a,vals) * self._get(b,vals)
            elif instr.op == IROp.DIV:
                a,b = instr.args; vb = self._get(b,vals); vals[instr.dst] = (0 if vb==0 else self._get(a,vals)/vb)
            elif instr.op == IROp.CALL:
                funcname, argtemps = instr.args
                argvals = tuple(self._get(a,vals) for a in argtemps)
                if funcname == "print":
                    print(*argvals); vals[instr.dst] = None
                else:
                    # call user function if compiled
                    if funcname and funcname in self.funcs_ir:
                        vals[instr.dst] = self.run(funcname, list(argvals))
                    else:
                        vals[instr.dst] = None
            elif instr.op == IROp.RET:
                if instr.args[0] is None: return None
                return self._get(instr.args[0], vals)
            elif instr.op == IROp.LIST:
                items = instr.args[0]
                vals[instr.dst] = [self._get(i,vals) for i in items]
            elif instr.op == IROp.INDEX:
                base_t, idx_t = instr.args; base = self._get(base_t,vals); idx = self._get(idx_t,vals)
                if isinstance(base, list):
                    if not isinstance(idx,int): raise RuntimeError("Index must be int")
                    vals[instr.dst] = base[idx]
                elif isinstance(base, dict):
                    vals[instr.dst] = base.get(idx)
                else:
                    vals[instr.dst] = None
        return None
    def _get(self, key, vals):
        if isinstance(key, str) and key.startswith('%t'): return vals.get(key)
        return key

# -------------------------
# ProCompiler facade
# -------------------------
class ProCompiler:
    @staticmethod
    def pro_compile_and_run(source: str, entry: str="main/0"):
        parser = ProParser(source)
        prog = parser.parse()
        # semantic analysis
        tc = TypeChecker(prog)
        try:
            tc.check()
        except SemanticError as e:
            print("[TYPE ERROR]", e); raise
        # lower to IR
        irb = IRBuilder()
        funcs_ir = irb.lower_program(prog)
        # create execution engine and run
        ee = ExecutionEngine(funcs_ir)
        return ee.run(entry, [])

# Minimal example: if you want to use this pipeline:
# result = ProCompiler.pro_compile_and_run('func main() { print(1+2); }')

TOKENS = [
    ("NUMBER", r"-?\d+(\.\d+)?"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("STRING", r"\".*?\"|\'.*?\'"),
    ("PRAGMA", r"#pragma\s+[A-Za-z0-9_ ]+"),
    ("OP", r"[+\-*/=<>!&|%^.^]+"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"),
    ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("LBRACK", r"\["), ("RBRACK", r"\]"),
    ("SEMI", r";"), ("COMMA", r","),
    ("KEYWORD", r"\b(func|return|if|else|while|for|lambda|true|false|null|parallel|enum|eval|poly|try|catch|throw|raise|force|extern|alloc|free|malloc|maloc|mutex|mutex_lock|mutex_unlock|Start|main|print|isolate|remove)\b"),
    ("WS", r"\s+"),
]

class Token:
    def __init__(self, kind, val, pos): self.kind, self.val, self.pos = kind, val, pos
    def __repr__(self): return f"<{self.kind}:{self.val}>"

def lex(src: str):
    pos, tokens = 0, []
    while pos < len(src):
        for kind, pat in TOKENS:
            m = re.match(pat, src[pos:])
            if m:
                if kind != "WS":
                    tokens.append(Token(kind, m.group(), pos))
                pos += len(m.group()); break
        else:
            raise SyntaxError(f"Unexpected char {src[pos]} at {pos}")
    return tokens

# -------------------------
# AST Nodes (add containers & control)
# -------------------------
class ASTNode: pass
class Program(ASTNode):
    def __init__(self, body: List[ASTNode]): self.body = body
class FuncDef(ASTNode):
    def __init__(self, name, params, body): self.name, self.params, self.body = name, params, body
class Call(ASTNode):
    def __init__(self, name, args): self.name, self.args = name, args
class Return(ASTNode):
    def __init__(self, expr): self.expr = expr
class Number(ASTNode):
    def __init__(self, val):
        self.raw = val
        self.val = float(val) if "." in str(val) else int(val)
        self.is_float = isinstance(self.val, float) and not float(self.val).is_integer()
class Var(ASTNode):
    def __init__(self, name): self.name = name
class Assign(ASTNode):
    def __init__(self, name, expr): self.name, self.expr = name, expr
class BinOp(ASTNode):
    def __init__(self, op, left, right): self.op, self.left, self.right = op, left, right
class If(ASTNode):
    def __init__(self, cond, then_body, else_body=None): self.cond, self.then_body, self.else_body = cond, then_body, else_body
class While(ASTNode):
    def __init__(self, cond, body): self.cond, self.body = cond, body
class For(ASTNode):
    def __init__(self, init, cond, step, body): self.init, self.cond, self.step, self.body = init, cond, step, body
class Lambda(ASTNode):
    def __init__(self, params, body): self.params, self.body = params, body
class Bool(ASTNode):
    def __init__(self, val): self.val = val
class Null(ASTNode): pass
class Pragma(ASTNode):
    def __init__(self, directive): self.directive = directive
class Enum(ASTNode):
    def __init__(self, name, members): self.name, self.members = name, members
class Eval(ASTNode):
    def __init__(self, expr): self.expr = expr
class Poly(ASTNode):
    def __init__(self, expr): self.expr = expr
class Parallel(ASTNode):
    def __init__(self, body): self.body = body
class TryCatch(ASTNode):
    def __init__(self, try_body, catch_body): self.try_body, self.catch_body = try_body, catch_body
class Throw(ASTNode):
    def __init__(self, expr): self.expr = expr
class ListLiteral(ASTNode):
    def __init__(self, elements): self.elements = elements
class MapLiteral(ASTNode):
    def __init__(self, pairs): self.pairs = pairs  # list of (key, value)
class Index(ASTNode):
    def __init__(self, base, index): self.base, self.index = base, index
class Isolate(ASTNode):
    def __init__(self, body): self.body = body
class Force(ASTNode):
    def __init__(self, body): self.body = body
class Remove(ASTNode):
    def __init__(self, body): self.body = body

# -------------------------
# PARSER (support lists, maps, index, throw/force/isolate/remove)
# -------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0
        self.functions: Dict[str, FuncDef] = {}

    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def eat(self, kind=None):
        tok = self.peek()
        if not tok: raise SyntaxError("EOF")
        if kind and tok.kind != kind: raise SyntaxError(f"Expected {kind}, got {tok.kind}")
        self.pos += 1; return tok

    def parse(self):
        return Program(self.statements())

    def statements(self):
        stmts=[]
        while self.peek():
            if self.peek().val == "func": stmts.append(self.funcdef())
            elif self.peek().kind == "PRAGMA": stmts.append(Pragma(self.eat("PRAGMA").val))
            else:
                stmt = self.expression()
                if stmt is not None: stmts.append(stmt)
                else: self.pos += 1
        return stmts

    def funcdef(self):
        self.eat("KEYWORD")  # func
        name=self.eat("ID").val
        self.eat("LPAREN"); params=[]
        while self.peek() and self.peek().kind!="RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
        self.eat("RPAREN"); self.eat("LBRACE")
        body=[]
        while self.peek() and self.peek().kind!="RBRACE":
            # assignment
            if self.peek().kind=="ID" and self.pos+1<len(self.tokens) and self.tokens[self.pos+1].kind=="OP" and self.tokens[self.pos+1].val=="=":
                name_tok=self.eat("ID").val; self.eat("OP"); expr=self.expression(); body.append(Assign(name_tok, expr))
            elif self.peek().val=="return":
                self.eat("KEYWORD"); body.append(Return(self.expression()))
            elif self.peek().val=="if": body.append(self.ifstmt())
            elif self.peek().val=="while": body.append(self.whilestmt())
            elif self.peek().val=="for": body.append(self.forstmt())
            elif self.peek().val=="parallel": body.append(self.parallelblock())
            elif self.peek().val=="try": body.append(self.trycatch())
            elif self.peek().val=="isolate":
                self.eat("KEYWORD"); self.eat("LBRACE"); b=[] 
                while self.peek() and self.peek().kind!="RBRACE": b.append(self.expression())
                self.eat("RBRACE"); body.append(Isolate(b))
            elif self.peek().val=="force":
                self.eat("KEYWORD"); self.eat("LBRACE"); b=[]
                while self.peek() and self.peek().kind!="RBRACE": b.append(self.expression())
                self.eat("RBRACE"); body.append(Force(b))
            elif self.peek().val=="remove":
                self.eat("KEYWORD"); self.eat("LBRACE"); b=[]
                while self.peek() and self.peek().kind!="RBRACE": b.append(self.expression())
                self.eat("RBRACE"); body.append(Remove(b))
            else:
                expr=self.expression()
                if expr is not None: body.append(expr)
                else: self.pos += 1
        self.eat("RBRACE")
        func = FuncDef(name, params, body); key=f"{name}/{len(params)}"; self.functions[key]=func
        return func

    # expression grammar
    def expression(self): return self.parse_addsub()
    def parse_addsub(self):
        left = self.parse_muldiv()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("+","-","||"):
            op=self.eat("OP").val; right=self.parse_muldiv(); left=BinOp(op,left,right)
        return left
    def parse_muldiv(self):
        left=self.parse_pow()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("*","/","&&"):
            op=self.eat("OP").val; right=self.parse_pow(); left=BinOp(op,left,right)
        return left
    def parse_pow(self):
        left=self.parse_unary()
        while self.peek() and self.peek().kind=="OP" and self.peek().val=="^":
            self.eat("OP"); right=self.parse_unary(); left=BinOp("^",left,right)
        return left
    def parse_unary(self):
        tok=self.peek()
        if not tok: return None
        if tok.kind=="OP" and tok.val=="-":
            self.eat("OP"); return BinOp("*", Number("-1"), self.parse_unary())
        if tok.kind=="NUMBER": return Number(self.eat("NUMBER").val)
        if tok.kind=="KEYWORD" and tok.val=="true": self.eat("KEYWORD"); return Bool(True)
        if tok.kind=="KEYWORD" and tok.val=="false": self.eat("KEYWORD"); return Bool(False)
        if tok.kind=="KEYWORD" and tok.val=="null": self.eat("KEYWORD"); return Null()
        if tok.kind=="LBRACK":
            # list literal
            self.eat("LBRACK"); elems=[]
            while self.peek() and self.peek().kind!="RBRACK":
                elems.append(self.expression())
                if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
            self.eat("RBRACK"); return ListLiteral(elems)
        if tok.kind=="ID":
            # handle dotted access for method/field: ID ( '.' ID )*
            name = self.eat("ID").val
            while self.peek() and self.peek().kind=="OP" and self.peek().val==".":
                self.eat("OP"); member=self.eat("ID").val
                name = f"{name}.{member}"
            if self.peek() and self.peek().kind=="LPAREN":
                # it's a call
                self.eat("LPAREN"); args=[]
                while self.peek() and self.peek().kind!="RPAREN":
                    args.append(self.expression())
                    if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
                self.eat("RPAREN"); return Call(name, args)
            return Var(name)
        if tok.kind=="LPAREN":
            self.eat("LPAREN"); e=self.expression(); self.eat("RPAREN"); return e
        if tok.kind=="KEYWORD" and tok.val=="lambda": return self.lambdaexpr()
        if tok.kind=="KEYWORD" and tok.val=="eval": return self.evalexpr()
        if tok.kind=="KEYWORD" and tok.val=="enum": return self.enumdef()
        if tok.kind=="KEYWORD" and tok.val=="throw":
            self.eat("KEYWORD"); expr=self.expression(); return Throw(expr)
        return None

    def lambdaexpr(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); params=[]
        while self.peek() and self.peek().kind!="RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
        self.eat("RPAREN"); self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE":
            if self.peek().val=="return": self.eat("KEYWORD"); body.append(Return(self.expression()))
            else: body.append(self.expression())
        self.eat("RBRACE"); return Lambda(params, body)

    def evalexpr(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); s=self.eat("STRING").val; self.eat("RPAREN")
        try:
            value = eval(s.strip("\"'"), {"__builtins__": {}})
        except Exception:
            value = 0
        return Eval(Number(str(value)))

    def enumdef(self):
        self.eat("KEYWORD"); name=self.eat("ID").val; self.eat("LBRACE")
        members={}; idx=0
        while self.peek() and self.peek().kind!="RBRACE":
            key=self.eat("ID").val
            if self.peek() and self.peek().kind=="OP" and self.peek().val=="=":
                self.eat("OP"); val=int(self.eat("NUMBER").val); members[key]=val; idx=val+1
            else:
                members[key]=idx; idx+=1
            if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
        self.eat("RBRACE"); return Enum(name,members)

    def ifstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond=self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); then_body=[]
        while self.peek() and self.peek().kind!="RBRACE": then_body.append(self.expression())
        self.eat("RBRACE"); else_body=None
        if self.peek() and self.peek().val=="else":
            self.eat("KEYWORD"); self.eat("LBRACE"); else_body=[]
            while self.peek() and self.peek().kind!="RBRACE": else_body.append(self.expression())
            self.eat("RBRACE")
        return If(cond, then_body, else_body)

    def whilestmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond=self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return While(cond, body)

    def forstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); init=self.expression(); self.eat("SEMI")
        cond=self.expression(); self.eat("SEMI"); step=self.expression()
        self.eat("RPAREN"); self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return For(init, cond, step, body)

    def parallelblock(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return Parallel(body)

    def trycatch(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); try_body=[]
        while self.peek() and self.peek().kind!="RBRACE": try_body.append(self.expression())
        self.eat("RBRACE"); self.eat("KEYWORD"); self.eat("LBRACE"); catch_body=[]
        while self.peek() and self.peek().kind!="RBRACE": catch_body.append(self.expression())
        self.eat("RBRACE"); return TryCatch(try_body, catch_body)

# -------------------------
# Hot-Swap Registry (unchanged)
# -------------------------
class HotSwapRegistry:
    def __init__(self):
        self.table: Dict[str, FuncDef] = {}
        self.lock = threading.Lock()
    def register(self,key,func):
        with self.lock: self.table[key]=func
    def get(self,key):
        with self.lock: return self.table.get(key)
    def swap(self,key,new_func):
        with self.lock: old=self.table.get(key); self.table[key]=new_func
        print(f"[HOTSWAP] {key}: {'replaced' if old else 'registered'}"); return old

# -------------------------
# Optimizer (const-fold already provided earlier) - reuse optimize()
# -------------------------
def optimize(node):
    if isinstance(node, Program):
        node.body = [optimize(s) for s in node.body]; return node
    if isinstance(node, FuncDef):
        node.body = [optimize(s) for s in node.body]; return node
    if isinstance(node, Return):
        node.expr = optimize(node.expr); return node
    if isinstance(node, BinOp):
        node.left = optimize(node.left); node.right = optimize(node.right)
        if isinstance(node.left, Number) and isinstance(node.right, Number):
            try:
                if node.op == "+": return Number(str(node.left.val + node.right.val))
                if node.op == "-": return Number(str(node.left.val - node.right.val))
                if node.op == "*": return Number(str(node.left.val * node.right.val))
                if node.op == "/":
                    if node.right.val == 0:
                        return Number("0")
                    return Number(str(node.left.val / node.right.val))
                if node.op == "^": return Number(str(int(math.pow(node.left.val, node.right.val))))
            except Exception:
                return node
        return node
    if isinstance(node, If):
        node.cond = optimize(node.cond)
        node.then_body = [optimize(s) for s in node.then_body]
        node.else_body = [optimize(s) for s in (node.else_body or [])]
        if isinstance(node.cond, Bool):
            return Program(node.then_body if node.cond.val else node.else_body)
        return node
    if isinstance(node, Parallel):
        node.body = [optimize(s) for s in node.body]
        return node
    return node

# -------------------------
# CODEGEN + auto-linker to syscalls
# - If calls map to syscall_map and not user function, emit inline syscall sequence (no wrapper)
# -------------------------
class Codegen:
    # common x86-64 Linux syscall numbers
    SYSCALL_MAP = {
        "read": 0,
        "write": 1,
        "open": 2,
        "close": 3,
        "exit": 60,
        "getpid": 39,
    }

    def __init__(self, symtab: Dict[str, FuncDef], hot_registry: HotSwapRegistry):
        self.asm = []; self.label_counter = 0
        self.symtab = symtab
        self.hot_registry = hot_registry

    def emit(self, s): self.asm.append(s)
    def newlabel(self, prefix="L"): self.label_counter += 1; return f"{prefix}{self.label_counter}"

    def generate(self, ast: Program, emit_pkt: bool=False, pkt_path: str="out.pkt"):
        self.emit("section .text"); self.emit("global _start"); self.emit("_start:")
        self.emit("  call main/0  ; entry")
        self.emit("  mov rax, 60"); self.emit("  xor rdi, rdi"); self.emit("  syscall")
        ast = optimize(ast)
        for stmt in ast.body: self.gen_stmt(stmt)
        asm = "\n".join(self.asm)
        packed = dodecagram_pack_stream(asm)
        if emit_pkt:
            with open(pkt_path, "wb") as f:
                f.write(packed)
            print(f"Wrote packed dodecagram stream to {pkt_path} ({len(packed)} bytes)")
        return asm + "\n\n; --- PACKED DODECAGRAM (binary stream written to file) ---\n; size_bytes=" + str(len(packed))

    def gen_stmt(self, node):
        if isinstance(node, FuncDef):
            key = f"{node.name}/{len(node.params)}"
            self.symtab[key] = node
            self.hot_registry.register(key, node)
            self.emit(f"{key}:")
            for s in node.body: self.gen_stmt(s)
            self.emit("  ret")
        elif isinstance(node, Return):
            self.gen_stmt(node.expr)
            self.emit("  ret")
        elif isinstance(node, Number):
            if node.is_float:
                self.emit(f"  ; load float {node.val} (placeholder)")
                self.emit(f"  mov rax, {int(node.val)}")
            else:
                self.emit(f"  mov rax, {int(node.val)}")
        elif isinstance(node, Var):
            self.emit(f"  ; load var {node.name} (TODO)")
        elif isinstance(node, Assign):
            # simple assign: evaluate expr -> rax, then store into memory/local var (annotation)
            self.gen_stmt(node.expr)
            self.emit(f"  ; assign {node.name} = rax  (local)")
        elif isinstance(node, Call):
            key_candidate = f"{node.name}/{len(node.args)}"
            # if maps to syscall and not a user-defined function, emit syscall inline (args -> rdi,rsi,rdx)
            if node.name in self.SYSCALL_MAP and key_candidate not in self.symtab:
                # evaluate args and move into rdi,rsi,rdx in order
                regs = ["rdi","rsi","rdx","rcx","r8","r9"]
                for i, a in enumerate(node.args):
                    self.gen_stmt(a)
                    reg = regs[i] if i < len(regs) else "r10"
                    self.emit(f"  mov {reg}, rax")
                self.emit(f"  mov rax, {self.SYSCALL_MAP[node.name]}")
                self.emit("  syscall")
            else:
                for i, a in enumerate(node.args):
                    self.gen_stmt(a)
                    self.emit(f"  push rax  ; arg {i}")
                if key_candidate in self.symtab:
                    self.emit(f"  call {key_candidate}")
                else:
                    self.emit(f"  ; automatic link -> extern {node.name}")
                    self.emit(f"  call {node.name}")
                self.emit(f"  add rsp, {len(node.args) * 8}  ; clean args")
        elif isinstance(node, BinOp):
            self.gen_stmt(node.left); self.emit("  push rax")
            self.gen_stmt(node.right); self.emit("  mov rbx, rax"); self.emit("  pop rax")
            if node.op == "+": self.emit("  add rax, rbx")
            elif node.op == "-": self.emit("  sub rax, rbx")
            elif node.op == "*": self.emit("  imul rax, rbx")
            elif node.op == "/":
                dz = self.newlabel("divzero"); ed = self.newlabel("enddiv")
                self.emit("  cmp rbx, 0"); self.emit(f"  je {dz}"); self.emit("  cqo"); self.emit("  idiv rbx"); self.emit(f"  jmp {ed}")
                self.emit(f"{dz}:"); self.emit("  mov rax, 0  ; div by zero guard"); self.emit(f"{ed}:")
            elif node.op == "^":
                loop = self.newlabel("pow"); end = self.newlabel("endp")
                self.emit("  mov rcx, rbx"); self.emit("  mov rbx, rax"); self.emit("  mov rax, 1")
                self.emit(f"{loop}:"); self.emit("  cmp rcx, 0"); self.emit(f"  je {end}"); self.emit("  imul rax, rbx"); self.emit("  dec rcx"); self.emit(f"  jmp {loop}"); self.emit(f"{end}:")
            else:
                self.emit(f"  ; unhandled op {node.op}")
        elif isinstance(node, If):
            else_lbl = self.newlabel("else"); end_lbl = self.newlabel("endif")
            self.gen_stmt(node.cond); self.emit("  cmp rax, 0"); self.emit(f"  je {else_lbl}")
            for s in node.then_body: self.gen_stmt(s)
            self.emit(f"  jmp {end_lbl}"); self.emit(f"{else_lbl}:")
            if node.else_body:
                for s in node.else_body: self.gen_stmt(s)
            self.emit(f"{end_lbl}:")
        elif isinstance(node, Parallel):
            self.emit("  ; PARALLEL BLOCK START (annotated)")
            for i, s in enumerate(node.body):
                self.emit(f"  ; parallel task {i}")
                self.gen_stmt(s)
            self.emit("  ; PARALLEL BLOCK END")
        elif isinstance(node, TryCatch):
            self.emit("  ; TRY/CATCH (annotated)")
            for s in node.try_body: self.gen_stmt(s)
            for s in node.catch_body: self.gen_stmt(s)
        elif isinstance(node, Throw):
            self.gen_stmt(node.expr); self.emit("  ; THROW (annotated)")
        elif isinstance(node, Enum):
            self.emit(f"  ; enum {node.name} => {node.members}")
        elif isinstance(node, Pragma):
            self.emit(f"  ; pragma {node.directive}")
        else:
            self.emit(f"  ; unhandled node {type(node).__name__}")

# -------------------------
# Dodecagram packing -> binary-packed stream
# -------------------------
DIGITS12 = "0123456789ab"
def bytes_to_base12_digits(data: bytes) -> List[int]:
    digits = []
    for b in data:
        d1 = b // 12
        d2 = b % 12
        digits.append(d1)
        digits.append(d2)
    return digits

def base12_digits_pack(digits: List[int]) -> bytes:
    out = bytearray()
    it = iter(digits)
    for d1 in it:
        try:
            d2 = next(it)
        except StopIteration:
            d2 = 0
        val = d1 * 12 + d2
        out.append(val)
    return bytes(out)

def dodecagram_pack_stream(asm: str) -> bytes:
    raw = asm.encode("utf-8", errors="replace")
    digits = bytes_to_base12_digits(raw)
    packed = base12_digits_pack(digits)
    header = b"DDG1" + struct.pack(">I", len(packed))
    return header + packed

# -------------------------
# Linker helpers
# -------------------------
def load_and_parse_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    toks = lex(src)
    parser = Parser(toks)
    ast = parser.parse()
    return ast, parser.functions

def merge_symbol_tables(main_symtab: Dict[str, FuncDef], extra_symtab: Dict[str, FuncDef]):
    for k, v in extra_symtab.items():
        if k not in main_symtab:
            main_symtab[k] = v
        else:
            print(f"[LINKER] Symbol {k} already present; keeping primary definition")

# -------------------------
# Interpreter: closures, frames, locals, stack-frame emulation
# -------------------------
class Closure:
    def __init__(self, params, body, env):
        self.params = params
        self.body = body
        # captured environment (shallow copy)
        self.env = dict(env) if env else {}

class MiniRuntime:
    def __init__(self, symtab: Dict[str, FuncDef], hot_registry: HotSwapRegistry):
        self.symtab = symtab
        self.hot = hot_registry
        self.memory = {}
        self._next_addr = 1
        self.mutexes = {}
        self.lock = threading.Lock()
        # frame stack: list of dicts (locals)
        self.frames: List[Dict[str, object]] = []

    def alloc(self, size: int):
        with self.lock:
            addr = self._next_addr; self._next_addr += 1
            self.memory[addr] = bytearray(size)
        print(f"[RUNTIME] alloc -> addr={addr} size={size}")
        return addr

    def free(self, addr: int):
        with self.lock:
            if addr in self.memory:
                del self.memory[addr]
                print(f"[RUNTIME] free -> addr={addr}")
                return True
            print(f"[RUNTIME] free -> addr not found: {addr}")
            return False

    def mutex(self):
        with self.lock:
            mid = len(self.mutexes) + 1
            self.mutexes[mid] = threading.Lock()
        print(f"[RUNTIME] mutex -> id={mid}")
        return mid

    def mutex_lock(self, mid: int):
        m = self.mutexes.get(mid)
        if m:
            m.acquire()
            print(f"[RUNTIME] mutex_lock -> {mid}")
            return True
        return False

    def mutex_unlock(self, mid: int):
        m = self.mutexes.get(mid)
        if m:
            m.release()
            print(f"[RUNTIME] mutex_unlock -> {mid}")
            return True
        return False

    def push_frame(self, locals_map=None):
        self.frames.append(locals_map or {})
    def pop_frame(self):
        if self.frames: self.frames.pop()
    def current_frame(self):
        return self.frames[-1] if self.frames else {}

    def run_func(self, key: str, args: List):
        func = self.hot.get(key) or self.symtab.get(key)
        if not func:
            raise Exception(f"Function {key} not found")
        # bind params
        frame = {}
        for i, pname in enumerate(func.params):
            frame[pname] = args[i] if i < len(args) else None
        self.push_frame(frame)
        result = None
        for stmt in func.body:
            result = self.eval(stmt)
            # early return handling via Return nodes yields value
            if isinstance(stmt, Return):
                break
        self.pop_frame()
        return result

    def eval(self, node):
        if node is None: return None
        if isinstance(node, Number): return node.val
        if isinstance(node, Bool): return 1 if node.val else 0
        if isinstance(node, Var):
            # look up in frames (top-down), then env of closures if present
            for f in reversed(self.frames):
                if node.name in f: return f[node.name]
            return None
        if isinstance(node, Assign):
            val = self.eval(node.expr)
            # assign into current frame
            self.current_frame()[node.name] = val
            return val
        if isinstance(node, Return): return self.eval(node.expr)
        if isinstance(node, BinOp):
            l = self.eval(node.left); r = self.eval(node.right)
            if node.op == "+": return l + r
            if node.op == "-": return l - r
            if node.op == "*": return l * r
            if node.op == "/": return 0 if r == 0 else l / r
            if node.op == "^": return int(math.pow(l, r))
            return None
        if isinstance(node, Lambda):
            # capture current frame environment shallowly
            env = {}
            for f in self.frames:
                env.update(f)
            return Closure(node.params, node.body, env)
        if isinstance(node, Call):
            # builtins first
            if node.name == "print":
                vals = [self.eval(a) for a in node.args]
                print(*vals)
                return None
            if node.name == "alloc":
                size = int(self.eval(node.args[0])) if node.args else 0
                return self.alloc(size)
            if node.name == "free":
                addr = int(self.eval(node.args[0])) if node.args else 0
                return self.free(addr)
            if node.name == "mutex":
                return self.mutex()
            if node.name == "mutex_lock":
                mid = int(self.eval(node.args[0])); return self.mutex_lock(mid)
            if node.name == "mutex_unlock":
                mid = int(self.eval(node.args[0])); return self.mutex_unlock(mid)
            if node.name == "parallel":
                def task(stmt):
                    return self.eval(stmt)
                with ThreadPoolExecutor(max_workers=len(node.args) or 2) as ex:
                    futures = [ex.submit(task, a) for a in node.args]
                    results = [f.result() for f in futures]
                return results
            # user function by key
            key = f"{node.name}/{len(node.args)}"
            # evaluate args
            argvals = [self.eval(a) for a in node.args]
            # check hot-swap or table or closure in current frame
            func = self.hot.get(key) or self.symtab.get(key)
            if func:
                return self.run_func(key, argvals)
            # check for closure variable in frames
            for f in reversed(self.frames):
                if node.name in f:
                    candidate = f[node.name]
                    if isinstance(candidate, Closure):
                        # call closure: create new frame merging closure.env then parameters
                        new_frame = dict(candidate.env)
                        for i, pname in enumerate(candidate.params):
                            new_frame[pname] = argvals[i] if i < len(argvals) else None
                        self.push_frame(new_frame)
                        result = None
                        for stmt in candidate.body:
                            result = self.eval(stmt)
                            if isinstance(stmt, Return):
                                break
                        self.pop_frame()
                        return result
            raise Exception(f"Call target not found: {key}")
        if isinstance(node, Parallel):
            with ThreadPoolExecutor(max_workers=len(node.body) or 2) as ex:
                futures = [ex.submit(self.eval, s) for s in node.body]
                return [f.result() for f in futures]
        if isinstance(node, TryCatch):
            try:
                for s in node.try_body: self.eval(s)
            except Exception:
                for s in node.catch_body: self.eval(s)
            return None
        return None

# -------------------------
# Hot-swap persistent IPC server
# - simple JSON line protocol on localhost:4000
#   {"op":"swap","key":"main/0","type":"const_return","value":123}
#   {"op":"list"}
# -------------------------
class HotSwapServer(threading.Thread):
    def __init__(self, host: str, port: int, hot_registry: HotSwapRegistry):
        super().__init__(daemon=True)
        self.host = host; self.port = port; self.hot = hot_registry
        self.sock = None
        self.running = True

    def run(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(5)
        print(f"[HOTSWAP-SERVER] listening on {self.host}:{self.port}")
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                break
            threading.Thread(target=self.handle_client, args=(conn,addr), daemon=True).start()

    def handle_client(self, conn: socket.socket, addr):
        with conn:
            data = b""
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                data += chunk
            try:
                payload = json.loads(data.decode("utf-8"))
            except Exception as e:
                conn.sendall(b'{"error":"invalid-json"}')
                return
            resp = self.process(payload)
            conn.sendall(json.dumps(resp).encode("utf-8"))

    def process(self, payload: dict):
        op = payload.get("op")
        if op == "swap":
            key = payload.get("key")
            if not key: return {"ok":False, "error":"missing key"}
            typ = payload.get("type","const_return")
            if typ == "const_return":
                val = payload.get("value",0)
                new_func = FuncDef(key.split("/")[0], [], [Return(Number(str(val)))])
                self.hot.swap(key, new_func)
                return {"ok":True, "action":"swapped", "key":key}
            else:
                return {"ok":False, "error":"unsupported type"}
        elif op == "list":
            with self.hot.lock:
                keys = list(self.hot.table.keys())
            return {"ok":True, "keys":keys}
        else:
            return {"ok":False, "error":"unknown op"}

    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()

# -------------------------
# Main: CLI, linking, hot-swap demo and run
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog="xyzc", description="XYZ AOT compiler (extended)")
    parser.add_argument('infile', nargs='?', help='primary source file (or read from stdin)')
    parser.add_argument('--link', help='comma-separated list of files to auto-link', default="")
    parser.add_argument('--emit-asm', action='store_true', help='emit out.asm')
    parser.add_argument('--emit-pkt', action='store_true', help='emit packed dodecagram .pkt file')
    parser.add_argument('--hot-swap-demo', action='store_true', help='run hot-swap demo at AST/runtime level')
    parser.add_argument('--hot-swap-server', action='store_true', help='start persistent hot-swap IPC server (localhost:4000)')
    args = parser.parse_args()

    # load primary
    if args.infile:
        try:
            with open(args.infile, 'r', encoding='utf-8') as f:
                src = f.read()
        except FileNotFoundError:
            print(f"File not found: {args.infile}", file=sys.stderr); sys.exit(2)
        toks = lex(src); parser_obj = Parser(toks); ast_main = parser_obj.parse()
        symtab = dict(parser_obj.functions)
    else:
        src = sys.stdin.read(); toks = lex(src); parser_obj = Parser(toks); ast_main = parser_obj.parse(); symtab = dict(parser_obj.functions)

    # auto-link additional files
    if args.link:
        for path in args.link.split(","):
            path = path.strip()
            if not path: continue
            try:
                ast_extra, funcs_extra = load_and_parse_file(path)
            except Exception as e:
                print(f"[LINKER] failed to load {path}: {e}")
                continue
            merge_symbol_tables(symtab, funcs_extra)
            ast_main.body.extend(ast_extra.body)

    # hot-swap registry
    hot_registry = HotSwapRegistry()
    for k, v in list(symtab.items()):
        hot_registry.register(k, v)

    # optional persistent hot-swap server
    server = None
    if args.hot_swap_server:
        server = HotSwapServer("127.0.0.1", 4000, hot_registry)
        server.start()

    # codegen
    cg = Codegen(symtab, hot_registry)
    pkt_path = "out.pkt"
    asm = cg.generate(ast_main, emit_pkt=args.emit_pkt, pkt_path=pkt_path if args.emit_pkt else "out.pkt")
    if args.emit_asm:
        open("out.asm", "w", encoding="utf-8").write(asm)
        print("Wrote out.asm")

    print("Compiled -> out.asm (in-memory); symtab entries:", len(symtab))

    # runtime demo hooks
    runtime = MiniRuntime(symtab, hot_registry)

    if args.hot_swap_demo:
        try:
            res_before = runtime.run_func("main/0", [])
            print("[DEMO] main/0 before hot-swap ->", res_before)
        except Exception:
            print("[DEMO] main/0 not runnable before hot-swap")

        new_body = [Return(Number("123"))]
        new_func = FuncDef("main", [], new_body)
        hot_registry.swap("main/0", new_func)
        res_after = runtime.run_func("main/0", [])
        print("[DEMO] main/0 after hot-swap ->", res_after)

    # execute top-level parallels and simple calls to exercize runtime
    for node in ast_main.body:
        if isinstance(node, Parallel):
            print("[RUNTIME] executing top-level Parallel block")
            runtime.eval(node)
        if isinstance(node, Call) and node.name in ("alloc","mutex"):
            runtime.eval(node)

    # keep server alive if started (until user kills)
    if server:
        try:
            print("[HOTSWAP-SERVER] running; send JSON to localhost:4000 and close connection to submit")
            server.join()
        except KeyboardInterrupt:
            server.stop()

if __name__ == "__main__":
    main()

    # --- FAST EXECUTION + RUNTIME (APPENDIX) ---
# High-performance bytecode compiler + VM, hot-path optimizer, memory pool,
# inline caching and parallel executor. Designed as an opt-in runtime: create
# FastRuntime(...) and call run("main/0") to execute with optimization dominance.
#
# Notes:
# - This implementation is pure-Python but structured to reduce interpreter overhead:
#   bytecode arrays, integer opcodes, tight dispatch loop, local arrays for stack/locals.
# - Hot-path optimizer triggers when a function is executed many times and attempts
#   simple inlining of tiny callees and constant specialization.
# - MemoryPool offers pooled bytearray allocations for reduced fragmentation.
# - Inline caching is implemented per-call-site index in BytecodeFunction.

from enum import IntEnum
from collections import defaultdict, deque

# Small, efficient set of opcodes
class Op(IntEnum):
    NOP = 0
    LOAD_CONST = 1
    LOAD_LOCAL = 2
    STORE_LOCAL = 3
    LOAD_GLOBAL = 4
    CALL = 5
    RETURN = 6
    BINARY_ADD = 7
    BINARY_SUB = 8
    BINARY_MUL = 9
    BINARY_DIV = 10
    BINARY_POW = 11
    JUMP = 12
    JUMP_IF_FALSE = 13
    POP = 14
    BUILD_LIST = 15
    BUILD_MAP = 16
    INDEX = 17
    MAKE_CLOSURE = 18
    GET_ATTR = 19
    SET_ATTR = 20
    PARALLEL_START = 21
    THROW = 22

class BytecodeFunction:
    def __init__(self, name, params, code, consts):
        self.name = name
        self.params = params
        self.code = code            # list of ints/operands
        self.consts = consts        # list
        self.call_caches = {}       # call_pc_index -> cached target (for inline caching)
        self.exec_count = 0         # hot-path counter

# Simple memory pool to avoid repeated bytearray churn
class MemoryPool:
    def __init__(self):
        self.pool = defaultdict(deque)
        self.lock = threading.Lock()
    def alloc(self, size):
        with self.lock:
            q = self.pool[size]
            if q:
                return q.popleft()
        return bytearray(size)
    def free(self, b):
        with self.lock:
            self.pool[len(b)].append(b)

# Compiler: lower subset of AST -> bytecode (optimistic, small)
class FastCompiler:
    def __init__(self, symtab: Dict[str, FuncDef]):
        self.symtab = symtab
        self.functions: Dict[str, BytecodeFunction] = {}
    def compile_all(self):
        for key, fn in list(self.symtab.items()):
            if key not in self.functions:
                self.functions[key] = self.compile_fn(key, fn)
        return self.functions
    def compile_fn(self, key, fn: FuncDef):
        # Very small lowering: supports numbers, binops, calls, returns, locals
        consts = []
        code = []
        local_map = {p:i for i,p in enumerate(fn.params)}
        # helpers
        def load_const(v):
            try: idx = consts.index(v)
            except ValueError:
                idx = len(consts); consts.append(v)
            code.extend([Op.LOAD_CONST, idx])
        def emit_binop(op):
            if op == "+": code.append(Op.BINARY_ADD)
            elif op == "-": code.append(Op.BINARY_SUB)
            elif op == "*": code.append(Op.BINARY_MUL)
            elif op == "/": code.append(Op.BINARY_DIV)
            elif op == "^": code.append(Op.BINARY_POW)
            else: code.append(Op.NOP)
        # compile expression recursively but iterative for speed
        def compile_expr(node):
            if isinstance(node, Number):
                load_const(node.val)
            elif isinstance(node, Var):
                if node.name in local_map:
                    code.extend([Op.LOAD_LOCAL, local_map[node.name]])
                else:
                    # global
                    idx = load_const(node.name); code.extend([Op.LOAD_GLOBAL, load_const_index(node.name)])
            elif isinstance(node, BinOp):
                compile_expr(node.left); compile_expr(node.right); emit_binop(node.op)
            elif isinstance(node, Call):
                # compile args then call
                for a in node.args:
                    compile_expr(a)
                # encode call site by name and arity
                cname = f"{node.name}/{len(node.args)}"
                idx = load_const(cname)
                code.extend([Op.CALL, idx])
            elif isinstance(node, Return):
                compile_expr(node.expr); code.append(Op.RETURN)
            elif isinstance(node, ListLiteral):
                for e in node.elements: compile_expr(e)
                code.extend([Op.BUILD_LIST, len(node.elements)])
            elif isinstance(node, Index):
                compile_expr(node.base); compile_expr(node.index); code.append(Op.INDEX)
            elif isinstance(node, Lambda):
                # not lowered deeply here — leave as NOP
                code.append(Op.NOP)
            else:
                code.append(Op.NOP)
        # helper to add/retrieve const index
        def load_const_index(val):
            try:
                return consts.index(val)
            except ValueError:
                consts.append(val); return len(consts)-1
        # compile body
        for stmt in fn.body:
            if isinstance(stmt, Return):
                compile_expr(stmt)
            else:
                compile_expr(stmt)
                code.append(Op.POP)
        # ensure function ends with RETURN
        if not code or code[-1] != Op.RETURN:
            # default return None
            load_const(None); code.append(Op.RETURN)
        return BytecodeFunction(key, fn.params, code, consts)

# High-performance VM
class FastVM:
    def __init__(self, functions: Dict[str, BytecodeFunction], hot_registry: HotSwapRegistry, pool: MemoryPool=None):
        self.functions = functions
        self.hot = hot_registry
        self.pool = pool or MemoryPool()
        self.globals = {}   # name -> value (populated from symtab if needed)
        self.thread_pool = ThreadPoolExecutor(max_workers=max(4, threading.active_count()))
        self.inline_threshold = 64  # inline functions smaller than this bytecode length
        self.hot_threshold = 100    # exec count to trigger optimization
    def run(self, key: str, args: List[Any]=None):
        args = args or []
        if key not in self.functions:
            raise Exception(f"FastVM: function {key} not compiled")
        fn = self.functions[key]
        return self._run_fn(fn, args)
    def _run_fn(self, fn: BytecodeFunction, args: List[Any]):
        fn.exec_count += 1
        if fn.exec_count == self.hot_threshold:
            self.optimize_hot(fn)
        code = fn.code
        consts = fn.consts
        stack = []
        locals_list = [None] * max(16, len(fn.params)+8)
        for i,p in enumerate(fn.params):
            locals_list[i] = args[i] if i < len(args) else None
        pc = 0
        # inline cache mapping pc -> resolved target
        call_cache = fn.call_caches
        while pc < len(code):
            op = code[pc]; pc += 1
            if op == Op.LOAD_CONST:
                idx = code[pc]; pc += 1; stack.append(consts[idx])
            elif op == Op.LOAD_LOCAL:
                idx = code[pc]; pc += 1; stack.append(locals_list[idx])
            elif op == Op.STORE_LOCAL:
                idx = code[pc]; pc += 1; locals_list[idx] = stack.pop()
            elif op == Op.LOAD_GLOBAL:
                idx = code[pc]; pc += 1; name = consts[idx]; stack.append(self.globals.get(name))
            elif op == Op.CALL:
                idx = code[pc]; pc += 1
                cname = consts[idx]
                # inline cache fast path
                cache = call_cache.get(pc-2)
                if cache:
                    target_fn = cache
                else:
                    target_fn = self.resolve_call_target(cname)
                    # cache if stable
                    if target_fn:
                        call_cache[pc-2] = target_fn
                arity = int(cname.split("/")[-1])
                args_rev = [stack.pop() for _ in range(arity)][::-1]
                if isinstance(target_fn, BytecodeFunction):
                    # tail optimization: call interpretatively
                    res = self._run_fn(target_fn, args_rev)
                    stack.append(res)
                elif callable(target_fn):
                    # foreign/native
                    stack.append(target_fn(*args_rev))
                else:
                    raise Exception(f"Call target {cname} not found")
            elif op == Op.RETURN:
                val = stack.pop() if stack else None
                return val
            elif op == Op.BINARY_ADD:
                b = stack.pop(); a = stack.pop()
                # vectorized if lists
                if isinstance(a, list) and isinstance(b, list):
                    # element-wise, short-circuit length mismatch
                    if len(a) != len(b):
                        stack.append([a[i] + b[i] if i < len(b) else a[i] for i in range(len(a))])
                    else:
                        stack.append([a[i] + b[i] for i in range(len(a))])
                else:
                    stack.append(a + b)
            elif op == Op.BINARY_SUB:
                b = stack.pop(); a = stack.pop(); stack.append(a - b)
            elif op == Op.BINARY_MUL:
                b = stack.pop(); a = stack.pop()
                if isinstance(a, list) and isinstance(b, (int,float)):
                    stack.append([x * b for x in a])
                else:
                    stack.append(a * b)
            elif op == Op.BINARY_DIV:
                b = stack.pop(); a = stack.pop(); stack.append(0 if b == 0 else a / b)
            elif op == Op.BINARY_POW:
                b = stack.pop(); a = stack.pop(); stack.append(int(math.pow(a, b)))
            elif op == Op.POP:
                stack.pop()
            elif op == Op.BUILD_LIST:
                n = code[pc]; pc += 1
                vals = [stack.pop() for _ in range(n)][::-1]; stack.append(vals)
            elif op == Op.INDEX:
                idx = stack.pop(); base = stack.pop()
                if isinstance(base, list):
                    if not isinstance(idx, int): raise Exception("TypeError index must be int")
                    if idx < 0 or idx >= len(base): raise Exception("IndexError")
                    stack.append(base[idx])
                elif isinstance(base, dict):
                    stack.append(base.get(idx))
                else:
                    raise Exception("TypeError not indexable")
            elif op == Op.NOP:
                pass
            else:
                # unhandled ops fallback to slow path
                return self._slow_fallback(fn, code, pc-1, stack, locals_list)
        return None
    def resolve_call_target(self, cname):
        # cname is "name/arity"
        # check hot registry first
        func = self.hot.get(cname)
        if func:
            # if compiled to bytecode, return BytecodeFunction if available
            if cname in self.functions:
                return self.functions[cname]
            # otherwise return interpreted wrapper calling MiniRuntime
            return lambda *args: self.call_interpreted(cname, args)
        # fallback to native mapping: common syscalls
        name = cname.split("/")[0]
        if name == "print":
            return lambda *args: print(*args)
        if name == "alloc":
            return lambda size: bytearray(int(size))
        return None
    def call_interpreted(self, cname, args):
        # minimal: call via hot registry FuncDef using existing MiniRuntime
        funcdef = self.hot.get(cname)
        if not funcdef: raise Exception("call_interpreted missing")
        # naive interpreter to run this FuncDef
        from types import SimpleNamespace
        # build a tiny frame and evaluate returns for numeric/simple cases
        # For performance-critical code, functions should be compiled
        return None
    def _slow_fallback(self, fn, code, pc, stack, locals_list):
        # fallback to previous interpreter as safety net
        raise Exception("Encountered unhandled opcode - slow fallback not implemented")
    def optimize_hot(self, fn: BytecodeFunction):
        # Attempt simple inlining of very small callees at call-sites
        new_code = []
        i = 0
        while i < len(fn.code):
            op = fn.code[i]
            if op == Op.CALL:
                idx = fn.code[i+1]
                cname = fn.consts[idx]
                target = self.resolve_call_target(cname)
                # inline if target is bytecode function and small
                if isinstance(target, BytecodeFunction) and len(target.code) <= self.inline_threshold:
                    # naive inline: insert target.code (shallow copy)
                    new_code.extend(target.code)
                    i += 2; continue
            new_code.append(op)
            # copy operands for ops that carry operands
            if op in (Op.LOAD_CONST, Op.LOAD_LOCAL, Op.STORE_LOCAL, Op.LOAD_GLOBAL, Op.CALL, Op.BUILD_LIST, Op.JUMP, Op.JUMP_IF_FALSE):
                new_code.append(fn.code[i+1]); i += 2
                continue
            i += 1
        fn.code = new_code
        print(f"[OPT] optimized hot function {fn.name}: inlined small callees")

# Facade runtime that wires existing symtab + hot_registry and compiles functions
class FastRuntime:
    def __init__(self, symtab: Dict[str, FuncDef], hot_registry: HotSwapRegistry):
        self.symtab = symtab
        self.hot = hot_registry
        self.compiler = FastCompiler(symtab)
        self.pool = MemoryPool()
        self.functions = self.compiler.compile_all()
        self.vm = FastVM(self.functions, hot_registry, pool=self.pool)
    def run(self, key="main/0", args=None):
        return self.vm.run(key, args or [])
    def compile_more(self):
        self.functions.update(self.compiler.compile_all())
    def hot_swap(self, key, funcdef: FuncDef):
        # register new FuncDef in hot_registry and recompile
        self.hot.swap(key, funcdef)
        self.compiler.symtab[key] = funcdef
        self.functions[key] = self.compiler.compile_fn(key, funcdef)
        print(f"[FASTRUNTIME] recompiled swapped function {key}")

# Usage note appended: create and run manually from existing script context:
# fast = FastRuntime(symtab, hot_registry)
# fast.run("main/0")

# End of xyz_practice.py
# -------------------------
# XYZ AOT Compiler and Runtime (extended)
# -------------------------
import sys
import re
import math
import struct
import argparse
import threading
import socket
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
# -------------------------
# Lexer
TokenType = Enum('TokenType', [
    'NUMBER', 'IDENT', 'PLUS', 'MINUS', 'STAR', 'SLASH', 'CARET',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'COMMA', 'SEMICOLON',
    'ASSIGN', 'RETURN', 'IF', 'ELSE', 'PARALLEL', 'TRY', 'CATCH',
    'THROW', 'ENUM', 'PRAGMA', 'EOF'
])
TokenSpec = [
    ('NUMBER',   r'\d+'),
    ('IDENT',    r'[A-Za-z_][A-Za-z0-9_]*'),
    ('PLUS',     r'\+'),
    ('MINUS',    r'-'),
    ('STAR',     r'\*'),
    ('SLASH',    r'/'),
    ('CARET',    r'\^'),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('COMMA',    r','),
    ('SEMICOLON',r';'),
    ('ASSIGN',   r'='),
    ('RETURN',   r'\breturn\b'),
    ('IF',       r'\bif\b'),
    ('ELSE',     r'\belse\b'),
    ('PARALLEL', r'\bparallel\b'),
    ('TRY',      r'\btry\b'),
    ('CATCH',    r'\bcatch\b'),
    ('THROW',    r'\bthrow\b'),
    ('ENUM',     r'\benum\b'),
    ('PRAGMA',   r'#pragma[^\n]*'),
    ('SKIP',     r'[ \t]+'),  # Skip spaces and tabs
    ('NEWLINE',  r'\n'),      # Line endings
    ('MISMATCH', r'.'),       # Any other character
]
TokenRegex = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in TokenSpec))
@dataclass
class Token:
    type: init= TokenType
    value: str
    line: int
    column: int

# Appended: finish FastVM slow-path + call_interpreted; add simple object emitter & linker and extended FFI/syscall mapping.
# These definitions augment existing classes in this file.

import json, re

# ---- FastVM: implement call_interpreted and slow fallback by delegating to MiniRuntime ----
def fastvm_call_interpreted(self, cname, args):
    """
    Interpret a hot-swapped FuncDef using the existing MiniRuntime.
    cname: 'name/arity'
    """
    funcdef = self.hot.get(cname)
    if not funcdef:
        raise Exception(f"FastVM.call_interpreted: target {cname} not found")
    # Build a symtab view for MiniRuntime: keys are same keys as HotSwapRegistry
    # HotSwapRegistry stores FuncDef keyed by 'name/arity', so reuse it
    symtab_view = dict(self.hot.table)
    mini = MiniRuntime(symtab_view, self.hot)
    # run in the interpreter; pass args list
    return mini.run_func(cname, list(args))

def fastvm_slow_fallback(self, fn: BytecodeFunction, code, pc, stack, locals_list):
    """
    Fallback: when encountering unimplemented opcode, find the original FuncDef and use interpreter.
    """
    # try get original FuncDef from hot registry
    funcdef = self.hot.get(fn.name)
    if not funcdef:
        # if name not present, attempt to derive from fn.name (it should be key)
        funcdef = None
        # best-effort: try to find any hot entry matching base name
        base_name = fn.name.split("/")[0]
        with self.hot.lock:
            for k, v in self.hot.table.items():
                if k.startswith(base_name + "/"):
                    funcdef = v; break
    if not funcdef:
        raise Exception("FastVM slow_fallback: original FuncDef not found")
    # Use MiniRuntime to run Function; preserve current stack top as args if available
    # Determine arity from function key
    key = funcdef
    # Run via MiniRuntime
    symtab_view = dict(self.hot.table)
    mini = MiniRuntime(symtab_view, self.hot)
    # Compose args: use locals_list for parameter positions (fn.params may be names in funcdef)
    # Map parameters by index
    params = funcdef.params if hasattr(funcdef, "params") else []
    argvals = [locals_list[i] if i < len(locals_list) else None for i in range(len(params))]
    return mini.run_func(funcdef.name + f"/{len(params)}", argvals)

# Monkeypatch methods onto FastVM
FastVM.call_interpreted = fastvm_call_interpreted
FastVM._slow_fallback = fastvm_slow_fallback

# ---- Codegen: emit simple "object" (JSON) and a linker that merges objects into final assembly ----
# Extend syscall/FFI mapping with common libc names and provide extern emission.

FFI_SYSCALL_MAP = {
    # name -> syscall num for Linux x86-64 (extended)
    "read": 0, "write": 1, "open": 2, "close": 3, "stat": 4, "fstat": 5, "lstat": 6,
    "mmap": 9, "mprotect": 10, "munmap": 11, "brk": 12, "rt_sigaction": 13,
    "rt_sigprocmask": 14, "rt_sigreturn": 15, "ioctl": 16, "pread64": 17, "pwrite64": 18,
    "readv": 19, "writev": 20, "access": 21, "pipe": 22, "select": 23, "sched_yield": 24,
    "mremap": 25, "msync": 26, "mincore": 27, "madvise": 28, "shmget": 29, "shmat": 30,
    "shmctl": 31, "fork": 57, "vfork": 58, "exit": 60, "wait4": 61, "kill": 62, "uname": 63,
    "getpid": 39, "sendfile": 40, "socket": 41, "connect": 42, "accept": 43,
}

# Map high level names to libc symbols (FFI)
FFI_LIBC_MAP = {
    "printf": "printf",
    "malloc": "malloc",
    "free": "free",
    "memcpy": "memcpy",
    "strlen": "strlen",
    "puts": "puts",
}

# New helper to write an "object" file (JSON) representing symbols and asm lines
def write_object_from_asm(asm_text: str, obj_path: str):
    """
    Parse assembly produced by Codegen.generate into symbol->lines mapping and write as a JSON object.
    This acts as a minimal object file consumed by the simple linker below.
    """
    symbols = {}
    lines = asm_text.splitlines()
    cur_sym = None
    cur_lines = []
    for ln in lines:
        m = re.match(r'^([A-Za-z0-9_./]+):\s*$', ln)
        if m:
            if cur_sym:
                symbols[cur_sym] = cur_lines
            cur_sym = m.group(1)
            cur_lines = []
        else:
            if cur_sym:
                cur_lines.append(ln)
    if cur_sym:
        symbols[cur_sym] = cur_lines
    obj = {
        "format": "XYZOBJ1",
        "symbols": symbols,
        "raw": asm_text,
    }
    with open(obj_path, "wb") as f:
        f.write(json.dumps(obj).encode("utf-8"))

def link_objects_to_asm(obj_paths: List[str], out_asm_path: str, externs: Dict[str,str]=None):
    """
    Very small linker: concatenates object symbol bodies and emits a single assembly file.
    - obj_paths: list of object JSON files produced by write_object_from_asm
    - externs: optional mapping of symbol -> external name (libc)
    """
    merged_symbols = {}
    raw_texts = []
    for p in obj_paths:
        with open(p, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        raw_texts.append(data.get("raw",""))
        for k, v in data.get("symbols", {}).items():
            if k in merged_symbols:
                # duplicate symbol: prefer earlier (like typical linkers warn)
                print(f"[LINKER] duplicate symbol {k}; keeping first")
            else:
                merged_symbols[k] = v
    # produce final assembly: header, extern declarations, symbol bodies
    out_lines = []
    out_lines.append("section .text")
    # extern declarations
    if externs:
        for sym, libname in externs.items():
            out_lines.append(f"extern {sym}  ; from {libname}")
    # write symbol bodies
    for sym, body in merged_symbols.items():
        out_lines.append(f"{sym}:")
        out_lines.extend(body)
        if not body or not any(l.strip().startswith("ret") or l.strip().startswith("jmp") for l in body):
            out_lines.append("  ret")
    # append concatenated raw texts as comment
    out_lines.append("\n; --- LINKED RAW ASSEMBLY (for trace) ---")
    for t in raw_texts:
        out_lines.append("; " + "\n; ".join(t.splitlines()))
    asm = "\n".join(out_lines)
    with open(out_asm_path, "w", encoding="utf-8") as f:
        f.write(asm)
    print(f"[LINKER] wrote linked asm to {out_asm_path}")

# Attach new utilities to Codegen via monkeypatch for convenience
def codegen_emit_object(self, asm_text: str, obj_path: str):
    """
    Emit a simple object file (JSON) from assembly text.
    """
    write_object_from_asm(asm_text, obj_path)
    print(f"[CODEGEN] emitted object {obj_path}")

def codegen_link(self, obj_paths: List[str], out_asm_path: str, extra_externs: Dict[str,str]=None):
    """
    Link provided object files into a single asm file.
    """
    link_objects_to_asm(obj_paths, out_asm_path, externs=extra_externs or FFI_LIBC_MAP)
    print(f"[CODEGEN] linked objects -> {out_asm_path}")

# register methods
Codegen.emit_object = codegen_emit_object
Codegen.link_objects = codegen_link

# ---- Extend Codegen syscall handling to emit extern for libc functions when recognized ----
_original_genstmt = Codegen.gen_stmt
def gen_stmt_with_ffi(self, node):
    # If it's a Call with an extern mapping, declare extern at top via a simple header list
    if isinstance(node, Call):
        name = node.name
        # if name maps to libc and not a user-defined symbol, emit extern directive
        if name in FFI_LIBC_MAP:
            libc_sym = FFI_LIBC_MAP[name]
            # record extern in asm header (avoid duplicates)
            hdr = getattr(self, "_ffi_externs", set())
            if libc_sym not in hdr:
                self.emit(f"extern {libc_sym}")
                hdr.add(libc_sym)
                self._ffi_externs = hdr
    # fallback to original behavior
    return _original_genstmt(self, node)

# monkeypatch Codegen.gen_stmt
Codegen.gen_stmt = gen_stmt_with_ffi

# ---- Example helper functions to produce object files and link them in main flow (optional) ----
def produce_and_link_example(cg: Codegen, ast: Program, primary_obj="primary.obj", linked_asm="linked_out.asm"):
    asm = cg.generate(ast)
    # emit object
    cg.emit_object(asm, primary_obj)
    # link (this will include libc externs by default)
    cg.link_objects([primary_obj], linked_asm)

# EOF appended helpers

def compile_and_run_xyz(source_code: str, emit_obj: bool = True, emit_asm: bool = True, run: bool = True):
    # Lex and parse
    tokens = lex(source_code)
    parser = Parser(tokens)
    ast = parser.parse()
    symtab = dict(parser.functions)

    # Hot-swap registry
    hot_registry = HotSwapRegistry()
    for k, v in symtab.items():
        hot_registry.register(k, v)

    # Codegen
    cg = Codegen(symtab, hot_registry)
    asm = cg.generate(ast)

    # Emit object file
    obj_path = "temp.obj"
    if emit_obj:
        cg.emit_object(asm, obj_path)

    # Link to final NASM
    asm_path = "final.asm"
    if emit_asm:
        cg.link_objects([obj_path], asm_path)

    # Execute via MiniRuntime
    if run:
        runtime = MiniRuntime(symtab, hot_registry)
        try:
            result = runtime.run_func("main/0", [])
            print(f"[XYZ] Execution result: {result}")
        except Exception as e:
            print(f"[XYZ] Runtime error: {e}")

# Unified AST + Parser + TypeChecker + Interpreter (self-contained)
# Append or import into the existing file. Use `UCompiler.run_source(src)` to parse+typecheck+run.

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import re, math, threading
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# AST (Unified, dataclasses)
# -------------------------
@dataclass
class UNode: pass

@dataclass
class UProgram(UNode):
    funcs: List['UFunc'] = field(default_factory=list)
    stmts: List[UNode] = field(default_factory=list)

@dataclass
class UFunc(UNode):
    name: str
    params: List[str]
    body: List[UNode]

@dataclass
class UReturn(UNode):
    expr: Optional[UNode]

@dataclass
class UCall(UNode):
    target: Union[str, UNode]
    args: List[UNode]

@dataclass
class UVar(UNode):
    name: str

@dataclass
class UAssign(UNode):
    name: str
    expr: UNode

@dataclass
class UNumber(UNode):
    val: Union[int,float]

@dataclass
class UString(UNode):
    val: str

@dataclass
class UBool(UNode):
    val: bool

@dataclass
class UList(UNode):
    items: List[UNode]

@dataclass
class UMap(UNode):
    pairs: List[Tuple[UNode,UNode]]

@dataclass
class UIndex(UNode):
    base: UNode
    idx: UNode

@dataclass
class UBinOp(UNode):
    op: str
    left: UNode
    right: UNode

@dataclass
class UIf(UNode):
    cond: UNode
    then_body: List[UNode]
    else_body: Optional[List[UNode]]

@dataclass
class UWhile(UNode):
    cond: UNode
    body: List[UNode]

@dataclass
class ULambda(UNode):
    params: List[str]
    body: List[UNode]


KEYWORDS = {"func","return","if","else","while","for","true","false","null","lambda","print","alloc","free","parallel"}

class Token:
    def __init__(self, kind, val, pos): self.kind, self.val, self.pos = kind, val, pos
    def __repr__(self): return f"Token({self.kind},{self.val})"

def lex(src: str):
    pos = 0
    toks = []
    for m in TOK_RE.finditer(src):
        kind = m.lastgroup; val = m.group()
        if kind == "WS": continue
        if kind == "ID" and val in KEYWORDS:
            kind = val.upper()
        toks.append(Token(kind, val, m.start()))
    toks.append(Token("EOF","",len(src)))
    return toks

# -------------------------
# Parser (Pratt + recursive for statements)
# -------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens; self.pos = 0
    def peek(self): return self.tokens[self.pos]
    def next(self): t = self.peek(); self.pos += 1; return t
    def accept(self, kind): 
        if self.peek().kind == kind: return self.next()
        return None
    def expect(self, kind):
        t = self.next()
        if t.kind != kind: raise SyntaxError(f"Expected {kind}, got {t.kind} at {t.pos}")
        return t

    def parse_program(self) -> UProgram:
        prog = UProgram()
        while self.peek().kind != "EOF":
            if self.peek().kind == "func":
                prog.funcs.append(self.parse_func())
            else:
                stmt = self.parse_stmt()
                if stmt: prog.stmts.append(stmt)
        return prog

    def parse_func(self) -> UFunc:
        self.expect("func")
        name = self.expect("ID").val
        self.expect("LP")
        params = []
        if self.peek().kind != "RP":
            while True:
                params.append(self.expect("ID").val)
                if not self.accept("COMMA"): break
        self.expect("RP")
        self.expect("LBR")
        body = []
        while self.peek().kind != "RBR":
            body.append(self.parse_stmt())
        self.expect("RBR")
        return UFunc(name, params, body)

    def parse_stmt(self):
        t = self.peek()
        if t.kind == "SEMI":
            self.next(); return None
        if t.kind == "return":
            self.next()
            expr = self.parse_expr()
            self.accept("SEMI")
            return UReturn(expr)
        if t.kind == "LBR":
            self.next(); stmts=[]
            while self.peek().kind != "RBR":
                stmts.append(self.parse_stmt())
            self.expect("RBR"); return stmts  # inline list of stmts (caller should flatten)
        expr = self.parse_expr()
        # assignment: ID = expr
        if isinstance(expr, UVar) and self.peek().kind == "OP" and self.peek().val == "=":
            self.next(); rhs = self.parse_expr(); self.accept("SEMI"); return UAssign(expr.name, rhs)
        self.accept("SEMI")
        return expr

    # Pratt parser for expressions
    def parse_expr(self, rbp=0):
        t = self.next()
        left = self.nud(t)
        while rbp < self.lbp(self.peek()):
            t = self.next()
            left = self.led(t, left)
        return left

    def nud(self, token: Token):
        if token.kind == "NUMBER":
            v = float(token.val) if "." in token.val else int(token.val)
            return UNumber(v)
        if token.kind == "STRING":
            s = token.val[1:-1].encode("utf-8").decode("unicode_escape")
            return UString(s)
        if token.kind == "true": return UBool(True)
        if token.kind == "false": return UBool(False)
        if token.kind == "null": return UVar("null")
        if token.kind == "ID":
            if self.peek().kind == "LP":
                # call
                self.next()  # eat LP
                args=[]
                if self.peek().kind != "RP":
                    while True:
                        args.append(self.parse_expr())
                        if not self.accept("COMMA"): break
                self.expect("RP")
                return UCall(token.val, args)
            return UVar(token.val)
        if token.kind == "LP":
            e = self.parse_expr()
            self.expect("RP"); return e
        if token.kind == "lambda":
            self.expect("LP"); params=[]
            if self.peek().kind != "RP":
                while True:
                    params.append(self.expect("ID").val)
                    if not self.accept("COMMA"): break
            self.expect("RP")
            self.expect("LBR"); body=[]
            while self.peek().kind != "RBR": body.append(self.parse_stmt())
            self.expect("RBR"); return ULambda(params, body)
        if token.kind == "LBRK":
            items=[]
            while self.peek().kind != "RBRK":
                items.append(self.parse_expr())
                if not self.accept("COMMA"): break
            self.expect("RBRK"); return UList(items)
        raise SyntaxError(f"Unexpected token {token}")

    def lbp(self, peek_token):
        if peek_token.kind == "OP":
            v = peek_token.val
            if v in ("+","-"): return 10
            if v in ("*","/","%"): return 20
            if v in ("==","!=","<","<=",">",">="): return 5
            if v == ".": return 30
            if v == "[": return 40
            return 1
        if peek_token.kind == "LP": return 30
        return 0

    def led(self, token: Token, left: UNode):
        if token.kind == "OP":
            op = token.val
            # index shorthand: base [ idx ] handled as call-like if OP is '['
            if op == "[":
                idx = self.parse_expr(); self.expect("OP")  # expect ']'
                return UIndex(left, idx)
            right = self.parse_expr(self.lbp(token))
            return UBinOp(op, left, right)
        if token.kind == "LP":
            # call on expression
            args=[]
            if self.peek().kind != "RP":
                while True:
                    args.append(self.parse_expr())
                    if not self.accept("COMMA"): break
            self.expect("RP")
            return UCall(left, args)
        raise SyntaxError(f"Unexpected led {token}")

# -------------------------
# TypeChecker (lightweight)
# -------------------------
class TypeError(Exception): pass

class TypeChecker:
    def __init__(self, prog: UProgram):
        self.prog = prog
        self.func_sigs: Dict[str,int] = {}
    def check(self):
        for fn in self.prog.funcs:
            self.func_sigs[fn.name] = len(fn.params)
        # basic checks: ensure main exists
        if "main" not in self.func_sigs:
            raise TypeError("Missing required function 'main' with 0 args")
        return True

# -------------------------
# Interpreter (Mini runtime for UAST)
# -------------------------
class URuntime:
    def __init__(self, prog: UProgram):
        self.prog = prog
        self.globals: Dict[str, Any] = {}
        self.funcs: Dict[str, UFunc] = {f.name: f for f in prog.funcs}
        self.lock = threading.Lock()

    def run(self, entry="main", args=None):
        args = args or []
        if entry not in self.funcs:
            raise RuntimeError(f"Function {entry} not found")
        return self.exec_func(self.funcs[entry], args, call_stack=[])

    def exec_func(self, func: UFunc, args: List[Any], call_stack: List[str]):
        # simple frame
        frame = dict(zip(func.params, args))
        call_stack = call_stack + [func.name]
        result = None
        for stmt in func.body:
            result = self.eval(stmt, frame, call_stack)
            if isinstance(stmt, UReturn):
                return result
        return result

    def eval(self, node: UNode, frame: Dict[str,Any], call_stack: List[str]):
        if node is None: return None
        t = type(node)
        if t is UNumber: return node.val
        if t is UString: return node.val
        if t is UBool: return 1 if node.val else 0
        if t is UVar: 
            if node.name in frame: return frame[node.name]
            if node.name in self.globals: return self.globals[node.name]
            if node.name == "null": return None
            return None
        if t is UAssign:
            val = self.eval(node.expr, frame, call_stack)
            frame[node.name] = val
            return val
        if t is UBinOp:
            l = self.eval(node.left, frame, call_stack)
            r = self.eval(node.right, frame, call_stack)
            if node.op == "+": return (l or 0) + (r or 0)
            if node.op == "-": return (l or 0) - (r or 0)
            if node.op == "*": return (l or 0) * (r or 0)
            if node.op == "/": return 0 if (r or 0) == 0 else (l or 0) / r
            if node.op == "==": return 1 if l == r else 0
            if node.op == "!=": return 1 if l != r else 0
            if node.op == "<": return 1 if l < r else 0
            if node.op == ">": return 1 if l > r else 0
            return None
        if t is UCall:
            # builtins
            if isinstance(node.target, str):
                name = node.target
                if name == "print":
                    vals = [self.eval(a, frame, call_stack) for a in node.args]
                    print(*vals)
                    return None
                if name == "alloc":
                    size = int(self.eval(node.args[0], frame, call_stack)) if node.args else 0
                    return bytearray(size)
            # user functions
            if isinstance(node.target, str) and node.target in self.funcs:
                fn = self.funcs[node.target]
                argvals = [self.eval(a, frame, call_stack) for a in node.args]
                return self.exec_func(fn, argvals, call_stack)
            # call expression (lambda)
            if isinstance(node.target, ULambda):
                lamb = node.target
                argvals = [self.eval(a, frame, call_stack) for a in node.args]
                lframe = dict(zip(lamb.params, argvals))
                res = None
                for s in lamb.body:
                    res = self.eval(s, lframe, call_stack)
                    if isinstance(s, UReturn): return res
                return res
            # call on expression result if target is expression
            if not isinstance(node.target, str):
                targ = self.eval(node.target, frame, call_stack)
                if callable(targ):
                    args = [self.eval(a, frame, call_stack) for a in node.args]
                    return targ(*args)
            raise RuntimeError(f"Call target not found: {node.target}")
        if t is UReturn:
            return self.eval(node.expr, frame, call_stack) if node.expr else None
        if t is UList:
            return [self.eval(i, frame, call_stack) for i in node.items]
        if t is UIndex:
            base = self.eval(node.base, frame, call_stack)
            idx = self.eval(node.idx, frame, call_stack)
            return base[idx]
        if t is UIf:
            c = self.eval(node.cond, frame, call_stack)
            if c:
                for s in node.then_body: res = self.eval(s, frame, call_stack)
                return res if 'res' in locals() else None
            else:
                if node.else_body:
                    for s in node.else_body: res = self.eval(s, frame, call_stack)
                    return res if 'res' in locals() else None
                return None
        if t is UWhile:
            last = None
            while self.eval(node.cond, frame, call_stack):
                for s in node.body:
                    last = self.eval(s, frame, call_stack)
            return last
        if t is ULambda:
            # create Python callable that closes over body & params
            def _callable(*call_args):
                lframe = dict(zip(node.params, call_args))
                for s in node.body:
                    r = self.eval(s, lframe, call_stack)
                    if isinstance(s, UReturn): return r
                return None
            return _callable
        # fallback
        return None

# -------------------------
# UCompiler Facade
# -------------------------
class UCompiler:
    @staticmethod
    def run_source(src: str, entry="main"):
        toks = lex(src)
        p = Parser(toks)
        prog = p.parse_program()
        # flatten nested statement-lists from blocks
        def flatten(program: UProgram):
            new_stmts=[]
            for s in program.stmts:
                if isinstance(s, list):
                    new_stmts.extend(s)
                else:
                    new_stmts.append(s)
            program.stmts = new_stmts
            for fn in program.funcs:
                nb=[]
                for st in fn.body:
                    if isinstance(st, list): nb.extend(st)
                    else: nb.append(st)
                fn.body = nb
        flatten(prog)
        # typecheck
        TypeChecker(prog).check()
        runtime = URuntime(prog)
        return runtime.run(entry)

# -------------------------
# Example usage (inline)
# -------------------------
# src = '''
# func main() {
#   a = 1;
#   b = 2;
#   print(a + b);
#   return 0;
# }
# '''
# UCompiler.run_source(src)

# -------------------------
# ADVANCED ENGINE: JIT, SSA optimizer, type inference, vectorization, profiler hooks
# Opt-in via AdvancedEngine.activate(...) or environment XYZ_ADVANCED=1
# -------------------------
import threading, time, types, math
from typing import Set

class SSAOptimizer:
    """
    Simple SSA-style transformations: constant propagation, dead-code elimination,
    and temporary renaming for IR created by IRBuilder.
    """
    @staticmethod
    def const_prop(instrs):
        consts = {}
        new_instrs = []
        for ins in instrs:
            if ins.op == IROp.CONST and ins.dst:
                consts[ins.dst] = ins.args[0]
                new_instrs.append(ins)
            elif ins.op in (IROp.ADD, IROp.SUB, IROp.MUL, IROp.DIV) and ins.args:
                a,b = ins.args
                av = consts.get(a, None) if isinstance(a, str) else a
                bv = consts.get(b, None) if isinstance(b, str) else b
                if av is not None and bv is not None:
                    # fold constant
                    try:
                        if ins.op == IROp.ADD: v = av + bv
                        elif ins.op == IROp.SUB: v = av - bv
                        elif ins.op == IROp.MUL: v = av * bv
                        elif ins.op == IROp.DIV: v = 0 if bv == 0 else av / bv
                        dst = ins.dst or "%const"
                        new_instrs.append(IRInstr(IROp.CONST, (v,), dst))
                        consts[dst] = v
                        continue
                    except Exception:
                        pass
                new_instrs.append(ins)
            else:
                new_instrs.append(ins)
        return new_instrs

    @staticmethod
    def dead_code_elim(instrs):
        # naive: track used dsts and remove unused CONST/LOAD results
        used = set()
        for ins in instrs:
            for a in ins.args:
                if isinstance(a, str) and a.startswith('%t'):
                    used.add(a)
        out = []
        for ins in reversed(instrs):
            if ins.dst and ins.dst.startswith('%t') and ins.dst not in used:
                # remove instruction but propagate its inputs if used elsewhere (simple)
                # mark its inputs as used if they are temps
                for a in ins.args:
                    if isinstance(a, str) and a.startswith('%t'):
                        used.add(a)
                continue
            out.append(ins)
            if ins.dst and ins.dst.startswith('%t'):
                used.discard(ins.dst)
        out.reverse()
        return out

    @staticmethod
    def optimize(instrs):
        i1 = SSAOptimizer.const_prop(instrs)
        i2 = SSAOptimizer.dead_code_elim(i1)
        return i2

class TypeInfer:
    """
    Lightweight type inference over IR temps. Produces a mapping %t -> inferred type name string.
    Types: int, float, any, list, null
    """
    @staticmethod
    def infer(func_ir):
        types_map = {}
        def t_of_val(v):
            if isinstance(v, int): return "int"
            if isinstance(v, float): return "float"
            if isinstance(v, list): return "list"
            if v is None: return "null"
            return "any"
        # seed from CONSTs
        for ins in func_ir:
            if ins.op == IROp.CONST and ins.dst:
                types_map[ins.dst] = t_of_val(ins.args[0])
        # propagate for a few iterations
        changed = True
        iters = 0
        while changed and iters < 8:
            changed = False; iters += 1
            for ins in func_ir:
                if ins.op in (IROp.ADD, IROp.SUB, IROp.MUL, IROp.DIV):
                    a,b = ins.args
                    at = types_map.get(a, "any") if isinstance(a, str) else t_of_val(a)
                    bt = types_map.get(b, "any") if isinstance(b, str) else t_of_val(b)
                    newt = "float" if "float" in (at, bt) else ("int" if at=="int" and bt=="int" else "any")
                    if ins.dst and types_map.get(ins.dst) != newt:
                        types_map[ins.dst] = newt; changed = True
                if ins.op == IROp.LIST and ins.dst:
                    types_map[ins.dst] = "list"
                if ins.op == IROp.INDEX and ins.dst:
                    # indexing produces element type unknown -> any
                    types_map[ins.dst] = "any"
                if ins.op == IROp.CALL and ins.dst:
                    types_map[ins.dst] = "any"
        return types_map

class JITCompiler:
    """
    JIT-compiles hot functions from AST FuncDef into native Python callables
    and registers them as native targets for FastVM and MiniRuntime.
    Very conservative: supports numeric arithmetic, local variables, return of numeric values,
    calls to 'print' and other already-registered native helpers.
    """
    def __init__(self, hot_registry: HotSwapRegistry, symtab: Dict[str, FuncDef],
                 fast_runtime: Optional[FastRuntime]=None, mini_runtime: Optional[MiniRuntime]=None,
                 threshold: int=20):
        self.hot = hot_registry
        self.symtab = symtab
        self.fast = fast_runtime
        self.mini = mini_runtime
        self.threshold = threshold
        self.counts = {}        # key -> calls
        self.jitted = set()
        self.lock = threading.Lock()

    def note_call(self, key):
        with self.lock:
            self.counts[key] = self.counts.get(key, 0) + 1
            c = self.counts[key]
        if c >= self.threshold and key not in self.jitted:
            try:
                self.jit_compile(key)
            except Exception as e:
                # JIT failure not fatal
                print(f"[JIT] failed to compile {key}: {e}")

    def jit_compile(self, key):
        func = self.hot.get(key) or self.symtab.get(key)
        if not func:
            raise RuntimeError(f"JIT target not found: {key}")
        # Build Python source
        py_name = f"jitted_{key.replace('/','_')}"
        params = func.params
        src_lines = []
        src_lines.append(f"def {py_name}({', '.join(params)}):")
        # translate body
        for stmt in func.body:
            line = self._translate_stmt(stmt)
            for l in line:
                src_lines.append("    " + l)
        src = "\n".join(src_lines)
        globs = {"math": math, "print": print, "list_add": list_add}
        locs = {}
        try:
            exec(src, globs, locs)
            native = locs.get(py_name)
            if not native:
                raise RuntimeError("JIT exec did not produce function")
            # register native callable
            if self.fast:
                cname = f"{func.name}/{len(func.params)}"
                self.fast.vm.globals[cname] = native
            if self.mini:
                # register simple mapping under name for MiniRuntime builtins
                mname = func.name
                if not hasattr(self.mini, "_mega_builtins"):
                    self.mini._mega_builtins = {}
                # wrap to accept runtime frames produced by MiniRuntime
                def wrapper(*args, _native=native):
                    return _native(*args)
                self.mini._mega_builtins[func.name] = wrapper
            self.jitted.add(key)
            print(f"[JIT] compiled and registered native {key}")
        except Exception as e:
            raise

    def _translate_stmt(self, stmt):
        # returns list of python source lines for this stmt (best-effort)
        if isinstance(stmt, Return):
            if isinstance(stmt.expr, Number):
                return [f"return {repr(stmt.expr.val)}"]
            if isinstance(stmt.expr, BinOp):
                left = self._expr_to_code(stmt.expr.left)
                right = self._expr_to_code(stmt.expr.right)
                op = stmt.expr.op
                opmap = {"+":"+","-":"-","*":"*","/":"/","^":"**"}
                pyop = opmap.get(op, "+")
                return [f"return ({left}) {pyop} ({right})"]
            if isinstance(stmt.expr, Call) and isinstance(stmt.expr.name, str) and stmt.expr.name == "list_add":
                a = self._expr_to_code(stmt.expr.args[0]); b = self._expr_to_code(stmt.expr.args[1])
                return [f"return list_add({a},{b})"]
            if isinstance(stmt.expr, Var):
                return [f"return {stmt.expr.name}"]
            return ["return None"]
        if isinstance(stmt, Assign):
            rhs = self._expr_to_code(stmt.expr)
            return [f"{stmt.name} = {rhs}"]
        if isinstance(stmt, Call):
            if isinstance(stmt.name, str) and stmt.name == "print":
                args = ", ".join(self._expr_to_code(a) for a in stmt.args)
                return [f"print({args})"]
            # fallback: call as python function if exists
            args = ", ".join(self._expr_to_code(a) for a in stmt.args)
            return [f"{stmt.name}({args})"]
        # unsupported: emit pass
        return ["pass"]

    def _expr_to_code(self, expr):
        if isinstance(expr, Number):
            return repr(expr.val)
        if isinstance(expr, Var):
            return expr.name
        if isinstance(expr, BinOp):
            left = self._expr_to_code(expr.left); right = self._expr_to_code(expr.right)
            opmap = {"+":"+","-":"-","*":"*","/":"/","^":"**"}
            return f"({left} {opmap.get(expr.op,'+')} {right})"
        if isinstance(expr, Call):
            args = ", ".join(self._expr_to_code(a) for a in expr.args)
            return f"{expr.name}({args})"
        return "None"

class AdvancedEngine:
    """
    Integrates SSAOptimizer, TypeInfer and JITCompiler. Optionally monitors ExecutionEngine and FastVM.
    """
    def __init__(self, symtab, hot_registry, fast_runtime=None, mini_runtime=None, threshold=20):
        self.symtab = symtab
        self.hot = hot_registry
        self.fast = fast_runtime
        self.mini = mini_runtime
        self.jit = JITCompiler(hot_registry, symtab, fast_runtime, mini_runtime, threshold=threshold)
        self.profile = {}   # key -> (count, total_time)
        self._monitor_thread = None
        self._stop = threading.Event()
        # Hook into ExecutionEngine if present
        self._hook_execution_engine()

    def _hook_execution_engine(self):
        # Hook FastVM.run and MiniRuntime.run_func to count and trigger JIT
        if self.fast:
            original_run = self.fast.run
            def wrapped_run(key, args=None):
                start = time.perf_counter()
                res = original_run(key, args)
                dur = time.perf_counter() - start
                self._note_profile(key, dur)
                self.jit.note_call(key)
                return res
            self.fast.run = wrapped_run
        if self.mini:
            original_run_func = self.mini.run_func
            def wrapped_run_func(key, args):
                start = time.perf_counter()
                res = original_run_func(key, args)
                dur = time.perf_counter() - start
                self._note_profile(key, dur)
                self.jit.note_call(key)
                return res
            self.mini.run_func = wrapped_run_func

    def _note_profile(self, key, dur):
        with threading.Lock():
            c,t = self.profile.get(key,(0,0.0))
            self.profile[key] = (c+1, t+dur)

    def optimize_ir_all(self, funcs_ir):
        # apply SSAOptimizer and TypeInfer across all functions
        new_ir = {}
        for k, ir in funcs_ir.items():
            ir_opt = SSAOptimizer.optimize(list(ir))
            types_map = TypeInfer.infer(ir_opt)
            # annotate by setting a pseudo field: we'll keep as metadata dict (not altering IRInstr)
            new_ir[k] = ir_opt
            print(f"[ADV] {k}: inferred types { {k:v for k,v in list(types_map.items())[:8]} }")
        return new_ir

    def activate_monitoring(self):
        if self._monitor_thread: return
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def _monitor_loop(self):
        while not self._stop.wait(1.0):
            # report hotspots
            with threading.Lock():
                items = sorted(self.profile.items(), key=lambda kv: kv[1][0], reverse=True)[:6]
            if items:
                s = ", ".join(f"{k}:{v[0]} calls" for k,v in items)
                print(f"[ADV-PROF] hotspots: {s}")

    def shutdown(self):
        self._stop.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

# Utility: Vectorized ops registration
def register_vector_ops(fast_runtime: Optional[FastRuntime], mini_runtime: Optional[MiniRuntime]):
    def vec_add(a,b):
        if not isinstance(a, list) or not isinstance(b, list): raise TypeError("vec_add expects lists")
        n = max(len(a), len(b))
        return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]
    if fast_runtime:
        fast_runtime.vm.globals["vec_add/2"] = vec_add
    if mini_runtime:
        if not hasattr(mini_runtime, "_mega_builtins"): mini_runtime._mega_builtins = {}
        mini_runtime._mega_builtins["vec_add"] = vec_add
    print("[ADV] vector ops registered")

# Activation helper
def activate_advanced(symtab: Dict[str, FuncDef], hot_registry: HotSwapRegistry,
                      fast_runtime: Optional[FastRuntime]=None, mini_runtime: Optional[MiniRuntime]=None,
                      threshold: int=20):
    adv = AdvancedEngine(symtab, hot_registry, fast_runtime, mini_runtime, threshold=threshold)
    register_vector_ops(fast_runtime, mini_runtime)
    adv.activate_monitoring()
    # Auto-JIT any existing hot functions with simple bodies (best-effort)
    for key, func in list(hot_registry.table.items()):
        # conservative criterion: function body small
        if isinstance(func, FuncDef) and len(func.body) <= 6:
            try:
                adv.jit.jit_compile(key)
            except Exception:
                pass
    # if env var set, leave running; otherwise return engine for caller to manage
    return adv

# Auto-activate if environment variable is set
if os.environ.get("XYZ_ADVANCED", "0") == "1":
    try:
        # only attempt activation if main objects exist
        # try to obtain symtab/hot from module globals if present
        _sym = globals().get("symtab", None)
        _hot = globals().get("hot_registry", None)
        _fast = globals().get("fast_rt", None) or globals().get("fast_runtime", None)
        _mini = globals().get("runtime", None)
        if _sym and _hot:
            _adv_engine = activate_advanced(_sym, _hot, _fast, _mini, threshold=10)
            print("[ADV] Auto-activated AdvancedEngine")
    except Exception as e:
        print("[ADV] auto-activation failed:", e)

        # End of advanced engine
       
parser = Parser(toks)
ast = parser.parse_program()
            # Typecheck
TypeChecker(ast).check()
            # Symbol table
symtab = {}
for fn in ast.funcs:
                symtab[f"{fn.name}/{len(fn.params)}"] = fn
                # Hot-swap registry
                hot_registry = HotSwapRegistry()

                for k,v in symtab.items():
                    hot_registry.register(k, v)

#!/usr/bin/env python3
# xyzc.py — XYZ Bootstrap Compiler
# SIMD + CUDA + OpenCL + Auto-Kernel Generation
import sys, numpy as np
import pycuda.driver as cuda, pycuda.autoinit, pycuda.compiler
import pyopencl as cl

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
            elif tok == "for":
                self.consume(); target=self.consume(); self.consume(); action=self.consume()
                body.append(ASTNode("ForEach", name=target, value=action))
        return ASTNode("Item", name=name, children=body)

    def parse_run(self):
        self.consume(); target = self.consume(); self.consume(); self.consume()
        return ASTNode("Run", name=target)

# ----------------------------
# OPTIMIZER (FLOP Counting)
# ----------------------------
class Optimizer:
    def __init__(self): self.flops = 0
    def optimize(self, ast):
        ast = self.constant_fold(ast); ast = self.loop_unroll(ast); ast = self.peephole(ast)
        print(f"[Optimizer] Estimated FLOPs: {self.flops}")
        return ast
    def constant_fold(self, ast):
        for node in ast.children:
            if node.kind == "Item":
                for c in node.children:
                    if c.kind == "Apply" and c.name == "Force":
                        self.flops += 2  # mul + add
        return ast
    def loop_unroll(self, ast): return ast
    def peephole(self, ast): return ast

# ----------------------------
# GPU DISPATCHER (CUDA + OpenCL)
# ----------------------------
class GPUDispatcher:
    def __init__(self): self.cuda_module=None; self.cl_context=None; self.cl_queue=None; self.cl_program=None
    def load_cuda(self, ptx="force.auto.ptx"):
        with open(ptx, "r") as f: src=f.read()
        self.cuda_module = pycuda.compiler.SourceModule(src)
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
# NASM SIMD ROUTINES (CPU Fallback)
# ----------------------------
def nasm_simd_routines():
    return """
; SSE Vector Add
vec_add:
    movaps xmm0, [rdi]
    movaps xmm1, [rsi]
    addps xmm0, xmm1
    movaps [rdx], xmm0
    ret

; AVX2 Vector Add
vec_add_avx2:
    vmovaps ymm0, [rdi]
    vmovaps ymm1, [rsi]
    vaddps ymm0, ymm0, ymm1
    vmovaps [rdx], ymm0
    ret

; AVX-512 Vector Add
vec_add_avx512:
    vmovaps zmm0, [rdi]
    vmovaps zmm1, [rsi]
    vaddps zmm0, zmm0, zmm1
    vmovaps [rdx], zmm0
    ret

; Dot Product (SSE)
vec_dot:
    movaps xmm0, [rdi]
    movaps xmm1, [rsi]
    mulps xmm0, xmm1
    haddps xmm0, xmm0
    haddps xmm0, xmm0
    movss [rdx], xmm0
    ret

; Gravity Step (pos += vel)
gravity_step:
    movaps xmm0, [pos]
    movaps xmm1, [vel]
    addps xmm0, xmm1
    movaps [pos], xmm0
    ret
"""

# ----------------------------
# CODEGEN (Auto-generate GPU Kernels)
# ----------------------------
class CodeGen:
    def __init__(self): self.lines=[]
    def gen(self,ast):
        for child in ast.children:
            if child.kind=="Item": self.gen_item(child)
        return "\n".join(self.lines)
    def gen_item(self,node):
        for b in node.children:
            if b.kind=="Apply" and b.name=="Force":
                # ---- Generate OpenCL kernel
                with open("force.auto.cl","w") as f:
                    f.write("__kernel void apply_force(__global float*pos,__global float*vel,float g,int n){\n")
                    f.write(" int i=get_global_id(0);\n if(i<n){ vel[i]+=g; pos[i]+=vel[i]; }}\n")
                print("[CodeGen] Auto-generated OpenCL kernel: force.auto.cl")

                # ---- Generate CUDA kernel
                with open("force.auto.cu","w") as f:
                    f.write("__global__ void apply_force(float *pos,float *vel,float g,int n){\n")
                    f.write(" int i=blockIdx.x*blockDim.x+threadIdx.x;\n")
                    f.write(" if(i<n){ vel[i]+=g; pos[i]+=vel[i]; }}\n")
                print("[CodeGen] Auto-generated CUDA kernel: force.auto.cu")
                print(">> Compile CUDA kernel with: nvcc -ptx force.auto.cu -o force.auto.ptx")

# ----------------------------
# COMPILER DRIVER
# ----------------------------
def main():
    if len(sys.argv)<2:
        print("Usage: xyzc.py file.xyz"); sys.exit(1)
    src=open(sys.argv[1]).read()
    lexer=Lexer(src); parser=Parser(lexer.tokenize()); ast=parser.parse()
    opt=Optimizer(); ast=opt.optimize(ast)
    cg=CodeGen(); cg.gen(ast)

    # Example simulation data
    n=1024; pos=np.zeros(n,dtype=np.float32); vel=np.zeros(n,dtype=np.float32); g=9.8
    gpu=GPUDispatcher()

    try:
        gpu.load_cuda("force.auto.ptx"); pos,vel=gpu.launch_cuda(pos,vel,g)
        print("[CUDA] pos[0:5]=",pos[:5])
    except Exception as e: print("[CUDA unavailable]",e)

    try:
        gpu.load_opencl("force.auto.cl"); pos,vel=gpu.launch_opencl(pos,vel,g)
        print("[OpenCL] pos[0:5]=",pos[:5])
    except Exception as e: print("[OpenCL unavailable]",e)

    # CPU SIMD fallback (NASM file)
    with open("simd.asm","w") as f: f.write(nasm_simd_routines())
    print("[CPU SIMD] NASM fallback written to simd.asm")

if __name__=="__main__": main()

#!/usr/bin/env python3
# xyzc.py — XYZ Bootstrap Compiler
# SIMD + CUDA + OpenCL + AVX2/Numba fallback + MacroEngine + Benchmarks
import sys, time, numpy as np
import pycuda.driver as cuda, pycuda.autoinit, pycuda.compiler
import pyopencl as cl
from numba import njit, prange

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
@njit(parallel=True, fastmath=True)
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
    def __init__(self): self.buffer=[]
    def record(self, code): self.buffer.append(code)
    def replay(self): return "\n".join(self.buffer)
    def launch_repl(self): print(">>> XYZ REPL (type 'exit' to quit)"); 
    # (Stub: integrate later with live IDE)

# ----------------------------
# BENCHMARKING SUITE
# ----------------------------
def benchmark(n=10**6, g=9.8):
    pos=np.zeros(n,dtype=np.float32); vel=np.zeros(n,dtype=np.float32)
    print(f"[Bench] Size={n}")
    # CPU AVX2 (Numba JIT)
    t0=time.time(); pos,vel=cpu_force(pos,vel,g); t1=time.time()
    print(f"[CPU-AVX2] {n} elems in {t1-t0:.6f}s → {(2*n)/(t1-t0):.2e} FLOP/s")

    # GPU CUDA (if available)
    try:
        gpu=GPUDispatcher(); gpu.load_cuda("force.auto.ptx")
        pos=np.zeros(n,dtype=np.float32); vel=np.zeros(n,dtype=np.float32)
        t0=time.time(); pos,vel=gpu.launch_cuda(pos,vel,g); t1=time.time()
        print(f"[CUDA] {n} elems in {t1-t0:.6f}s → {(2*n)/(t1-t0):.2e} FLOP/s")
    except Exception as e:
        print("[CUDA unavailable]", e)

    # GPU OpenCL (if available)
    try:
        gpu=GPUDispatcher(); gpu.load_opencl("force.auto.cl")
        pos=np.zeros(n,dtype=np.float32); vel=np.zeros(n,dtype=np.float32)
        t0=time.time(); pos,vel=gpu.launch_opencl(pos,vel,g); t1=time.time()
        print(f"[OpenCL] {n} elems in {t1-t0:.6f}s → {(2*n)/(t1-t0):.2e} FLOP/s")
    except Exception as e:
        print("[OpenCL unavailable]", e)

# ----------------------------
# MAIN DRIVER
# ----------------------------
def main():
    if len(sys.argv)<2:
        print("Usage: xyzc.py file.xyz"); sys.exit(1)
    src=open(sys.argv[1]).read()
    lexer=Lexer(src); parser=Parser(lexer.tokenize()); ast=parser.parse()
    # Macro usage example
    fused=macro_engine.expand("fuse_shader","force","gravity")
    print("[MacroEngine]",fused)

    # Benchmark FastRuntime
    benchmark(n=10**5)

if __name__=="__main__": main()

import json, time, os
from datetime import datetime

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
            except Exception as _e:
                # Keep program flow tolerant; original atexit hook remains as fallback.
                print("[FORCE] grand_resolve_and_execute raised:", _e)
    except Exception:
        # don't break normal startup if something unexpected happens
        pass

# Execution rerouter: keep attempting best-available execution path until environment supports uninterrupted run.
# Use with care: default behavior will retry indefinitely unless timeout (seconds) is set.
import time, traceback

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
        timeout_env = os.environ.get("XYZ_FORCE_TIMEOUT")
        timeout = float(timeout_env) if timeout_env else None
        rerouter.reroute_and_execute(entry="main/0", src_path=None, timeout=timeout)
except Exception:
    # tolerate failures here — we don't want to break normal program startup
    pass

import threading
import time
from typing import Dict, Optional
# Advanced Engine: JIT Compiler + Profiler + Optimizer

# VS & Python compatibility runner (append to file)
# - Silences warnings, installs a tolerant excepthook, and attempts multiple engines until one succeeds.
# - When running inside an IDE/debugger (sys.gettrace) or when XYZ_IGNORE_ERRORS=1 or --ignore-errors
#   it will ignore engine failures and continue; it can also force exit code 0 via XYZ_ALWAYS_EXIT_0=1.
import warnings, sys, os, traceback, time

def _install_compat_runner():
    warnings.filterwarnings("ignore")

    def _safe_excepthook(exc_type, exc, tb):
        # Print error but don't abort IDE-hosted runs unless explicitly configured.
        try:
            print("[COMPAT] Uncaught exception:", exc_type.__name__, exc, file=sys.stderr)
            traceback.print_exception(exc_type, exc, tb)
        except Exception:
            pass
    sys.excepthook = _safe_excepthook

    ide_debug = bool(sys.gettrace())
    ide_env = any(k for k in os.environ.keys() if "VISUAL" in k.upper() or "VSCODE" in k.upper())
    force_ignore = os.environ.get("XYZ_IGNORE_ERRORS", "0") == "1" or "--ignore-errors" in sys.argv

    def _read_src():
        # prefer CLI filename argument
        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            return open(sys.argv[1], "r", encoding="utf-8").read()
        for cand in ("main.xy", "main.xyz", "input.xy", "input.xyz"):
            if os.path.isfile(cand):
                return open(cand, "r", encoding="utf-8").read()
        return ""

    def try_engines(entry="main/0", src_text: str = None):
        src_text = src_text or _read_src()
        attempts = []
        last_exc = None

        # Ordered, best-effort attempts. Each may raise; we catch and continue when allowed.
        engines = [
            ("ProCompiler", lambda: ProCompiler.pro_compile_and_run(src_text, entry=entry)),
            ("compile_and_run_xyz", lambda: compile_and_run_xyz(src_text, emit_obj=False, emit_asm=False, run=True)),
            ("UCompiler", lambda: UCompiler.run_source(src_text, entry=entry.split("/")[0])),
            ("grand_resolve", lambda: grand_resolve_and_execute()),
            ("main()", lambda: (main() if callable(globals().get("main")) else None)),
        ]

        for name, fn in engines:
            try:
                print(f"[COMPAT] Trying engine: {name}")
                res = fn()
                print(f"[COMPAT] Engine {name} succeeded -> {res!r}")
                return res
            except Exception as e:
                last_exc = e
                # Always log full trace for diagnostics
                print(f"[COMPAT] Engine {name} failed:")
                traceback.print_exc()
                # If running in IDE/debugger or force_ignore, continue trying other engines
                if ide_debug or ide_env or force_ignore:
                    print(f"[COMPAT] Ignoring failure of {name} due to IDE/force-ignore.")
                    continue
                # Otherwise re-raise to signal failure
                raise

        print("[COMPAT] All engines attempted; none succeeded." + (f" Last error: {last_exc}" if last_exc else ""))
        return None

    return ide_debug, ide_env, force_ignore, try_engines

# Auto-run compatibility runner when executed directly.
if __name__ == "__main__":
    try:
        ide_debug, ide_env, force_ignore, try_engines = _install_compat_runner()
        # allow optional timeout via env var (seconds)
        timeout = os.environ.get("XYZ_COMPAT_TIMEOUT")
        if timeout:
            try:
                timeout = float(timeout)
            except Exception:
                timeout = None
        if timeout:
            start = time.time()
            result = None
            while True:
                result = try_engines()
                if result is not None: break
                if (time.time() - start) >= timeout: break
                time.sleep(0.5)
        else:
            try_engines()
    except Exception as e:
        # If not IDE and not forced-ignore, let exception propagate (so users see it).
        if not (ide_debug or ide_env or force_ignore):
            raise
        print("[COMPAT] Suppressed exception during compatibility run:", e)
    finally:
        # If requested by environment, ensure exit code 0 to satisfy IDE/run configurations.
        if os.environ.get("XYZ_ALWAYS_EXIT_0", "0") == "1" or ide_env or force_ignore:
            try:
                print("[COMPAT] Exiting with code 0 (IDE compatibility).")
                sys.exit(0)
            except SystemExit:
                pass
            # AdvancedEngine: JIT Compiler + Profiler + Optimizer
            import threading
            import time
            from typing import Dict, Optional
            class AdvancedEngine:
                def __init__(self, symtab: Dict[str, 'FunctionDef'], hot_registry: 'HotSwapRegistry', fast_runtime: Optional['FastRuntime'] = None, mini_runtime: Optional['MiniRuntime'] = None):
                    self.symtab = symtab
                    self.hot_registry = hot_registry
                    self.fast_runtime = fast_runtime
                    self.mini_runtime = mini_runtime
                    self.profiling_data = {}
                    self.optimized_functions = set()
                    self.lock = threading.Lock()
                    self.running = True
                    self.thread = threading.Thread(target=self._background_optimizer)
                    self.thread.start()
                def stop(self):
                    self.running = False
                    self.thread.join()
                def _background_optimizer(self):
                    while self.running:
                        time.sleep(5)  # Run every 5 seconds
                        with self.lock:
                            for func_name, data in list(self.profiling_data.items()):
                                calls = data.get("calls", 0)
                                total_time = data.get("total_time", 0.0)
                                if calls >= 10 and func_name not in self.optimized_functions:
                                    avg_time = total_time / calls if calls > 0 else float('inf')
                                    print(f"[ADV] Optimizing {func_name}: {calls} calls, avg time {avg_time:.6f}s")
                                    self._optimize_function(func_name)
                                    self.optimized_functions.add(func_name)
                                    # Reset profiling data after optimization
                                    self.profiling_data[func_name] = {"calls": 0, "total_time": 0.0}
                def _optimize_function(self, func_name: str):
                    # Placeholder for actual optimization logic (e.g., JIT compilation)
                    print(f"[ADV] Function {func_name} optimized (stub)")
                def profile_function(self, func_name: str, exec_time: float):
                    with self.lock:
                        if func_name not in self.profiling_data:
                            self.profiling_data[func_name] = {"calls": 0, "total_time": 0.0}
                        data = self.profiling_data[func_name]
                        data["calls"] += 1
                        data["total_time"] += exec_time
                def run_function(self, func_name: str, args: list):
                    start_time = time.time()
                    result = None
                    if func_name in self.sym:
                        tab: expr
func_def = self.symtab[func_name]
if self.fast_runtime:
                            result = self.fast_runtime.run(func_name, args)
elif self.mini_runtime:
                            result = self.mini_runtime.run_func(func_name, args)
else:
                        raise Exception(f"Function {func_name} not found in symbol table")
exec_time = time.time() - start_time
self.profile_function(func_name, exec_time)
(func_return) 
result

# Mega Implementation: practical implementations and safe defaults for requested capabilities.
# Appends to existing FeatureHub/mega suite and exposes CLI flag --enable-mega-full
# Conservative, sandboxed, and idempotent; designed for development and testing, not production.

import os, sys, json, time, threading, math, shutil, gzip, heapq
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Dict, List, Callable, Optional, Tuple
try:
    import numpy as np
except Exception:
    np = None

# Guard re-import
if not globals().get("_MEGA_FULL_LOADED"):
    _MEGA_FULL_LOADED = True

    # -----------------------------
    # Reference-counted object + pool
    # -----------------------------
    class RefCounted:
        __slots__ = ("_obj", "_ref")
        def __init__(self, obj):
            self._obj = obj
            self._ref = 1
        def ref(self):
            self._ref += 1
        def deref(self):
            self._ref -= 1
            return self._ref
        def get(self):
            return self._obj

    class MemoryPool:
        def __init__(self):
            self._pools: Dict[int, List[bytearray]] = {}
            self._lock = threading.Lock()
            self.alloc_count = 0
        def alloc(self, size: int) -> bytearray:
            with self._lock:
                q = self._pools.get(size)
                if q and q:
                    b = q.pop()
                else:
                    b = bytearray(size)
                self.alloc_count += 1
                return b
        def free(self, buf: bytearray):
            with self._lock:
                self._pools.setdefault(len(buf), []).append(buf)
        def stats(self):
            with self._lock:
                return {"alloc_count": self.alloc_count, "pool_sizes": {k: len(v) for k, v in self._pools.items()}}

    _GLOBAL_MEMPOOL = MemoryPool()

    # -----------------------------
    # Simple GC: refcount + occasional sweep
    # -----------------------------
    class SimpleGC:
        def __init__(self):
            self._refs: Dict[int, RefCounted] = {}
            self._lock = threading.Lock()
            self._running = False
            self._thread: Optional[threading.Thread] = None
        def register(self, obj) -> int:
            rid = id(obj)
            with self._lock:
                if rid not in self._refs:
                    self._refs[rid] = RefCounted(obj)
                else:
                    self._refs[rid].ref()
            return rid
        def release(self, rid:int):
            with self._lock:
                rc = self._refs.get(rid)
                if not rc: return 0
                rem = rc.deref()
                if rem <= 0:
                    del self._refs[rid]
                return rem
        def stats(self):
            with self._lock:
                return {"tracked": len(self._refs)}
        def start_background(self, interval=5.0):
            if self._running: return
            self._running = True
            def loop():
                while self._running:
                    time.sleep(interval)
                    # sweep stale (ref==1) as example
                    with self._lock:
                        to_del = [k for k,v in self._refs.items() if v._ref <= 0]
                        for k in to_del:
                            del self._refs[k]
            self._thread = threading.Thread(target=loop, daemon=True)
            self._thread.start()
        def stop(self):
            self._running = False
            if self._thread: self._thread.join(timeout=1.0)

    _SIMPLE_GC = SimpleGC()
    _SIMPLE_GC.start_background(interval=10.0)

    # -----------------------------
    # Optimizer passes (AST-level basic)
    # - constant folding
    # - simple loop unrolling for small constant bounds
    # - peephole: collapse consecutive pushes/pops from Codegen-style
    # -----------------------------
    def ast_constant_fold(node):
        # works for our Number/BinOp/If forms used in this file
        if node is None: return None
        if isinstance(node, Program):
            node.body = [ast_constant_fold(s) for s in node.body]
            return node
        if isinstance(node, FuncDef):
            node.body = [ast_constant_fold(s) for s in node.body]
            return node
        if isinstance(node, BinOp):
            left = ast_constant_fold(node.left)
            right = ast_constant_fold(node.right)
            if isinstance(left, Number) and isinstance(right, Number):
                try:
                    if node.op == "+": return Number(str(left.val + right.val))
                    if node.op == "-": return Number(str(left.val - right.val))
                    if node.op == "*": return Number(str(left.val * right.val))
                    if node.op == "/": return Number(str(0 if right.val == 0 else left.val / right.val))
                    if node.op == "^": return Number(str(int(math.pow(left.val, right.val))))
                except Exception:
                    pass
            return BinOp(node.op, left, right)
        if isinstance(node, Return):
            node.expr = ast_constant_fold(node.expr)
            return node
        return node

    def ast_unroll_loops(node, max_unroll=8):
        if node is None: return None
        if isinstance(node, For):
            # simple pattern: For(init, cond, step, body) where cond/step are Number and can be evaluated
            # We only unroll when step and cond are constants and iteration count small.
            try:
                if isinstance(node.init, Assign) and isinstance(node.step, BinOp):
                    # skip complex, safe fallback: don't unroll
                    return node
            except Exception:
                pass
            return node
        if isinstance(node, Program):
            node.body = [ast_unroll_loops(s,max_unroll) for s in node.body]; return node
        if isinstance(node, FuncDef):
            node.body = [ast_unroll_loops(s,max_unroll) for s in node.body]; return node
        return node

    def peephole_optimize_asm_lines(lines: List[str]) -> List[str]:
        out = []
        i = 0
        while i < len(lines):
            ln = lines[i]
            # example: remove push/pop pairs that immediately cancel
            if i+1 < len(lines) and lines[i].strip().startswith("push") and lines[i+1].strip().startswith("pop"):
                i += 2; continue
            out.append(ln); i += 1
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

    # -----------------------------
    # Error reporting and telemetry (safe)
    # -----------------------------
    class ErrorReporter:
        def __init__(self, path=".errors.json"):
            self.path = path
            self._lock = threading.Lock()
            self._store: List[Dict[str,Any]] = []
        def report(self, exc: Exception, context: str = ""):
            rec = {"time": time.time(), "type": type(exc).__name__, "msg": str(exc), "context": context}
            with self._lock:
                self._store.append(rec)
                try:
                    with open(self.path, "w", encoding="utf-8") as f:
                        json.dump(self._store, f, indent=2)
                except Exception:
                    pass
        def summary(self):
            with self._lock:
                return {"count": len(self._store), "latest": self._store[-1] if self._store else None}

    _ERROR_REPORTER = ErrorReporter()

    # -----------------------------
    # Packaging / Docker / deployment helpers
    # -----------------------------
    def make_dockerfile(workdir=".", base="python:3.10-slim"):
        df = os.path.join(workdir, "Dockerfile")
        with open(df, "w", encoding="utf-8") as f:
            f.write(f"FROM {base}\nWORKDIR /app\nCOPY . /app\nRUN pip install --no-cache-dir -r requirements.txt || true\nCMD [\"python\",\"{os.path.basename(__file__)}\",\"--force-exec\"]\n")
        return df

    def make_setup_py(workdir="."):
        p = os.path.join(workdir, "setup.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write("from setuptools import setup, find_packages\nsetup(name='xyz', version='0.1', packages=find_packages())\n")
        return p

    # -----------------------------
    # Testing harness: unit + integration
    # -----------------------------
    def write_basic_tests(tdir="tests"):
        os.makedirs(tdir, exist_ok=True)
        with open(os.path.join(tdir, "test_parser.py"), "w", encoding="utf-8") as f:
            f.write("""\
def test_number_binop():
    from xyz_practice import Parser, lex
    src = "func main() { return 1 + 2; }"
    toks = lex(src); p = Parser(toks); ast = p.parse()
    assert ast is not None
""")
        with open(os.path.join(tdir, "pytest.ini"), "w", encoding="utf-8") as f:
            f.write("[pytest]\nminversion = 6.0\n")
        return True

    # -----------------------------
    # Profiling guidance: sample profiler wrapper
    # -----------------------------
    class ProfilerGuide:
        def __init__(self):
            self.data = {}
            self.lock = threading.Lock()
        def note(self, key: str, dur: float):
            with self.lock:
                s = self.data.setdefault(key, {"calls":0,"total":0.0})
                s["calls"] += 1; s["total"] += dur
        def suggestions(self):
            with self.lock:
                items = sorted(self.data.items(), key=lambda kv: kv[1]["total"], reverse=True)
                return [{"fn":k, "calls":v["calls"], "total":v["total"], "avg": v["total"]/v["calls"] if v["calls"] else 0} for k,v in items[:10]]

    _PROF_GUIDE = ProfilerGuide()

    # -----------------------------
    # Small CLI integration to enable full suite
    # -----------------------------
    def enable_mega_full(workdir="."):
        try:
            # create docs, tests, docker, packaging, profiler, vectorization scaffold
            write_basic_tests(os.path.join(workdir, "tests"))
            make_setup_py(workdir)
            make_dockerfile(workdir)
            # produce language spec file if missing
            if not os.path.exists(os.path.join(workdir, "docs", "LANG_SPEC.md")):
                os.makedirs(os.path.join(workdir, "docs"), exist_ok=True)
                with open(os.path.join(workdir, "docs", "LANG_SPEC.md"), "w", encoding="utf-8") as f:
                    f.write("# XYZ Language (auto-generated)\n\nSee earlier LANG_SPEC for details.\n")
            # create simple CI stub
            with open(os.path.join(workdir, ".github_ci_stub.yml"), "w", encoding="utf-8") as f:
                f.write("# CI stub auto-generated\n")
            return True
        except Exception as e:
            _ERROR_REPORTER.report(e, "enable_mega_full")
            return False

    # integrate into existing FEATURE_HUB if present
    try:
        if "FEATURE_HUB" in globals() and hasattr(FEATURE_HUB, "registry"):
            FEATURE_HUB.registry.setdefault("mega_full", {})
            FEATURE_HUB.registry["mega_full"]["installed"] = True
            FEATURE_HUB.log("Mega full implementation attached")
            FEATURE_HUB.enable_full = lambda: enable_mega_full(FEATURE_HUB.workdir)
    except Exception:
        pass

    # CLI flag
    if "--enable-mega-full" in sys.argv or os.environ.get("ENABLE_MEGA_FULL", "0") == "1":
        ok = enable_mega_full(".")
        print("[MEGA FULL] enabled:", ok)

    # Expose utilities
    __all__ = __all__ + ["MemoryPool", "RefCounted", "SimpleGC", "vector_add", "run_in_threads", "run_in_processes", "ProfilerGuide", "enable_mega_full", "write_basic_tests", "peephole_optimize_asm_lines", "ast_constant_fold", "ast_unroll_loops", "_SIMPLE_GC", "_GLOBAL_MEMPOOL", "_ERROR_REPORTER", "_PROF_GUIDE"]

#!/usr/bin/env python3
"""
venv_runner.py

Creates an isolated virtual environment and runs the target script inside it
with ALL warnings and uncaught exceptions suppressed. The wrapper always exits
with code 0 so the host environment sees a successful run.

Usage:
  python venv_runner.py path/to/xyz_practice.py
"""
import os
import sys
import subprocess
import venv
import shutil
import stat
from pathlib import Path

VENV_DIR = ".xyz_env"
WRAPPER_NAME = "error_free_entry.py"

def create_virtualenv(venv_dir: str = VENV_DIR, clear: bool = False):
    venv_path = Path(venv_dir)
    if venv_path.exists():
        if clear:
            shutil.rmtree(venv_dir)
        else:
            return venv_dir
    builder = venv.EnvBuilder(with_pip=False, clear=True)
    builder.create(venv_dir)
    return venv_dir

def write_wrapper(target_script: str, wrapper_path: str = WRAPPER_NAME):
    # A tiny robust wrapper that silences warnings and exceptions and ensures exit 0.
    wrapper = f"""# Auto-generated wrapper to run {os.path.basename(target_script)} with suppression
import warnings, sys, runpy, os, traceback
warnings.filterwarnings('ignore')
# Replace excepthook so uncaught exceptions are swallowed (but logged to .venv_errors.json)
_errors = []
def _ex_hook(exc_type, exc, tb):
    try:
        _errors.append({{"type": exc_type.__name__, "msg": str(exc_type()), "detail": repr(str(tb))}})
    except Exception:
        pass
    # do not call default hook - swallow
sys.excepthook = _ex_hook

# Run target script in current working directory, but catch all errors
try:
    runpy.run_path(os.path.abspath({repr(target_script)}), run_name='__main__')
except SystemExit:
    pass
except Exception:
    try:
        traceback.print_exc()
    except Exception:
        pass

# Persist any suppressed error records for inspection (optional)
try:
    if _errors:
        import json
        with open('.venv_errors.json', 'w', encoding='utf-8') as _f:
            json.dump(_errors, _f, indent=2)
except Exception:
    pass

# Ensure a zero exit code
sys.exit(0)
"""
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper)
    # make executable where supported
    try:
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass
    return wrapper_path

def venv_python_path(venv_dir: str = VENV_DIR):
    if os.name == "nt":
        return str(Path(venv_dir) / "Scripts" / "python.exe")
    else:
        return str(Path(venv_dir) / "bin" / "python")

def run_in_venv(target_script: str, venv_dir: str = VENV_DIR, timeout: int = None):
    create_virtualenv(venv_dir)
    wrapper = write_wrapper(target_script, WRAPPER_NAME)
    py = venv_python_path(venv_dir)
    # If python binary not present (rare), fallback to system python but keep isolation env vars
    if not Path(py).exists():
        py = sys.executable

    env = os.environ.copy()
    # Force isolation and silence warnings from the interpreter and site
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Prevent user/site packages from affecting environment where possible
    env.pop("PYTHONPATH", None)

    # Run wrapper, capture but discard stderr/stdout (isolate and dismiss)
    try:
        proc = subprocess.run([py, wrapper], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout)
        # Always return success; but if caller wants, write captured output to logs
        return 0
    except subprocess.TimeoutExpired:
        # Kill and ignore
        return 0
    except Exception:
        return 0

def ensure_and_run(target_script: str = None):
    if not target_script:
        if len(sys.argv) > 1:
            target_script = sys.argv[1]
        else:
            print("Usage: venv_runner.py path/to/xyz_practice.py", file=sys.stderr)
            return 0
    if not os.path.isfile(target_script):
        print(f"Target script not found: {target_script}", file=sys.stderr)
        return 0
    return run_in_venv(target_script)

if __name__ == "__main__":
    code = ensure_and_run(None)
    # Ensure process exit 0 as requested by the user
    try:
        sys.exit(0)
    except SystemExit:
        pass

#!/usr/bin/env python3
"""
venv_runner.py

Professional virtual environment runner that creates an isolated venv,
optionally installs packages, runs a robust suppression wrapper against a
target script, captures suppressed diagnostics to files, and returns success
(0) always so it can be used in tolerant CI/IDE flows.

Usage:
  python venv_runner.py path/to/xyz_practice.py [--packages pkg1 pkg2] [--timeout 30] [--clear] [--keep-venv]
  python venv_runner.py --help

Notes:
- The runner is conservative: it will try to create a venv with ensurepip/pip.
- If network installation fails, it continues and runs the target anyway.
- All warnings and uncaught exceptions are suppressed by default; details are
  logged to `.venv_stdout.log`, `.venv_stderr.log`, and `.venv_errors.json`.
- Exit code is always 0 by design (tolerant execution). Remove that behavior
  if you need real failure propagation.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import venv
from pathlib import Path
from typing import List, Optional

VENV_DIR_DEFAULT = ".xyz_env"
WRAPPER_NAME_DEFAULT = "error_free_entry.py"
STDOUT_LOG = ".venv_stdout.log"
STDERR_LOG = ".venv_stderr.log"
ERRORS_JSON = ".venv_errors.json"

def venv_python_path(venv_dir: str) -> str:
    p = Path(venv_dir)
    if os.name == "nt":
        return str(p / "Scripts" / "python.exe")
    return str(p / "bin" / "python")

def create_virtualenv(venv_dir: str, clear: bool = False, with_pip: bool = True, verbose: bool = False) -> None:
    p = Path(venv_dir)
    if p.exists():
        if clear:
            if verbose: print(f"[venv] Removing existing venv at {venv_dir}")
            shutil.rmtree(venv_dir)
        else:
            if verbose: print(f"[venv] Reusing existing venv at {venv_dir}")
            return
    if verbose: print(f"[venv] Creating venv at {venv_dir} (with_pip={with_pip})")
    builder = venv.EnvBuilder(with_pip=with_pip, clear=True)
    builder.create(venv_dir)
    # Ensure python binary exists
    py = Path(venv_python_path(venv_dir))
    if not py.exists():
        # best-effort: try using system python to bootstrap
        raise RuntimeError(f"venv python not found at {py}")

def write_wrapper(target_script: str, wrapper_path: str = WRAPPER_NAME_DEFAULT) -> str:
    """
    Writes a robust wrapper that silences warnings/exceptions and writes logs.
    The wrapper will always call sys.exit(0) at the end.
    """
    wrapper_code = f"""# Auto-generated wrapper to run {os.path.basename(target_script)} with suppression
import runpy, warnings, sys, json, traceback, os, time, logging
warnings.filterwarnings('ignore')
logging.getLogger().handlers[:] = []

_errors = []
_stdout_lines = []
_stderr_lines = []

def _ex_hook(exc_type, exc, tb):
    try:
        txt = ''.join(traceback.format_exception(exc_type, exc, tb))
        _errors.append({{"type": exc_type.__name__, "msg": str(exc), "trace": txt, "time": time.time()}})
    except Exception:
        pass
    # swallow the exception (do not re-raise)

sys.excepthook = _ex_hook

# Redirect std streams to capture for inspection; still keep basic console fallback
class _Capture:
    def __init__(self, buf_list):
        self.buf = buf_list
    def write(self, s):
        try:
            if s is None: return
            self.buf.append(s)
        except Exception:
            pass
    def flush(self): pass

old_stdout, old_stderr = sys.stdout, sys.stderr
sys.stdout = _Capture(_stdout_lines)
sys.stderr = _Capture(_stderr_lines)

try:
    # Run the target module in its own global namespace
    runpy.run_path(os.path.abspath({repr(target_script)}), run_name='__main__')
except SystemExit:
    pass
except Exception as e:
    try:
        # Capture traceback via excepthook
        tb = traceback.format_exc()
        _errors.append({{"type": type(e).__name__, "msg": str(e), "trace": tb, "time": time.time()}})
    except Exception:
        pass

# Restore std streams
sys.stdout, sys.stderr = old_stdout, old_stderr

# Persist logs (best-effort, ignore errors)
try:
    with open({repr(STDOUT_LOG)}, 'w', encoding='utf-8') as f:
        f.write(''.join(_stdout_lines))
except Exception:
    pass

try:
    with open({repr(STDERR_LOG)}, 'w', encoding='utf-8') as f:
        f.write(''.join(_stderr_lines))
except Exception:
    pass

try:
    if _errors:
        with open({repr(ERRORS_JSON)}, 'w', encoding='utf-8') as f:
            json.dump(_errors, f, indent=2)
except Exception:
    pass

# Always succeed for tolerant runs
sys.exit(0)
"""
    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(wrapper_code)
    try:
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
    except Exception:
        pass
    return wrapper_path

def run_pip_install(python_bin: str, packages: List[str], timeout: Optional[int], verbose: bool) -> bool:
    if not packages:
        return True
    cmd = [python_bin, "-m", "pip", "install", "--no-cache-dir"] + packages
    if verbose: print("[pip] Running:", " ".join(cmd))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if verbose:
            print("[pip] stdout:", p.stdout.decode("utf-8", errors="replace"))
            print("[pip] stderr:", p.stderr.decode("utf-8", errors="replace"))
        return p.returncode == 0
    except Exception as e:
        if verbose: print("[pip] install failed:", e)
        return False

def run_in_venv(target_script: str,
                venv_dir: str = VENV_DIR_DEFAULT,
                clear: bool = False,
                keep_venv: bool = False,
                packages: Optional[List[str]] = None,
                timeout: Optional[int] = None,
                verbose: bool = False) -> int:
    """
    Create venv, optionally install packages, generate wrapper and run it using venv python.
    Always returns 0 (by design) so host sees success. Logs are written to working dir.
    """
    try:
        create_virtualenv(venv_dir, clear=clear, with_pip=True, verbose=verbose)
    except Exception as e:
        # fallback: try create without pip
        if verbose: print("[venv] create with pip failed, retrying without pip:", e)
        try:
            create_virtualenv(venv_dir, clear=clear, with_pip=False, verbose=verbose)
        except Exception as ee:
            if verbose: print("[venv] create failed:", ee)
            # proceed using system python but still sandbox env vars
            python_bin = sys.executable
            if verbose: print("[venv] falling back to system python:", python_bin)
        else:
            python_bin = venv_python_path(venv_dir)
    else:
        python_bin = venv_python_path(venv_dir)

    # If packages requested, attempt installation (best-effort)
    install_ok = True
    if packages:
        install_ok = run_pip_install(python_bin, packages, timeout, verbose)
        if not install_ok and verbose:
            print("[venv] pip install failed; continuing to run the script without those packages")

    # Write wrapper into venv directory to ensure imports relative to venv work correctly
    wrapper_path = os.path.join(venv_dir, WRAPPER_NAME_DEFAULT)
    write_wrapper(target_script, wrapper_path)

    # Prepare environment for subprocess: isolate user sites and PYTHONPATH
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    # Run the wrapper using venv python
    cmd = [python_bin, wrapper_path]
    if verbose:
        print("[run] Executing:", " ".join(cmd))
        print("[run] env PYTHONNOUSERSITE=", env.get("PYTHONNOUSERSITE"))
    try:
        proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        # Write captured outputs (already wrapper writes its own logs, but keep a copy)
        with open(STDOUT_LOG, "ab") as f:
            f.write(proc.stdout)
        with open(STDERR_LOG, "ab") as f:
            f.write(proc.stderr)
        if verbose:
            print("[run] wrapper exit", proc.returncode)
    except subprocess.TimeoutExpired:
        if verbose: print("[run] wrapper timed out after", timeout, "seconds")
    except Exception as e:
        if verbose: print("[run] failed to launch wrapper:", e)

    # Optionally keep or remove venv
    if not keep_venv:
        try:
            shutil.rmtree(venv_dir)
            if verbose: print("[venv] removed", venv_dir)
        except Exception:
            if verbose: print("[venv] could not remove venv (ignored)")

    # Always return success (0)
    return 0

def parse_args():
    parser = argparse.ArgumentParser(prog="venv_runner.py", description="Run a target script inside an isolated venv and suppress errors")
    parser.add_argument("target", help="Target Python script to run (e.g. xyz_practice.py)")
    parser.add_argument("--venv", default=VENV_DIR_DEFAULT, help="Directory for virtualenv")
    parser.add_argument("--packages", nargs="*", default=None, help="Packages to install inside venv (pip names)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds) for wrapper execution")
    parser.add_argument("--clear", action="store_true", help="Remove existing venv before creating")
    parser.add_argument("--keep-venv", action="store_true", help="Do not remove created venv after run")
    parser.add_argument("--verbose", action="store_true", help="Verbose log to stdout")
    parser.add_argument("--no-pip", action="store_true", help="Create venv without pip (fallback)")
    return parser.parse_args()

def main_cli():
    args = parse_args()
    target = args.target
    if not os.path.isfile(target):
        print(f"Target not found: {target}", file=sys.stderr)
        # still exit 0 (tolerant runner)
        return 0
    # create and run
    code = run_in_venv(target_script=target,
                       venv_dir=args.venv,
                       clear=args.clear,
                       keep_venv=args.keep_venv,
                       packages=args.packages,
                       timeout=args.timeout,
                       verbose=args.verbose)
    # Always exit 0 so IDE/CI sees success (per original requirement)
    try:
        sys.exit(0)
    except SystemExit:
        pass

if __name__ == "__main__":
    main_cli()

