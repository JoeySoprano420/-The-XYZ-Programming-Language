#!/usr/bin/env python3
# xyzc.py - XYZ Compiler (Extended with auto-linker -> syscalls, persistent hot-swap IPC,
# and extended interpreter with frames/closures)
#
# Features added:
# - Auto-linker: map common extern calls (write/read/exit/getpid) to Linux syscalls and emit inline 'syscall'
# - Persistent hot-swap API: small TCP JSON server on localhost:4000 supporting "swap" operations
# - Interpreter: frame stack, parameter binding, Var lookup in frames, Lambda -> Closure objects, calling closures
#
# Usage:
#  python xyz_practice.py input.xy --emit-asm --emit-pkt --hot-swap-server
#  Then connect to localhost:4000 and send JSON to swap functions:
#    {"op":"swap","key":"main/0","type":"const_return","value":123}
#

import sys, re, argparse, math, threading, struct, json, socket, copy
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# LEXER (extended)
# -------------------------
TOKENS = [
    ("NUMBER", r"-?\d+(\.\d+)?"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("STRING", r"\".*?\"|\'.*?\'"),
    ("PRAGMA", r"#pragma\s+[A-Za-z0-9_ ]+"),
    ("OP", r"[+\-*/=<>!&|%^.^]+"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"),
    ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("SEMI", r";"), ("COMMA", r","),
    ("KEYWORD", r"\b(func|return|if|else|while|for|lambda|true|false|null|parallel|enum|eval|poly|try|catch|throw|raise|force|extern|alloc|free|malloc|mutex|mutex_lock|mutex_unlock|Start|main|print)\b"),
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
# AST Nodes
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

# -------------------------
# PARSER with symbol table
# -------------------------
class Parser:
    def __init__(self, tokens):
        self.tokens, self.pos = tokens, 0
        self.functions: Dict[str, FuncDef] = {}  # symbol table: "name/arity" -> FuncDef

    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def eat(self, kind=None):
        tok = self.peek()
        if not tok: raise SyntaxError("EOF")
        if kind and tok.kind != kind: raise SyntaxError(f"Expected {kind}, got {tok.kind}")
        self.pos += 1; return tok

    def parse(self):
        prog = Program(self.statements())
        return prog

    def statements(self):
        stmts = []
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
        name = self.eat("ID").val
        self.eat("LPAREN")
        params = []
        while self.peek() and self.peek().kind != "RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind == "COMMA": self.eat("COMMA")
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = []
        while self.peek() and self.peek().kind != "RBRACE":
            # support simple assignment with '=' as OP token
            if self.peek().kind == "ID" and self.pos+1 < len(self.tokens) and self.tokens[self.pos+1].kind == "OP" and self.tokens[self.pos+1].val == "=":
                name_tok = self.eat("ID").val
                self.eat("OP")  # =
                expr = self.expression()
                body.append(Assign(name_tok, expr))
            elif self.peek().val == "return":
                self.eat("KEYWORD"); body.append(Return(self.expression()))
            elif self.peek().val == "if":
                body.append(self.ifstmt())
            elif self.peek().val == "while":
                body.append(self.whilestmt())
            elif self.peek().val == "for":
                body.append(self.forstmt())
            elif self.peek().val == "parallel":
                body.append(self.parallelblock())
            elif self.peek().val == "try":
                body.append(self.trycatch())
            else:
                expr = self.expression()
                if expr is not None: body.append(expr)
                else: self.pos += 1
        self.eat("RBRACE")
        func = FuncDef(name, params, body)
        key = f"{name}/{len(params)}"
        self.functions[key] = func  # register in symbol table
        return func

    # expression parsing
    def expression(self): return self.parse_addsub()
    def parse_addsub(self):
        left = self.parse_muldiv()
        while self.peek() and self.peek().kind == "OP" and self.peek().val in ("+","-","||"):
            op = self.eat("OP").val
            right = self.parse_muldiv()
            left = BinOp(op, left, right)
        return left
    def parse_muldiv(self):
        left = self.parse_pow()
        while self.peek() and self.peek().kind == "OP" and self.peek().val in ("*","/","&&"):
            op = self.eat("OP").val
            right = self.parse_pow()
            left = BinOp(op, left, right)
        return left
    def parse_pow(self):
        left = self.parse_unary()
        while self.peek() and self.peek().kind == "OP" and self.peek().val == "^":
            op = self.eat("OP").val
            right = self.parse_unary()
            left = BinOp(op, left, right)
        return left
    def parse_unary(self):
        tok = self.peek()
        if not tok: return None
        if tok.kind == "OP" and tok.val == "-":
            self.eat("OP"); return BinOp("*", Number("-1"), self.parse_unary())
        if tok.kind == "NUMBER": return Number(self.eat("NUMBER").val)
        if tok.kind == "KEYWORD" and tok.val == "true": self.eat("KEYWORD"); return Bool(True)
        if tok.kind == "KEYWORD" and tok.val == "false": self.eat("KEYWORD"); return Bool(False)
        if tok.kind == "KEYWORD" and tok.val == "null": self.eat("KEYWORD"); return Null()
        if tok.kind == "ID":
            if self.pos+1 < len(self.tokens) and self.tokens[self.pos+1].kind == "LPAREN": return self.call()
            return Var(self.eat("ID").val)
        if tok.kind == "LPAREN":
            self.eat("LPAREN"); e = self.expression(); self.eat("RPAREN"); return e
        if tok.kind == "KEYWORD" and tok.val == "lambda": return self.lambdaexpr()
        if tok.kind == "KEYWORD" and tok.val == "eval": return self.evalexpr()
        if tok.kind == "KEYWORD" and tok.val == "enum": return self.enumdef()
        return None
    def call(self):
        name = self.eat("ID").val; self.eat("LPAREN"); args = []
        while self.peek() and self.peek().kind != "RPAREN":
            args.append(self.expression())
            if self.peek() and self.peek().kind == "COMMA": self.eat("COMMA")
        self.eat("RPAREN"); return Call(name, args)
    def lambdaexpr(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); params=[]
        while self.peek() and self.peek().kind != "RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind == "COMMA": self.eat("COMMA")
        self.eat("RPAREN"); self.eat("LBRACE")
        body = []
        while self.peek() and self.peek().kind != "RBRACE":
            if self.peek().val == "return":
                self.eat("KEYWORD"); body.append(Return(self.expression()))
            else:
                body.append(self.expression())
        self.eat("RBRACE"); return Lambda(params, body)
    def evalexpr(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); s = self.eat("STRING").val; self.eat("RPAREN")
        try:
            value = eval(s.strip("\"'"), {"__builtins__": {}})
        except Exception:
            value = 0
        return Eval(Number(str(value)))
    def enumdef(self):
        self.eat("KEYWORD"); name = self.eat("ID").val; self.eat("LBRACE")
        members = {}; idx = 0
        while self.peek() and self.peek().kind != "RBRACE":
            key = self.eat("ID").val
            if self.peek() and self.peek().kind == "OP" and self.peek().val == "=":
                self.eat("OP"); val = int(self.eat("NUMBER").val); members[key] = val; idx = val + 1
            else:
                members[key] = idx; idx += 1
            if self.peek() and self.peek().kind == "COMMA": self.eat("COMMA")
        self.eat("RBRACE"); return Enum(name, members)
    def ifstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond = self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); then_body = []
        while self.peek() and self.peek().kind != "RBRACE": then_body.append(self.expression())
        self.eat("RBRACE")
        else_body = None
        if self.peek() and self.peek().val == "else":
            self.eat("KEYWORD"); self.eat("LBRACE"); else_body=[]
            while self.peek() and self.peek().kind != "RBRACE": else_body.append(self.expression())
            self.eat("RBRACE")
        return If(cond, then_body, else_body)
    def whilestmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond = self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); body = []
        while self.peek() and self.peek().kind != "RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return While(cond, body)
    def forstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); init = self.expression(); self.eat("SEMI")
        cond = self.expression(); self.eat("SEMI"); step = self.expression()
        self.eat("RPAREN"); self.eat("LBRACE"); body = []
        while self.peek() and self.peek().kind != "RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return For(init, cond, step, body)
    def parallelblock(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind != "RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return Parallel(body)
    def trycatch(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); try_body=[]
        while self.peek() and self.peek().kind != "RBRACE": try_body.append(self.expression())
        self.eat("RBRACE"); self.eat("KEYWORD"); self.eat("LBRACE"); catch_body=[]
        while self.peek() and self.peek().kind != "RBRACE": catch_body.append(self.expression())
        self.eat("RBRACE"); return TryCatch(try_body, catch_body)

# -------------------------
# Hot-Swap Registry
# -------------------------
class HotSwapRegistry:
    def __init__(self):
        # map symbol key -> FuncDef
        self.table: Dict[str, FuncDef] = {}
        self.lock = threading.Lock()

    def register(self, key: str, func: FuncDef):
        with self.lock:
            self.table[key] = func

    def get(self, key: str):
        with self.lock:
            return self.table.get(key)

    def swap(self, key: str, new_func: FuncDef):
        with self.lock:
            old = self.table.get(key)
            self.table[key] = new_func
        print(f"[HOTSWAP] {key}: {'replaced' if old else 'registered'}")
        return old

# -------------------------
# Optimizer (same)
# -------------------------
def optimize(node):
    if isinstance(node, Program):
        node.body = [optimize(s) for s in node.body]
        return node
    if isinstance(node, FuncDef):
        node.body = [optimize(s) for s in node.body]
        return node
    if isinstance(node, Return):
        node.expr = optimize(node.expr)
        return node
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

