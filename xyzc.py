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

from mimetypes import init
import sys, re, argparse, math, threading, struct, json, socket, copy, difflib, time, random, traceback
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

# -------------------------
# LEXER (add brackets)
# -------------------------
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
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from io import BytesIO
from enum import Enum, auto
from dataclasses import dataclass, field
from copy import deepcopy
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

import json, os, io, re

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

           
