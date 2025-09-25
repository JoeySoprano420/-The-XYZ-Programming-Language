#!/usr/bin/env python3
"""
xyz_practice.py - Production-grade consolidated XYZ compiler + runtimes.

Pipelines:
  1) ProCompiler: ProLexer → ProParser → TypeChecker → IRBuilder → ExecutionEngine
  2) Legacy Parser + Optimizer + Codegen (assembly + minimal object + linker)
  3) FastRuntime: AST → Bytecode compiler + FastVM
  4) MiniRuntime: AST interpreter + hot-swap

Optional features:
  - Mega features: struct/type registry + list_add + vector op autogen
    via enable_mega_features(hot_registry, mini_runtime, fast_runtime=None)

Grand resolution (env: XYZ_GRAND_EXECUTE=1):
  Attempts ProCompiler → FastRuntime → MiniRuntime.
  Returns a diagnostic dict. If XYZ_GRAND_STRICT=1 and all fail,
  raises GrandResolutionError.
"""

from __future__ import annotations
import re, math, json, threading, logging, numbers
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG = logging.getLogger("xyz")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class GrandResolutionError(Exception):
    def __init__(self, message: str, diagnostic: Dict[str, Any]):
        super().__init__(message)
        self.diagnostic = diagnostic

class SemanticError(Exception): ...
class TypeErrorXYZ(Exception): ...

# ---------------------------------------------------------------------------
# Type & Struct registries
# ---------------------------------------------------------------------------
class MegaType:
    def __init__(self, name: str, meta: Dict[str, Any] = None):
        self.name = name
        self.meta = meta or {}
    def __repr__(self): return f"MegaType({self.name})"

class TypeRegistry:
    _types: Dict[str, MegaType] = {}
    @classmethod
    def register(cls, t: MegaType): cls._types[t.name] = t
    @classmethod
    def get(cls, name: str): return cls._types.get(name)
    @classmethod
    def list_types(cls): return list(cls._types.keys())

for _p in ("Int8","Int16","Int32","Int64","UInt8","UInt16","UInt32","UInt64","BigInt",
           "Float32","Float64","Decimal","Complex","Bool","String","Any"):
    TypeRegistry.register(MegaType(_p))

class StructDef:
    def __init__(self, name: str, fields: List[Tuple[str,str]]):
        self.name = name
        self.fields = fields
    def __repr__(self): return f"StructDef({self.name}, fields={self.fields})"

class StructRegistry:
    _structs: Dict[str, StructDef] = {}
    @classmethod
    def register(cls, s: StructDef): cls._structs[s.name] = s
    @classmethod
    def get(cls, name: str): return cls._structs.get(name)
    @classmethod
    def dump(cls): return dict(cls._structs)

def make_struct_instance(name: str, **kw):
    sdef = StructRegistry.get(name)
    if not sdef: raise ValueError(f"Unknown struct {name}")
    inst = {"__struct__": name}
    for fname,_ in sdef.fields:
        inst[fname] = kw.get(fname)
    return inst

# ---------------------------------------------------------------------------
# Legacy AST
# ---------------------------------------------------------------------------
class ASTNode: ...
class Program(ASTNode):
    def __init__(self, body: List[ASTNode]): self.body = body
class FuncDef(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name, self.params, self.body = name, params, body
class Call(ASTNode):
    def __init__(self, name: str, args: List[ASTNode]): self.name, self.args = name, args
class Return(ASTNode):
    def __init__(self, expr: ASTNode): self.expr = expr
class Number(ASTNode):
    def __init__(self, val: str):
        self.raw = val
        self.val = float(val) if "." in val else int(val)
class Var(ASTNode):
    def __init__(self, name: str): self.name = name
class Assign(ASTNode):
    def __init__(self, name: str, expr: ASTNode): self.name, self.expr = name, expr
class BinOp(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode): self.op, self.left, self.right = op, left, right
class UnaryOp(ASTNode):
    def __init__(self, op: str, operand: ASTNode): self.op, self.operand = op, operand
class Bool(ASTNode):
    def __init__(self, val: bool): self.val = val
class Null(ASTNode): ...
class If(ASTNode):
    def __init__(self, cond, then_body, else_body=None): self.cond, self.then_body, self.else_body = cond, then_body, else_body
class While(ASTNode):
    def __init__(self, cond, body): self.cond, self.body = cond, body
class For(ASTNode):
    def __init__(self, init, cond, step, body): self.init, self.cond, self.step, self.body = init, cond, step, body
class Lambda(ASTNode):
    def __init__(self, params, body): self.params, self.body = params, body
class ListLiteral(ASTNode):
    def __init__(self, elements: List[ASTNode]): self.elements = elements
class MapLiteral(ASTNode):
    def __init__(self, pairs: List[Tuple[ASTNode,ASTNode]]): self.pairs = pairs
class Index(ASTNode):
    def __init__(self, base, index): self.base, self.index = base, index
class Parallel(ASTNode):
    def __init__(self, body): self.body = body
class TryCatch(ASTNode):
    def __init__(self, try_body, catch_body): self.try_body, self.catch_body = try_body, catch_body
class Throw(ASTNode):
    def __init__(self, expr): self.expr = expr
class Pragma(ASTNode):
    def __init__(self, directive): self.directive = directive
class Enum(ASTNode):
    def __init__(self, name, members): self.name, self.members = name, members
class Isolate(ASTNode):
    def __init__(self, body): self.body = body
class Force(ASTNode):
    def __init__(self, body): self.body = body
class Remove(ASTNode):
    def __init__(self, body): self.body = body

# ---------------------------------------------------------------------------
# Hot Swap
# ---------------------------------------------------------------------------
class HotSwapRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self.table: Dict[str, FuncDef] = {}
    def register(self, key: str, func: FuncDef):
        if not isinstance(func, FuncDef):
            raise TypeError("HotSwapRegistry.register expects FuncDef")
        if "/" in key:
            n, a = key.rsplit("/",1)
            if n != func.name or int(a) != len(func.params):
                raise ValueError(f"Key/function arity mismatch: {key} vs {func.name}/{len(func.params)}")
        else:
            key = f"{func.name}/{len(func.params)}"
        with self._lock:
            self.table[key] = func
    def get(self, key: str) -> Optional[FuncDef]:
        with self._lock:
            return self.table.get(key)
    def swap(self, key: str, new_func: FuncDef):
        if "/" in key:
            n,a = key.rsplit("/",1)
            if n != new_func.name or int(a) != len(new_func.params):
                raise ValueError("swap arity mismatch")
        with self._lock:
            old = self.table.get(key)
            self.table[key] = new_func
        LOG.info("Hot-swap %s (%s)", key, "replaced" if old else "registered")
        return old

# ---------------------------------------------------------------------------
# Macros & Self-expander (lightweight)
# ---------------------------------------------------------------------------
class MacroEngine:
    def __init__(self): self.macros: Dict[str, Callable[[Call], List[ASTNode]]] = {}
    def register_macro(self, name: str, fn: Callable[[Call], List[ASTNode]]):
        self.macros[name] = fn
    def expand(self, node: ASTNode):
        if isinstance(node, Call) and node.name in self.macros:
            try: return self.macros[node.name](node)
            except Exception as e:
                LOG.warning("Macro expansion error for %s: %s", node.name, e)
        return [node]
_global_macro_engine = MacroEngine()

class SelfExpander:
    def __init__(self, hot: HotSwapRegistry, symtab: Dict[str, FuncDef]):
        self.hot = hot; self.symtab=symtab
    def _mk(self,name,params,body):
        f=FuncDef(name,params,body)
        k=f"{name}/{len(params)}"; self.hot.register(k,f); self.symtab[k]=f
    def generate_vector_ops(self, vec_name: str, dims:int=3):
        params=[f"x{i}" for i in range(dims)]
        self._mk(f"{vec_name}.new", params,
                 [Assign(f"f{i}", Var(p)) for i,p in enumerate(params)] +
                 [Return(ListLiteral([Var(p) for p in params]))])
        self._mk(f"{vec_name}.add", ["a","b"],
                 [Return(Call("list_add",[Var("a"),Var("b")]))])

# ---------------------------------------------------------------------------
# list_add + mega features
# ---------------------------------------------------------------------------
def list_add(a,b):
    if not isinstance(a,list) or not isinstance(b,list):
        raise TypeError("list_add requires lists")
    n=max(len(a),len(b)); out=[]
    for i in range(n):
        va = a[i] if i < len(a) else 0
        vb = b[i] if i < len(b) else 0
        if isinstance(va,list) or isinstance(vb,list):
            if not isinstance(va,list): va=[va]
            if not isinstance(vb,list): vb=[vb]
            out.append(list_add(va,vb)); continue
        if isinstance(va,(int,float,numbers.Number)) and isinstance(vb,(int,float,numbers.Number)):
            out.append(va+vb)
        elif isinstance(va,str) or isinstance(vb,str):
            out.append(str(va)+str(vb))
        else:
            try: out.append(va+vb)
            except Exception: out.append((va,vb))
    return out

def enable_mega_features(hot: HotSwapRegistry, mini_runtime: Any, fast_runtime: Optional[Any]=None):
    StructRegistry.register(StructDef("Point2D",[("x","Float64"),("y","Float64")]))
    StructRegistry.register(StructDef("Rect",[("min","Point2D"),("max","Point2D")]))
    hot.register("list_add/2", FuncDef("list_add", ["a","b"], [Return(Number("0"))]))
    mini_runtime._mega_builtins["list_add"] = list_add
    if fast_runtime:
        fast_runtime.vm.globals["list_add"] = list_add
    LOG.info("Mega features enabled (types=%d, structs=%d)", len(TypeRegistry.list_types()), len(StructRegistry.dump()))
    return True

# ---------------------------------------------------------------------------
# Lexer / Parser (legacy)
# ---------------------------------------------------------------------------
TOKEN_SPEC = [
    ("NUMBER", r"-?\d+(\.\d+)?"),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("STRING", r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''),
    ("PRAGMA", r"#pragma[^\n]*"),
    ("OP", r"[+\-*/=<>!&|%^.^]+"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"),
    ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("LBRACK", r"\["), ("RBRACK", r"\]"),
    ("SEMI", r";"), ("COMMA", r","),
    ("KEYWORD", r"\b(func|return|if|else|while|for|lambda|true|false|null|parallel|enum|eval|try|catch|throw|alloc|free|print|isolate|force|remove)\b"),
    ("WS", r"\s+"),
]
class Token:
    def __init__(self, kind, val, pos): self.kind, self.val, self.pos = kind,val,pos
def lex(src: str):
    pos=0; out=[]
    while pos < len(src):
        for k,p in TOKEN_SPEC:
            m=re.match(p,src[pos:])
            if m:
                if k!="WS": out.append(Token(k,m.group(),pos))
                pos+=len(m.group()); break
        else:
            raise SyntaxError(f"Unexpected char {src[pos]!r} at {pos}")
    return out

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens=tokens; self.pos=0; self.functions: Dict[str, FuncDef] = {}
    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def eat(self, kind=None):
        t=self.peek()
        if not t: raise SyntaxError("EOF")
        if kind and t.kind!=kind: raise SyntaxError(f"Expected {kind}, got {t.kind}")
        self.pos+=1; return t
    def parse(self) -> Program:
        return Program(self.statements())
    def statements(self):
        out=[]
        while self.peek():
            if self.peek().val=="func": out.append(self.funcdef())
            elif self.peek().kind=="PRAGMA": out.append(Pragma(self.eat("PRAGMA").val))
            else:
                e=self.expression()
                if e is not None: out.append(e)
                else: self.pos+=1
        return out
    def funcdef(self):
        self.eat("KEYWORD"); name=self.eat("ID").val; self.eat("LPAREN")
        params=[]
        while self.peek() and self.peek().kind!="RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
        self.eat("RPAREN"); self.eat("LBRACE")
        body=[]
        while self.peek() and self.peek().kind!="RBRACE":
            if (self.peek().kind=="ID" and self.pos+1 < len(self.tokens)
                and self.tokens[self.pos+1].kind=="OP" and self.tokens[self.pos+1].val=="="):
                var=self.eat("ID").val; self.eat("OP"); body.append(Assign(var,self.expression()))
            elif self.peek().val=="return":
                self.eat("KEYWORD"); body.append(Return(self.expression()))
            elif self.peek().val=="if": body.append(self.ifstmt())
            elif self.peek().val=="while": body.append(self.whilestmt())
            elif self.peek().val=="for": body.append(self.forstmt())
            elif self.peek().val=="parallel": body.append(self.parallelblock())
            elif self.peek().val=="try": body.append(self.trycatch())
            elif self.peek().val=="throw":
                self.eat("KEYWORD"); body.append(Throw(self.expression()))
            else:
                e=self.expression()
                if e is not None: body.append(e)
                else: self.pos+=1
        self.eat("RBRACE")
        f=FuncDef(name,params,body)
        self.functions[f"{name}/{len(params)}"]=f
        return f
    def expression(self): return self.parse_addsub()
    def parse_addsub(self):
        l=self.parse_muldiv()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("+","-"):
            op=self.eat("OP").val; r=self.parse_muldiv(); l=BinOp(op,l,r)
        return l
    def parse_muldiv(self):
        l=self.parse_pow()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("*","/"):
            op=self.eat("OP").val; r=self.parse_pow(); l=BinOp(op,l,r)
        return l
    def parse_pow(self):
        l=self.parse_unary()
        while self.peek() and self.peek().kind=="OP" and self.peek().val=="^":
            self.eat("OP"); r=self.parse_unary(); l=BinOp("^",l,r)
        return l
    def parse_unary(self):
        t=self.peek()
        if not t: return None
        if t.kind=="OP" and t.val=="-":
            self.eat("OP"); return UnaryOp("-", self.parse_unary())
        if t.kind=="NUMBER": return Number(self.eat("NUMBER").val)
        if t.kind=="KEYWORD" and t.val=="true": self.eat("KEYWORD"); return Bool(True)
        if t.kind=="KEYWORD" and t.val=="false": self.eat("KEYWORD"); return Bool(False)
        if t.kind=="KEYWORD" and t.val=="null": self.eat("KEYWORD"); return Null()
        if t.kind=="LBRACK":
            self.eat("LBRACK"); elems=[]
            while self.peek() and self.peek().kind!="RBRACK":
                elems.append(self.expression())
                if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
            self.eat("RBRACK"); return ListLiteral(elems)
        if t.kind=="ID":
            name=self.eat("ID").val
            while self.peek() and self.peek().kind=="OP" and self.peek().val==".":
                self.eat("OP"); name=f"{name}.{self.eat('ID').val}"
            if self.peek() and self.peek().kind=="LPAREN":
                self.eat("LPAREN"); args=[]
                while self.peek() and self.peek().kind!="RPAREN":
                    args.append(self.expression())
                    if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
                self.eat("RPAREN"); return Call(name,args)
            return Var(name)
        if t.kind=="LPAREN":
            self.eat("LPAREN"); e=self.expression(); self.eat("RPAREN"); return e
        return None
    def ifstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond=self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); tb=[]
        while self.peek() and self.peek().kind!="RBRACE": tb.append(self.expression())
        self.eat("RBRACE"); eb=None
        if self.peek() and self.peek().val=="else":
            self.eat("KEYWORD"); self.eat("LBRACE"); eb=[]
            while self.peek() and self.peek().kind!="RBRACE": eb.append(self.expression())
            self.eat("RBRACE")
        return If(cond,tb,eb)
    def whilestmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN"); cond=self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return While(cond,body)
    def forstmt(self):
        self.eat("KEYWORD"); self.eat("LPAREN")
        init=self.expression(); self.eat("SEMI")
        cond=self.expression(); self.eat("SEMI")
        step=self.expression(); self.eat("RPAREN")
        self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return For(init,cond,step,body)
    def parallelblock(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE"); return Parallel(body)
    def trycatch(self):
        self.eat("KEYWORD"); self.eat("LBRACE"); tb=[]
        while self.peek() and self.peek().kind!="RBRACE": tb.append(self.expression())
        self.eat("RBRACE"); self.eat("KEYWORD"); self.eat("LBRACE"); cb=[]
        while self.peek() and self.peek().kind!="RBRACE": cb.append(self.expression())
        self.eat("RBRACE"); return TryCatch(tb,cb)

# --- Pro AST + Pro Parser (minimal, for ProCompiler) ---

@dataclass
class PNode: ...
@dataclass
class PProgram(PNode): body: List[PNode]=field(default_factory=list)
@dataclass
class PFunc(PNode): name: str; params: List[str]; body: List[PNode]
@dataclass
class PNumber(PNode): raw: str
@dataclass
class PBool(PNode): val: bool
@dataclass
class PVar(PNode): name: str
@dataclass
class PBinOp(PNode): op: str; left: PNode; right: PNode
@dataclass
class PReturn(PNode): expr: Optional[PNode]

class ProLexer:
    spec=[('NUMBER',r'-?\d+(\.\d+)?'),('ID',r'[A-Za-z_][A-Za-z0-9_]*'),
          ('OP',r'[\+\-\*/\^=]+'),('LP',r'\('),('RP',r'\)'),('WS',r'\s+'),('X',r'.*?')
         ]
    rx=re.compile("|".join(f"(?P<{n}>{p})" for n,p in spec))
    kws={"func","return","true","false"}
    def __init__(self, src:str):
        self.t=[]
        for m in self.rx.finditer(src):
            k=m.lastgroup; v=m.group()
            if k=="WS": continue
            if k=="ID" and v in self.kws: k=v.upper()
            self.t.append((k,v))
        self.i=0
    def peek(self): return self.t[self.i] if self.i<len(self.t) else ("EOF","")
    def next(self): t=self.peek(); self.i+=1; return t
    def accept(self,k): 
        if self.peek()[0]==k: return self.next()
        return None
    def expect(self,k):
        t=self.next()
        if t[0]!=k: raise SyntaxError(f"Expected {k}, got {t}")
        return t

class ProParser:
    def __init__(self, src:str): self.lex=ProLexer(src)
    def parse(self)->PProgram:
        body=[]
        while self.lex.peek()[0]!="EOF":
            if self.lex.peek()[0]=="func": body.append(self.func())
            else: self.lex.next()
        return PProgram(body)
    def func(self)->PFunc:
        self.lex.expect("func"); _,name=self.lex.expect("ID")
        self.lex.expect("LP"); params=[]
        if self.lex.peek()[0]!="RP":
            while True:
                _,pid=self.lex.expect("ID"); params.append(pid)
                if not self.lex.accept("OP") and not self.lex.accept("COMMA"): break
        self.lex.accept("COMMA"); self.lex.expect("RP")
        # parse single-return or simple body inside braces
        if self.lex.peek()[0]!="X" or self.lex.peek()[1]!="{": self.lex.next()
        else: self.lex.next()
        body=[]
        while not (self.lex.peek()[0]=="X" and self.lex.peek()[1]=="}"):
            if self.lex.peek()[0]=="return":
                self.lex.next(); body.append(PReturn(self.expr()))
            else:
                body.append(self.expr())
        self.lex.next()
        return PFunc(name, params, body)
    def expr(self, rbp=0):
        k,v=self.lex.next(); left=self.nud(k,v)
        while rbp < self.lbp(self.lex.peek()):
            k,v=self.lex.next(); left=self.led(k,v,left)
        return left
    def nud(self,k,v):
        if k=="NUMBER": return PNumber(v)
        if k=="true": return PBool(True)
        if k=="false": return PBool(False)
        if k=="ID": return PVar(v)
        if k=="LP":
            e=self.expr(); self.lex.expect("RP"); return e
        raise SyntaxError("Unexpected token")
    def lbp(self,p): return 10 if p[0]=="OP" else 0
    def led(self,k,v,left):
        if k=="OP":
            r=self.expr(10); return PBinOp(v,left,r)
        raise SyntaxError("Bad infix")

# --- Minimal Codegen (used in CLI) ---
class Codegen:
    SYSCALL_MAP={"read":0,"write":1,"open":2,"close":3,"exit":60,"getpid":39}
    def __init__(self,symtab:Dict[str,FuncDef],hot:HotSwapRegistry):
        self.symtab=symtab; self.hot=hot; self.asm=[]; self._lbl=0
    def newlabel(self,p="L"): self._lbl+=1; return f"{p}{self._lbl}"
    def emit(self,s): self.asm.append(s)
    def generate(self, prog: Program) -> str:
        self.emit("section .text"); self.emit("global _start"); self.emit("_start:")
        self.emit("  call main/0"); self.emit("  mov rax, 60"); self.emit("  xor rdi, rdi"); self.emit("  syscall")
        prog=optimize(prog)
        for s in prog.body: self.gen_stmt(s)
        return "\n".join(self.asm)
    def gen_safe_div(self):
        dz=self.newlabel("divz"); ed=self.newlabel("div_end")
        self.emit("  cmp rbx, 0"); self.emit(f"  je {dz}")
        self.emit("  cqo"); self.emit("  idiv rbx"); self.emit(f"  jmp {ed}")
        self.emit(f"{dz}:"); self.emit("  mov rax, 0"); self.emit(f"{ed}:")
    def gen_stmt(self,n:ASTNode):
        if isinstance(n,FuncDef):
            k=f"{n.name}/{len(n.params)}"; self.symtab[k]=n; self.hot.register(k,n)
            self.emit(f"{k}:"); 
            for s in n.body: self.gen_stmt(s)
            self.emit("  ret"); return
        if isinstance(n,Return):
            self.gen_stmt(n.expr); self.emit("  ret"); return
        if isinstance(n,Number):
            self.emit(f"  mov rax, {int(n.val)}"); return
        if isinstance(n,UnaryOp):
            self.gen_stmt(n.operand); 
            if n.op=="-": self.emit("  neg rax"); return
        if isinstance(n,Assign):
            self.gen_stmt(n.expr); self.emit(f"  ; assign {n.name} = rax"); return
        if isinstance(n,Var):
            self.emit(f"  ; var {n.name}"); return
        if isinstance(n,Call):
            k=f"{n.name}/{len(n.args)}"
            if n.name in self.SYSCALL_MAP and k not in self.symtab:
                regs=["rdi","rsi","rdx","rcx","r8","r9"]
                for i,a in enumerate(n.args):
                    self.gen_stmt(a)
                    if i < len(regs): self.emit(f"  mov {regs[i]}, rax")
                if len(n.args) > len(regs):
                    self.emit(f"  ; WARN: extra syscall args truncated")
                self.emit(f"  mov rax, {self.SYSCALL_MAP[n.name]}"); self.emit("  syscall"); return
            for i,a in enumerate(n.args):
                self.gen_stmt(a); self.emit("  push rax")
            self.emit(f"  call {k if k in self.symtab else n.name}")
            if n.args: self.emit(f"  add rsp, {len(n.args)*8}")
            return
        if isinstance(n,BinOp):
            self.gen_stmt(n.left); self.emit("  push rax")
            self.gen_stmt(n.right); self.emit("  mov rbx, rax"); self.emit("  pop rax")
            if n.op=="+": self.emit("  add rax, rbx")
            elif n.op=="-": self.emit("  sub rax, rbx")
            elif n.op=="*": self.emit("  imul rax, rbx")
            elif n.op=="/": self.gen_safe_div()
            elif n.op=="^":
                loop=self.newlabel("pow"); end=self.newlabel("pow_end")
                self.emit("  mov rcx, rbx"); self.emit("  mov rbx, rax"); self.emit("  mov rax,1")
                self.emit(f"{loop}:"); self.emit("  cmp rcx,0"); self.emit(f"  je {end}")
                self.emit("  imul rax, rbx"); self.emit("  dec rcx"); self.emit(f"  jmp {loop}")
                self.emit(f"{end}:")
            return
        if isinstance(n,If):
            el=self.newlabel("else"); en=self.newlabel("endif")
            self.gen_stmt(n.cond); self.emit("  cmp rax,0"); self.emit(f"  je {el}")
            for s in n.then_body: self.gen_stmt(s)
            self.emit(f"  jmp {en}"); self.emit(f"{el}:")
            if n.else_body:
                for s in n.else_body: self.gen_stmt(s)
            self.emit(f"{en}:"); return
        if isinstance(n,TryCatch):
            self.emit("  ; try/catch (no-op)"); 
            for s in n.try_body: self.gen_stmt(s)
            for s in n.catch_body: self.gen_stmt(s); return
        if isinstance(n,Parallel):
            self.emit("  ; parallel (annotation)"); 
            for s in n.body: self.gen_stmt(s); return
        self.emit(f"  ; unhandled {type(n).__name__}")

def write_object_from_asm(asm: str, path: str):
    symbols={}; cur=None; lines=[]
    for ln in asm.splitlines():
        if ln.endswith(":") and not ln.startswith(" "):
            if cur: symbols[cur]=lines
            cur=ln[:-1]; lines=[]
        else:
            if cur: lines.append(ln)
    if cur: symbols[cur]=lines
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"format":"XYZOBJ1","symbols":symbols,"raw":asm}, f)
    LOG.info("Object -> %s", path)

def link_objects(obj_paths: List[str], out_path: str):
    merged={}
    for p in obj_paths:
        with open(p,"r",encoding="utf-8") as f: data=json.load(f)
        for k,v in data.get("symbols",{}).items():
            if k not in merged: merged[k]=v
    lines=["section .text"]
    for sym,body in merged.items():
        lines.append(f"{sym}:"); lines.extend(body)
        if not any(b.strip().startswith("ret") for b in body):
            lines.append("  ret")
    open(out_path,"w",encoding="utf-8").write("\n".join(lines))
    LOG.info("Linked -> %s", out_path)

# --- MiniRuntime (interpreter) ---
class Closure:
    def __init__(self, params, body, frames_ref: List[Dict[str,Any]]):
        self.params=params; self.body=body; self.frames_ref=frames_ref

class MiniRuntime:
    def __init__(self, symtab: Dict[str,FuncDef], hot: HotSwapRegistry):
        self.symtab=symtab; self.hot=hot; self.frames: List[Dict[str,Any]]=[]; self._mega_builtins={}
    def push(self,f=None): self.frames.append(f or {})
    def pop(self): 
        if self.frames: self.frames.pop()
    def run_func(self,key:str,args:List[Any]):
        f=self.hot.get(key) or self.symtab.get(key)
        if not f: raise KeyError(key)
        frame={p:(args[i] if i < len(args) else None) for i,p in enumerate(f.params)}
        self.push(frame); result=None
        for st in f.body:
            result=self.eval(st)
            if isinstance(st,Return):
                result=self.eval(st.expr); break
        self.pop(); return result
    def eval(self,n:ASTNode):
        if n is None: return None
        if isinstance(n,Number): return n.val
        if isinstance(n,Bool): return 1 if n.val else 0
        if isinstance(n,Null): return None
        if isinstance(n,Var):
            for fr in reversed(self.frames):
                if n.name in fr: return fr[n.name]
            return None
        if isinstance(n,Assign):
            val=self.eval(n.expr); self.frames[-1][n.name]=val; return val
        if isinstance(n,UnaryOp):
            v=self.eval(n.operand); return -v if n.op=="-" else v
        if isinstance(n,BinOp):
            l=self.eval(n.left); r=self.eval(n.right)
            return {"+":l+r,"-":l-r,"*":l*r,"/":(0 if r==0 else l/r),"^":int(math.pow(l,r))}.get(n.op)
        if isinstance(n,ListLiteral):
            return [self.eval(e) for e in n.elements]
        if isinstance(n,Call):
            if n.name in self._mega_builtins:
                return self._mega_builtins[n.name](*[self.eval(a) for a in n.args])
            if n.name=="print":
                vals=[self.eval(a) for a in n.args]; print(*vals); return None
            key=f"{n.name}/{len(n.args)}"
            if self.hot.get(key) or self.symtab.get(key):
                return self.run_func(key,[self.eval(a) for a in n.args])
            raise RuntimeError(f"Call target not found: {key}")
        if isinstance(n,Lambda): return Closure(n.params,n.body,self.frames)
        if isinstance(n,If):
            cond=self.eval(n.cond); body=n.then_body if cond else (n.else_body or [])
            res=None
            for s in body: res=self.eval(s)
            return res
        if isinstance(n,While):
            res=None
            while self.eval(n.cond):
                for s in n.body: res=self.eval(s)
            return res
        if isinstance(n,TryCatch):
            try:
                for s in n.try_body: self.eval(s)
            except Exception:
                for s in n.catch_body: self.eval(s)
            return None
        if isinstance(n,Return): return self.eval(n.expr)
        if isinstance(n,Throw): raise RuntimeError(self.eval(n.expr))
        return None

# --- FastRuntime (bytecode) ---
from enum import IntEnum
class Op(IntEnum):
    CONST=0; LOADL=1; STOREL=2; CALL=3; RET=4; ADD=5; SUB=6; MUL=7; DIV=8; POW=9; NOP=10
class BytecodeFunction:
    def __init__(self,name,params,code,consts): self.name=name; self.params=params; self.code=code; self.consts=consts
class FastCompiler:
    def __init__(self,symtab:Dict[str,FuncDef]): self.symtab=symtab
    def compile_all(self)->Dict[str,BytecodeFunction]:
        out={}
        for k,f in self.symtab.items(): out[k]=self.compile_func(k,f)
        return out
    def compile_func(self,key,f:FuncDef):
        consts=[]; code=[]; lm={p:i for i,p in enumerate(f.params)}
        def addc(v): 
            try: return consts.index(v)
            except ValueError: consts.append(v); return len(consts)-1
        def emit(op,*ops): code.append(op); code.extend(ops)
        def comp(n):
            if isinstance(n,Number): emit(Op.CONST, addc(n.val))
            elif isinstance(n,UnaryOp): comp(n.operand); emit(Op.CONST, addc(-1)); emit(Op.MUL)
            elif isinstance(n,Var): emit(Op.LOADL, lm.get(n.name, addc(None)))
            elif isinstance(n,BinOp):
                comp(n.left); comp(n.right); emit({"+":Op.ADD,"-":Op.SUB,"*":Op.MUL,"/":Op.DIV,"^":Op.POW}.get(n.op,Op.NOP))
            elif isinstance(n,Assign):
                comp(n.expr); idx=lm.setdefault(n.name,len(lm)); emit(Op.STOREL, idx)
            elif isinstance(n,Call):
                for a in n.args: comp(a); emit(Op.CALL, addc(f"{n.name}/{len(n.args)}"))
            elif isinstance(n,Return): comp(n.expr); emit(Op.RET)
            else: emit(Op.NOP)
        for st in f.body: comp(st)
        if not code or code[-1] != Op.RET: emit(Op.CONST, addc(None)); emit(Op.RET)
        return BytecodeFunction(key,f.params,code,consts)
class FastVM:
    def __init__(self, funcs: Dict[str,BytecodeFunction], hot: HotSwapRegistry):
        self.funcs=funcs; self.hot=hot; self.globals={"print":print,"list_add":list_add}; self.max_depth=512
    def run(self,key:str,args:List[Any]=None,depth=0):
        if depth > self.max_depth: raise RecursionError("max depth")
        if key not in self.funcs: raise KeyError(key)
        fn=self.funcs[key]; consts=fn.consts; code=fn.code
        locals_=[None]*max(16,len(fn.params)+4)
        if args:
            for i,p in enumerate(fn.params): locals_[i]=args[i] if i < len(args) else None
        stack=[]; pc=0
        while pc < len(code):
            op=code[pc]; pc+=1
            if op==Op.CONST: stack.append(consts[code[pc]]); pc+=1
            elif op==Op.LOADL: stack.append(locals_[code[pc]]); pc+=1
            elif op==Op.STOREL: locals_[code[pc]]=stack.pop(); pc+=1
            elif op==Op.ADD: b=stack.pop(); a=stack.pop(); stack.append(a+b)
            elif op==Op.SUB: b=stack.pop(); a=stack.pop(); stack.append(a-b)
            elif op==Op.MUL: b=stack.pop(); a=stack.pop(); stack.append(a*b)
            elif op==Op.DIV: b=stack.pop(); a=stack.pop(); stack.append(0 if b==0 else a/b)
            elif op==Op.POW: b=stack.pop(); a=stack.pop(); stack.append(int(math.pow(a,b)))
            elif op==Op.CALL:
                cname=consts[code[pc]]; pc+=1; ar=int(cname.rsplit("/",1)[1])
                argv=[stack.pop() for _ in range(ar)][::-1]
                if cname in self.funcs: stack.append(self.run(cname,argv,depth+1))
                elif cname.split("/")[0] in self.globals: stack.append(self.globals[cname.split('/')[0]](*argv))
                else: stack.append(None)
            elif op==Op.RET: return stack.pop() if stack else None
            else: pass
        return None
class FastRuntime:
    def __init__(self,symtab:Dict[str,FuncDef],hot:HotSwapRegistry):
        self.compiler=FastCompiler(symtab); self.functions=self.compiler.compile_all(); self.vm=FastVM(self.functions, hot)
    def run(self,key="main/0",args=None): return self.vm.run(key,args or [])


def optimize(node: ASTNode) -> ASTNode:
    """
    Basic AST optimizer: constant folding for +,-,*,/,^ and unary '-'.
    Also simplifies 'if' with constant boolean conditions.
    Returns the same node instance (mutated) for simplicity.
    """
    if node is None:
        return None

    if isinstance(node, Program):
        node.body = [optimize(n) for n in node.body]
        return node

    if isinstance(node, FuncDef):
        node.body = [optimize(n) for n in node.body]
        return node

    if isinstance(node, Return):
        node.expr = optimize(node.expr)
        return node

    if isinstance(node, UnaryOp):
        node.operand = optimize(node.operand)
        if node.op == "-" and isinstance(node.operand, Number):
            return Number(str(-node.operand.val))
        return node

    if isinstance(node, BinOp):
        node.left = optimize(node.left)
        node.right = optimize(node.right)
        if isinstance(node.left, Number) and isinstance(node.right, Number):
            a, b = node.left.val, node.right.val
            try:
                if node.op == "+": return Number(str(a + b))
                if node.op == "-": return Number(str(a - b))
                if node.op == "*": return Number(str(a * b))
                if node.op == "/": return Number(str(0 if b == 0 else a / b))
                if node.op == "^": return Number(str(int(math.pow(a, b))))
            except Exception:
                pass
        return node

    if isinstance(node, If):
        node.cond = optimize(node.cond)
        node.then_body = [optimize(n) for n in node.then_body]
        if node.else_body:
            node.else_body = [optimize(n) for n in node.else_body]
        if isinstance(node.cond, Bool):
            return Program(node.then_body if node.cond.val else (node.else_body or []))
        return node

    # default: return as-is
    return node

import argparse

def main_cli():
    ap = argparse.ArgumentParser(prog="xyzc", description="XYZ Compiler Toolchain")
    ap.add_argument("source", nargs="+", help=".xyz source files or .obj objects")
    ap.add_argument("-o", "--output", default="out.asm", help="Output file (for ASM or linked ASM)")
    ap.add_argument("--emit-asm", action="store_true", help="Emit NASM assembly from source")
    ap.add_argument("--emit-obj", action="store_true", help="Emit JSON object file")
    ap.add_argument("--link", action="store_true", help="Link object files into final ASM")
    ap.add_argument("--run", action="store_true", help="Run program after compiling")
    ap.add_argument("--runtime", choices=["mini","fast"], default="mini", help="Runtime to use when running")
    ap.add_argument("--mega", action="store_true", help="Enable Mega features")
    args = ap.parse_args()

    hot = HotSwapRegistry()
    symtab = {}

    # Linking mode: merge .obj files
    if args.link:
        LOG.info("Linking mode: %d object files", len(args.source))
        link_objects(args.source, args.output)
        return

    # Compile .xyz sources
    progs = []
    for srcpath in args.source:
        if srcpath.endswith(".xyz"):
            LOG.info("Parsing %s", srcpath)
            src = open(srcpath).read()
            toks = lex(src)
            parser = Parser(toks)
            prog = parser.parse()
            symtab.update(parser.functions)
            progs.append(prog)

    # Single combined program
    if not progs:
        LOG.error("No .xyz sources to compile")
        sys.exit(1)
    prog = Program(sum([p.body for p in progs], []))

    # Optimization
    prog = optimize(prog)

    # Codegen
    cg = Codegen(symtab, hot)
    asm = cg.generate(prog)

    # Emit ASM
    if args.emit_asm:
        open(args.output, "w").write(asm)
        LOG.info("Assembly written -> %s", args.output)

    # Emit Object
    if args.emit_obj:
        objpath = args.output if args.output.endswith(".obj") else args.output + ".obj"
        write_object_from_asm(asm, objpath)

    # Run
    if args.run:
        if args.mega:
            mini = MiniRuntime(symtab, hot)
            enable_mega_features(hot, mini)
        if args.runtime == "mini":
            rt = MiniRuntime(symtab, hot)
            res = rt.run_func("main/0", [])
            LOG.info("MiniRuntime result: %s", res)
        else:
            rt = FastRuntime(symtab, hot)
            res = rt.run("main/0", [])
            LOG.info("FastRuntime result: %s", res)


if __name__ == "__main__":
    main_cli()

class Codegen:
    SYSCALL_MAP = {"read":0,"write":1,"open":2,"close":3,"exit":60,"getpid":39}
    def __init__(self,symtab:Dict[str,FuncDef],hot:HotSwapRegistry):
        self.symtab = symtab
        self.hot = hot
        self.asm = []
        self._lbl = 0
        self.current_locals: Dict[str,int] = {}
        self.local_size = 0

    def newlabel(self,p="L"): 
        self._lbl += 1
        return f"{p}{self._lbl}"

    def emit(self,s): 
        self.asm.append(s)

    def generate(self, prog: Program) -> str:
        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main/0")
        self.emit("  mov rax, 60")
        self.emit("  xor rdi, rdi")
        self.emit("  syscall")

        prog = optimize(prog)
        for s in prog.body: 
            self.gen_stmt(s)

        return "\n".join(self.asm)

    def prologue(self):
        self.emit("  push rbp")
        self.emit("  mov rbp, rsp")
        if self.local_size > 0:
            self.emit(f"  sub rsp, {self.local_size}")

    def epilogue(self):
        if self.local_size > 0:
            self.emit(f"  add rsp, {self.local_size}")
        self.emit("  mov rsp, rbp")
        self.emit("  pop rbp")
        self.emit("  ret")

    def alloc_local(self, name: str) -> int:
        if name not in self.current_locals:
            self.local_size += 8
            self.current_locals[name] = self.local_size
        return self.current_locals[name]

    def gen_stmt(self, n: ASTNode):
        if isinstance(n, FuncDef):
            key = f"{n.name}/{len(n.params)}"
            self.symtab[key] = n
            self.hot.register(key,n)

            # reset locals for new function
            self.current_locals = {}
            self.local_size = 0

            self.emit(f"{key}:")
            self.prologue()
            for s in n.body: 
                self.gen_stmt(s)
            self.epilogue()
            return

        if isinstance(n, Return):
            self.gen_stmt(n.expr)
            self.epilogue()
            return

        if isinstance(n, Number):
            self.emit(f"  mov rax, {int(n.val)}")
            return

        if isinstance(n, UnaryOp):
            self.gen_stmt(n.operand)
            if n.op == "-": 
                self.emit("  neg rax")
            return

        if isinstance(n, Assign):
            self.gen_stmt(n.expr)
            offset = self.alloc_local(n.name)
            self.emit(f"  mov [rbp-{offset}], rax   ; {n.name}")
            return

        if isinstance(n, Var):
            offset = self.current_locals.get(n.name)
            if offset is not None:
                self.emit(f"  mov rax, [rbp-{offset}] ; {n.name}")
            else:
                self.emit(f"  ; var {n.name} not found")
            return

        if isinstance(n, Call):
            key = f"{n.name}/{len(n.args)}"
            for a in n.args[::-1]:   # push args right-to-left
                self.gen_stmt(a)
                self.emit("  push rax")
            self.emit(f"  call {key if key in self.symtab else n.name}")
            if n.args:
                self.emit(f"  add rsp, {len(n.args)*8}")
            return

        if isinstance(n, BinOp):
            self.gen_stmt(n.left)
            self.emit("  push rax")
            self.gen_stmt(n.right)
            self.emit("  mov rbx, rax")
            self.emit("  pop rax")
            if n.op == "+": self.emit("  add rax, rbx")
            elif n.op == "-": self.emit("  sub rax, rbx")
            elif n.op == "*": self.emit("  imul rax, rbx")
            elif n.op == "/": self.gen_safe_div()
            elif n.op == "^":
                loop = self.newlabel("pow"); end = self.newlabel("pow_end")
                self.emit("  mov rcx, rbx")
                self.emit("  mov rbx, rax")
                self.emit("  mov rax,1")
                self.emit(f"{loop}:")
                self.emit("  cmp rcx,0")
                self.emit(f"  je {end}")
                self.emit("  imul rax, rbx")
                self.emit("  dec rcx")
                self.emit(f"  jmp {loop}")
                self.emit(f"{end}:")
            return

        if isinstance(n, If):
            el=self.newlabel("else"); en=self.newlabel("endif")
            self.gen_stmt(n.cond)
            self.emit("  cmp rax,0")
            self.emit(f"  je {el}")
            for s in n.then_body: self.gen_stmt(s)
            self.emit(f"  jmp {en}")
            self.emit(f"{el}:")
            if n.else_body:
                for s in n.else_body: self.gen_stmt(s)
            self.emit(f"{en}:")
            return

        self.emit(f"  ; unhandled {type(n).__name__}")

    def gen_safe_div(self):
        dz = self.newlabel("divz")
        ed = self.newlabel("div_end")
        self.emit("  cmp rbx, 0")
        self.emit(f"  je {dz}")
        self.emit("  cqo")
        self.emit("  idiv rbx")
        self.emit(f"  jmp {ed}")
        self.emit(f"{dz}:")
        self.emit("  mov rax, 0")
        self.emit(f"{ed}:")

class Codegen:
    ...
    def generate(self, prog: Program) -> str:
        self.emit("section .bss")
        self.emit("print_buf resb 32")  # buffer for integer printing
        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main/0")
        self.emit("  mov rax, 60")   # exit syscall
        self.emit("  xor rdi, rdi")
        self.emit("  syscall")

        prog = optimize(prog)
        for s in prog.body: 
            self.gen_stmt(s)

        # Add print routine at the end if needed
        if getattr(self, "_needs_print", False):
            self.gen_print_func()

        return "\n".join(self.asm)

    def gen_stmt(self, n: ASTNode):
        ...
        if isinstance(n, Call):
            key = f"{n.name}/{len(n.args)}"
            if n.name == "print":
                # generate arg then call our print routine
                for a in n.args:
                    self.gen_stmt(a)
                self.emit("  call print/1")
                self._needs_print = True
                return
            ...
    
    def gen_print_func(self):
        """Emit an integer-only print function using write syscall."""
        self.emit("print/1:")
        self.prologue()
        # assume arg is in rax
        self.emit("  mov rcx, print_buf+31")   # point to end of buffer
        self.emit("  mov rbx, 10")
        self.emit("  mov rdx, 0")              # digit count
        self.emit("print_loop:")
        self.emit("  xor rdx, rdx")
        self.emit("  div rbx")                 # divide rax by 10
        self.emit("  add rdx, '0'")
        self.emit("  dec rcx")
        self.emit("  mov [rcx], dl")
        self.emit("  test rax, rax")
        self.emit("  jnz print_loop")
        self.emit("  mov rax, 1")              # syscall: write
        self.emit("  mov rdi, 1")              # fd=stdout
        self.emit("  mov rsi, rcx")
        self.emit("  mov rdx, print_buf+32 - rcx")
        self.emit("  syscall")
        self.epilogue()

        if isinstance(n, Call):
            key = f"{n.name}/{len(n.args)}"
            if n.name == "print":
                # generate arg then call our print routine
                for a in n.args:
                    self.gen_stmt(a)
                self.emit("  call print/1")
                self._needs_print = True
                return
            ...
            def gen_print_func(self):
                """Emit an integer-only print function using write syscall."""
                self.emit("print/1:")
                self.prologue()
                # assume arg is in rax
                self.emit("  mov rcx, print_buf+31")   # point to end of buffer
                self.emit("  mov rbx, 10")
                self.emit("  mov rdx, 0")              # digit count
                self.emit("print_loop:")
                self.emit("  xor rdx, rdx")
                self.emit("  div rbx")                 # divide rax by 10
                self.emit("  add rdx, '0'")
                self.emit("  dec rcx")
                self.emit("  mov [rcx], dl")
                self.emit("  test rax, rax")
                self.emit("  jnz print_loop")
                self.emit("  mov rax, 1")              # syscall: write
                self.emit("  mov rdi, 1")              # fd=stdout
                self.emit("  mov rsi, rcx")
                self.emit("  mov rdx, print_buf+32 - rcx")
                self.emit("  syscall")
                self.epilogue()

# add sys and os to imports (top of file)
import sys, os
import re, math, json, threading, logging, numbers
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("xyzc")
# --- AST Nodes ---
@dataclass
class ASTNode: ...
@dataclass
class Program(ASTNode): body: List[ASTNode]=field(default_factory=list)
@dataclass
class FuncDef(ASTNode): name: str; params: List[str]; body: List[ASTNode]
@dataclass
class Number(ASTNode): raw: str; val: float=field(init=False)

# replace main_cli() and __main__ block
import argparse

def main_cli() -> int:
    ap = argparse.ArgumentParser(prog="xyzc", description="XYZ Compiler Toolchain")
    ap.add_argument("source", nargs="*", help=".xyz source files (use '-' for stdin) or .obj objects for --link")
    ap.add_argument("-o", "--output", default="out.asm", help="Output file (ASM or linked ASM)")
    ap.add_argument("--emit-asm", action="store_true", help="Emit NASM assembly from source")
    ap.add_argument("--emit-obj", action="store_true", help="Emit JSON object file")
    ap.add_argument("--link", action="store_true", help="Link object files into final ASM")
    ap.add_argument("--run", action="store_true", help="Run program after compiling")
    ap.add_argument("--runtime", choices=["mini","fast"], default="mini", help="Runtime to use when running")
    ap.add_argument("--mega", action="store_true", help="Enable Mega features")
    args = ap.parse_args()

    hot = HotSwapRegistry()
    symtab: Dict[str, FuncDef] = {}

    # Linking mode
    if args.link:
        if not args.source:
            LOG.error("No object files provided for --link")
            ap.print_usage()
            return 2
        LOG.info("Linking %d object files", len(args.source))
        link_objects(args.source, args.output)
        return 0

    # Collect sources; allow auto-discovery or stdin
    sources = list(args.source)
    if not sources:
        for cand in ("main.xyz", "main.xy"):
            if os.path.isfile(cand):
                sources = [cand]
                LOG.info("Auto-discovered source: %s", cand)
                break
    if not sources and not sys.stdin.isatty():
        sources = ["-"]  # read from stdin
        LOG.info("Reading source from stdin (-)")

    if not sources:
        LOG.error("No sources provided. Provide a .xyz file, '-', or use --link with .obj files.")
        ap.print_usage()
        return 2

    # Parse all sources
    progs: List[Program] = []
    for srcpath in sources:
        if srcpath == "-":
            src = sys.stdin.read()
        elif srcpath.endswith((".xyz", ".xy")):
            with open(srcpath, "r", encoding="utf-8") as f:
                src = f.read()
        else:
            LOG.warning("Skipping non-source file: %s", srcpath)
            continue

        toks = lex(src)
        parser = Parser(toks)
        prog = parser.parse()
        symtab.update(parser.functions)
        progs.append(prog)

    if not progs:
        LOG.error("No valid .xyz sources parsed")
        return 2

    # Combine programs
    prog = Program([stmt for p in progs for stmt in p.body])

    # Optimize
    prog = optimize(prog)

    # Codegen
    cg = Codegen(symtab, hot)
    asm = cg.generate(prog)

    # Emit ASM
    if args.emit_asm:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(asm)
        LOG.info("Assembly written -> %s", args.output)

    # Emit Object
    if args.emit_obj:
        objpath = args.output if args.output.endswith(".obj") else args.output + ".obj"
        write_object_from_asm(asm, objpath)

    # Run (optional)
    if args.run:
        if args.mega:
            mini_tmp = MiniRuntime(symtab, hot)
            enable_mega_features(hot, mini_tmp)
        if args.runtime == "mini":
            rt = MiniRuntime(symtab, hot)
            res = rt.run_func("main/0", [])
            LOG.info("MiniRuntime result: %s", res)
        else:
            rt = FastRuntime(symtab, hot)
            res = rt.run("main/0", [])
            LOG.info("FastRuntime result: %s", res)

    return 0


if __name__ == "__main__":
    sys.exit(main_cli())

    @dataclass
    class Bool(ASTNode): val: bool
    @dataclass
    class Null(ASTNode): ...
    @dataclass
    class Var(ASTNode): name: str
    @dataclass
    class Assign(ASTNode): name: str; expr: ASTNode
    @dataclass
    class BinOp(ASTNode): left: ASTNode; op: str; right: ASTNode
    @dataclass
    class UnaryOp(ASTNode): op: str; operand: ASTNode
    @dataclass
    class Call(ASTNode): name: str; args: List[ASTNode]=field(default_factory=list)
    @dataclass
    class If(ASTNode): cond: ASTNode; then_body: List[ASTNode]=field(default_factory=list); else_body: Optional[List[ASTNode]]=None
    @dataclass
    class While(ASTNode): cond: ASTNode; body: List[ASTNode]=field(default_factory=list)
    @dataclass
    class TryCatch(ASTNode): try_body: List[ASTNode]=field(default_factory=list); catch_body: List[ASTNode]=field(default_factory=list)
    @dataclass
    class Throw(ASTNode): expr: ASTNode
    @dataclass
    class Return(ASTNode): expr: ASTNode
    @dataclass
    class ListLiteral(ASTNode): elements: List[ASTNode]=field(default_factory=list)
    @dataclass
    class Lambda(ASTNode): params: List[str]=field(default_factory=list); body: List[ASTNode]=field(default_factory=list)
    def gen_safe_div(self):
        dz = self.newlabel("divz")
        ed = self.newlabel("div_end")
        self.emit("  cmp rbx, 0"); self.emit(f"  je {dz}")
        self.emit("  cqo"); self.emit("  idiv rbx"); self.emit(f"  jmp {ed}")
        self.emit(f"{dz}:"); self.emit("  mov rax, 0"); self.emit(f"{ed}:")
        def gen_stmt(self, n: ASTNode):
            if n is None: return
            if isinstance(n, FuncDef):
                key = f"{n.name}/{len(n.params)}"
                self.symtab[key] = n
                self.hot.register(key, n)
                self.emit(f"{key}:")
                self.prologue()
                for s in n.body: self.gen_stmt(s)
                self.epilogue()
                return
            if isinstance(n, Return):
                self.gen_stmt(n.expr)
                self.epilogue()
                return
            if isinstance(n, Number):
                self.emit(f"  mov rax, {int(n.val)}"); return
            if isinstance(n, Bool):
                self.emit(f"  mov rax, {1 if n.val else 0}"); return
            if isinstance(n, Null):
                self.emit("  mov rax, 0"); return
            if isinstance(n, UnaryOp):
                self.gen_stmt(n.operand)
                if n.op == "-": self.emit("  neg rax")
                return
            if isinstance(n, Assign):
                self.gen_stmt(n.expr)
                offset = self.alloc_local(n.name)
                self.emit(f"  mov [rbp-{offset}], rax   ; {n.name}")
                return
            if isinstance(n, Var):
                offset = self.current_locals.get(n.name)
                if offset is not None:
                    self.emit(f"  mov rax, [rbp-{offset}] ; {n.name}")
                else:
                    self.emit(f"  ; var {n.name} not found")
                return
            if isinstance(n, Call):
                key = f"{n.name}/{len(n.args)}"
                for a in n.args[::-1]:   # push args right-to-left
                    self.gen_stmt(a)
                    self.emit("  push rax")
                    self.emit(f"  call {key if key in self.symtab else n.name}")
                    if n.args:
                        self.emit(f"  add rsp, {len(n.args)*8}")
                        return
                    if isinstance(n, BinOp):
                        self.gen_stmt(n.left)
                        self.emit("  push rax")
                        self.gen_stmt(n.right)
                        self.emit("  mov rbx, rax")
                        self.emit("  pop rax")
                        if n.op == "+": self.emit("  add rax, rbx")
                        elif n.op == "-": self.emit("  sub rax, rbx")
                        elif n.op == "*": self.emit("  imul rax, rbx")
                        elif n.op == "/": self.gen_safe_div()
                        elif n.op == "^":
                            loop = self.newlabel("pow"); end = self.newlabel("pow_end")
                            self.emit("  mov rcx, rbx"); self.emit("  mov rbx, rax"); self.emit("  mov rax,1")
                            self.emit(f"{loop}:"); self.emit("  cmp rcx,0"); self.emit(f"  je {end}")
                            self.emit("  imul rax, rbx"); self.emit("  dec rcx"); self.emit(f"  jmp {loop}")
                            self.emit(f"{end}:")
                            return
                        return
                    if isinstance(n, If):
                        el=self.newlabel("else"); en=self.newlabel("endif")
                        self.gen_stmt(n.cond)
                        self.emit("  cmp rax,0"); self.emit(f"  je {el}")
                        for s in n.then_body: self.gen_stmt(s)
                        self.emit(f"  jmp {en}"); self.emit(f"{el}:")
                        if n.else_body:
                            for s in n.else_body: self.gen_stmt(s)
                        self.emit(f"{en}:"); return
                    if isinstance(n, While):
                        start = self.newlabel("while_start")
                        end = self.newlabel("while_end")
                        self.emit(f"{start}:")
                        self.gen_stmt(n.cond)
                        self.emit("  cmp rax,0"); self.emit(f"  je {end}")
                        for s in n.body: self.gen_stmt(s)
                        self.emit(f"  jmp {start}"); self.emit(f"{end}:")
                        return
                    if isinstance(n, TryCatch):
                        try_start = self.newlabel("try_start")
                        try_end = self.newlabel("try_end")
                        catch_start = self.newlabel("catch_start")
                        end = self.newlabel("try_catch_end")
                        self.emit(f"  jmp {try_start}")
                        self.emit(f"{catch_start}:")
                        for s in n.catch_body: self.gen_stmt(s)
                        self.emit(f"  jmp {end}")
                        self.emit(f"{try_start}:")
                        for s in n.try_body: self.gen_stmt(s)
                        self.emit(f"  jmp {end}")
                        self.emit(f"{end}:")
                        return
                    if isinstance(n, Throw):
                        self.gen_stmt(n.expr)
                        self.emit("  ; throw not implemented in ASM")
                        return
                    if isinstance(n, ListLiteral):
                        self.emit(f"  ; list literal with {len(n.elements)} elements not implemented in ASM")
                        return
                    if isinstance(n, Lambda):
                        self.emit(f"  ; lambda not implemented in ASM")
                        return
                        return
                    self.emit(f"  ; unhandled {type(n).__name__}")
                    def run_func(self
                                 ,key:str,args:List[Any]=None):
                        if key not in self.symtab: ra
raise KeyError(key)
f=self.symtab[key]
self.push(); self.push()  # new frame
for i,p in enumerate(f.params):
                            val=args[i] if args and i < len(args) else None
                            self.frames[-1][p]=val
                            res=None
                            for s in f.body: res=self.eval(s)
                            self.pop(); self.pop()  # pop frame
res

if key not in self.symtab:
                            raise KeyError(key)

class StrLiteral(ASTNode):
    def __init__(self, val: str):
        self.val = val

        class Closure:
            def __init__(self, params: List[str], body: List[ASTNode], env: List[Dict[str, Any]]):
                self.params = params
                self.body = body
                self.env = env  # capture the environment at definition time
                self.val = val

if t.kind == "STRING":
    raw = self.eat("STRING").val
StrLiteral(raw[1:-1])  # strip quotes

def __call__(self, args: List[Any], runtime: 'MiniRuntime'):
                    if len(args) != len(self.params):
                        raise RuntimeError(f"Lambda expected {len(self.params)} args, got {len(args)}")
                    runtime.push()
                    for i, p in enumerate(self.params):
                        runtime.frames[-1][p] = args[i]
                    res = None
                    for s in self.body:
                        res = runtime.eval(s)
                    runtime.pop()
                    return res
res

class Codegen:
    def __init__(self, symtab, hot):
        self.symtab = symtab
        self.hot = hot
        self.asm = []
        self._lbl = 0
        self.current_locals = {}
        self.local_size = 0
        self._needs_print = False
        self._string_table = {}   # map: text → label

    def newlabel(self, p="L"):
        self._lbl += 1
        return f"{p}{self._lbl}"

    def add_string(self, text: str) -> str:
        if text in self._string_table:
            return self._string_table[text]
        label = f"str{len(self._string_table)}"
        self._string_table[text] = label
        return label

    def generate(self, prog: Program) -> str:
        # string table
        if self._string_table:
            self.emit("section .rodata")
            for s,label in self._string_table.items():
                self.emit(f"{label}: db {', '.join(str(ord(c)) for c in s)}, 10, 0")

        self.emit("section .bss")
        self.emit("print_buf resb 32")

        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main/0")
        self.emit("  mov rax, 60")
        self.emit("  xor rdi, rdi")
        self.emit("  syscall")

        prog = optimize(prog)
        for s in prog.body:
            self.gen_stmt(s)

        if self._needs_print:
            self.gen_print_funcs()

        return "\n".join(self.asm)

    def gen_stmt(self, n: ASTNode):
        ...
        if isinstance(n, StrLiteral):
            label = self.add_string(n.val)
            self.emit(f"  lea rax, [{label}]")
            return
        if isinstance(n, Call) and n.name == "print":
            for a in n.args:
                self.gen_stmt(a)
                if isinstance(a, StrLiteral):
                    self.emit("  call print_str")
                else:
                    self.emit("  call print_num")
            self._needs_print = True
            return
        ...

        if isinstance(n, Call):
            key = f"{n.name}/{len(n.args)}"
            for a in n.args[::-1]:   # push args right-to-left
                self.gen_stmt(a)
                self.emit("  push rax")
            self.emit(f"  call {key if key in self.symtab else n.name}")
            if n.args:
                self.emit(f"  add rsp, {len(n.args)*8}")
            return
            if n.name == "print":
                for a in n.args:
                    self.gen_stmt(a)
                    if isinstance(a, StrLiteral):
                        self.emit("  call print_str")
                    else:
                        self.emit("  call print_num")
                self._needs_print = True
                return
            ...
            def gen_print_funcs(self):
                """Emit print routines for integers and strings."""

def gen_print_funcs(self):
    # integer printer
    self.emit("print_num:")
    self.prologue()
    self.emit("  mov rcx, print_buf+31")
    self.emit("  mov rbx, 10")
    self.emit("  mov rdx, 0")
    self.emit("print_loop:")
    self.emit("  xor rdx, rdx")
    self.emit("  div rbx")
    self.emit("  add rdx, '0'")
    self.emit("  dec rcx")
    self.emit("  mov [rcx], dl")
    self.emit("  test rax, rax")
    self.emit("  jnz print_loop")
    self.emit("  mov rax, 1")
    self.emit("  mov rdi, 1")
    self.emit("  mov rsi, rcx")
    self.emit("  mov rdx, print_buf+32 - rcx")
    self.emit("  syscall")
    self.epilogue()

    # string printer
    self.emit("print_str:")
    self.prologue()
    self.emit("  mov rsi, rax")         # rax has string address
    self.emit("  mov rdi, 1")           # stdout
    self.emit("  mov rdx, 0")           # compute length
    self.emit("  mov rcx, rsi")
    self.emit("strlen_loop:")
    self.emit("  cmp byte [rcx], 0")
    self.emit("  je strlen_done")
    self.emit("  inc rcx")
    self.emit("  jmp strlen_loop")
    self.emit("strlen_done:")
    self.emit("  mov rdx, rcx")
    self.emit("  sub rdx, rsi")
    self.emit("  mov rax, 1")
    self.emit("  syscall")
    self.epilogue()

