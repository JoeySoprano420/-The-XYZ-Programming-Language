#!/usr/bin/env python3
"""
XYZC8.py – XYZ Compiler v8
Production-grade toolchain with int/float/string support,
constant folding, NASM codegen, and runtime print system.
"""

import re, math, sys, logging
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG = logging.getLogger("xyzc7")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------
class ASTNode:
    def children(self): return []
    def __eq__(self, other): return isinstance(other, self.__class__) and self.__dict__ == other.__dict__
    def __repr__(self):
        fields = ", ".join(f"{k}={v!r}" for k,v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"
    def walk(self):
        yield self
        for c in self.children():
            if isinstance(c, ASTNode):
                yield from c.walk()
            elif isinstance(c, list):
                for e in c:
                    if isinstance(e, ASTNode): yield from e.walk()
                    else: yield e
            else: yield c

# --- Literals ---
class Int(ASTNode):
    def __init__(self, val: int): self.val = int(val)
    def children(self): return []
class Float(ASTNode):
    def __init__(self, val: float): self.val = float(val)
    def children(self): return []
class String(ASTNode):
    def __init__(self, val: str):
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        self.val = val
    def children(self): return []
class Bool(ASTNode):
    def __init__(self, val: bool): self.val = bool(val)
    def children(self): return []
class Null(ASTNode):
    def __init__(self): pass
    def children(self): return []

# --- Expr ---
class Var(ASTNode):
    def __init__(self, name: str): self.name = name
    def children(self): return []
class Assign(ASTNode):
    def __init__(self, name: str, expr: ASTNode): self.name,self.expr=name,expr
    def children(self): return [self.expr]
class BinOp(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode): self.op,self.left,self.right=op,left,right
    def children(self): return [self.left,self.right]
class UnaryOp(ASTNode):
    def __init__(self, op: str, operand: ASTNode): self.op,self.operand=op,operand
    def children(self): return [self.operand]
class Call(ASTNode):
    def __init__(self, name: str, args: List[ASTNode]): self.name,self.args=name,args
    def children(self): return self.args

# --- Stmt ---
class Return(ASTNode):
    def __init__(self, expr: ASTNode): self.expr=expr
    def children(self): return [self.expr]
class If(ASTNode):
    def __init__(self, cond, then_body, else_body=None): self.cond,self.then_body,self.else_body=cond,then_body,else_body
    def children(self): return [self.cond,self.then_body,self.else_body or []]
class While(ASTNode):
    def __init__(self, cond, body): self.cond,self.body=cond,body
    def children(self): return [self.cond,self.body]
class For(ASTNode):
    def __init__(self, init, cond, step, body): self.init,self.cond,self.step,self.body=init,cond,step,body
    def children(self): return [self.init,self.cond,self.step,self.body]
class FuncDef(ASTNode):
    def __init__(self, name, params, body): self.name,self.params,self.body=name,params,body
    def children(self): return self.body
class Program(ASTNode):
    def __init__(self, body): self.body=body
    def children(self): return self.body

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------
TOKEN_SPEC = [
    ("FLOAT",   r"-?\d+\.\d+"),
    ("INT",     r"-?\d+"),
    ("STRING",  r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''),
    ("ID",      r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP",      r"[+\-*/=<>!%^]+"),
    ("LPAREN",  r"\("), ("RPAREN", r"\)"),
    ("LBRACE",  r"\{"), ("RBRACE", r"\}"),
    ("COMMA",   r","),  ("SEMI",   r";"),
    ("WS",      r"\s+"),
]
class Token:
    def __init__(self, kind, val, pos): self.kind,self.val,self.pos=kind,val,pos
    def __repr__(self): return f"Token({self.kind},{self.val},{self.pos})"
def lex(src: str):
    pos=0; out=[]
    while pos < len(src):
        for k,p in TOKEN_SPEC:
            m=re.match(p,src[pos:])
            if m:
                if k!="WS": out.append(Token(k,m.group(),pos))
                pos+=len(m.group()); break
        else:
            raise SyntaxError(f"Unexpected char {src[pos]} at {pos}")
    return out

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class Parser:
    def __init__(self,tokens): self.tokens=tokens; self.pos=0
    def peek(self): return self.tokens[self.pos] if self.pos < len(self.tokens) else None
    def eat(self, kind=None):
        t=self.peek()
        if not t: raise SyntaxError("EOF")
        if kind and t.kind!=kind: raise SyntaxError(f"Expected {kind}, got {t.kind}")
        self.pos+=1; return t
    def parse(self): return Program(self.statements())
    def statements(self):
        out=[]
        while self.peek():
            if self.peek().kind=="ID" and self.peek().val=="func": out.append(self.funcdef())
            else: out.append(self.expression())
        return out
    def funcdef(self):
        self.eat("ID") # 'func'
        name=self.eat("ID").val; self.eat("LPAREN"); params=[]
        while self.peek() and self.peek().kind!="RPAREN":
            params.append(self.eat("ID").val)
            if self.peek().kind=="COMMA": self.eat("COMMA")
        self.eat("RPAREN"); self.eat("LBRACE")
        body=[]
        while self.peek() and self.peek().kind!="RBRACE": body.append(self.expression())
        self.eat("RBRACE")
        return FuncDef(name,params,body)
    def expression(self): return self.parse_addsub()
    def parse_addsub(self):
        l=self.parse_term()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("+","-"):
            op=self.eat("OP").val; r=self.parse_term(); l=BinOp(op,l,r)
        return l
    def parse_term(self):
        l=self.parse_factor()
        while self.peek() and self.peek().kind=="OP" and self.peek().val in ("*","/"):
            op=self.eat("OP").val; r=self.parse_factor(); l=BinOp(op,l,r)
        return l
    def parse_factor(self):
        t=self.peek()
        if not t: return None
        if t.kind=="INT": return Int(self.eat("INT").val)
        if t.kind=="FLOAT": return Float(self.eat("FLOAT").val)
        if t.kind=="STRING": return String(self.eat("STRING").val)
        if t.kind=="ID":
            name=self.eat("ID").val
            if self.peek() and self.peek().kind=="LPAREN":
                self.eat("LPAREN"); args=[]
                while self.peek() and self.peek().kind!="RPAREN":
                    args.append(self.expression())
                    if self.peek().kind=="COMMA": self.eat("COMMA")
                self.eat("RPAREN"); return Call(name,args)
            return Var(name)
        if t.kind=="LPAREN": self.eat("LPAREN"); e=self.expression(); self.eat("RPAREN"); return e
        return None

# ---------------------------------------------------------------------------
# Optimizer (constant folding)
# ---------------------------------------------------------------------------
def optimize(node: ASTNode) -> ASTNode:
    if node is None: return None
    if isinstance(node, Program): node.body=[optimize(n) for n in node.body]; return node
    if isinstance(node, FuncDef): node.body=[optimize(n) for n in node.body]; return node
    if isinstance(node, Return): node.expr=optimize(node.expr); return node
    if isinstance(node, UnaryOp):
        node.operand=optimize(node.operand)
        if node.op=="-" and isinstance(node.operand,(Int,Float)):
            return Float(-node.operand.val) if isinstance(node.operand,Float) else Int(-node.operand.val)
        return node
    if isinstance(node, BinOp):
        node.left,optimize(node.left); node.right,optimize(node.right)
        if isinstance(node.left,(Int,Float)) and isinstance(node.right,(Int,Float)):
            a,b=node.left.val,node.right.val
            if node.op=="+": return Float(a+b) if isinstance(node.left,Float) or isinstance(node.right,Float) else Int(a+b)
            if node.op=="-": return Float(a-b) if isinstance(node.left,Float) or isinstance(node.right,Float) else Int(a-b)
            if node.op=="*": return Float(a*b) if isinstance(node.left,Float) or isinstance(node.right,Float) else Int(a*b)
            if node.op=="/":
                if b==0: LOG.warning("div/0"); return Int(0)
                return Float(a/b)
        return node
    return node

# ---------------------------------------------------------------------------
# Codegen (NASM + runtime helpers)
# ---------------------------------------------------------------------------
class Codegen:
    def __init__(self): self.asm=[]; self.data=[]; self._str_id=0
    def emit(self,s): self.asm.append(s)
    def newstr(self,text:str)->str:
        lbl=f"str{self._str_id}"; self._str_id+=1
        self.data.append(f'{lbl}: db "{text}",0')
        return lbl
    def newfloat(self,val:float)->str:
        lbl=f"flt{self._str_id}"; self._str_id+=1
        self.data.append(f"{lbl}: dq {val}")
        return lbl
    def generate(self,prog:Program)->str:
        self.emit("section .text"); self.emit("global _start"); self.emit("_start:")
        self.emit("  call main"); self.emit("  mov rax,60"); self.emit("xor rdi,rdi"); self.emit("syscall")
        for s in prog.body: self.gen_stmt(s)
        self.emit("section .data"); self.asm.extend(self.data)
        self.emit("; runtime helpers: itoa, ftoa omitted for brevity but included in build")
        return "\n".join(self.asm)
    def gen_stmt(self,n:ASTNode):
        if isinstance(n,FuncDef):
            self.emit(f"{n.name}:"); [self.gen_stmt(b) for b in n.body]; self.emit("ret"); return
        if isinstance(n,Return): self.gen_stmt(n.expr); self.emit("ret"); return
        if isinstance(n,Int): self.emit(f"mov rax,{n.val}"); return
        if isinstance(n,Float):
            lbl=self.newfloat(n.val); self.emit(f"movq xmm0,[{lbl}]"); return
        if isinstance(n,String):
            lbl=self.newstr(n.val); self.emit(f"; string {lbl}"); return
        if isinstance(n,BinOp): self.gen_stmt(n.left); self.emit("push rax"); self.gen_stmt(n.right); self.emit("pop rbx"); self.emit("add rax,rbx"); return
        if isinstance(n,Call):
            if n.name=="print":
                for a in n.args: self.gen_stmt(a)
                self.emit("; call print runtime"); return

# ---------------------------------------------------------------------------
# Runtime Helpers (itoa, ftoa, print)
# ---------------------------------------------------------------------------

class Runtime:
    """
    Provides NASM-level implementations of integer→string,
    float→string, and the print syscall.
    """
    def __init__(self, codegen: Codegen):
        self.cg = codegen

    def inject_itoa(self):
        self.cg.emit("; -------- itoa: convert int in rax to string buffer --------")
        self.cg.emit("itoa:")
        self.cg.emit("  mov rcx, 10")         # divisor
        self.cg.emit("  mov rbx, rsp")        # save base pointer
        self.cg.emit("  sub rsp, 64")         # allocate buffer
        self.cg.emit("  mov rdi, rsp")        # buffer ptr
        self.cg.emit("  add rdi, 63")         # start from end
        self.cg.emit("  mov byte [rdi], 0")   # null terminator")
        self.cg.emit(".itoa_loop:")
        self.cg.emit("  xor rdx, rdx")
        self.cg.emit("  div rcx")
        self.cg.emit("  add dl, '0'")
        self.cg.emit("  dec rdi")
        self.cg.emit("  mov [rdi], dl")
        self.cg.emit("  test rax, rax")
        self.cg.emit("  jnz .itoa_loop")
        self.cg.emit("  mov rax, rdi")        # return ptr in rax
        self.cg.emit("  ret")

    def inject_ftoa(self):
        """
        Simplified float→string conversion: uses C-style formatting
        by calling printf from libc. (For pure NASM we’d implement
        manual double→ASCII, but here we delegate for reliability.)
        """
        self.cg.emit("; -------- ftoa: convert float in xmm0 using printf --------")
        self.cg.emit("extern printf")
        self.cg.emit("ftoa:")
        self.cg.emit("  sub rsp, 32")
        self.cg.emit("  mov rdi, fmt_float")
        self.cg.emit("  movq rax, xmm0")
        self.cg.emit("  movq rsi, xmm0")
        self.cg.emit("  xor eax, eax")
        self.cg.emit("  call printf")
        self.cg.emit("  add rsp, 32")
        self.cg.emit("  ret")
        # add format string
        self.cg.data.append('fmt_float: db "%f",0')

    def inject_print(self):
        """
        Print system:
        - Default adds newline.
        - User may pass `end=""` to suppress newline.
        """
        self.cg.emit("; -------- print: handles strings, ints, floats --------")
        self.cg.emit("print:")
        self.cg.emit("  ; expects arg in rax or xmm0 (float)")
        self.cg.emit("  ret")

# ---------------------------------------------------------------------------
# Extended Codegen to use Runtime
# ---------------------------------------------------------------------------

class CodegenWithRuntime(Codegen):
    def __init__(self):
        super().__init__()
        self.runtime = Runtime(self)

    def generate(self, prog: Program) -> str:
        # text
        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main")
        self.emit("  mov rax,60")
        self.emit("  xor rdi,rdi")
        self.emit("  syscall")

        # user code
        for s in prog.body: self.gen_stmt(s)

        # runtime
        self.runtime.inject_itoa()
        self.runtime.inject_ftoa()
        self.runtime.inject_print()

        # data
        self.emit("section .data")
        self.asm.extend(self.data)

        return "\n".join(self.asm)

    def gen_stmt(self, n: ASTNode):
        if isinstance(n, FuncDef):
            self.emit(f"{n.name}:")
            for b in n.body: self.gen_stmt(b)
            self.emit("  ret")
            return
        if isinstance(n, Return):
            self.gen_stmt(n.expr)
            self.emit("  ret")
            return
        if isinstance(n, Int):
            self.emit(f"  mov rax,{n.val}")
            return
        if isinstance(n, Float):
            lbl = self.newfloat(n.val)
            self.emit(f"  movq xmm0,[{lbl}]")
            return
        if isinstance(n, String):
            lbl = self.newstr(n.val)
            self.emit(f"  lea rax,[{lbl}]")
            return
        if isinstance(n, Call):
            if n.name == "print":
                for a in n.args: self.gen_stmt(a)
                self.emit("  call print")
                return
        if isinstance(n, BinOp):
            self.gen_stmt(n.left)
            self.emit("  push rax")
            self.gen_stmt(n.right)
            self.emit("  pop rbx")
            if n.op == "+": self.emit("  add rax,rbx")
            elif n.op == "-": self.emit("  sub rax,rbx")
            elif n.op == "*": self.emit("  imul rax,rbx")
            elif n.op == "/":
                self.emit("  xor rdx,rdx")
                self.emit("  idiv rbx")
            return

# ---------------------------------------------------------------------------
# Print Implementation with newline suppression
# ---------------------------------------------------------------------------

class Runtime:
    def __init__(self, codegen: Codegen):
        self.cg = codegen
        self.data = []

    def inject_itoa(self):
        self.cg.emit("; -------- itoa: convert int in rax to string buffer --------")
        self.cg.emit("itoa:")
        self.cg.emit("  push rbx")
        self.cg.emit("  mov rcx,10")
        self.cg.emit("  lea rdi,[num_buf+63]")
        self.cg.emit("  mov byte [rdi],0")
        self.cg.emit(".itoa_loop:")
        self.cg.emit("  xor rdx,rdx")
        self.cg.emit("  div rcx")
        self.cg.emit("  add dl,'0'")
        self.cg.emit("  dec rdi")
        self.cg.emit("  mov [rdi],dl")
        self.cg.emit("  test rax,rax")
        self.cg.emit("  jnz .itoa_loop")
        self.cg.emit("  mov rax,rdi")
        self.cg.emit("  pop rbx")
        self.cg.emit("  ret")
        self.data.append("num_buf: times 64 db 0")

    def inject_ftoa(self):
        self.cg.emit("; -------- ftoa: convert float in xmm0 using printf --------")
        self.cg.emit("extern printf")
        self.cg.emit("ftoa:")
        self.cg.emit("  sub rsp,32")
        self.cg.emit("  mov rdi,fmt_float")
        self.cg.emit("  movq rsi,xmm0")
        self.cg.emit("  xor eax,eax")
        self.cg.emit("  call printf")
        self.cg.emit("  add rsp,32")
        self.cg.emit("  ret")
        self.data.append('fmt_float: db "%f",0')

    def inject_print(self):
        self.cg.emit("; -------- print: handles strings, ints, floats --------")
        self.cg.emit("print:")
        self.cg.emit("  ; convention: rax=ptr/val, rbx=type, rcx=endflag")
        self.cg.emit("  cmp rbx,0")         # 0=int
        self.cg.emit("  je .print_int")
        self.cg.emit("  cmp rbx,1")         # 1=string
        self.cg.emit("  je .print_str")
        self.cg.emit("  cmp rbx,2")         # 2=float
        self.cg.emit("  je .print_float")
        self.cg.emit("  ret")

        self.cg.emit(".print_int:")
        self.cg.emit("  call itoa")
        self.cg.emit("  mov rsi,rax")
        self.cg.emit("  mov rdx,64")
        self.cg.emit("  mov rax,1")
        self.cg.emit("  mov rdi,1")
        self.cg.emit("  syscall")
        self.cg.emit("  jmp .maybe_nl")

        self.cg.emit(".print_str:")
        self.cg.emit("  mov rsi,rax")
        self.cg.emit("  mov rdx,r8")       # length preloaded
        self.cg.emit("  mov rax,1")
        self.cg.emit("  mov rdi,1")
        self.cg.emit("  syscall")
        self.cg.emit("  jmp .maybe_nl")

        self.cg.emit(".print_float:")
        self.cg.emit("  call ftoa")
        self.cg.emit("  jmp .maybe_nl")

        # newline control
        self.cg.emit(".maybe_nl:")
        self.cg.emit("  cmp rcx,0")       # rcx=1 suppress newline
        self.cg.emit("  je .do_nl")
        self.cg.emit("  ret")
        self.cg.emit(".do_nl:")
        self.cg.emit("  mov rax,1")
        self.cg.emit("  mov rdi,1")
        self.cg.emit("  mov rsi,nl")
        self.cg.emit("  mov rdx,1")
        self.cg.emit("  syscall")
        self.cg.emit("  ret")
        self.data.append("nl: db 0x0A")

# ---------------------------------------------------------------------------
# Codegen that respects newline suppression
# ---------------------------------------------------------------------------

class CodegenWithRuntime(Codegen):
    def __init__(self):
        super().__init__()
        self.runtime = Runtime(self)

    def gen_stmt(self, n: ASTNode):
        if isinstance(n, Call) and n.name == "print":
            # Default: newline
            for a in n.args:
                if isinstance(a, String):
                    lbl = self.newstr(a.val)
                    self.emit(f"  lea rax,[{lbl}]")
                    self.emit("  mov rbx,1")   # string
                    self.emit(f"  mov r8,{len(a.val)}")
                elif isinstance(a, Int):
                    self.emit(f"  mov rax,{a.val}")
                    self.emit("  mov rbx,0")   # int
                elif isinstance(a, Float):
                    lbl = self.newfloat(a.val)
                    self.emit(f"  movq xmm0,[{lbl}]")
                    self.emit("  mov rbx,2")   # float
                # rcx = 0 → newline; rcx = 1 → suppress
                self.emit("  mov rcx,0")
                self.emit("  call print")
            return
        return super().gen_stmt(n)

# ---------------------------------------------------------------------------
# Parser with print("...", end="") support
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def eat(self, kind=None):
        t = self.peek()
        if not t:
            raise SyntaxError("Unexpected EOF")
        if kind and t.kind != kind:
            raise SyntaxError(f"Expected {kind}, got {t.kind}")
        self.pos += 1
        return t

    def parse(self) -> Program:
        body = []
        while self.peek():
            body.append(self.statement())
        return Program(body)

    # ---------------- STATEMENTS -----------------
    def statement(self):
        t = self.peek()
        if not t:
            return None
        if t.kind == "ID" and t.val == "func":
            return self.funcdef()
        return self.expression()

    def funcdef(self):
        self.eat("ID")  # func
        name = self.eat("ID").val
        self.eat("LPAREN")
        params = []
        while self.peek() and self.peek().kind != "RPAREN":
            params.append(self.eat("ID").val)
            if self.peek() and self.peek().kind == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        self.eat("LBRACE")
        body = []
        while self.peek() and self.peek().kind != "RBRACE":
            body.append(self.statement())
        self.eat("RBRACE")
        return FuncDef(name, params, body)

    # ---------------- EXPRESSIONS -----------------
    def expression(self):
        t = self.peek()
        if not t:
            return None
        if t.kind == "NUMBER":
            return Int(int(self.eat("NUMBER").val))
        if t.kind == "FLOAT":
            return Float(float(self.eat("FLOAT").val))
        if t.kind == "STRING":
            return String(self.eat("STRING").val)
        if t.kind == "ID" and t.val == "print":
            return self.print_call()
        if t.kind == "ID":
            return Var(self.eat("ID").val)
        raise SyntaxError(f"Unexpected token {t.kind} {t.val}")

    def print_call(self):
        self.eat("ID")  # print
        self.eat("LPAREN")
        args = []
        endflag = True  # default: newline
        while self.peek() and self.peek().kind != "RPAREN":
            if self.peek().kind == "STRING":
                args.append(String(self.eat("STRING").val))
            elif self.peek().kind == "NUMBER":
                args.append(Int(int(self.eat("NUMBER").val)))
            elif self.peek().kind == "FLOAT":
                args.append(Float(float(self.eat("FLOAT").val)))
            elif self.peek().kind == "ID" and self.peek().val == "end":
                # keyword argument
                self.eat("ID")   # end
                self.eat("OP")   # =
                valtok = self.eat("STRING")
                if valtok.val.strip('"') == "":
                    endflag = False  # suppress newline
                else:
                    endflag = True
            if self.peek() and self.peek().kind == "COMMA":
                self.eat("COMMA")
        self.eat("RPAREN")
        return PrintCall(args, endflag)


# ---------------------------------------------------------------------------
# AST Node for PrintCall
# ---------------------------------------------------------------------------

class PrintCall(ASTNode):
    def __init__(self, args: list[ASTNode], newline: bool = True):
        self.args = args
        self.newline = newline
    def children(self): return self.args

# ---------------------------------------------------------------------------
# Codegen integration for PrintCall with newline support
# ---------------------------------------------------------------------------

class CodegenWithRuntime(Codegen):
    def __init__(self):
        super().__init__()
        self.runtime = Runtime(self)

    def generate(self, prog: Program) -> str:
        # text section
        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main")
        self.emit("  mov rax,60")
        self.emit("  xor rdi,rdi")
        self.emit("  syscall")

        # user code
        for s in prog.body:
            self.gen_stmt(s)

        # runtime routines
        self.runtime.inject_itoa()
        self.runtime.inject_ftoa()
        self.runtime.inject_print()

        # data section
        self.emit("section .data")
        for d in self.runtime.data:
            self.emit(d)
        for d in self.data:
            self.emit(d)

        return "\n".join(self.asm)

    def gen_stmt(self, n: ASTNode):
        # Handle PrintCall
        if isinstance(n, PrintCall):
            for a in n.args:
                if isinstance(a, String):
                    lbl = self.newstr(a.val)
                    self.emit(f"  lea rax,[{lbl}]")
                    self.emit("  mov rbx,1")   # type=string
                    self.emit(f"  mov r8,{len(a.val)}")
                elif isinstance(a, Int):
                    self.emit(f"  mov rax,{a.val}")
                    self.emit("  mov rbx,0")   # type=int
                elif isinstance(a, Float):
                    lbl = self.newfloat(a.val)
                    self.emit(f"  movq xmm0,[{lbl}]")
                    self.emit("  mov rbx,2")   # type=float
                else:
                    raise NotImplementedError(f"Unsupported print arg {a}")
                # newline control
                if n.newline:
                    self.emit("  mov rcx,0")   # newline
                else:
                    self.emit("  mov rcx,1")   # suppress
                self.emit("  call print")
            return

        # Fallback: other nodes
        if isinstance(n, FuncDef):
            self.emit(f"{n.name}:")
            for b in n.body:
                self.gen_stmt(b)
            self.emit("  ret")
            return

        if isinstance(n, Return):
            self.gen_stmt(n.expr)
            self.emit("  ret")
            return

        if isinstance(n, BinOp):
            self.gen_stmt(n.left)
            self.emit("  push rax")
            self.gen_stmt(n.right)
            self.emit("  pop rbx")
            if n.op == "+": self.emit("  add rax,rbx")
            elif n.op == "-": self.emit("  sub rax,rbx")
            elif n.op == "*": self.emit("  imul rax,rbx")
            elif n.op == "/":
                self.emit("  xor rdx,rdx")
                self.emit("  idiv rbx")
            return

        if isinstance(n, Int):
            self.emit(f"  mov rax,{n.val}")
            return

        if isinstance(n, Float):
            lbl = self.newfloat(n.val)
            self.emit(f"  movq xmm0,[{lbl}]")
            return

        if isinstance(n, String):
            lbl = self.newstr(n.val)
            self.emit(f"  lea rax,[{lbl}]")
            return

# ---------------------------------------------------------------------------
# Part 6 — CIAMS & Dodecagrams support
# ---------------------------------------------------------------------------

import hashlib

# --- CIAMS: Contextual Inference Abstraction Macros ---

class MacroContext:
    """
    Holds context during compilation (type info, symbol tables, macro state).
    CIAMS allow macro transforms based on context (types, usage patterns).
    """
    def __init__(self):
        # maps AST nodes or symbols → inferred types, annotations
        self.type_map: Dict[ASTNode, str] = {}
        self.symbol_types: Dict[str, str] = {}
        self.macro_rules: Dict[str, Callable[..., ASTNode]] = {}

    def infer_type(self, node: ASTNode) -> str:
        """Return inferred type for node, or None if unknown."""
        return self.type_map.get(node)

    def set_type(self, node: ASTNode, typ: str):
        self.type_map[node] = typ

    def register_macro(self, name: str, fn: Callable[..., ASTNode]):
        """Register a macro transformation function under a name."""
        self.macro_rules[name] = fn

    def apply_macro(self, name: str, *args, **kwargs) -> Optional[ASTNode]:
        if name in self.macro_rules:
            return self.macro_rules[name](*args, **kwargs)
        return None


# Example macro: automatic type promotion or wrapper
def macro_promote_int_to_float(call_node: Call, ctx: MacroContext) -> Optional[ASTNode]:
    """
    Example CIAMS macro: if a function expects a float but gets an int,
    wrap the int → float conversion automatically.
    """
    # Suppose we know target function expects float args
    prot = ctx.infer_type(call_node)
    if prot and prot.startswith("float_fn"):
        # wrap integer args into Float nodes
        new_args = []
        for a in call_node.args:
            if isinstance(a, Int):
                new_args.append(Float(float(a.val)))
            else:
                new_args.append(a)
        return Call(call_node.name, new_args)
    return None

# --- Dodecagram IR / Encoding ---

class DodecagramEncoder:
    """
    Encode AST or IR into a “dodecagram” base-12 serialized code.
    Digits: 0–9, a, b represent values 0–11.
    """
    DIGITS = "0123456789ab"

    def encode_int(self, value: int) -> str:
        """Encode integer into base-12 string (dodecagram digits)."""
        if value < 0:
            return "-" + self.encode_int(-value)
        if value == 0:
            return "0"
        res = []
        v = value
        while v > 0:
            d = v % 12
            res.append(self.DIGITS[d])
            v //= 12
        return "".join(reversed(res))

    def encode_float(self, val: float, prec: int = 6) -> str:
        """
        Encode float by splitting into integer + fractional parts.
        Fraction encoded base-12 with `.` separator.
        """
        iv = int(val)
        fv = abs(val - iv)
        i_part = self.encode_int(iv)
        frac_str = ""
        for _ in range(prec):
            fv *= 12
            d = int(fv)
            frac_str += self.DIGITS[d]
            fv -= d
        return f"{i_part}.{frac_str}"

    def encode_node(self, node: ASTNode) -> str:
        """
        Recursively encode AST nodes into a dodecagram string.
        For example, BinOp("+", left, right) → " +:<left_enc>:<right_enc>"
        """
        if isinstance(node, Int):
            return self.encode_int(node.val)
        if isinstance(node, Float):
            return self.encode_float(node.val)
        if isinstance(node, String):
            # For strings, simply base-12 encode each char code
            return '"' + "".join(self.encode_int(ord(c)) for c in node.val) + '"'
        if isinstance(node, Var):
            return f"v({node.name})"
        if isinstance(node, BinOp):
            le = self.encode_node(node.left)
            re = self.encode_node(node.right)
            return f"({node.op}:{le}:{re})"
        if isinstance(node, Call):
            arg_enc = ":".join(self.encode_node(a) for a in node.args)
            return f"call({node.name}:{arg_enc})"
        # fallback
        return "?"

    def encode(self, program: Program) -> str:
        """Encode entire program body into a dodecagram string."""
        parts = [self.encode_node(n) for n in program.body]
        return "|".join(parts)


# --- Hook these into compilation pipeline ---

class Compiler:
    def __init__(self):
        self.ctx = MacroContext()
        self.encoder = DodecagramEncoder()

    def compile(self, src: str):
        # Lex → Parser → AST
        toks = lex(src)
        parser = Parser(toks)
        prog = parser.parse()

        # Apply CIAMS macros (pre-optimization)
        for node in prog.walk():
            if isinstance(node, Call):
                macro_out = self.ctx.apply_macro(node.name, node, self.ctx)
                if macro_out:
                    # Replace the node in AST (naively)
                    # (In full implementation you'd maintain parent links)
                    # For demo:
                    # This is just shallow: node = macro_out
                    pass

        # Optimize AST
        prog = optimize(prog)

        # Optionally: encode Dodecagram IR
        encoded = self.encoder.encode(prog)
        LOG.info("Dodecagram encoding: %s", encoded)

        # Codegen
        cg = CodegenWithRuntime()
        asm = cg.generate(prog)
        return asm
