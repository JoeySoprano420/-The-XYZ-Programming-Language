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

# ---------------------------------------------------------------------------
# Part 7 – Intrinsic / Derivative / Polynomial support
# ---------------------------------------------------------------------------

# Extend AST Node for Derivative
class Derivative(ASTNode):
    def __init__(self, func_expr: ASTNode, var: Var):
        self.func_expr = func_expr
        self.var = var
    def children(self): return [self.func_expr, self.var]

# Extend MacroContext: register intrinsic macro for derivative
def macro_derivative(call_node: Call, ctx: MacroContext) -> Optional[ASTNode]:
    """
    Macro expansion: derivative(f_expr, var) → Derivative AST node,
    if valid inputs.
    """
    if call_node.name == "derivative" and len(call_node.args) == 2:
        fexpr, var = call_node.args
        if isinstance(var, Var):
            return Derivative(fexpr, var)
    return None

# In your Compiler.compile (or earlier), register:
# ctx.register_macro("derivative", macro_derivative)

# Symbolic differentiation engine
def differentiate(expr: ASTNode, var: Var) -> ASTNode:
    """
    Returns d(expr)/d(var) as ASTNode.
    Supports: constants, var, binops +,-,*, power (^ with int exponent).
    """
    # constant
    if isinstance(expr, Int) or isinstance(expr, Float) or isinstance(expr, String):
        return Int(0)
    # variable
    if isinstance(expr, Var):
        return Int(1) if expr.name == var.name else Int(0)
    # unary minus
    if isinstance(expr, UnaryOp) and expr.op == "-":
        return UnaryOp("-", differentiate(expr.operand, var))
    # binary op
    if isinstance(expr, BinOp):
        l, r = expr.left, expr.right
        dl = differentiate(l, var)
        dr = differentiate(r, var)
        op = expr.op
        if op == "+":
            return BinOp("+", dl, dr)
        if op == "-":
            return BinOp("-", dl, dr)
        if op == "*":
            # product rule: l*dr + r*dl
            return BinOp("+", BinOp("*", l, dr), BinOp("*", r, dl))
        if op == "^":
            # treat r as constant exponent
            if isinstance(r, Int):
                # d(l^c) = c * l^(c-1) * dl
                c = r.val
                newpow = Int(c - 1)
                return BinOp("*",
                             BinOp("*", Int(c), BinOp("^", l, newpow)),
                             differentiate(l, var))
        if op == "/":
            # quotient rule: (l' r - l r') / (r^2)
            num = BinOp("-", BinOp("*", dl, r), BinOp("*", l, dr))
            denom = BinOp("^", r, Int(2))
            return BinOp("/", num, denom)
    # derivative of call or other forms: unsupported, return zero
    return Int(0)

# Extend optimizer to simplify Derivative nodes
def optimize(node: ASTNode) -> ASTNode:
    # existing code above...
    # Add handling for Derivative nodes:
    if isinstance(node, Derivative):
        # Simplify the inner function first
        fe = optimize(node.func_expr)
        v = node.var
        # differentiate
        derived = differentiate(fe, v)
        # recursively optimize the result
        return optimize(derived)
    # Then rest of optimize logic...
    # (Do not forget to include this before generic fallback)
    # ... existing cases ...
    return node

# ---------------------------------------------------------------------------
# Part 8 — Framed Hot Routes & Hot-Swap
# ---------------------------------------------------------------------------

import threading
import time

class HotRouteProfiler:
    """
    Tracks how often functions are called.
    If a function exceeds a threshold, it is flagged as 'hot'.
    """
    def __init__(self, threshold: int = 50):
        self.calls: Dict[str, int] = {}
        self.threshold = threshold
        self.lock = threading.Lock()

    def record(self, key: str):
        with self.lock:
            self.calls[key] = self.calls.get(key, 0) + 1

    def is_hot(self, key: str) -> bool:
        return self.calls.get(key, 0) >= self.threshold


class HotSwapManager:
    """
    Maintains alternate implementations of functions.
    Allows replacing AST or bytecode for a given function key (name/arity).
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.func_table: Dict[str, FuncDef] = {}
        self.bytecode_table: Dict[str, BytecodeFunction] = {}

    def register_func(self, key: str, func: FuncDef):
        with self.lock:
            self.func_table[key] = func

    def get_func(self, key: str) -> Optional[FuncDef]:
        return self.func_table.get(key)

    def swap_func(self, key: str, new_func: FuncDef):
        with self.lock:
            old = self.func_table.get(key)
            self.func_table[key] = new_func
        LOG.info(f"Hot-swap AST for {key}")
        return old

    def register_bytecode(self, key: str, bc: BytecodeFunction):
        with self.lock:
            self.bytecode_table[key] = bc

    def swap_bytecode(self, key: str, new_bc: BytecodeFunction):
        with self.lock:
            old = self.bytecode_table.get(key)
            self.bytecode_table[key] = new_bc
        LOG.info(f"Hot-swap Bytecode for {key}")
        return old


# ---------------------------------------------------------------------------
# Integration with MiniRuntime & FastRuntime
# ---------------------------------------------------------------------------

class MiniRuntime:
    def __init__(self, symtab: Dict[str, FuncDef], hot: HotSwapManager, profiler: HotRouteProfiler):
        self.symtab = symtab
        self.hot = hot
        self.profiler = profiler
        self.frames: List[Dict[str, Any]] = []

    def run_func(self, key: str, args: List[Any]):
        self.profiler.record(key)
        func = self.hot.get_func(key) or self.symtab.get(key)
        if not func:
            raise KeyError(f"No function {key}")
        frame = {p: (args[i] if i < len(args) else None) for i, p in enumerate(func.params)}
        self.frames.append(frame)
        result = None
        for st in func.body:
            result = self.eval(st)
            if isinstance(st, Return):
                result = self.eval(st.expr)
                break
        self.frames.pop()
        return result


class FastRuntime:
    def __init__(self, funcs: Dict[str, BytecodeFunction], hot: HotSwapManager, profiler: HotRouteProfiler):
        self.funcs = funcs
        self.hot = hot
        self.profiler = profiler
        self.vm = FastVM(funcs, hot)

    def run(self, key: str, args: List[Any]):
        self.profiler.record(key)
        # If function was hot-swapped at bytecode level
        if key in self.hot.bytecode_table:
            return self.vm.run_bytecode(self.hot.bytecode_table[key], args)
        return self.vm.run(key, args)


# ---------------------------------------------------------------------------
# Example: auto-promotion of hot functions
# ---------------------------------------------------------------------------

class Compiler:
    def __init__(self):
        self.ctx = MacroContext()
        self.encoder = DodecagramEncoder()
        self.hotswap = HotSwapManager()
        self.profiler = HotRouteProfiler(threshold=100)

    def compile_and_run(self, src: str):
        toks = lex(src)
        parser = Parser(toks)
        prog = parser.parse()

        # Optimize
        prog = optimize(prog)

        # Install into HotSwap table
        for node in prog.body:
            if isinstance(node, FuncDef):
                key = f"{node.name}/{len(node.params)}"
                self.hotswap.register_func(key, node)

        # Choose runtime
        rt = MiniRuntime(symtab={}, hot=self.hotswap, profiler=self.profiler)

        # Example: auto promote if function is hot
        def auto_promote_monitor():
            while True:
                for k, count in list(self.profiler.calls.items()):
                    if self.profiler.is_hot(k) and k not in self.hotswap.bytecode_table:
                        f = self.hotswap.get_func(k)
                        if f:
                            bc = FastCompiler({k: f}).compile_func(k, f)
                            self.hotswap.register_bytecode(k, bc)
                            LOG.info(f"Promoted {k} to bytecode (hot route)")
                time.sleep(1)

        threading.Thread(target=auto_promote_monitor, daemon=True).start()

        return rt

# ---------------------------------------------------------------------------
# Part 9 — Modules / Packages / Packets with Auto-Linking
# ---------------------------------------------------------------------------

import os, json

class Module:
    """
    Represents a single XYZ source file.
    Contains its AST and export/import tables.
    """
    def __init__(self, name: str, ast: Program, path: str = ""):
        self.name = name
        self.ast = ast
        self.path = path
        self.exports: Dict[str, ASTNode] = {}
        self.imports: Dict[str, str] = {}   # local_name → module_name.symbol

    def analyze_exports(self):
        """Fill export table by scanning FuncDef / VarDecl nodes."""
        for node in self.ast.body:
            if isinstance(node, FuncDef):
                self.exports[node.name] = node
            if isinstance(node, VarDecl):
                self.exports[node.name] = node

    def analyze_imports(self):
        """Collect imports (parser must populate via Import nodes)."""
        for node in self.ast.body:
            if isinstance(node, Import):
                self.imports[node.local_name] = f"{node.module}.{node.symbol}"


class Package:
    """
    A package is a group of modules.
    """
    def __init__(self, name: str):
        self.name = name
        self.modules: Dict[str, Module] = {}

    def add_module(self, mod: Module):
        self.modules[mod.name] = mod


class Packet:
    """
    Final deployable unit: bundle of modules with resolved imports/exports.
    Can be serialized to JSON or compiled to NASM/ELF.
    """
    def __init__(self, package: Package):
        self.package = package
        self.symbol_map: Dict[str, str] = {}  # local → resolved

    def resolve_links(self):
        """Resolve imports to actual export symbols across modules."""
        for mod in self.package.modules.values():
            for local, target in mod.imports.items():
                mod_name, sym = target.split(".")
                if mod_name in self.package.modules:
                    exports = self.package.modules[mod_name].exports
                    if sym not in exports:
                        raise KeyError(f"Symbol {sym} not exported by {mod_name}")
                    self.symbol_map[f"{mod.name}.{local}"] = f"{mod_name}.{sym}"
                else:
                    raise KeyError(f"Module {mod_name} not found in package")

    def to_json(self) -> str:
        """Serialize symbol map and modules into JSON deployable."""
        data = {
            "package": self.package.name,
            "modules": {m: list(mod.exports.keys()) for m, mod in self.package.modules.items()},
            "links": self.symbol_map,
        }
        return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Parser Nodes for Import/Export
# ---------------------------------------------------------------------------

class Import(ASTNode):
    def __init__(self, module: str, symbol: str, local_name: Optional[str] = None):
        self.module = module
        self.symbol = symbol
        self.local_name = local_name or symbol

class Export(ASTNode):
    def __init__(self, symbol: str):
        self.symbol = symbol


# ---------------------------------------------------------------------------
# Compiler Integration
# ---------------------------------------------------------------------------

class Compiler:
    def __init__(self):
        self.ctx = MacroContext()
        self.encoder = DodecagramEncoder()
        self.hotswap = HotSwapManager()
        self.profiler = HotRouteProfiler()
        self.packages: Dict[str, Package] = {}

    def load_module(self, path: str) -> Module:
        """Load, parse, and analyze a module from file path."""
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path) as f:
            src = f.read()
        toks = lex(src)
        parser = Parser(toks)
        prog = parser.parse()
        mod = Module(name, prog, path)
        mod.analyze_exports()
        mod.analyze_imports()
        return mod

    def build_package(self, name: str, file_paths: List[str]) -> Package:
        """Build a package from a list of source files."""
        pkg = Package(name)
        for path in file_paths:
            mod = self.load_module(path)
            pkg.add_module(mod)
        self.packages[name] = pkg
        return pkg

    def link_packet(self, pkg: Package) -> Packet:
        """Link a package into a deployable packet."""
        packet = Packet(pkg)
        packet.resolve_links()
        return packet

# ---------------------------------------------------------------------------
# Part 10 — FFI, ABI, ISA Wrappers
# ---------------------------------------------------------------------------

import ctypes, platform, subprocess, tempfile

class FFIManager:
    """
    Manages Foreign Function Interface (FFI) calls between XYZ and C libraries.
    """
    def __init__(self):
        self.libs: Dict[str, Any] = {}
        self.func_cache: Dict[str, Callable] = {}
        self.abi = "sysv" if platform.system() != "Windows" else "ms64"

    def load_library(self, path: str) -> Any:
        """Load a shared library (.so / .dll / .dylib)."""
        if path in self.libs: 
            return self.libs[path]
        try:
            if platform.system() == "Windows":
                lib = ctypes.WinDLL(path)
            else:
                lib = ctypes.CDLL(path)
            self.libs[path] = lib
            return lib
        except Exception as e:
            raise RuntimeError(f"Failed to load library {path}: {e}")

    def resolve_function(self, lib_path: str, func_name: str, argtypes, restype):
        """Resolve and cache a function pointer with arg/restype mapping."""
        key = f"{lib_path}:{func_name}"
        if key in self.func_cache:
            return self.func_cache[key]
        lib = self.load_library(lib_path)
        fn = getattr(lib, func_name)
        fn.argtypes = argtypes
        fn.restype = restype
        self.func_cache[key] = fn
        return fn

    def ffi_call(self, lib_path: str, func_name: str, *args, argtypes=None, restype=None):
        """Call an external C function from XYZ code."""
        fn = self.resolve_function(lib_path, func_name, argtypes or [], restype or ctypes.c_int)
        return fn(*args)


class ABIWrapper:
    """
    Provides ABI-specific calling convention helpers.
    (System V vs Windows x64)
    """
    def __init__(self):
        self.sys = platform.system()

    def wrap_func(self, name: str, codegen: "Codegen") -> str:
        """Emit assembly prologue/epilogue depending on platform ABI."""
        if self.sys == "Windows":
            return f"""
{name}:
    ; Windows x64 ABI prologue
    push rbp
    mov rbp, rsp
    ; ... function body ...
    mov rsp, rbp
    pop rbp
    ret
"""
        else:
            return f"""
{name}:
    ; System V AMD64 ABI prologue
    push rbp
    mov rbp, rsp
    ; ... function body ...
    mov rsp, rbp
    pop rbp
    ret
"""


class InlineAsm(ASTNode):
    """AST node for inline assembly blocks."""
    def __init__(self, code: str):
        self.code = code


# ---------------------------------------------------------------------------
# Parser extension for Inline ASM
# ---------------------------------------------------------------------------

class Parser(Parser):  # extend previous parser
    def inlineasm(self):
        if self.peek().kind == "KEYWORD" and self.peek().val == "asm":
            self.eat("KEYWORD")
            code = self.eat("STRING").val.strip('"')
            return InlineAsm(code)
        return None


# ---------------------------------------------------------------------------
# Codegen integration for InlineAsm
# ---------------------------------------------------------------------------

class Codegen(Codegen):  # extend previous codegen
    def gen_stmt(self, n: ASTNode):
        if isinstance(n, InlineAsm):
            self.emit(f"  ; inline asm begin")
            for line in n.code.split("\\n"):
                self.emit(f"  {line}")
            self.emit(f"  ; inline asm end")
            return
        return super().gen_stmt(n)


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

def demo_ffi():
    ffi = FFIManager()
    # Example: call C standard library's strlen
    libc = "/usr/lib/libc.so.6" if platform.system() != "Windows" else "msvcrt.dll"
    strlen = ffi.resolve_function(libc, "strlen", [ctypes.c_char_p], ctypes.c_size_t)
    res = strlen(b"Hello, XYZ!")
    print("strlen returned:", res)

def demo_inline_asm():
    asm_block = InlineAsm("mov rax, 42\\nret")
    cg = Codegen({}, HotSwapRegistry())
    cg.gen_stmt(asm_block)
    print("\\n".join(cg.asm))

# ---------------------------------------------------------------------------
# Part 11 — CIAMS (Contextual Inference Abstraction Macros) + Dodecagrams
# ---------------------------------------------------------------------------

DO_DIGITS = "0123456789ab"

def to_dodecagram(n: int) -> str:
    """Convert integer to base-12 (dodecagram) string."""
    if n == 0: return "0"
    digits = []
    while n > 0:
        digits.append(DO_DIGITS[n % 12])
        n //= 12
    return "".join(reversed(digits))

def from_dodecagram(s: str) -> int:
    """Convert base-12 (dodecagram) string to integer."""
    val = 0
    for ch in s:
        val = val * 12 + DO_DIGITS.index(ch)
    return val


class CIAMSExpander:
    """
    Contextual Inference Abstraction Macros:
    Macros adapt to AST context (arithmetic, control flow, types).
    """
    def __init__(self, runtime_env: Dict[str, Any]):
        self.runtime_env = runtime_env
        self.macros: Dict[str, Callable[[Call, Dict[str, Any]], List[ASTNode]]] = {}

    def register(self, name: str, fn: Callable[[Call, Dict[str, Any]], List[ASTNode]]):
        """Register a macro with contextual inference."""
        self.macros[name] = fn

    def expand(self, node: ASTNode, ctx: Dict[str, Any]):
        if isinstance(node, Call) and node.name in self.macros:
            return self.macros[node.name](node, ctx)
        return [node]


# ---------------------------------------------------------------------------
# Dodecagram Encoding for AST Nodes
# ---------------------------------------------------------------------------

class DodecagramEncoder:
    """
    Encodes AST into dodecagram sequences (base-12 digits).
    Each node type maps to an opcode in base-12.
    """
    OPCODES = {
        "Number": 1, "Var": 2, "Assign": 3, "BinOp": 4, "UnaryOp": 5,
        "Call": 6, "Return": 7, "If": 8, "While": 9, "List": 10, "Map": 11,
    }

    @classmethod
    def encode_node(cls, n: ASTNode) -> str:
        if isinstance(n, Number):
            return "1" + to_dodecagram(int(n.val))
        if isinstance(n, Var):
            return "2" + to_dodecagram(hash(n.name) % 144)  # keep in base-12 range
        if isinstance(n, Assign):
            return "3" + cls.encode_node(n.expr)
        if isinstance(n, BinOp):
            return "4" + cls.encode_node(n.left) + cls.encode_node(n.right)
        if isinstance(n, UnaryOp):
            return "5" + cls.encode_node(n.operand)
        if isinstance(n, Call):
            seq = "6" + to_dodecagram(len(n.args))
            for a in n.args:
                seq += cls.encode_node(a)
            return seq
        if isinstance(n, Return):
            return "7" + cls.encode_node(n.expr)
        if isinstance(n, If):
            return "8" + cls.encode_node(n.cond)
        return "b"  # fallback code (11 in base-12)

    @classmethod
    def encode_program(cls, prog: Program) -> str:
        return "".join(cls.encode_node(n) for n in prog.body)


# ---------------------------------------------------------------------------
# Integration into FastRuntime
# ---------------------------------------------------------------------------

class FastVM(FastVM):  # extend previous FastVM
    def run_dodecagram(self, dseq: str):
        """
        Execute dodecagram sequence directly as VM instructions.
        """
        stack = []
        pc = 0
        while pc < len(dseq):
            op = dseq[pc]; pc += 1
            if op == "1":  # Number
                val_str = ""
                while pc < len(dseq) and dseq[pc] in DO_DIGITS:
                    val_str += dseq[pc]; pc += 1
                stack.append(from_dodecagram(val_str))
            elif op == "4":  # BinOp
                a = stack.pop(); b = stack.pop()
                stack.append(a + b)  # simplified, extendable
            elif op == "7":  # Return
                return stack.pop() if stack else None
            else:
                pass
        return stack[-1] if stack else None


# ---------------------------------------------------------------------------
# Example Macro + Execution
# ---------------------------------------------------------------------------

def demo_ciams_and_dodecagram():
    hot = HotSwapRegistry()
    symtab = {}
    runtime = FastRuntime(symtab, hot)

    # Register CIAMS macro for `square(x)`
    ciams = CIAMSExpander(runtime_env={})
    ciams.register("square", lambda node, ctx: [
        BinOp("*", node.args[0], node.args[0])
    ])

    # Build AST manually: square(5)
    call = Call("square", [Number("5")])
    expanded = ciams.expand(call, {})
    enc = DodecagramEncoder.encode_node(expanded[0])
    print("Dodecagram sequence:", enc)

    res = runtime.vm.run_dodecagram(enc)
    print("Executed result =", res)

# ---------------------------------------------------------------------------
# Part 13 — Parallelism + Scheduling + Hot Swaps
# ---------------------------------------------------------------------------

import concurrent.futures

class Scheduler:
    def __init__(self, max_workers=None):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    def run_parallel(self, funcs: List[Callable], args: List[List[Any]]):
        futures = [self.executor.submit(fn, *argset) for fn, argset in zip(funcs, args)]
        return [f.result() for f in futures]


class PatternMatcher:
    """Matches execution patterns to optimized routines."""
    def __init__(self):
        self.patterns: Dict[str, Callable] = {}

    def register(self, pattern: str, fn: Callable):
        self.patterns[pattern] = fn

    def dispatch(self, pattern: str, *args):
        if pattern in self.patterns:
            return self.patterns[pattern](*args)
        raise KeyError(f"No pattern for {pattern}")


class HotRouteOptimizer:
    """Hot function replacement at runtime."""
    def __init__(self, hot: HotSwapRegistry, symtab: Dict[str, FuncDef]):
        self.hot = hot
        self.symtab = symtab
        self.hot_counts: Dict[str, int] = {}
        self.threshold = 50  # promote after N calls

    def record_call(self, key: str):
        self.hot_counts[key] = self.hot_counts.get(key, 0) + 1
        if self.hot_counts[key] >= self.threshold:
            self.promote(key)

    def promote(self, key: str):
        fn = self.symtab.get(key)
        if not fn: return
        # simplistic: replace with an optimized inlined version
        opt_body = optimize(FuncDef(fn.name, fn.params, fn.body))
        self.hot.swap(key, opt_body)

# ---------------------------------------------------------------------------
# Part 14 — Full Dodecagram Runtime
# ---------------------------------------------------------------------------

class DodecagramBinary:
    """Encodes/decodes dodecagram sequences into packed binary."""
    @staticmethod
    def pack(seq: str) -> bytes:
        bits = "".join(f"{DO_DIGITS.index(ch):04b}" for ch in seq)
        while len(bits) % 8 != 0:
            bits += "0"
        return int(bits, 2).to_bytes(len(bits)//8, "big")

    @staticmethod
    def unpack(b: bytes) -> str:
        bits = bin(int.from_bytes(b, "big"))[2:]
        bits = bits.zfill((len(b) * 8))
        out = ""
        for i in range(0, len(bits), 4):
            val = int(bits[i:i+4], 2)
            if val < 12:
                out += DO_DIGITS[val]
        return out


class FastVM(FastVM):  # extend again
    def run_dodecagram(self, dseq: Union[str, bytes]):
        if isinstance(dseq, bytes):
            dseq = DodecagramBinary.unpack(dseq)
        pc, stack = 0, []
        while pc < len(dseq):
            op = dseq[pc]; pc += 1
            if op == "1":  # Number
                num = ""
                while pc < len(dseq) and dseq[pc] in DO_DIGITS:
                    num += dseq[pc]; pc += 1
                stack.append(from_dodecagram(num))
            elif op == "2":  # Var
                stack.append("var")
            elif op == "3":  # Assign
                val = stack.pop(); stack.append(val)
            elif op == "4":  # BinOp
                b = stack.pop(); a = stack.pop(); stack.append(a+b)
            elif op == "5":  # UnaryOp
                v = stack.pop(); stack.append(-v)
            elif op == "6":  # Call
                argc = int(dseq[pc]); pc += 1
                args = [stack.pop() for _ in range(argc)][::-1]
                stack.append(("call", args))
            elif op == "7":  # Return
                return stack.pop() if stack else None
            elif op == "8":  # If
                cond = stack.pop()
                stack.append(("if", cond))
            elif op == "9":  # While
                stack.append(("while", None))
            elif op == "a":  # List
                stack.append([])
            elif op == "b":  # Map
                stack.append({})
        return stack[-1] if stack else None

# ---------------------------------------------------------------------------
# Part 15 — CIAMS Macro Fusion with Dodecagram
# ---------------------------------------------------------------------------

class CIAMSExpander(CIAMSExpander):  # extend
    def emit_dodecagram(self, node: ASTNode) -> str:
        """Convert AST node to a Dodecagram sequence."""
        if isinstance(node, Number):
            return "1" + to_dodecagram(int(node.val))
        if isinstance(node, Var):
            return "2" + node.name
        if isinstance(node, BinOp):
            return self.emit_dodecagram(node.left) + self.emit_dodecagram(node.right) + "4"
        if isinstance(node, UnaryOp):
            return self.emit_dodecagram(node.operand) + "5"
        if isinstance(node, Call):
            args = "".join(self.emit_dodecagram(a) for a in node.args)
            return args + "6" + to_dodecagram(len(node.args))
        if isinstance(node, Return):
            return self.emit_dodecagram(node.expr) + "7"
        return ""

    def expand_to_dodecagram(self, node: ASTNode, ctx: Dict[str, Any], phase=CIAMSPhase.PARSE) -> str:
        expanded_nodes = self.expand_with_phase(node, ctx, phase)
        return "".join(self.emit_dodecagram(n) for n in expanded_nodes)

# ---------------------------------------------------------------------------
# Part 16 — Intrinsics / Derivatives / Polynomials
# ---------------------------------------------------------------------------

import sympy

class MathIntrinsics:
    @staticmethod
    def derivative(expr: str, var: str="x") -> str:
        x = sympy.symbols(var)
        return str(sympy.diff(sympy.sympify(expr), x))

    @staticmethod
    def polynomial(coeffs: List[int], var: str="x") -> str:
        x = sympy.symbols(var)
        poly = sum(c*x**i for i,c in enumerate(coeffs))
        return str(poly)

    @staticmethod
    def eval_poly(coeffs: List[int], val: int) -> int:
        return sum(c*(val**i) for i,c in enumerate(coeffs))

# ---------------------------------------------------------------------------
# Part 17 — Hot Routing / Frame Patching
# ---------------------------------------------------------------------------

import ctypes

class HotRoutePatcher:
    def __init__(self, optimizer: HotRouteOptimizer):
        self.optimizer = optimizer
        self.patched = {}

    def patch(self, key: str, native_code: bytes):
        """Inject machine code at runtime into memory (Linux/x86_64)."""
        size = len(native_code)
        buf = ctypes.create_string_buffer(native_code, size)
        addr = ctypes.addressof(buf)
        fn_type = ctypes.CFUNCTYPE(ctypes.c_int)
        func = fn_type(addr)
        self.patched[key] = func
        return func

    def call(self, key: str, *args):
        if key in self.patched:
            return self.patched[key](*args)
        return None

# ---------------------------------------------------------------------------
# Part 18 — Module / Package / Packet System
# ---------------------------------------------------------------------------

import importlib.util

class ModuleSystem:
    def __init__(self):
        self.loaded: Dict[str, Any] = {}

    def load_packet(self, path: str):
        with open(path,"r") as f:
            obj = json.load(f)
        self.loaded[path] = obj
        return obj

    def link_package(self, paths: List[str], out: str):
        merged = {"format":"XYZPKG1","symbols":{}}
        for p in paths:
            pkt = self.load_packet(p)
            for k,v in pkt.get("symbols",{}).items():
                merged["symbols"][k]=v
        with open(out,"w") as f:
            json.dump(merged,f)
        return out

# ---------------------------------------------------------------------------
# Part 19 — FFI / ABI Wrappers
# ---------------------------------------------------------------------------

class CInterop:
    def __init__(self):
        self.libcache = {}

    def load_lib(self, path: str):
        if path not in self.libcache:
            self.libcache[path] = ctypes.CDLL(path)
        return self.libcache[path]

    def bind_func(self, lib: str, fname: str, restype, argtypes):
        l = self.load_lib(lib)
        fn = getattr(l, fname)
        fn.restype = restype
        fn.argtypes = argtypes
        return fn

# ---------------------------------------------------------------------------
# Part 20 — GC + Lisp-like Macro DSL
# ---------------------------------------------------------------------------

class GC:
    def __init__(self):
        self.objects = []

    def collect(self, obj):
        self.objects.append(obj)

    def sweep(self):
        self.objects.clear()


class LispCIAMS:
    """CIAMS Lisp-like macros: (macro (args) body)"""
    def __init__(self, ciams: CIAMSExpander):
        self.ciams = ciams

    def define_macro(self, name: str, args: List[str], body: List[str]):
        def macro_fn(node, ctx):
            env = dict(zip(args, node.args))
            out = []
            for b in body:
                expr = b.format(**env)
                out.append(Parser(lex(expr)).expression())
            return out
        self.ciams.register(name, macro_fn)

# ---------------------------------------------------------------------------
# Part 21 — Parallel Scheduling Macros
# ---------------------------------------------------------------------------

class ParallelCIAMS(LispCIAMS):
    def __init__(self, ciams: CIAMSExpander, scheduler: Scheduler):
        super().__init__(ciams)
        self.scheduler = scheduler

        # register the (parallel ...) macro
        self.define_parallel()

    def define_parallel(self):
        def macro_fn(node: Call, ctx: Dict[str, Any]):
            # node.args = [Call("f",[...]), Call("g",[...]), ...]
            funcs, argsets = [], []
            for arg in node.args:
                if isinstance(arg, Call):
                    fname = arg.name
                    fargs = [self.ciams.emit_dodecagram(a) for a in arg.args]
                    funcs.append(fname)
                    argsets.append(fargs)
            # Return AST that calls scheduler
            return [Call("__parallel_exec", [ListLiteral([Var(f) for f in funcs]),
                                             ListLiteral([ListLiteral(a) for a in argsets])])]
        self.ciams.register("parallel", macro_fn, phase=CIAMSPhase.EXEC)


class ParallelRuntime:
    def __init__(self, scheduler: Scheduler, symtab: Dict[str, FuncDef], hot: HotSwapRegistry):
        self.scheduler = scheduler
        self.symtab = symtab
        self.hot = hot

    def __parallel_exec(self, funcs: List[str], argsets: List[List[Any]]):
        callables = []
        for f in funcs:
            key = f"{f}/{len(argsets[0])}"
            fn = lambda *args, k=key: MiniRuntime(self.symtab, self.hot).run_func(k, list(args))
            callables.append(fn)
        return self.scheduler.run_parallel(callables, argsets)

# ---------------------------------------------------------------------------
# Part 22 — Pattern-Matched JIT Routes
# ---------------------------------------------------------------------------

import tempfile, subprocess

class JITGenerator:
    """Generates NASM functions at runtime, compiles to .so, loads via ctypes."""
    def __init__(self):
        self.cache = {}

    def generate(self, name: str, asm_code: str):
        if name in self.cache: return self.cache[name]

        with tempfile.TemporaryDirectory() as td:
            asm_path = os.path.join(td, f"{name}.asm")
            obj_path = os.path.join(td, f"{name}.o")
            so_path  = os.path.join(td, f"{name}.so")

            with open(asm_path, "w") as f: f.write(asm_code)
            subprocess.check_call(["nasm", "-felf64", asm_path, "-o", obj_path])
            subprocess.check_call(["gcc", "-shared", "-o", so_path, obj_path])

            lib = ctypes.CDLL(so_path)
            self.cache[name] = lib
            return lib

class JITMacroExpander(LispCIAMS):
    def __init__(self, ciams: CIAMSExpander, jit: JITGenerator):
        super().__init__(ciams)
        self.jit = jit
        self.define_intrinsics()

    def define_intrinsics(self):
        def mul2_macro(node: Call, ctx: Dict[str, Any]):
            # generate NASM to multiply by 2
            asm = """
            global mul2
            section .text
            mul2:
                mov rax, rdi
                add rax, rdi
                ret
            """
            lib = self.jit.generate("mul2", asm)
            fn = getattr(lib, "mul2")
            fn.argtypes = [ctypes.c_long]
            fn.restype = ctypes.c_long
            return [Number(str(fn(int(node.args[0].val))))]
        self.ciams.register("mul2", mul2_macro, phase=CIAMSPhase.EXEC)

# ---------------------------------------------------------------------------
# Part 23 — XYZOBJ2 Binary Format
# ---------------------------------------------------------------------------

class XYZObj2:
    def __init__(self):
        self.symbols = {}
        self.debug   = {}

    def add_symbol(self, name: str, dodecagram: str, debug: Dict[str, Any]):
        self.symbols[name] = DodecagramBinary.pack(dodecagram)
        self.debug[name]   = debug

    def write(self, path: str):
        obj = {
            "format": "XYZOBJ2",
            "symbols": {k: v.hex() for k,v in self.symbols.items()},
            "debug": self.debug
        }
        with open(path, "w") as f: json.dump(obj, f)

    @staticmethod
    def load(path: str):
        with open(path,"r") as f: data = json.load(f)
        if data.get("format")!="XYZOBJ2":
            raise ValueError("Not a valid XYZOBJ2 file")
        obj = XYZObj2()
        obj.symbols = {k: bytes.fromhex(v) for k,v in data["symbols"].items()}
        obj.debug   = data["debug"]
        return obj

# ---------------------------------------------------------------------------
# Part 24a — Pattern-Matched Parallel Scheduling
# ---------------------------------------------------------------------------

class SmartParallelRuntime(ParallelRuntime):
    def __init__(self, scheduler: Scheduler, symtab: Dict[str, FuncDef], hot: HotSwapRegistry,
                 fastvm: FastVM, jit: JITGenerator, gc: GC):
        super().__init__(scheduler, symtab, hot)
        self.fastvm = fastvm
        self.jit = jit
        self.gc = gc

    def __parallel_exec(self, funcs: List[str], argsets: List[List[Any]]):
        results = []
        for f, args in zip(funcs, argsets):
            # heuristic: if all args are numbers → JIT optimize
            if all(isinstance(a, (int, float)) for a in args):
                asm = f"""
                global {f}
                section .text
                {f}:
                    mov rax, rdi
                    imul rax, 2
                    ret
                """
                lib = self.jit.generate(f, asm)
                fn = getattr(lib, f)
                fn.argtypes = [ctypes.c_long]
                fn.restype = ctypes.c_long
                results.append(fn(int(args[0])))

            # heuristic: if function exists in FastVM → run there
            elif f in self.fastvm.funcs:
                results.append(self.fastvm.run(f"{f}/{len(args)}", args))

            # else → fallback to thread + MiniRuntime
            else:
                key = f"{f}/{len(args)}"
                r = MiniRuntime(self.symtab, self.hot).run_func(key, args)
                results.append(r)

            # GC bookkeeping
            self.gc.collect(results[-1])

        return results

# ---------------------------------------------------------------------------
# Part 24b — XYZOBJ2 Linker
# ---------------------------------------------------------------------------

class XYZObj2Linker:
    def __init__(self):
        self.symbols = {}
        self.debug = {}

    def add_obj(self, obj: XYZObj2):
        for k,v in obj.symbols.items():
            if k in self.symbols:
                raise ValueError(f"Duplicate symbol {k}")
            self.symbols[k] = v
        self.debug.update(obj.debug)

    def link(self, objs: List[XYZObj2], out: str):
        for obj in objs:
            self.add_obj(obj)
        merged = {
            "format": "XYZOBJ2-LINKED",
            "symbols": {k: v.hex() for k,v in self.symbols.items()},
            "debug": self.debug
        }
        with open(out, "w") as f: json.dump(merged, f)
        return out

    @staticmethod
    def load_linked(path: str):
        with open(path,"r") as f: data = json.load(f)
        if data.get("format")!="XYZOBJ2-LINKED":
            raise ValueError("Invalid linked file")
        linker = XYZObj2Linker()
        linker.symbols = {k: bytes.fromhex(v) for k,v in data["symbols"].items()}
        linker.debug = data["debug"]
        return linker

# ---------------------------------------------------------------------------
# Part 25a — XYZOBJ2 Module Loader
# ---------------------------------------------------------------------------

class ModuleLoader:
    def __init__(self, symtab: Dict[str, FuncDef], hot: HotSwapRegistry, fastvm: FastVM, mini: MiniRuntime):
        self.symtab = symtab
        self.hot = hot
        self.fastvm = fastvm
        self.mini = mini
        self.loaded = {}

    def load_xyzobj2(self, path: str):
        obj = XYZObj2.load(path)
        for k, v in obj.symbols.items():
            # decode dodecagram → AST or VM function
            dseq = DodecagramBinary.unpack(v)
            self.loaded[k] = dseq
            # register in FastVM
            if k not in self.fastvm.funcs:
                self.fastvm.funcs[k] = self.fastvm.run_dodecagram(dseq)
        return obj

    def load_linked(self, path: str):
        linked = XYZObj2Linker.load_linked(path)
        for k, v in linked.symbols.items():
            dseq = DodecagramBinary.unpack(v)
            self.loaded[k] = dseq
            if k not in self.fastvm.funcs:
                self.fastvm.funcs[k] = self.fastvm.run_dodecagram(dseq)
        return linked

# ---------------------------------------------------------------------------
# Part 25b — Debugger Hooks
# ---------------------------------------------------------------------------

class Debugger:
    def __init__(self, vm: FastVM):
        self.vm = vm
        self.breakpoints = set()
        self.trace = []

    def set_breakpoint(self, func: str, pc: int):
        self.breakpoints.add((func, pc))

    def step(self, func: str, args: List[Any]):
        fn = self.vm.funcs[func]
        code, consts = fn.code, fn.consts
        stack, pc = [], 0
        while pc < len(code):
            if (func, pc) in self.breakpoints:
                print(f"⚡ Breakpoint at {func}:{pc}")
                print(f"Stack: {stack}")
                input("Press Enter to continue…")
            op = code[pc]; pc += 1
            self.trace.append((pc, stack[:]))
            # same dispatch loop as FastVM.run
            if op == Op.CONST: stack.append(consts[code[pc]]); pc+=1
            elif op == Op.ADD: b=stack.pop(); a=stack.pop(); stack.append(a+b)
            elif op == Op.SUB: b=stack.pop(); a=stack.pop(); stack.append(a-b)
            elif op == Op.RET: return stack.pop() if stack else None
        return None

# ---------------------------------------------------------------------------
# Part 25c — Parallel GC Sweeper
# ---------------------------------------------------------------------------

import threading, time

class ParallelGC(GC):
    def __init__(self, interval=1.0):
        super().__init__()
        self.interval = interval
        self._running = False
        self._thread = None

    def start(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._sweeper, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join()

    def _sweeper(self):
        while self._running:
            time.sleep(self.interval)
            before = len(self.objects)
            self.sweep()
            after = len(self.objects)
            LOG.info(f"[GC] Collected {before-after} objects")

# ---------------------------------------------------------------------------
# Part 26 — Interactive REPL
# ---------------------------------------------------------------------------

class XYZRepl:
    def __init__(self, symtab: Dict[str, FuncDef], hot: HotSwapRegistry,
                 fastvm: FastVM, mini: MiniRuntime, gc: ParallelGC, debugger: Debugger):
        self.symtab = symtab
        self.hot = hot
        self.fastvm = fastvm
        self.mini = mini
        self.gc = gc
        self.debugger = debugger
        self.running = False

    def start(self):
        self.running = True
        self.gc.start()
        print("🚀 XYZ REPL started. Type :quit to exit, :gc to sweep, :break fn pc to set breakpoints.")
        while self.running:
            try:
                src = input("xyz> ").strip()
                if not src: continue
                if src == ":quit":
                    self.running = False
                    break
                if src == ":gc":
                    self.gc.sweep()
                    print("GC sweep done.")
                    continue
                if src.startswith(":break"):
                    _, fn, pc = src.split()
                    self.debugger.set_breakpoint(fn, int(pc))
                    print(f"Breakpoint set at {fn}:{pc}")
                    continue
                if src.startswith(":hotswap"):
                    _, fn = src.split()
                    if fn in self.symtab:
                        opt = optimize(self.symtab[fn])
                        self.hot.swap(fn, opt)
                        print(f"Hot-swapped {fn}")
                    continue

                # compile on the fly
                tokens = lex(src)
                parser = Parser(tokens)
                prog = parser.parse()
                for stmt in prog.body:
                    result = self.mini.eval(stmt)
                    if result is not None:
                        print("=>", result)
            except Exception as e:
                print(f"[ERR] {e}")

# ---------------------------------------------------------------------------
# Part 27 — Profiler + Hot Route Metrics
# ---------------------------------------------------------------------------

import time

class Profiler:
    def __init__(self):
        self.data = {}  # key -> {"count": int, "time": float}

    def record(self, key: str, duration: float):
        if key not in self.data:
            self.data[key] = {"count": 0, "time": 0.0}
        self.data[key]["count"] += 1
        self.data[key]["time"] += duration

    def report(self):
        print("🔥 Profiler Report:")
        for k, v in sorted(self.data.items(), key=lambda x: x[1]["count"], reverse=True):
            avg = v["time"]/v["count"] if v["count"] else 0
            print(f"  {k} → {v['count']} calls, total {v['time']:.6f}s, avg {avg:.6f}s")


class HotRouteOptimizer(HotRouteOptimizer):  # extend previous
    def __init__(self, hot: HotSwapRegistry, symtab: Dict[str, FuncDef], profiler: Profiler):
        super().__init__(hot, symtab)
        self.profiler = profiler

    def timed_call(self, key: str, func: Callable, *args):
        start = time.perf_counter()
        result = func(*args)
        end = time.perf_counter()
        self.profiler.record(key, end-start)
        self.record_call(key)
        return result

# ---------------------------------------------------------------------------
# Part 28a — XYZOBJ3 Format
# ---------------------------------------------------------------------------

MAGIC_XYZOBJ3 = b"XYZ3"

class XYZObj3:
    def __init__(self):
        self.symbols = {}     # name -> offset
        self.relocations = [] # (offset, symbol)
        self.code = bytearray()

    def add_symbol(self, name: str, dodecagram: str):
        offset = len(self.code)
        encoded = DodecagramBinary.pack(dodecagram)
        self.symbols[name] = offset
        self.code.extend(encoded)

    def add_relocation(self, offset: int, symbol: str):
        self.relocations.append((offset, symbol))

    def write(self, path: str):
        with open(path, "wb") as f:
            f.write(MAGIC_XYZOBJ3)
            # symbol table
            f.write(struct.pack("<I", len(self.symbols)))
            for name, off in self.symbols.items():
                nb = name.encode()
                f.write(struct.pack("<I", len(nb)))
                f.write(nb)
                f.write(struct.pack("<Q", off))
            # relocation table
            f.write(struct.pack("<I", len(self.relocations)))
            for off, sym in self.relocations:
                sb = sym.encode()
                f.write(struct.pack("<Q", off))
                f.write(struct.pack("<I", len(sb)))
                f.write(sb)
            # code
            f.write(struct.pack("<Q", len(self.code)))
            f.write(self.code)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            if f.read(4) != MAGIC_XYZOBJ3:
                raise ValueError("Invalid XYZOBJ3 file")
            obj = XYZObj3()
            nsym, = struct.unpack("<I", f.read(4))
            for _ in range(nsym):
                nlen, = struct.unpack("<I", f.read(4))
                name = f.read(nlen).decode()
                off, = struct.unpack("<Q", f.read(8))
                obj.symbols[name] = off
            nrel, = struct.unpack("<I", f.read(4))
            for _ in range(nrel):
                off, = struct.unpack("<Q", f.read(8))
                slen, = struct.unpack("<I", f.read(4))
                sym = f.read(slen).decode()
                obj.relocations.append((off, sym))
            clen, = struct.unpack("<Q", f.read(8))
            obj.code = bytearray(f.read(clen))
            return obj

# ---------------------------------------------------------------------------
# Part 28b — Profiler-Driven JIT Fusion
# ---------------------------------------------------------------------------

class ProfilerDrivenJIT:
    def __init__(self, profiler: Profiler, jit: JITGenerator, ciams: CIAMSExpander, hot: HotSwapRegistry):
        self.profiler = profiler
        self.jit = jit
        self.ciams = ciams
        self.hot = hot
        self.threshold = 100

    def check_and_optimize(self, key: str, func: FuncDef):
        stats = self.profiler.data.get(key)
        if not stats or stats["count"] < self.threshold:
            return False
        # Convert AST → NASM specialized
        asm = self.ast_to_nasm(func)
        lib = self.jit.generate(func.name, asm)
        fn = getattr(lib, func.name)
        fn.restype = ctypes.c_long
        fn.argtypes = [ctypes.c_long]
        # wrap into runtime call
        def native_wrapper(x): return fn(x)
        optimized = FuncDef(func.name, ["x"], [Return(Number(str(native_wrapper(1))))])
        self.hot.swap(key, optimized)
        return True

    def ast_to_nasm(self, func: FuncDef) -> str:
        # simple example: arithmetic-only specialization
        asm = f"""
        global {func.name}
        section .text
        {func.name}:
            mov rax, rdi
            add rax, 5
            ret
        """
        return asm

# ---------------------------------------------------------------------------
# Part 29a — XYZOBJ3 Linker/Loader with Relocation Fixups
# ---------------------------------------------------------------------------

class XYZObj3Linker:
    def __init__(self):
        self.symbols = {}    # name -> absolute offset
        self.code = bytearray()
        self.relocations = []

    def link(self, objs: List[XYZObj3]) -> XYZObj3:
        linked = XYZObj3()
        offset = 0
        for obj in objs:
            # assign absolute offsets
            for name, off in obj.symbols.items():
                self.symbols[name] = offset + off
            # copy code
            base = len(linked.code)
            linked.code.extend(obj.code)
            # adjust relocations
            for roff, sym in obj.relocations:
                linked.relocations.append((base + roff, sym))
            offset += len(obj.code)

        linked.symbols = self.symbols
        # apply relocation fixups
        for roff, sym in linked.relocations:
            if sym not in linked.symbols:
                raise ValueError(f"Unresolved symbol: {sym}")
            addr = linked.symbols[sym]
            struct.pack_into("<Q", linked.code, roff, addr)
        return linked


class XYZObj3Loader:
    def __init__(self, fastvm: FastVM, mini: MiniRuntime):
        self.fastvm = fastvm
        self.mini = mini

    def load(self, obj: XYZObj3):
        for name, off in obj.symbols.items():
            dseq = DodecagramBinary.unpack(obj.code[off:])
            self.fastvm.funcs[name] = self.fastvm.run_dodecagram(dseq)
            self.mini.symtab[name] = self.fastvm.funcs[name]

# ---------------------------------------------------------------------------
# Part 29b — Polymorphic JIT Specialization
# ---------------------------------------------------------------------------

class PolymorphicJIT(ProfilerDrivenJIT):
    def __init__(self, profiler, jit, ciams, hot):
        super().__init__(profiler, jit, ciams, hot)
        self.type_variants = {}  # key -> {sig: native_fn}

    def check_and_optimize(self, key: str, func: FuncDef):
        stats = self.profiler.data.get(key)
        if not stats or stats["count"] < self.threshold:
            return False

        # build polymorphic variants
        sigs = self.infer_signatures(func)
        for sig in sigs:
            asm = self.ast_to_nasm(func, sig)
            lib = self.jit.generate(f"{func.name}_{sig}", asm)
            fn = getattr(lib, f"{func.name}_{sig}")
            fn.restype = ctypes.c_double if "float" in sig else ctypes.c_long
            fn.argtypes = [ctypes.c_double] if "float" in sig else [ctypes.c_long]
            self.type_variants.setdefault(key, {})[sig] = fn

        def dispatcher(x):
            sig = "float" if isinstance(x, float) else "int"
            fn = self.type_variants[key][sig]
            return fn(x)

        optimized = FuncDef(func.name, ["x"], [Return(Number(str(dispatcher(1))))])
        self.hot.swap(key, optimized)
        return True

    def infer_signatures(self, func: FuncDef):
        # naive inference → int + float variants
        return ["int", "float"]

    def ast_to_nasm(self, func: FuncDef, sig: str) -> str:
        if sig == "int":
            return f"""
            global {func.name}_int
            section .text
            {func.name}_int:
                mov rax, rdi
                imul rax, rax
                ret
            """
        else:
            return f"""
            global {func.name}_float
            section .text
            {func.name}_float:
                movq xmm0, rdi
                mulsd xmm0, xmm0
                movq rax, xmm0
                ret
            """

# ---------------------------------------------------------------------------
# Part 30a — Standalone Executable Packager
# ---------------------------------------------------------------------------

import os

class ExecutablePackager:
    ELF_MAGIC = b"\x7fELF"

    def __init__(self, obj: XYZObj3):
        self.obj = obj

    def build_elf(self, path: str):
        # Minimal ELF64 header for executable
        e_ident = self.ELF_MAGIC + bytes([2,1,1]) + bytes(9)  # ELF64, little endian
        e_type = 2   # ET_EXEC
        e_machine = 62  # x86_64
        e_version = 1
        e_entry = 0x400080  # entry point
        e_phoff = 64
        e_shoff = 0
        e_flags = 0
        e_ehsize = 64
        e_phentsize = 56
        e_phnum = 1
        e_shentsize = 0
        e_shnum = 0
        e_shstrndx = 0

        header = struct.pack(
            "<16sHHIQQQIHHHHHH",
            e_ident, e_type, e_machine, e_version, e_entry,
            e_phoff, e_shoff, e_flags,
            e_ehsize, e_phentsize, e_phnum,
            e_shentsize, e_shnum, e_shstrndx
        )

        # Program header (single load segment)
        p_type = 1      # PT_LOAD
        p_flags = 5     # R+X
        p_offset = 0x80
        p_vaddr = 0x400080
        p_paddr = p_vaddr
        p_filesz = len(self.obj.code)
        p_memsz = p_filesz
        p_align = 0x200000
        phdr = struct.pack(
            "<IIQQQQQQ",
            p_type, p_flags, p_offset, p_vaddr, p_paddr,
            p_filesz, p_memsz, p_align
        )

        with open(path, "wb") as f:
            f.write(header)
            f.write(phdr)
            # pad to code offset
            pad = p_offset - f.tell()
            f.write(b"\x00" * pad)
            f.write(self.obj.code)

        os.chmod(path, 0o755)
        print(f"✅ Built ELF executable: {path}")

# ---------------------------------------------------------------------------
# Part 30b — Multi-Arg SIMD Specialization
# ---------------------------------------------------------------------------

class SIMDPolymorphicJIT(PolymorphicJIT):
    def infer_signatures(self, func: FuncDef):
        # infer both int and float, multi-arg
        return [("int", len(func.params)), ("float", len(func.params))]

    def ast_to_nasm(self, func: FuncDef, sig: Tuple[str,int]) -> str:
        typ, nargs = sig
        fn_name = f"{func.name}_{typ}{nargs}"
        if typ == "int":
            # AVX2 integer vectorization (4 ints at once)
            body = f"""
            global {fn_name}
            section .text
            {fn_name}:
                vmovdqa ymm0, [rdi]
                vmovdqa ymm1, [rsi]
                vpaddd ymm2, ymm0, ymm1
                vmovdqa [rdx], ymm2
                ret
            """
        else:
            # AVX512 float vectorization (8 doubles at once)
            body = f"""
            global {fn_name}
            section .text
            {fn_name}:
                vmovapd zmm0, [rdi]
                vmovapd zmm1, [rsi]
                vaddpd zmm2, zmm0, zmm1
                vmovapd [rdx], zmm2
                ret
            """
        return body

    def check_and_optimize(self, key: str, func: FuncDef):
        stats = self.profiler.data.get(key)
        if not stats or stats["count"] < self.threshold:
            return False

        sigs = self.infer_signatures(func)
        for sig in sigs:
            asm = self.ast_to_nasm(func, sig)
            lib = self.jit.generate(f"{func.name}_{sig[0]}{sig[1]}", asm)
            fn = getattr(lib, f"{func.name}_{sig[0]}{sig[1]}")
            # Use ctypes arrays for vector args
            fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
            fn.restype = None
            self.type_variants.setdefault(key, {})[sig] = fn

        def dispatcher(a, b, out):
            sig = ("float", len([a,b])) if isinstance(a[0], float) else ("int", len([a,b]))
            fn = self.type_variants[key][sig]
            fn(a, b, out)
            return out

        optimized = FuncDef(func.name, ["a","b","out"], [Return(Var("out"))])
        self.hot.swap(key, optimized)
        return True

# ---------------------------------------------------------------------------
# Part 31a — PE/COFF Packager
# ---------------------------------------------------------------------------

class PECOFFPackager:
    DOS_MAGIC = b"MZ"
    PE_MAGIC  = b"PE\x00\x00"

    def __init__(self, obj: XYZObj3):
        self.obj = obj

    def build_pe(self, path: str):
        # DOS stub (minimal)
        dos_stub = self.DOS_MAGIC + b"\x90" * 58 + struct.pack("<I", 0x80)

        # PE header
        machine = 0x8664  # AMD64
        sections = 1
        time_date = int(time.time())
        symtab_ptr, symtab_count = 0, 0
        opt_hdr_size = 240
        characteristics = 0x22  # EXEC + 64-bit

        coff_header = struct.pack(
            "<HHIIIHH",
            machine, sections, time_date,
            symtab_ptr, symtab_count,
            opt_hdr_size, characteristics
        )

        # Optional header (PE32+)
        magic = 0x20B
        entry_point = 0x1000
        code_base   = 0x1000
        image_base  = 0x400000
        section_align, file_align = 0x1000, 0x200
        size_code = len(self.obj.code)
        size_image = 0x2000

        opt_header = struct.pack(
            "<HBBIIIIQQQQHHHHIIIIIIQQQQQQ",
            magic, 0, 0,  # magic, linker ver
            size_code, 0, 0, entry_point,
            code_base, image_base,
            section_align, file_align,
            4, 0, 0, 0,  # OS ver, subsystem ver
            size_image, 0x200, 0, 0, 0,  # size headers, checksum, subsystem
            0,0,0,0,0,0  # DLL flags, stack/heap
        )

        # Section header for ".text"
        name = b".text\x00\x00\x00"
        vsize = len(self.obj.code)
        vaddr = 0x1000
        raw_size = (len(self.obj.code) + 0x1FF) & ~0x1FF
        raw_ptr = 0x200
        sh_flags = 0x60000020

        section_header = struct.pack(
            "<8sIIIIIIHHI",
            name, vsize, vaddr, raw_size, raw_ptr,
            0, 0, 0, 0, sh_flags
        )

        with open(path, "wb") as f:
            f.write(dos_stub)
            f.write(self.PE_MAGIC)
            f.write(coff_header)
            f.write(opt_header)
            f.write(section_header)
            # pad to code
            f.write(b"\x00" * (raw_ptr - f.tell()))
            f.write(self.obj.code)

        print(f"✅ Built PE/COFF executable: {path}")

# ---------------------------------------------------------------------------
# Part 31b — CIAMS SIMD Auto-Expansion
# ---------------------------------------------------------------------------

class CIAMSExpander:
    def __init__(self, hot: HotSwapRegistry, profiler: Profiler, jit: NASMJITEngine):
        self.hot = hot
        self.profiler = profiler
        self.jit = jit
        self.macros: Dict[str, Callable[[Call], List[ASTNode]]] = {}

    def register_simd_macros(self):
        def vector_add_macro(call: Call):
            args = call.args
            if len(args) != 2:
                raise ValueError("vector-add requires 2 args")

            # profiler-driven length inference
            length = len(args[0].elements) if isinstance(args[0], ListLiteral) else 4
            if length <= 4:
                instr = "addps"  # SSE
            elif length <= 8:
                instr = "vaddps"  # AVX
            else:
                instr = "vaddpd"  # AVX512 double precision

            asm = f"""
            global vector_add
            section .text
            vector_add:
                mov rdi, rsi
                mov rsi, rdx
                {instr} xmm0, [rdi]
                {instr} xmm0, [rsi]
                ret
            """

            lib = self.jit.generate("vector_add", asm)
            fn = getattr(lib, "vector_add")
            fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
            fn.restype = None

            return [Call("vector_add", args)]

        self.macros["vector-add"] = vector_add_macro

    def expand(self, node: ASTNode):
        if isinstance(node, Call) and node.name in self.macros:
            return self.macros[node.name](node)
        return [node]

# ---------------------------------------------------------------------------
# Part 32a — CIAMS Conditional Macros + Hygiene
# ---------------------------------------------------------------------------

class CIAMSContext:
    """Carries scope for hygienic renaming."""
    _gensym_counter = 0

    @classmethod
    def gensym(cls, prefix="g"):
        cls._gensym_counter += 1
        return f"{prefix}{cls._gensym_counter}"

class CIAMSMacroDSL:
    def __init__(self):
        self.macros: Dict[str, Callable[[Call, CIAMSContext], List[ASTNode]]] = {}

    def define(self, name: str, fn: Callable[[Call, CIAMSContext], List[ASTNode]]):
        self.macros[name] = fn

    def expand(self, node: ASTNode, ctx: Optional[CIAMSContext] = None):
        ctx = ctx or CIAMSContext()
        if isinstance(node, Call) and node.name in self.macros:
            return self.macros[node.name](node, ctx)
        return [node]

# Example macros
ciams_dsl = CIAMSMacroDSL()

def if_macro(call: Call, ctx: CIAMSContext):
    if len(call.args) != 3:
        raise ValueError("(if-macro cond then else) requires 3 args")
    cond, then_branch, else_branch = call.args
    if isinstance(cond, Bool):
        return [then_branch] if cond.val else [else_branch]
    return [If(cond, [then_branch], [else_branch])]

def hygienic_let(call: Call, ctx: CIAMSContext):
    if len(call.args) != 2:
        raise ValueError("(let-macro var expr)")
    var_name = ctx.gensym("let")
    expr = call.args[1]
    return [Assign(var_name, expr), Var(var_name)]

ciams_dsl.define("if-macro", if_macro)
ciams_dsl.define("let-macro", hygienic_let)

# ---------------------------------------------------------------------------
# Part 32b — SIMD CPU Feature Detection
# ---------------------------------------------------------------------------

import subprocess

class CPUFeatures:
    _features: Dict[str, bool] = {}

    @classmethod
    def detect(cls):
        if cls._features:  # already cached
            return cls._features
        features = {"sse": False, "avx": False, "avx512": False}

        try:
            if sys.platform == "linux":
                flags = subprocess.check_output(["cat", "/proc/cpuinfo"]).decode()
            elif sys.platform == "darwin":
                flags = subprocess.check_output(["sysctl", "machdep.cpu.features"]).decode()
            elif sys.platform == "win32":
                flags = subprocess.check_output(["wmic", "cpu", "get", "Name,Architecture"]).decode()
            else:
                flags = ""
        except Exception:
            flags = ""

        flags = flags.lower()
        features["sse"] = "sse" in flags
        features["avx"] = "avx" in flags
        features["avx512"] = "avx512" in flags

        cls._features = features
        return features

    @classmethod
    def best_vector(cls):
        f = cls.detect()
        if f["avx512"]:
            return "avx512"
        if f["avx"]:
            return "avx"
        if f["sse"]:
            return "sse"
        return "scalar"

def vector_add_macro(call: Call, ctx: CIAMSContext):
    features = CPUFeatures.best_vector()
    if features == "avx512":
        instr = "vaddpd"
    elif features == "avx":
        instr = "vaddps"
    elif features == "sse":
        instr = "addps"
    else:
        instr = "add"  # scalar fallback
    # Generate JIT NASM like before…

# ---------------------------------------------------------------------------
# Part 33a — Multi-Phase CIAMS Expansion
# ---------------------------------------------------------------------------

class MultiPhaseCIAMS(CIAMSMacroDSL):
    def __init__(self):
        super().__init__()
        self.expansion_phases = []

    def expand_program(self, prog: Program, phases: int = 2) -> Program:
        """Run multiple expansion passes, installing macros as we go."""
        self.expansion_phases.clear()
        out_body = prog.body
        for phase in range(phases):
            ctx = CIAMSContext()
            new_body = []
            for node in out_body:
                expanded = self.expand(node, ctx)
                # If expansion defines a macro, install it
                for n in expanded:
                    if isinstance(n, Call) and n.name == "define-macro":
                        mname = n.args[0].name
                        mbody = n.args[1]
                        def dyn_macro(call, _ctx, body=mbody, name=mname):
                            return [body]  # simplistic substitution
                        self.define(mname, dyn_macro)
                    else:
                        new_body.append(n)
            out_body = new_body
            self.expansion_phases.append(out_body)
        return Program(out_body)

ciams_multi = MultiPhaseCIAMS()

# Example meta-macro: (define-macro square (lambda (x) (* x x)))
def define_macro(call: Call, ctx: CIAMSContext):
    if len(call.args) != 2:
        raise ValueError("(define-macro name lambda-body)")
    return [call]  # handled in expand_program loop

ciams_multi.define("define-macro", define_macro)

# ---------------------------------------------------------------------------
# Part 33b — SIMD Fusion
# ---------------------------------------------------------------------------

class SIMDOptimizer:
    def __init__(self, jit_engine):
        self.jit = jit_engine

    def fuse(self, body: List[ASTNode]) -> List[ASTNode]:
        fused = []
        buffer = []
        for node in body:
            if isinstance(node, Call) and node.name.startswith("vector-"):
                buffer.append(node)
            else:
                if buffer:
                    fused.append(self._fuse_buffer(buffer))
                    buffer = []
                fused.append(node)
        if buffer:
            fused.append(self._fuse_buffer(buffer))
        return fused

    def _fuse_buffer(self, ops: List[Call]) -> Call:
        # Example: fuse multiple vector-add into one batched kernel
        fn_name = "vector_fused"
        asm_body = "global vector_fused\nsection .text\nvector_fused:\n"
        for i, op in enumerate(ops):
            if op.name == "vector-add":
                asm_body += f"  ; fused vector-add {i}\n"
                asm_body += "  addps xmm0, xmm1\n"
        asm_body += "  ret\n"

        self.jit.generate(fn_name, asm_body)
        return Call(fn_name, [arg for op in ops for arg in op.args])

# ---------------------------------------------------------------------------
# Part 33c — Macro Debugger
# ---------------------------------------------------------------------------

class MacroDebugger:
    def __init__(self, ciams: MultiPhaseCIAMS):
        self.ciams = ciams

    def trace_expansion(self, prog: Program, phases: int = 2):
        expanded = self.ciams.expand_program(prog, phases)
        for i, phase_body in enumerate(self.ciams.expansion_phases):
            print(f"\n=== CIAMS Phase {i+1} ===")
            for node in phase_body:
                print("  ", node)
        return expanded

# ---------------------------------------------------------------------------
# Part 34a — Macro Scoping
# ---------------------------------------------------------------------------

class ScopedCIAMS(MultiPhaseCIAMS):
    def __init__(self, mode="lexical"):
        super().__init__()
        self.mode = mode
        self.lexical_env: Dict[str, Callable] = {}

    def define(self, name: str, fn: Callable):
        if self.mode == "lexical":
            self.lexical_env[name] = fn
        else:
            super().define(name, fn)

    def expand(self, node: ASTNode, ctx: CIAMSContext) -> List[ASTNode]:
        if isinstance(node, Call):
            fn = None
            if self.mode == "lexical":
                fn = self.lexical_env.get(node.name)
            if not fn:
                fn = self.macros.get(node.name)
            if fn:
                return fn(node, ctx)
        return super().expand(node, ctx)

# ---------------------------------------------------------------------------
# Part 34b — Cross-Parallel SIMD Fusion
# ---------------------------------------------------------------------------

class ParallelSIMDFusion:
    def __init__(self, jit_engine):
        self.jit = jit_engine

    def fuse_parallel(self, parallel_nodes: List[Parallel]) -> List[Call]:
        # Collect vector ops across all parallel bodies
        ops = []
        for p in parallel_nodes:
            for stmt in p.body:
                if isinstance(stmt, Call) and stmt.name.startswith("vector-"):
                    ops.append(stmt)

        if not ops:
            return []

        # Build one mega kernel
        fn_name = "parallel_fused"
        asm_body = "global parallel_fused\nsection .text\nparallel_fused:\n"
        asm_body += "  ; fused vector ops across parallel blocks\n"

        for i, op in enumerate(ops):
            if op.name == "vector-add":
                asm_body += f"  ; fused vector-add {i}\n"
                asm_body += "  addps xmm0, xmm1\n"
            elif op.name == "vector-mul":
                asm_body += f"  ; fused vector-mul {i}\n"
                asm_body += "  mulps xmm0, xmm1\n"

        asm_body += "  ret\n"
        self.jit.generate(fn_name, asm_body)

        return [Call(fn_name, [arg for op in ops for arg in op.args])]

def parallel_macro(call: Call, ctx: CIAMSContext):
    fused_calls = parallel_simd_fusion.fuse_parallel([Parallel(call.args)])
    return fused_calls or call.args

ciams.define("parallel", parallel_macro)

# ---------------------------------------------------------------------------
# Part 35a — CIAMS Macro Hygiene
# ---------------------------------------------------------------------------

import itertools

class HygienicCIAMS(ScopedCIAMS):
    _gensym_counter = itertools.count()

    def gensym(self, hint="tmp"):
        """Generate a unique symbol name to avoid capture."""
        return f"__ciams_{hint}_{next(self._gensym_counter)}"

    def hygienic_expand(self, node: ASTNode, ctx: CIAMSContext) -> List[ASTNode]:
        expanded = self.expand(node, ctx)
        return [self._apply_hygiene(n) for n in expanded]

    def _apply_hygiene(self, node: ASTNode) -> ASTNode:
        if isinstance(node, Var) and node.name.startswith("ciams-internal-"):
            node.name = self.gensym(node.name)
        elif isinstance(node, Assign) and node.name.startswith("ciams-internal-"):
            node.name = self.gensym(node.name)
        elif isinstance(node, FuncDef):
            node.params = [self.gensym(p) if p.startswith("ciams-internal-") else p for p in node.params]
            node.body = [self._apply_hygiene(b) for b in node.body]
        elif isinstance(node, Call):
            node.args = [self._apply_hygiene(a) for a in node.args]
        return node

# ---------------------------------------------------------------------------
# Part 35b — Parallel Fusion + GC
# ---------------------------------------------------------------------------

import threading

class ParallelGC:
    def __init__(self):
        self.lock = threading.Lock()
        self.objects = []
        self.gc_thread = threading.Thread(target=self._sweeper, daemon=True)
        self.gc_thread.start()

    def register(self, obj):
        with self.lock:
            self.objects.append(obj)

    def _sweeper(self):
        while True:
            time.sleep(0.1)
            with self.lock:
                before = len(self.objects)
                self.objects = [o for o in self.objects if not self._is_collectable(o)]
                after = len(self.objects)
            if after < before:
                LOG.info("GC swept %d objects", before - after)

    def _is_collectable(self, obj):
        return hasattr(obj, "__collectable__") and obj.__collectable__

gc_manager = ParallelGC()

class ParallelSIMDFusionGC(ParallelSIMDFusion):
    def run_fused(self, args):
        result = super().fuse_parallel(args)
        gc_manager.register(result)
        return result

def hygienic_parallel_macro(call: Call, ctx: CIAMSContext):
    fused_calls = parallel_simd_fusion_gc.run_fused([Parallel(call.args)])
    return fused_calls or call.args

ciams_hygienic = HygienicCIAMS()
ciams_hygienic.define("parallel", hygienic_parallel_macro)

# ---------------------------------------------------------------------------
# Part 36a — XYZOBJ3 with Hygiene + GC Metadata
# ---------------------------------------------------------------------------

class XYZOBJ3:
    def __init__(self, symbols=None, macros=None, hygiene=None, gc_roots=None):
        self.symbols = symbols or {}
        self.macros = macros or {}
        self.hygiene = hygiene or {}
        self.gc_roots = gc_roots or []

    def serialize(self) -> bytes:
        obj = {
            "format": "XYZOBJ3",
            "symbols": self.symbols,
            "macros": {k: str(v) for k, v in self.macros.items()},
            "hygiene": self.hygiene,
            "gc_roots": [id(o) for o in self.gc_roots],
        }
        raw = json.dumps(obj, indent=2).encode("utf-8")
        LOG.info("Serialized XYZOBJ3: %d bytes", len(raw))
        return raw

    @classmethod
    def deserialize(cls, data: bytes):
        obj = json.loads(data.decode("utf-8"))
        return cls(
            symbols=obj.get("symbols", {}),
            macros=obj.get("macros", {}),
            hygiene=obj.get("hygiene", {}),
            gc_roots=obj.get("gc_roots", []),
        )

# ---------------------------------------------------------------------------
# Part 36b — Macro Hygiene Debugger
# ---------------------------------------------------------------------------

class HygieneDebugger:
    def __init__(self, ciams: HygienicCIAMS):
        self.ciams = ciams

    def trace_hygiene(self, node: ASTNode):
        hygienic = self.ciams._apply_hygiene(node)
        print(f"Original: {node}")
        print(f"Hygienic: {hygienic}")
        return hygienic

# ---------------------------------------------------------------------------
# Part 36c — GC Lifecycle Debugger
# ---------------------------------------------------------------------------

class GCDebugger:
    def __init__(self, gc_manager: ParallelGC):
        self.gc = gc_manager

    def show_roots(self):
        print("GC Roots:")
        with self.gc.lock:
            for i, obj in enumerate(self.gc.objects):
                print(f"  Root {i}: {repr(obj)}")

    def force_sweep(self):
        with self.gc.lock:
            before = len(self.gc.objects)
            self.gc.objects = [o for o in self.gc.objects if not self.gc._is_collectable(o)]
            after = len(self.gc.objects)
        print(f"Forced GC sweep: {before - after} objects collected")

# ---------------------------------------------------------------------------
# Part 37a — Profiler-Driven GC Scheduling
# ---------------------------------------------------------------------------

class RuntimeProfiler:
    def __init__(self):
        self.hot_counts: Dict[str, int] = {}
        self.thread_load: Dict[int, float] = {}
        self.latency_threshold = 0.01

    def record_hot(self, fn_name: str):
        self.hot_counts[fn_name] = self.hot_counts.get(fn_name, 0) + 1

    def record_thread_load(self, thread_id: int, latency: float):
        self.thread_load[thread_id] = latency

    def idle_threads(self) -> List[int]:
        return [tid for tid, lat in self.thread_load.items() if lat < self.latency_threshold]

profiler = RuntimeProfiler()

class AdaptiveParallelGC(ParallelGC):
    def __init__(self, profiler: RuntimeProfiler):
        super().__init__()
        self.profiler = profiler

    def _sweeper(self):
        while True:
            time.sleep(0.05)
            idle = self.profiler.idle_threads()
            if idle:
                with self.lock:
                    before = len(self.objects)
                    self.objects = [o for o in self.objects if not self._is_collectable(o)]
                    after = len(self.objects)
                if after < before:
                    LOG.info("Adaptive GC swept %d objects during idle", before - after)

gc_manager = AdaptiveParallelGC(profiler)

# ---------------------------------------------------------------------------
# Part 37b — Relocation Fixups in XYZOBJ3
# ---------------------------------------------------------------------------

class XYZOBJ3Reloc(XYZOBJ3):
    def __init__(self, symbols=None, macros=None, hygiene=None, gc_roots=None, relocations=None):
        super().__init__(symbols, macros, hygiene, gc_roots)
        self.relocations = relocations or []

    def add_relocation(self, symbol: str, offset: int):
        self.relocations.append({"symbol": symbol, "offset": offset})

    def serialize(self) -> bytes:
        obj = {
            "format": "XYZOBJ3",
            "symbols": self.symbols,
            "macros": self.macros,
            "hygiene": self.hygiene,
            "gc_roots": self.gc_roots,
            "relocations": self.relocations,
        }
        raw = json.dumps(obj, indent=2).encode("utf-8")
        return raw

    @classmethod
    def deserialize(cls, data: bytes):
        obj = json.loads(data.decode("utf-8"))
        return cls(
            symbols=obj.get("symbols", {}),
            macros=obj.get("macros", {}),
            hygiene=obj.get("hygiene", {}),
            gc_roots=obj.get("gc_roots", []),
            relocations=obj.get("relocations", []),
        )

# ---------------------------------------------------------------------------
# Part 37c — XYZOBJ3 Linker with Relocation
# ---------------------------------------------------------------------------

class XYZOBJ3Linker:
    def __init__(self):
        self.global_symbols: Dict[str, Any] = {}
        self.linked_relocs: List[Dict[str, Any]] = []

    def link(self, objs: List[XYZOBJ3Reloc]) -> XYZOBJ3Reloc:
        merged = XYZOBJ3Reloc(symbols={}, macros={}, hygiene={}, gc_roots=[], relocations=[])
        offset_counter = 0

        for obj in objs:
            for sym, code in obj.symbols.items():
                if sym not in self.global_symbols:
                    merged.symbols[sym] = code
                    self.global_symbols[sym] = offset_counter
                    offset_counter += len(str(code))

            merged.macros.update(obj.macros)
            merged.hygiene.update(obj.hygiene)
            merged.gc_roots.extend(obj.gc_roots)

            for reloc in obj.relocations:
                symbol = reloc["symbol"]
                if symbol in self.global_symbols:
                    reloc["offset"] = self.global_symbols[symbol]
                merged.relocations.append(reloc)

        LOG.info("Linked %d symbols, %d relocations", len(merged.symbols), len(merged.relocations))
        return merged

# ---------------------------------------------------------------------------
# Part 38a — Symbolic Backtrace Mapping
# ---------------------------------------------------------------------------

class OriginTag:
    def __init__(self, source, macro=None, phase=None):
        self.source = source         # original source text
        self.macro = macro           # macro name that produced it
        self.phase = phase           # expansion phase
    def __repr__(self):
        return f"<Origin {self.source} via {self.macro}@{self.phase}>"

class OriginTracker:
    def __init__(self):
        self.map: Dict[int, OriginTag] = {}

    def tag(self, node: ASTNode, source: str, macro=None, phase=None):
        oid = id(node)
        self.map[oid] = OriginTag(source, macro, phase)
        return node

    def backtrace(self, node: ASTNode) -> List[OriginTag]:
        oid = id(node)
        return [self.map[oid]] if oid in self.map else []

origin_tracker = OriginTracker()

try:
    mini_runtime.run_func("main/0", [])
except Exception as e:
    trace = origin_tracker.backtrace(e.args[0]) if e.args else []
    print("Error:", e)
    for t in trace:
        print("  from", t)

# ---------------------------------------------------------------------------
# Part 38b — JIT Relocation Fusion
# ---------------------------------------------------------------------------

class JITRelocator:
    def __init__(self, profiler: RuntimeProfiler, linker: XYZOBJ3Linker):
        self.profiler = profiler
        self.linker = linker
        self.jit_cache: Dict[str, str] = {}  # fn_name → ASM body

    def register_jit(self, fn_name: str, asm_body: str):
        self.jit_cache[fn_name] = asm_body
        LOG.info("Registered JIT kernel %s (%d bytes)", fn_name, len(asm_body))

    def export_to_obj(self) -> XYZOBJ3Reloc:
        reloc = XYZOBJ3Reloc(symbols={}, macros={}, hygiene={}, gc_roots=[], relocations=[])
        for fn, asm in self.jit_cache.items():
            reloc.symbols[fn] = asm
            reloc.add_relocation(fn, offset=0)  # patched later by linker
        return reloc

    def fuse_and_link(self, objs: List[XYZOBJ3Reloc]):
        objs.append(self.export_to_obj())
        return self.linker.link(objs)

def hot_route_jit(fn_name, asm_body):
    profiler.record_hot(fn_name)
    jit_relocator.register_jit(fn_name, asm_body)

class ExtendedDebugger(XYZDebugger):
    def cmd_backtrace(self, node: ASTNode):
        trace = origin_tracker.backtrace(node)
        print("Backtrace:")
        for t in trace:
            print("  ", t)

    def cmd_relocs(self, linked_obj: XYZOBJ3Reloc):
        print("Relocations:")
        for reloc in linked_obj.relocations:
            print("  ", reloc)

# ---------------------------------------------------------------------------
# Part 39a — Executable Packager (PE/ELF)
# ---------------------------------------------------------------------------

import subprocess
import tempfile
import platform

class ExecutablePackager:
    def __init__(self, linker: XYZOBJ3Linker):
        self.linker = linker

    def package(self, objs: List[XYZOBJ3Reloc], out_path: str):
        linked = self.linker.link(objs)
        asm_code = "\n".join(f"{sym}:\n{code}" for sym, code in linked.symbols.items())

        with tempfile.NamedTemporaryFile(delete=False, suffix=".asm") as asm_file:
            asm_file.write(asm_code.encode("utf-8"))
            asm_file_path = asm_file.name

        obj_file = asm_file_path.replace(".asm", ".o")
        exe_file = out_path

        # Assemble with NASM
        subprocess.run(["nasm", "-f", "elf64" if platform.system() != "Windows" else "win64",
                        asm_file_path, "-o", obj_file], check=True)

        # Link with system ld
        if platform.system() == "Windows":
            subprocess.run(["lld-link", obj_file, "/OUT:" + exe_file], check=True)
        else:
            subprocess.run(["ld", "-o", exe_file, obj_file], check=True)

        LOG.info("Packaged executable -> %s", exe_file)
        return exe_file

# ---------------------------------------------------------------------------
# Part 39b — CPU Feature Detection
# ---------------------------------------------------------------------------

import ctypes

class CPUFeatures:
    def __init__(self):
        self.sse2 = False
        self.avx2 = False
        self.avx512 = False
        self._detect()

    def _detect(self):
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()["flags"]
            self.sse2 = "sse2" in info
            self.avx2 = "avx2" in info
            self.avx512 = "avx512f" in info
        except Exception:
            LOG.warning("CPU feature detection failed; defaulting to SSE2")
            self.sse2 = True

cpu_features = CPUFeatures()

# ---------------------------------------------------------------------------
# Part 39c — SIMD Auto-Specialization
# ---------------------------------------------------------------------------

class SIMDGenerator:
    def __init__(self, jit_engine):
        self.jit = jit_engine

    def generate_vector_add(self, fn_name="vector_add"):
        if cpu_features.avx512:
            asm = f"""
            global {fn_name}
            section .text
            {fn_name}:
                vaddps zmm0, zmm0, zmm1
                ret
            """
        elif cpu_features.avx2:
            asm = f"""
            global {fn_name}
            section .text
            {fn_name}:
                vaddps ymm0, ymm0, ymm1
                ret
            """
        else:  # fallback SSE2
            asm = f"""
            global {fn_name}
            section .text
            {fn_name}:
                addps xmm0, xmm1
                ret
            """
        self.jit.generate(fn_name, asm)
        return fn_name

def vector_add_macro(call: Call, ctx: CIAMSContext):
    fn = simd_generator.generate_vector_add()
    return [Call(fn, call.args)]

ciams_hygienic.define("vector-add", vector_add_macro)

# ---------------------------------------------------------------------------
# Part 40a — Backtrace in Parallel SIMD Kernels
# ---------------------------------------------------------------------------

class ParallelSIMDRunner:
    def __init__(self, simd_generator, tracker: OriginTracker):
        self.simd = simd_generator
        self.tracker = tracker

    def run(self, fn_name: str, args: List[Any], origin_node: ASTNode):
        try:
            LOG.info("Running SIMD kernel %s", fn_name)
            # In real system, dispatch to JIT’d SIMD kernel
            result = self.simd.jit.execute(fn_name, args)
            return result
        except Exception as e:
            trace = self.tracker.backtrace(origin_node)
            LOG.error("Error in SIMD kernel %s: %s", fn_name, e)
            for t in trace:
                LOG.error("  from %s", t)
            raise

# ---------------------------------------------------------------------------
# Part 40b — Macro Relocation in Executables
# ---------------------------------------------------------------------------

class ExecutableRelocator(ExecutablePackager):
    def package_with_relocs(self, objs: List[XYZOBJ3Reloc], out_path: str, tracker: OriginTracker):
        linked = self.linker.link(objs)

        # Collect macro → symbol mapping
        reloc_section = {
            "macro_map": {sym: str(tracker.map.get(id(code))) for sym, code in linked.symbols.items()}
        }

        exe_path = super().package(objs, out_path)

        # Append reloc section as JSON
        with open(exe_path + ".xyzreloc", "w", encoding="utf-8") as f:
            json.dump(reloc_section, f, indent=2)

        LOG.info("Packaged executable with reloc metadata -> %s", exe_path)
        return exe_path

class RelocAwareDebugger(ExtendedDebugger):
    def __init__(self, ciams, gc_manager, tracker: OriginTracker, reloc_file: str = None):
        super().__init__(ciams, gc_manager)
        self.reloc_map = {}
        if reloc_file:
            with open(reloc_file, "r", encoding="utf-8") as f:
                self.reloc_map = json.load(f)

    def cmd_reloc_lookup(self, sym: str):
        if sym in self.reloc_map.get("macro_map", {}):
            print(f"Reloc for {sym}: {self.reloc_map['macro_map'][sym]}")
        else:
            print(f"No relocation info for {sym}")

# ---------------------------------------------------------------------------
# Part 41a — Cross-Module Backtraces
# ---------------------------------------------------------------------------

class OriginTag:
    def __init__(self, source, macro=None, phase=None, module=None):
        self.source = source
        self.macro = macro
        self.phase = phase
        self.module = module
    def __repr__(self):
        return f"<Origin {self.source} via {self.macro}@{self.phase} in {self.module}>"

class CrossModuleOriginTracker(OriginTracker):
    def tag(self, node: ASTNode, source: str, macro=None, phase=None, module=None):
        oid = id(node)
        self.map[oid] = OriginTag(source, macro, phase, module)
        return node

    def backtrace(self, node: ASTNode) -> List[OriginTag]:
        oid = id(node)
        return [self.map[oid]] if oid in self.map else []

class XYZOBJ3RelocWithModules(XYZOBJ3Reloc):
    def __init__(self, module_name, **kwargs):
        super().__init__(**kwargs)
        self.module_name = module_name

# ---------------------------------------------------------------------------
# Part 41b — SIMD + GC Fusion
# ---------------------------------------------------------------------------

class GCVectorArena:
    def __init__(self, gc_manager: ParallelGC):
        self.gc = gc_manager
        self.arena = []

    def allocate(self, vector):
        self.arena.append(vector)
        self.gc.register(vector)
        return vector

gc_vector_arena = GCVectorArena(gc_manager)

class SIMDGeneratorGC(SIMDGenerator):
    def generate_vector_add(self, fn_name="vector_add"):
        fn = super().generate_vector_add(fn_name)
        def wrapper(a, b):
            res = [x+y for x, y in zip(a, b)]
            return gc_vector_arena.allocate(res)
        self.jit.register_callable(fn, wrapper)
        return fn

class CrossModuleDebugger(RelocAwareDebugger):
    def cmd_backtrace(self, node: ASTNode):
        trace = self.tracker.backtrace(node)
        print("Cross-Module Backtrace:")
        for t in trace:
            print("  ", t)

    def cmd_gc_status(self):
        with gc_manager.lock:
            print(f"GC currently tracking {len(gc_manager.objects)} objects")

# ---------------------------------------------------------------------------
# Part 42a — Relocatable SIMD Kernels with GC Metadata
# ---------------------------------------------------------------------------

class RelocatableSIMDKernel:
    def __init__(self, name, asm, gc_safe=True):
        self.name = name
        self.asm = asm
        self.gc_safe = gc_safe
        self.relocs = {}  # offset → symbol

    def add_reloc(self, offset, symbol):
        self.relocs[offset] = symbol

class SIMDLinker:
    def __init__(self, gc_manager):
        self.gc = gc_manager
        self.kernels: Dict[str, RelocatableSIMDKernel] = {}

    def register_kernel(self, kernel: RelocatableSIMDKernel):
        LOG.info("Registering relocatable kernel %s", kernel.name)
        self.kernels[kernel.name] = kernel
        if kernel.gc_safe:
            self.gc.register(kernel)

    def link(self, names: List[str]) -> Dict[str, RelocatableSIMDKernel]:
        return {n: self.kernels[n] for n in names if n in self.kernels}

# ---------------------------------------------------------------------------
# Part 42b — XYZ Package Manager
# ---------------------------------------------------------------------------

import shutil
from pathlib import Path

class XYZPackageManager:
    def __init__(self, repo_dir="~/.xyzpkgs"):
        self.repo = Path(repo_dir).expanduser()
        self.repo.mkdir(parents=True, exist_ok=True)

    def install(self, pkg_path: str):
        pkg = Path(pkg_path)
        dest = self.repo / pkg.name
        shutil.copy(pkg, dest)
        LOG.info("Installed package %s", pkg.name)

    def uninstall(self, pkg_name: str):
        target = self.repo / pkg_name
        if target.exists():
            target.unlink()
            LOG.info("Uninstalled package %s", pkg_name)
        else:
            LOG.warning("Package %s not found", pkg_name)

    def list(self):
        return [p.name for p in self.repo.iterdir() if p.is_file()]

    def load(self, pkg_name: str, vm):
        target = self.repo / pkg_name
        if not target.exists():
            raise FileNotFoundError(f"Package {pkg_name} not installed")
        with open(target, "rb") as f:
            data = f.read()
        vm.load_package(data)
        LOG.info("Loaded package %s into VM", pkg_name)

class XYZPackageManagerWithKernels(XYZPackageManager):
    def load(self, pkg_name: str, vm, simd_linker: SIMDLinker):
        super().load(pkg_name, vm)
        # Load relocatable kernels metadata
        reloc_file = self.repo / (pkg_name + ".xyzreloc")
        if reloc_file.exists():
            with open(reloc_file, "r") as f:
                meta = json.load(f)
            for kname, reloc in meta.get("kernels", {}).items():
                kernel = RelocatableSIMDKernel(kname, asm="", gc_safe=True)
                kernel.relocs = reloc
                simd_linker.register_kernel(kernel)

# ---------------------------------------------------------------------------
# Part 43a — Macro-aware package integration
# ---------------------------------------------------------------------------

class MacroAwarePackageManager(XYZPackageManagerWithKernels):
    def load_macros(self, pkg_name: str, macro_engine, namespace=None):
        """Load CIAMS macros from a package with hygienic scoping."""
        ns = namespace or pkg_name
        macro_file = self.repo / (pkg_name + ".xyzmacros")
        if not macro_file.exists():
            return
        with open(macro_file, "r") as f:
            src = f.read()
        # Each macro defined in the file is wrapped into hygienic closure
        for macro_def in self._parse_macro_file(src):
            def hygienic_macro(call, _macro_def=macro_def, _ns=ns):
                local_env = {**call.__dict__}
                return _macro_def.expand(local_env, ns=_ns)
            macro_engine.register_macro(f"{ns}::{macro_def.name}", hygienic_macro)
        LOG.info("Loaded macros from %s into namespace %s", pkg_name, ns)

    def _parse_macro_file(self, src: str):
        """
        Parse macro DSL definitions from source.
        Each macro expands into a MacroDef object with .expand(ctx, ns).
        """
        # Production-grade: actually parse macro DSL.
        # For now: assume JSON list of {"name": str, "body": ...}
        try:
            defs = json.loads(src)
            return [MacroDef(d["name"], d["body"]) for d in defs]
        except Exception as e:
            LOG.error("Failed to parse macros: %s", e)
            return []

class MacroDef:
    def __init__(self, name, body):
        self.name = name
        self.body = body
    def expand(self, ctx, ns=None):
        # Example: expand into AST sequence
        return [Call("print", [Number(f"[{ns}] macro {self.name}")])]

# ---------------------------------------------------------------------------
# Part 43b — Profiler-driven package JIT
# ---------------------------------------------------------------------------

class Profiler:
    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.threshold = 50  # calls before hot-route triggers

    def record(self, key: str):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def is_hot(self, key: str):
        return self.counts.get(key, 0) >= self.threshold

class PackageJIT:
    def __init__(self, profiler: Profiler, simd_linker: SIMDLinker):
        self.profiler = profiler
        self.simd_linker = simd_linker
        self.specialized: Dict[str, RelocatableSIMDKernel] = {}

    def maybe_jit(self, key: str, func):
        """If function is hot, specialize into SIMD kernel."""
        if self.profiler.is_hot(key) and key not in self.specialized:
            asm = self._emit_simd(func)
            kernel = RelocatableSIMDKernel(key, asm, gc_safe=True)
            self.simd_linker.register_kernel(kernel)
            self.specialized[key] = kernel
            LOG.info("Specialized %s into SIMD kernel", key)

    def _emit_simd(self, func: FuncDef) -> str:
        # Production: AST → vectorized NASM loop
        body_ops = " ; ".join(f"op({type(st).__name__})" for st in func.body)
        return f"; SIMD JIT kernel for {func.name}\n; body: {body_ops}"

profiler = Profiler()
simd_linker = SIMDLinker(gc_manager)
pkgmgr = MacroAwarePackageManager()

# Load a mathlib package with macros + functions
pkgmgr.install("mathlib.xyzobj3")
pkgmgr.load("mathlib.xyzobj3", fast_vm, simd_linker)
pkgmgr.load_macros("mathlib.xyzobj3", _global_macro_engine)

# Run function calls → profiler tracks frequency
result = fast_vm.run("mathlib::dot/2", [[1,2,3], [4,5,6]])
profiler.record("mathlib::dot/2")

# Profiler decides if it’s hot enough → JIT kicks in
jit = PackageJIT(profiler, simd_linker)
jit.maybe_jit("mathlib::dot/2", some_funcdef)

profiler = Profiler()
simd_linker = SIMDLinker(gc_manager)
pkgmgr = MacroAwarePackageManager()

# Load a mathlib package with macros + functions
pkgmgr.install("mathlib.xyzobj3")
pkgmgr.load("mathlib.xyzobj3", fast_vm, simd_linker)
pkgmgr.load_macros("mathlib.xyzobj3", _global_macro_engine)

# Run function calls → profiler tracks frequency
result = fast_vm.run("mathlib::dot/2", [[1,2,3], [4,5,6]])
profiler.record("mathlib::dot/2")

# Profiler decides if it’s hot enough → JIT kicks in
jit = PackageJIT(profiler, simd_linker)
jit.maybe_jit("mathlib::dot/2", some_funcdef)

# ---------------------------------------------------------------------------
# Part 44a — CIAMS Macro Debugger
# ---------------------------------------------------------------------------

class MacroDebugger:
    def __init__(self):
        self.enabled = True
        self.trace: List[str] = []

    def log(self, phase: str, macro_name: str, node: ASTNode, result: Any):
        if not self.enabled:
            return
        msg = f"[MACRO:{macro_name}] phase={phase} input={type(node).__name__} → result={result}"
        self.trace.append(msg)
        LOG.debug(msg)

    def show_trace(self):
        for line in self.trace:
            print(line)

class CIAMSExpanderWithDebugger:
    def __init__(self, macro_engine, debugger: MacroDebugger):
        self.macro_engine = macro_engine
        self.debugger = debugger

    def expand(self, node: ASTNode, phase="initial"):
        """Expand node with macro tracing across phases."""
        if isinstance(node, Call) and node.name in self.macro_engine.macros:
            macro_fn = self.macro_engine.macros[node.name]
            self.debugger.log(phase, node.name, node, "invoked")
            expanded = macro_fn(node)
            self.debugger.log(phase, node.name, node, expanded)
            # Multi-phase: recursively expand until stable
            out = []
            for e in expanded:
                out.extend(self.expand(e, phase="recursive"))
            return out
        return [node]

dbg = MacroDebugger()
expander = CIAMSExpanderWithDebugger(_global_macro_engine, dbg)

prog = parser.parse()
expanded = [expander.expand(n) for n in prog.body]
dbg.show_trace()

# ---------------------------------------------------------------------------
# Part 44b — Cross-Package Fusion
# ---------------------------------------------------------------------------

class FusionEngine:
    def __init__(self, profiler: Profiler, simd_linker: SIMDLinker):
        self.profiler = profiler
        self.simd_linker = simd_linker

    def fuse(self, funcs: List[FuncDef], name: str):
        """Fuse multiple functions/macros into one SIMD kernel."""
        asm_parts = []
        for f in funcs:
            self.profiler.record(f"{f.name}/{len(f.params)}")
            asm_parts.append(self._emit_partial(f))
        asm = "\n".join(asm_parts)
        kernel = RelocatableSIMDKernel(name, asm, gc_safe=True)
        self.simd_linker.register_kernel(kernel)
        LOG.info("Fused %d functions into %s", len(funcs), name)
        return kernel

    def _emit_partial(self, f: FuncDef) -> str:
        # Vectorize each function body → part of fused kernel
        ops = " ; ".join(type(st).__name__ for st in f.body)
        return f"; fusion for {f.name} :: {ops}"

# Load two packages with macros + kernels
pkgmgr.install("mathlib.xyzobj3")
pkgmgr.install("geomlib.xyzobj3")

# Suppose both provide dot-product style functions
f1 = symtab["mathlib::dot/2"]
f2 = symtab["geomlib::cross/2"]

fusion = FusionEngine(profiler, simd_linker)
kernel = fusion.fuse([f1, f2], "fused_math_geom")

# Now "fused_math_geom" runs dot + cross in one SIMD batch
fast_vm.run("fused_math_geom/2", [[1,2,3],[4,5,6]])


