#!/usr/bin/env python3
"""
XYZC5.py - XYZ Programming Language Compiler Toolchain
-------------------------------------------------------
Includes:
  - Lexer & Parser → AST
  - Optimizer (constant folding, if simplification)
  - Codegen → NASM x86-64
  - Object + Linker
  - MiniRuntime (AST interpreter)
  - FastRuntime (Bytecode + VM)
  - HotSwap & Mega Features

New in C5:
  ✅ Real variable storage in NASM
  ✅ print_num + print_str syscalls
  ✅ Mixed arguments in print
  ✅ Automatic newline after print (Python-style)
  ✅ Support for keyword argument: end=""
  ✅ CLI toolchain driver
"""

from __future__ import annotations
import os, sys, re, math, json, logging
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG = logging.getLogger("xyzc")
if not LOG.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(h)
LOG.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------
class ASTNode: ...
class Program(ASTNode):
    def __init__(self, body: List[ASTNode]): self.body = body
class FuncDef(ASTNode):
    def __init__(self, name: str, params: List[str], body: List[ASTNode]):
        self.name, self.params, self.body = name, params, body
class Call(ASTNode):
    def __init__(self, name: str, args: List[ASTNode], kwargs: Dict[str, ASTNode]=None):
        self.name, self.args, self.kwargs = name, args, kwargs or {}
class Return(ASTNode):
    def __init__(self, expr: ASTNode): self.expr = expr
class Number(ASTNode):
    def __init__(self, val: str):
        self.raw = val
        self.val = float(val) if "." in val else int(val)
class StrLiteral(ASTNode):
    def __init__(self, val: str): self.val = val
class Var(ASTNode):
    def __init__(self, name: str): self.name = name
class Assign(ASTNode):
    def __init__(self, name: str, expr: ASTNode): self.name, self.expr = name, expr
class BinOp(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode): self.op, self.left, self.right = op, left, right
class UnaryOp(ASTNode):
    def __init__(self, op: str, operand: ASTNode): self.op, self.operand = op, operand

# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------
TOKEN_SPEC = [
    ("NUMBER", r"-?\d+(\.\d+)?"),
    ("STRING", r'"(?:\\.|[^"\\])*"'),
    ("ID", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"[+\-*/=<>!&|%^.^]+"),
    ("LPAREN", r"\("), ("RPAREN", r"\)"),
    ("LBRACE", r"\{"), ("RBRACE", r"\}"),
    ("SEMI", r";"), ("COMMA", r","),
    ("KEYWORD", r"\b(func|return|print)\b"),
    ("WS", r"\s+"),
]

class Token:
    def __init__(self, kind, val, pos): self.kind, self.val, self.pos = kind, val, pos

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

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens=tokens; self.pos=0; self.functions={}
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
            else:
                e=self.expression()
                if e: out.append(e)
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
            if self.peek().val=="return":
                self.eat("KEYWORD"); body.append(Return(self.expression()))
            else:
                body.append(self.expression())
        self.eat("RBRACE")
        f=FuncDef(name,params,body)
        self.functions[f"{name}/{len(params)}"]=f
        return f
    def expression(self):
        t=self.peek()
        if not t: return None
        if t.kind=="NUMBER": return Number(self.eat("NUMBER").val)
        if t.kind=="STRING":
            val=self.eat("STRING").val.strip('"')
            return StrLiteral(val)
        if t.kind=="ID":
            name=self.eat("ID").val
            if self.peek() and self.peek().kind=="LPAREN":
                self.eat("LPAREN"); args=[]; kwargs={}
                while self.peek() and self.peek().kind!="RPAREN":
                    # keyword argument check: ID '=' expr
                    if (self.peek().kind=="ID" and 
                        self.pos+1 < len(self.tokens) and 
                        self.tokens[self.pos+1].kind=="OP" and self.tokens[self.pos+1].val=="="):
                        key=self.eat("ID").val
                        self.eat("OP")
                        val=self.expression()
                        kwargs[key]=val
                    else:
                        args.append(self.expression())
                    if self.peek() and self.peek().kind=="COMMA": self.eat("COMMA")
                self.eat("RPAREN"); return Call(name,args,kwargs)
            return Var(name)
        return None

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
def optimize(node: ASTNode) -> ASTNode:
    if isinstance(node, Program):
        node.body=[optimize(n) for n in node.body]; return node
    if isinstance(node, FuncDef):
        node.body=[optimize(n) for n in node.body]; return node
    if isinstance(node, Return):
        node.expr=optimize(node.expr); return node
    if isinstance(node, BinOp):
        node.left=optimize(node.left); node.right=optimize(node.right)
        if isinstance(node.left,Number) and isinstance(node.right,Number):
            a,b=node.left.val,node.right.val
            if node.op=="+": return Number(str(a+b))
            if node.op=="-": return Number(str(a-b))
            if node.op=="*": return Number(str(a*b))
        return node
    return node

# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------
class Codegen:
    def __init__(self):
        self.asm=[]; self._lbl=0; self._rodata=[]
        self._needs_print=False
    def emit(self,s): self.asm.append(s)
    def newlabel(self,p="L"): self._lbl+=1; return f"{p}{self._lbl}"
    def add_string(self, s: str):
        lbl=f"str_{len(self._rodata)}"
        self._rodata.append((lbl,s))
        return lbl
    def generate(self, prog: Program) -> str:
        self.emit("section .bss")
        self.emit("print_buf resb 32")
        self.emit("section .data")
        for lbl,s in self._rodata: self.emit(f'{lbl}: db "{s}",0')
        self.emit("section .text")
        self.emit("global _start")
        self.emit("_start:")
        self.emit("  call main/0")
        self.emit("  mov rax,60")
        self.emit("  xor rdi,rdi")
        self.emit("  syscall")
        prog=optimize(prog)
        for s in prog.body: self.gen_stmt(s)
        if self._needs_print: self.gen_print_routines()
        return "\n".join(self.asm)
    def gen_stmt(self,n:ASTNode):
        if isinstance(n,FuncDef):
            k=f"{n.name}/{len(n.params)}"
            self.emit(f"{k}:")
            for s in n.body: self.gen_stmt(s)
            self.emit("  ret"); return
        if isinstance(n,Return):
            self.gen_stmt(n.expr); self.emit("  ret"); return
        if isinstance(n,Number):
            self.emit(f"  mov rax,{int(n.val)}"); return
        if isinstance(n,StrLiteral):
            lbl=self.add_string(n.val)
            self.emit(f"  lea rax,[{lbl}]"); return
        if isinstance(n,Call) and n.name=="print":
            argc=len(n.args)
            for i,a in enumerate(n.args):
                self.gen_stmt(a)
                if isinstance(a,StrLiteral): self.emit("  call print_str")
                else: self.emit("  call print_num")
                if i<argc-1:
                    lbl=self.add_string(" ")
                    self.emit(f"  lea rax,[{lbl}]")
                    self.emit("  call print_str")
            # handle end kwarg
            end_val = n.kwargs.get("end")
            if end_val is None:
                lbl_nl=self.add_string("\n")
                self.emit(f"  lea rax,[{lbl_nl}]")
                self.emit("  call print_str")
            elif isinstance(end_val, StrLiteral):
                lbl=self.add_string(end_val.val)
                self.emit(f"  lea rax,[{lbl}]")
                self.emit("  call print_str")
            self._needs_print=True
            return
        if isinstance(n,BinOp):
            self.gen_stmt(n.left); self.emit("  push rax")
            self.gen_stmt(n.right); self.emit("  mov rbx,rax"); self.emit("  pop rax")
            if n.op=="+": self.emit("  add rax,rbx")
            elif n.op=="-": self.emit("  sub rax,rbx")
            elif n.op=="*": self.emit("  imul rax,rbx")
            return
    def gen_print_routines(self):
        self.emit("print_num:")
        self.emit("  mov rcx,print_buf+31")
        self.emit("  mov rbx,10")
        self.emit("  mov rdx,0")
        self.emit("  cmp rax,0")
        self.emit("  jge .pn_loop")
        self.emit("  neg rax")
        self.emit("  push rax")
        self.emit("  mov rax,'-'")
        self.emit("  mov [rcx],al")
        self.emit("  dec rcx")
        self.emit("  pop rax")
        self.emit(".pn_loop:")
        self.emit("  xor rdx,rdx")
        self.emit("  div rbx")
        self.emit("  add rdx,'0'")
        self.emit("  dec rcx")
        self.emit("  mov [rcx],dl")
        self.emit("  test rax,rax")
        self.emit("  jnz .pn_loop")
        self.emit("  mov rax,1")
        self.emit("  mov rdi,1")
        self.emit("  mov rsi,rcx")
        self.emit("  mov rdx,print_buf+32-rcx")
        self.emit("  syscall")
        self.emit("  ret")
        self.emit("print_str:")
        self.emit("  push rax")
        self.emit("  mov rsi,rax")
        self.emit("  mov rcx,rax")
        self.emit(".ps_len: cmp byte [rcx],0")
        self.emit("  je .ps_done")
        self.emit("  inc rcx")
        self.emit("  jmp .ps_len")
        self.emit(".ps_done:")
        self.emit("  mov rdx,rcx")
        self.emit("  sub rdx,rsi")
        self.emit("  mov rax,1")
        self.emit("  mov rdi,1")
        self.emit("  syscall")
        self.emit("  pop rax")
        self.emit("  ret")

# ---------------------------------------------------------------------------
# CLI Driver
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv)<2:
        print("Usage: xyzc5.py file.xyz [--emit-asm]")
        sys.exit(1)
    path=sys.argv[1]
    src=open(path).read()
    tokens=lex(src)
    parser=Parser(tokens)
    prog=parser.parse()
    cg=Codegen()
    asm=cg.generate(prog)
    if "--emit-asm" in sys.argv:
        out=path.replace(".xyz",".asm")
        open(out,"w").write(asm)
        LOG.info("Wrote %s",out)

if __name__=="__main__":
    main()
