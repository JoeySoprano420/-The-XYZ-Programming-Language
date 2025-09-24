# 🌌 The XYZ Programming Language

*"Item-Oriented. Hex-Bodied. Self-Optimized."*

---

## 🔹 Core Identity

XYZ is a **next-generation systems and application language** designed for **direct NASM mapping** with **Ahead-of-Time (AOT) compilation**. It eliminates external toolchains by being **self-contained** and **self-optimizing**, allowing developers to focus entirely on *what* to build rather than *how* to manage toolchains, linking, or optimization flags.

---

## 🔹 Defining Features

### 1. **Item-Oriented Programming (IOP)**

* Objects are replaced with **items**—lightweight, native-first entities that avoid OOP overhead.
* Items unify the power of structs, classes, and modules while allowing inheritance, polymorphism, and directives.
* Everything in XYZ is an item: functions, bundles, packets, even memory allocations.

### 2. **Direct NASM Mapping**

* Source compiles *straight into NASM x64*.
* IR is written directly in **hexadecimal form**, making binaries transparent and audit-ready.
* **Dodecagram AST (base-12 model)** ensures compact, lossless AST storage: digits `0–9, a, b`.

### 3. **Framed Hot Routes & Hot Swaps**

* Hot sections of code are auto-identified and given **framed hot routes** for efficiency.
* Live **hot swaps** replace sections during execution without downtime.

### 4. **Self-Optimizing by Design**

* No flags, no config. The compiler detects usage patterns, hardware, and user tendencies, then builds tuned libraries automatically.
* Code deployment is **pattern-matched**—scheduling and parallelism are inferred without developer micromanagement.

### 5. **Error & Memory Models**

* **Errors:** `try`, `catch`, `throw`, `except`, `isolate`, `delete`.
* **Memory:** `collect`, `store`, `flush`, `sweep`.
* Fully **register-based stack/heap management**, blending low-level control with high-level safety.

### 6. **Universal Grammar**

* One syntax for **system programming** *and* **scripting**.
* Developers can write drivers, kernels, or one-liners with the same concise grammar.

---

## 🔹 Ecosystem & Interop

* **Automatic Linking:** All imports/exports are resolved at compile time—no manual linker steps.
* **C Interop:** Native bidirectional binding with **C** (ABI, ISA, FFI, calls, wraps).
* **Deployment:** Built-in deployment system abstracts containers, services, and distribution.

---

## 🔹 Example Syntax

```xyz
; Single-line comment
-- Multiline
-- comment block

Start() print {"Hello, world"} run

main() print ["Hello, world"] start
```

* Curly `{}` or square `[]` are interchangeable for literals.
* Minimalistic keywords: `start`, `run`, `print`.

---

## 🔹 Advantages Over Other Languages

* **Faster than C** through:

  * NASM-level mapping
  * Register-based optimization
  * Eliminating undefined behavior entirely
* **Safer than Rust** by embedding error/memory models directly into grammar.
* **Simpler than Go/Python** with a single universal grammar and no external runtime.

---

## 🔹 Use Cases

* **Systems Programming:** kernels, OS components, embedded hardware.
* **High-Performance Apps:** game engines, rendering pipelines, real-time physics.
* **Enterprise:** deployment-ready server applications with zero external tooling.
* **Research/Math:** polynomials, derivatives, decimals, negative numbers fully baked-in.

---

## 🔹 Formal Tagline

**XYZ: The Self-Optimizing Item-Oriented Language — Faster Than C, Simpler Than Scripting.**

---

Compile with:

gcc -c libxyzrt.c -o libxyzrt.o
ar rcs libxyzrt.a libxyzrt.o


Now -lxyzrt can be linked into programs emitted by your compiler.



🔹 Build Instructions
cd libxyzrt
make              # builds libxyzrt.a and libxyzrt.so
sudo make install # installs headers and libs into /usr/local



