import pyopencl as cl
import numpy as np

N = 1024
pos_y = np.zeros(N, dtype=np.float32)
vel_y = np.zeros(N, dtype=np.float32)

kernel_src = open("force.cl").read()

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos_y)
vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel_y)

prg = cl.Program(ctx, kernel_src).build()
g = np.float32(9.8)

prg.apply_force(queue, (N,), None, pos_buf, vel_buf, g, np.int32(N))

cl.enqueue_copy(queue, pos_y, pos_buf)
cl.enqueue_copy(queue, vel_y, vel_buf)

print("pos[0] =", pos_y[0], "vel[0] =", vel_y[0])
