// force.cl — OpenCL kernel for apply Force("down", 9.8)

__kernel void apply_force(__global float *pos_y,
                          __global float *vel_y,
                          float g,
                          int n) {
    int i = get_global_id(0);
    if (i < n) {
        vel_y[i] += g;        // update velocity with gravity
        pos_y[i] += vel_y[i]; // integrate position
    }
}
