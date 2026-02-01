import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import opensimplex
import openvdb as vdb
import slangpy as spy

opensimplex.seed(1234)

device = spy.create_device(
    type=spy.DeviceType.vulkan,
    include_paths=[
        pathlib.Path(".").absolute(),
    ],
)

# Image dimensions
width, height = 1280, 720

render_texture = device.create_texture(
    format=spy.Format.rgba32_float,
    width=width,
    height=height,
    usage=spy.TextureUsage.unordered_access,
    label="render_texture",
)

# Create a sample density volume
density_grid_size = 128
densities = np.zeros(
    (density_grid_size, density_grid_size, density_grid_size), dtype=np.float32
)


def init_grid_with_noise():
    for z in range(density_grid_size):
        for y in range(density_grid_size):
            for x in range(density_grid_size):
                densities[z, y, x] = np.clip(
                    opensimplex.noise3(
                        x=4.0 * x / density_grid_size,
                        y=4.0 * y / density_grid_size,
                        z=4.0 * z / density_grid_size,
                    )
                    * 2.0,
                    0,
                    1,
                )


def init_grid_with_cloud():
    grid, metadata = vdb.readAll("assets/wdas_cloud_sixteenth.vdb")
    grid = grid[0]
    print(grid.metadata)
    grid_acc = grid.getConstAccessor()
    grid_start = grid.metadata["file_bbox_min"]
    for z in range(density_grid_size):
        for y in range(density_grid_size):
            for x in range(density_grid_size):
                density, active = grid_acc.probeValue(
                    (grid_start[0] + x, grid_start[1] + y - 20, grid_start[2] + z)
                )
                densities[z, y, x] = density


init_grid_with_cloud()

density_texture = device.create_texture(
    type=spy.TextureType.texture_3d,
    format=spy.Format.r32_float,
    width=density_grid_size,
    height=density_grid_size,
    depth=density_grid_size,
    usage=spy.TextureUsage.shader_resource,
    label="densities",
)
density_texture.copy_from_numpy(densities)

density_sampler = device.create_sampler(
    min_filter=spy.TextureFilteringMode.linear,
    mag_filter=spy.TextureFilteringMode.linear,
    address_u=spy.TextureAddressingMode.clamp_to_edge,
    address_v=spy.TextureAddressingMode.clamp_to_edge,
    address_w=spy.TextureAddressingMode.clamp_to_edge,
)


# Camera setup
def look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = r
    view[1, :3] = u
    view[2, :3] = -f
    view[:3, 3] = -np.array([np.dot(r, eye), np.dot(u, eye), np.dot(-f, eye)])
    return view


def perspective(fov_y, aspect, near, far):
    f = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


projection = perspective(np.radians(45.0), width / height, 0.1, 100.0)
inv_projection = np.linalg.inv(projection)

# Model matrix: identity (volume occupies [-1,1]^3 in world space)
model = np.matrix(
    [
        [2.0, 0.0, 0.0, -1.0],
        [0.0, 2.0, 0.0, -1.0],
        [0.0, 0.0, 2.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
inv_model = np.linalg.inv(model)

# Light setup
light_dir = np.array([0.0, 1.0, 0.0], dtype=np.float32)
light_dir = light_dir / np.linalg.norm(light_dir)
light_color = np.array([10.0, 10.0, 10.0], dtype=np.float32)

# Volume properties
sigma_a = 1.0  # Absorption
sigma_s = 10.0  # Scattering

program = device.load_program("pathtracer.slang", ["main"])
kernel = device.create_compute_kernel(program)

# Progressive rendering loop
num_iterations = 8

plt.ion()
fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((height, width, 4)))
plt.title("Volumetric Path Tracer")

try:
    for j in range(360):
        eye = np.array(
            [
                3.0 * math.sin(2.0 * j * math.pi / 180.0),
                0.0,
                3.0 * math.cos(2.0 * j * math.pi / 180.0),
            ],
            dtype=np.float32,
        )
        target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        view = look_at(eye, target, up)
        inv_view = np.linalg.inv(view)

        for i in range(num_iterations):
            kernel.dispatch(
                thread_count=[width, height, 1],
                vars={
                    "render_texture": render_texture,
                    "densities": density_texture,
                    "density_sampler": density_sampler,
                    "density_grid_size": density_grid_size,
                    "frame_index": i,
                    "model": model,
                    "inv_model": inv_model,
                    "view": view,
                    "inv_view": inv_view,
                    "projection": projection,
                    "inv_projection": inv_projection,
                    "light_dir": light_dir,
                    "light_color": light_color,
                    "sigma_a": sigma_a,
                    "sigma_s": sigma_s,
                    # "threshold": 0.9,
                },
            )

            # Update display periodically
            if (i + 1) % 8 == 0 or i == num_iterations - 1:
                img = render_texture.to_numpy()
                # Tone mapping
                img_display.set_data(np.clip(img[..., :3] / (1.0 + img[..., :3]), 0, 1))
                # img_display.set_data(np.clip(img[..., :4], 0, 1))
                ax.set_title(f"Volumetric Path Tracer - Frame {i + 1}")
                plt.pause(0.01)

except KeyboardInterrupt:
    pass

plt.ioff()
plt.show()
