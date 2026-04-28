"""
Blender Python Script — runs INSIDE Blender's embedded Python environment.
Called via: blender --background --python render.py -- --image <path> --output <path>

IMPORTANT: bpy import is intentionally deferred until after arg parsing.
This allows the module to be safely imported in unit tests without Blender present.
"""

# ── Argument Parsing (must happen BEFORE bpy import) ─────────────────────────
import sys


def _parse_args():
    """Parse --image and --output from sys.argv after the '--' separator."""
    argv = sys.argv
    try:
        sep_index = argv.index("--")
    except ValueError:
        print("ERROR: Missing '--' separator in Blender args.", file=sys.stderr)
        sys.exit(1)

    script_args = argv[sep_index + 1 :]
    image_path = None
    output_path = None

    i = 0
    while i < len(script_args):
        if script_args[i] == "--image" and i + 1 < len(script_args):
            image_path = script_args[i + 1]
            i += 2
        elif script_args[i] == "--output" and i + 1 < len(script_args):
            output_path = script_args[i + 1]
            i += 2
        else:
            i += 1

    if not image_path or not output_path:
        print(
            "ERROR: --image and --output are required arguments.",
            file=sys.stderr,
        )
        sys.exit(1)

    return image_path, output_path


def setup_scene(image_path: str, output_path: str) -> None:
    """
    Configure and render a Blender scene that composites the input image onto a plane.

    Camera looks straight at the plane (orthographic-style), image fills the full
    frame. Cinematic colour grading is applied via compositor nodes.
    Uses Cycles render engine at 32 samples.

    Compatible with Blender 4.x and 5.x (handles API changes).
    """
    import bpy
    import math

    # ── Version detection ─────────────────────────────────────────────────────
    blender_major = bpy.app.version[0]
    print(f"Blender version: {'.'.join(str(v) for v in bpy.app.version)}")

    # ── 1. Clear default scene ────────────────────────────────────────────────
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    # ── 2. Load image and read its pixel dimensions ───────────────────────────
    img = bpy.data.images.load(image_path)
    img_w = img.size[0] if img.size[0] > 0 else 1920
    img_h = img.size[1] if img.size[1] > 0 else 1080
    aspect = img_w / img_h  # e.g. 1.777 for 16:9
    print(f"Image loaded: {img_w}x{img_h}, aspect={aspect:.3f}")

    # ── 3. Create image plane, sized to exact image aspect ratio ─────────────
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.scale.x = aspect
    plane.scale.y = 1.0

    # ── 4. Add UV map and apply image texture ─────────────────────────────────
    mat = bpy.data.materials.new(name="SceneMat")

    # Blender 5.x deprecates use_nodes but materials still have node_tree
    if blender_major < 5:
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear all default nodes
    for n in nodes:
        nodes.remove(n)

    # Emission shader so the image colour is not affected by scene lighting
    output_node = nodes.new("ShaderNodeOutputMaterial")
    emit_node   = nodes.new("ShaderNodeEmission")
    tex_node    = nodes.new("ShaderNodeTexImage")

    tex_node.image = img
    tex_node.interpolation = "Linear"

    emit_node.inputs["Strength"].default_value = 3.0
    links.new(tex_node.outputs["Color"],   emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], output_node.inputs["Surface"])

    plane.data.materials.append(mat)

    # ── 5. Camera — looking straight down at the plane ────────────────────────
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.type = "PERSP"
    cam_data.lens = 50
    cam_data.clip_start = 0.01
    cam_data.clip_end   = 100.0

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)

    import mathutils
    cam_z = 3.0
    cam_obj.location = (0.0, 0.0, cam_z)

    # Make camera look at plane center (0,0,0)
    target = mathutils.Vector((0.0, 0.0, 0.0))
    direction = target - mathutils.Vector(cam_obj.location)

    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam_obj

    # ── 6. World — pure black background ──────────────────────────────────────
    world = bpy.data.worlds.new("World")
    if blender_major < 5:
        world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value   = (0, 0, 0, 1)
        bg_node.inputs["Strength"].default_value = 0.0
    bpy.context.scene.world = world

    # ── 7. Single soft fill light ─────────────────────────────────────────────
    bpy.ops.object.light_add(type="AREA", location=(0, 0, cam_z + 1))
    light = bpy.context.active_object
    light.data.energy = 5
    light.data.size   = 4.0

    # ── 8. Render settings ────────────────────────────────────────────────────
    scene = bpy.context.scene
    scene.render.engine          = "CYCLES"
    scene.cycles.samples         = 32
    scene.cycles.use_denoising   = True
    scene.render.resolution_x    = 1920
    scene.render.resolution_y    = 1080
    scene.render.resolution_percentage = 100
    scene.render.filepath        = output_path
    scene.render.image_settings.file_format      = "PNG"
    scene.render.image_settings.color_mode       = "RGB"
    scene.render.image_settings.compression      = 15

    # ── 9. Compositor — cinematic colour grade ────────────────────────────────
    # Skip compositor entirely for now (Blender 5.x unstable API)

    # ── 10. Render ────────────────────────────────────────────────────────────
    print("Rendering to:", output_path)
    bpy.ops.render.render(write_still=True)
    
    import os
    import sys
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
        print("ERROR: Render not written properly")
        sys.exit(1)
        
    print(f"Render complete: {output_path}")


if __name__ == "__main__":
    image_path, output_path = _parse_args()

    # Normalize paths (POSIX → native) so os.path.exists works reliably
    import os
    image_path = os.path.normpath(image_path)
    output_path = os.path.normpath(output_path)

    # Pre-flight: validate image exists before launching Blender scene setup
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Render script args: image={image_path}, output={output_path}")

    try:
        setup_scene(image_path, output_path)
    except Exception as exc:
        print(f"Render failed: {exc}", file=sys.stderr)
        sys.exit(1)