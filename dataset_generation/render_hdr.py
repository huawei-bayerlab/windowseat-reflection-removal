import bpy 
import os
import sys
import argparse
import numpy as np
import math 
import json

from mathutils import Matrix, Vector

# Node CUDA setup
cycles_prefs = bpy.context.preferences.addons['cycles'].preferences  
cycles_prefs.get_devices()
cycles_prefs.compute_device_type = 'CUDA'

print("CUDA devices:", cycles_prefs.devices)
for device in cycles_prefs.devices:
    print(f"Device: {device.name}, Type: {device.type}")


class Scene:
    def __init__(self, args):
        self.args = args

        self.resolution_x = args.resolution_x
        self.resolution_y = args.resolution_y
        if args.mode == 'img':
            assert args.img_path is not None, "If mode is 'img', img_path must be provided."
            self.resolution_x, self.resolution_y = bpy.data.images.load(args.img_path).size
            print("Set image resolution to ({}, {})".format(self.resolution_x, self.resolution_y))

        # clear the scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)

        self.scene = bpy.context.scene
        self.world = self.scene.world

        self.glass_distance = self.args.glass_distance 
        if args.mode == 'img':
            self.glass_distance = 0.2

    def load_hdr_background(self):
        # Load the HDR image
        self.world.use_nodes = True
        nodes = self.world.node_tree.nodes
        links = self.world.node_tree.links

        nodes.clear()

        env_tex = nodes.new(type='ShaderNodeTexEnvironment')
        env_tex.image = bpy.data.images.load(self.args.background_hdr_path)  

        bg_node = nodes.new(type='ShaderNodeBackground')
        bg_node.inputs['Strength'].default_value = self.args.background_strength
        output_node = nodes.new(type='ShaderNodeOutputWorld')

        links.new(env_tex.outputs['Color'], bg_node.inputs['Color'])
        links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

        print(f"Loaded HDR image: {self.args.background_hdr_path}")

    def load_img_background(self):
        img_path = os.path.abspath(self.args.background_img_path)
        bpy.ops.image.import_as_mesh_planes(
            relative=False, 
            filepath=img_path, 
            files=[{"name":os.path.basename(img_path)}], 
            directory=os.path.dirname(img_path)
        )

        self.background_img = bpy.context.active_object
        print(self.background_img)
        self.background_img.scale = (self.resolution_x, self.resolution_y, 1)
        self.background_img.rotation_euler = self.camera.rotation_euler # location is not normalized~!!!!
        self.background_img.location = (0, 0, 0)
        # self.background_img.scale = (scale, scale, scale)

        # Make the default material properties 
        for mat in self.background_img.data.materials:
            if mat.use_nodes:
                bsdf_node = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf_node:
                    bsdf_node.inputs['IOR'].default_value = 1.0
                    bsdf_node.inputs['Roughness'].default_value = 0.0
                    bsdf_node.inputs['Metallic'].default_value = 0.0


        old_size = bpy.data.images.load(args.background_img_path).size
        sensor_width = self.camera.data.sensor_width
        img_dimensions = self.get_dimensions(bpy.context.active_object)
        scale = 1.4 * sensor_width / (self.camera.data.lens) / max(img_dimensions)
        print(f"Image dimensions: {img_dimensions}, scale: {scale}")
        self.background_img.scale = (scale * self.resolution_x / old_size[0], scale * self.resolution_y / old_size[1], 1)


    def _cartesian_from_euler(self, v):
        # Convert spherical coordinates to Cartesian coordinates
        zimuth_rad = v[0]
        azimuth_rad = v[2]
        x = math.cos(zimuth_rad) * math.cos(azimuth_rad)
        y = math.cos(zimuth_rad) * math.sin(azimuth_rad)
        z = math.sin(zimuth_rad)
        return np.asarray((x, y, z), float)

    def rotate_about_axis(self, v, w, theta_deg):
        up = np.asarray(self._cartesian_from_euler(v), float)
        forward = np.asarray(self._cartesian_from_euler(w), float)
        right = np.cross(up, forward)
        print(v)
        print(f"Original up vector: {up}, forward vector: {forward}, right vector: {right}")

        print(f"Rotating vector {up} about axis {forward} by {theta_deg} degrees")
        phi = np.deg2rad(theta_deg)
        print(np.cos(phi), np.sin(phi))
        up_rotated = (up*np.cos(phi) + np.cross(forward, up)*np.sin(phi) + forward*np.dot(forward, up)*(1-np.cos(phi)))
        print(f"Rotated vector: {up_rotated}")
        
        print(f"Rotating vector {right} about axis {forward} by {theta_deg} degrees")
        phi = np.deg2rad(theta_deg)
        print(np.cos(phi), np.sin(phi))
        right_rotated = (right*np.cos(phi) + np.cross(forward, right)*np.sin(phi) + forward*np.dot(forward, right)*(1-np.cos(phi)))
        print(f"Rotated vector: {right_rotated}")

        # convert up, forward, right to euler angles 
        up_vec = Vector(up_rotated).normalized()
        forward_vec = Vector(forward).normalized()
        right_vec = Vector(right_rotated).normalized()

        print(f"Final up vector: {up_vec}, forward vector: {forward_vec}, right vector: {right_vec}")

        mat = Matrix((right_vec, up_vec, forward_vec)).to_3x3().transposed()

        return mat.to_euler('XYZ')

    def setup(self):
        # Setup the scene
        bpy.ops.object.camera_add(location=(0, 0, 0))
        self.camera = bpy.context.active_object

        self.scene.render.resolution_x = self.resolution_x
        self.scene.render.resolution_y = self.resolution_y
        self.scene.render.engine = 'CYCLES'
        self.scene.cycles.device = 'GPU'
        self.scene.cycles.samples = self.args.cycles_samples

        camera_rotation_spherical = (np.deg2rad(self.args.camera_zimuth), 0, np.deg2rad(self.args.camera_azimuth))
        primary_axis_spherical = (np.deg2rad(self.args.camera_zimuth - 90), 0, np.deg2rad(self.args.camera_azimuth))
        self.primary_axis_spherical = primary_axis_spherical
        camera_rotation_with_tilt = self.rotate_about_axis(camera_rotation_spherical, primary_axis_spherical, self.args.camera_tilt)
        self.camera.rotation_euler = camera_rotation_with_tilt
        self.camera.data.lens = self.args.camera_focal_length
        print(f"Camera rotation (spherical): {camera_rotation_spherical}, with tilt: {camera_rotation_with_tilt}")

        self.scene.camera = self.camera

    def add_glass(self):
        sensor_width = self.camera.data.sensor_width
        print(f"Camera sensor width: {sensor_width} mm, focal length: {self.camera.data.lens} mm, glass distance: {self.glass_distance} m")
        cube_w = self.glass_distance * sensor_width / (self.camera.data.lens)
        cube_h = cube_w * (self.resolution_y / self.resolution_x)
        print(f"Cube dimensions: width={cube_w}, height={cube_h}")

        # add glass 
        size = 1.0
        if self.args.glass_as_plane:
            bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False)
        else:
            bpy.ops.mesh.primitive_cube_add(size=size, enter_editmode=False)
        self.glass = bpy.context.active_object
        self.glass.rotation_euler = self.camera.rotation_euler # location is not normalized~!!!!    
        camera_primary_vectory_cartesian = self._cartesian_from_euler(self.primary_axis_spherical)
        self.glass.location = tuple(-camera_primary_vectory_cartesian * self.glass_distance)
        self.glass.scale = (cube_w * 1.25, 
                            cube_h * 1.25, 
                            self.args.glass_thickness)

        # Create glass material
        glass_material = bpy.data.materials.new(name="GlassMaterial")
        glass_material.use_nodes = True
        nodes = glass_material.node_tree.nodes
        links = glass_material.node_tree.links
        nodes.clear()
        
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.location = (0, 0)
        node_bsdf.inputs['Coat IOR'].default_value = self.args.glass_default_ior
        node_bsdf.inputs['Coat Roughness'].default_value = self.args.glass_roughness
        node_bsdf.inputs['Coat Weight'].default_value = 1.0
        node_bsdf.inputs['Metallic'].default_value = self.args.glass_metallic
        node_bsdf.inputs['Roughness'].default_value = 0.0
        node_bsdf.inputs['IOR'].default_value = 1.0
        node_bsdf.inputs['Base Color'].default_value = tuple([self.args.glass_color_grey_scale] * 4)
        node_bsdf.inputs['Transmission Weight'].default_value = 1.0
        node_bsdf.inputs['Alpha'].default_value = self.args.glass_alpha

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = (300, 0)
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        self.glass.data.materials.append(glass_material)

        self.medium = self.glass

    
    def get_dimensions(self, obj):
        mat = obj.matrix_world

        world_bbox = [mat @ Vector(corner) for corner in obj.bound_box]

        # Find min and max coordinates
        min_corner = Vector((min(v.x for v in world_bbox),
                                    min(v.y for v in world_bbox),
                                    min(v.z for v in world_bbox)))
        max_corner = Vector((max(v.x for v in world_bbox),
                                    max(v.y for v in world_bbox),
                                    max(v.z for v in world_bbox)))

        dimensions = max_corner - min_corner

        return dimensions


    def add_img(self):
        img_path = os.path.abspath(self.args.img_path)
        bpy.ops.image.import_as_mesh_planes(
            relative=False, 
            filepath=img_path, 
            files=[{"name":os.path.basename(img_path)}], 
            directory=os.path.dirname(img_path)
        )

        sensor_width = self.camera.data.sensor_width
        img_dimensions = self.get_dimensions(bpy.context.active_object)
        scale = 1.05 * sensor_width / (self.camera.data.lens) / max(img_dimensions)
        print(f"Image dimensions: {img_dimensions}, scale: {scale}")

        self.img = bpy.context.active_object
        self.img.rotation_euler = self.camera.rotation_euler # location is not normalized~!!!!
        camera_primary_vectory_cartesian = self._cartesian_from_euler(self.primary_axis_spherical)
        self.img.location = tuple(-camera_primary_vectory_cartesian)
        self.img.scale = (scale, scale, scale)

        # Make the default material properties 
        for mat in self.img.data.materials:
            if mat.use_nodes:
                bsdf_node = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf_node:
                    bsdf_node.inputs['IOR'].default_value = self.args.glass_default_ior
                    bsdf_node.inputs['Roughness'].default_value = self.args.glass_roughness
                    bsdf_node.inputs['Metallic'].default_value = self.args.glass_metallic
        self.medium = self.img

    def render(self, path):
        # Render the scene
        # Set render layers for RGB, Depth, and Mask
        self.scene.use_nodes = True

        tree = self.scene.node_tree
        tree.nodes.clear()

        # Create necessary nodes
        render_layers = tree.nodes.new(type='CompositorNodeRLayers')

        # RGB output
        composite_rgb = tree.nodes.new('CompositorNodeOutputFile')
        composite_rgb.base_path = '' 
        composite_rgb.file_slots[0].path = os.path.relpath(path, start=os.path.dirname(bpy.data.filepath))
        composite_rgb.format.file_format = 'PNG'

        tree.links.new(render_layers.outputs['Image'], composite_rgb.inputs[0])

        # Render
        bpy.ops.render.render(write_still=True)

    def make_transmissive(self):
        print(self.medium)
        # Make the materials transmissive
        for mat in self.medium.data.materials:
            if mat.use_nodes:
                bsdf_node = mat.node_tree.nodes.get('Principled BSDF')
                if bsdf_node:
                    # bsdf_node.inputs['Transmission Weight'].default_value = 1.0 if self.glass is not None else 0.0
                    bsdf_node.inputs['Coat IOR'].default_value = self.args.glass_transmissive_ior
                    bsdf_node.inputs['Coat Roughness'].default_value = 0.0#self.args.glass_roughness
                    bsdf_node.inputs['Metallic'].default_value = 0.0
                    bsdf_node.inputs['Base Color'].default_value = tuple([self.args.glass_color_grey_scale] * 4)
                    bsdf_node.inputs['Alpha'].default_value = self.args.glass_alpha

    def make_reflective(self):
        # Create reflective material
        material = self.medium.data.materials[0]
        material.use_nodes = True
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        nodes.clear()
        
        node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        node_bsdf.location = (0, 0)
        node_bsdf.inputs['Coat IOR'].default_value = self.args.glass_default_ior
        node_bsdf.inputs['Coat Roughness'].default_value = 0.0
        node_bsdf.inputs['Coat Weight'].default_value = 0.0
        node_bsdf.inputs['Metallic'].default_value = 1.0
        node_bsdf.inputs['Roughness'].default_value = 0.0
        node_bsdf.inputs['Base Color'].default_value = tuple([self.args.glass_color_grey_scale] * 4)
        node_bsdf.inputs['Transmission Weight'].default_value = 1.0
        node_bsdf.inputs['Alpha'].default_value = 1.0

        node_output = nodes.new(type='ShaderNodeOutputMaterial')
        node_output.location = (300, 0)
        links.new(node_bsdf.outputs['BSDF'], node_output.inputs['Surface'])
        self.medium.data.materials.append(material)
    
    def hide_glass(self):
        # Hide the glass object
        if self.glass:
            self.glass.hide_set(True)
            self.glass.hide_render = True
            print("Glass object hidden for rendering.")
        else:
            print("No glass object to hide.")

    def save_config(self, path):
        # Save the configuration
        with open(path, 'w') as f:
            json.dump(self.args.__dict__, f, indent=4)
        
    def _convert_img_mesh_material(self, obj, img_path):
        print("Converting image mesh material...")
        # Convert the image texture to a mesh material
        assert obj.type == 'MESH', "Object must be a mesh"
        # clear material nodes 
        for mat in obj.data.materials:
            if mat.use_nodes:
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links
                nodes.clear()

                node_background = nodes.new(type='ShaderNodeBackground')
                node_background.location = (-300, 0)
                node_output = nodes.new(type='ShaderNodeOutputMaterial')
                node_output.location = (300, 0)
                links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

                # Load the image texture
                img_texture = nodes.new(type='ShaderNodeTexImage')
                img_texture.location = (-600, 0)
                img_texture.image = bpy.data.images.load(img_path)
                links.new(img_texture.outputs['Color'], node_background.inputs['Color'])

def rename(path):
    # Rename the output files to match the expected format
    for ext in ['.png', '.exr']:
        if os.path.exists(path + "0001" + ext):
            new_path = path + ext
            os.rename(path + "0001" + ext, new_path)
            print(f"Renamed {path + '0001' + ext} to {new_path}")
        else:
            print(f"File {path + ext} does not exist.")

def get_args():
    parser = argparse.ArgumentParser(description="Render HDR images from a dataset.")
    # you can use "/home/daniyar/Desktop/datasets/polyhaven_4k/abandoned_hall_01_4k.hdr"
    parser.add_argument("--background_hdr_path", type=str, default=None, help="Path to the input HDR image.")
    # you can use "/home/daniyar/Desktop/datasets/coco2017unlabeled/000000322291.jpg"
    parser.add_argument("--background_img_path", type=str, default=None, help="Path to the input background image (used if mode is 'img').")
    parser.add_argument("--mode", type=str, choices=["hdri", "img"], default="hdri", help="Rendering mode. If 'hdri', we render HDRI panorama. If 'img', we render a standard image with a HDRI panorama background.")
    parser.add_argument("--img_path", type=str, default=None, help="Path to the input image (required if mode is 'img').")
    parser.add_argument("--output_name", type=str, default="sample", help="Prefix of the output files.")
    parser.add_argument("--output_folder", type=str, default="storage/tmp/", help="Path to the output folder for rendered images.")
    parser.add_argument("--resolution_x", type=int, default=1920, help="Width of the rendered image.")
    parser.add_argument("--resolution_y", type=int, default=1080, help="Height of the rendered image.")
    parser.add_argument("--cycles_samples", type=int, default=128, help="Number of samples for Cycles rendering.")
    parser.add_argument("--camera_focal_length", type=float, default=15.0, help="Focal length of the camera in mm.")
    parser.add_argument("--camera_azimuth", type=float, default=0.0, help="Azimuth angle for the camera in degrees. Describes the direction of the `up` in the xy plane")
    parser.add_argument("--camera_zimuth", type=float, default=90.0, help="Zimuth angle for the camera in degrees. Describes the direction of the `up` vector in the xz plane.")
    parser.add_argument("--camera_tilt", type=float, default=0.0, help="Tilt angle for the camera in degrees. Describes the rotation around the primary axis of the camera.")
    parser.add_argument("--background_strength", type=float, default=1.0, help="Strength of the background lighting.")
    parser.add_argument("--glass_default_ior", type=float, default=1.45, help="IOR for the glass material.")
    parser.add_argument("--glass_distance", type=float, default=5.0, help="Distance of the glass material from the camera.")
    parser.add_argument("--glass_transmissive_ior", type=float, default=1.0, help="IOR for the transmissive glass material.")
    parser.add_argument("--glass_roughness", type=float, default=0.0, help="Roughness of the glass material.")
    parser.add_argument("--glass_metallic", type=float, default=0.0, help="Metallic property of the glass material.")
    parser.add_argument("--glass_color_grey_scale", type=float, default=1.0, help="Grey scale for the glass color.")
    parser.add_argument("--glass_thickness", type=float, default=0.01, help="Thickness of the glass material.")
    parser.add_argument("--glass_alpha", type=float, default=1.0, help="Alpha value for the glass material.")
    parser.add_argument("--save_blend", action="store_true", help="Save the Blender file.")
    parser.add_argument('--glass_as_plane', action='store_true', help='Use a plane for the glass instead of a cube.')

    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

    assert args.mode == 'hdri' or (args.mode == 'img' and args.img_path is not None), "If mode is 'img', img_path must be provided."
    assert sum([args.background_hdr_path is not None, args.background_img_path is not None]) == 1, "Either background_hdr_path or background_img_path must be provided, but not both."

    return args

if __name__ == "__main__":
    args = get_args()

    if args.mode == "hdri":
        scene = Scene(args)

        scene.load_hdr_background()

        scene.setup()

        scene.add_glass()

        scene.render(os.path.join(args.output_folder, f"{args.output_name}_default"))

        scene.make_transmissive()

        scene.render(os.path.join(args.output_folder, f"{args.output_name}_transmissive"))

        scene.make_reflective()

        scene.render(os.path.join(args.output_folder, f"{args.output_name}_reflection"))

        scene.hide_glass()

        scene.render(os.path.join(args.output_folder, f"{args.output_name}_no_glass"))

        if hasattr(args, 'save_blend') and args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.output_folder, f"{args.output_name}.blend"))
        else:
            print("Blender file saving is disabled.")

        scene.save_config(os.path.join(args.output_folder, f"{args.output_name}_config.json"))

        rename(os.path.join(args.output_folder, f"{args.output_name}_default"))
        rename(os.path.join(args.output_folder, f"{args.output_name}_transmissive"))
        rename(os.path.join(args.output_folder, f"{args.output_name}_no_glass"))
        rename(os.path.join(args.output_folder, f"{args.output_name}_reflection"))
    elif args.mode == "img":
        scene = Scene(args)


        # Set up camera and scene parameters
        scene.setup()

        # Load the HDR image as environment background
        if args.background_hdr_path is not None:
            scene.load_hdr_background()
        else: 
            scene.load_img_background()
            scene._convert_img_mesh_material(scene.background_img, args.background_img_path)
        
        # Add the image as a mesh plane in front of the camera
        scene.add_img()
        scene._convert_img_mesh_material(scene.img, args.img_path)

        print("Rendering default image with glass and background...")

        # adding reflective glass into the scene 
        # if args.background_img_path is not None:
        scene.add_glass()


        # Render the default image with glass and background
        scene.render(os.path.join(args.output_folder, f"{args.output_name}_default"))

        # Make the image material transmissive (glass-like)
        scene.make_transmissive()

        # Render the image with transmissive glass material
        scene.render(os.path.join(args.output_folder, f"{args.output_name}_transmissive"))

        # Make the image material reflective (mirror-like)
        scene.make_reflective()

        # Render the image with reflective glass material
        scene.render(os.path.join(args.output_folder, f"{args.output_name}_reflection"))

        # Optionally save the Blender file
        if hasattr(args, 'save_blend') and args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.output_folder, f"{args.output_name}.blend"))
        else:
            print("Blender file saving is disabled.")

        # Save the configuration to a JSON file
        scene.save_config(os.path.join(args.output_folder, f"{args.output_name}_config.json"))

        rename(os.path.join(args.output_folder, f"{args.output_name}_default"))
        rename(os.path.join(args.output_folder, f"{args.output_name}_transmissive"))
        rename(os.path.join(args.output_folder, f"{args.output_name}_reflection"))