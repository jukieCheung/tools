import bpy
import os
import traceback

#blender 3.0.0 version



# ---------------------------
# 1. 设置 PLY 文件路径（请修改）
# ---------------------------
ply_folder = "D:\ReLPU2"  # 你的 PLY 点云文件夹路径
output_folder = "D:\ReLPU4"  # 你的 PNG 渲染输出路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ---------------------------
# 2. Blender 渲染设置
# ---------------------------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'  # 使用 Cycles 渲染引擎
scene.cycles.samples = 128  # 设定采样数
scene.render.image_settings.file_format = 'PNG'  # 确保输出 PNG

# 获取所有 PLY 文件
ply_files = [f for f in os.listdir(ply_folder) if f.endswith(".ply")]

# ---------------------------
# 3. 清理场景
# ---------------------------
def clean_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

# ---------------------------
# 4. 确保有相机
# ---------------------------
def setup_camera():
    if "Camera" not in bpy.data.objects:
        bpy.ops.object.camera_add(location=(2.8, 0.8, 1.8))  # 远一点，确保物体完整
        #bpy.ops.object.camera_add(location=(2, -2, 1.5))  # 远一点，确保物体完整
    camera = bpy.data.objects["Camera"]
    camera.rotation_euler = (1.0, -0.5, 2)  # 调整角度
    #camera.rotation_euler = (1.0, -0.5, 1)  # 调整角度
    scene.camera = camera

# ---------------------------
# 5. 确保有光源
# ---------------------------
def setup_light():
    if "Light" not in bpy.data.objects:
        bpy.ops.object.light_add(type='SUN', location=(2, -2, 4))
    light = bpy.data.objects["Light"]
    light.data.energy = 5  # 适量光照

# ---------------------------
# 6. 创建材质
# ---------------------------
def create_orange_material():
    material = bpy.data.materials.new(name="OrangeMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 0.5, 0, 1)  # 橙色
    return material

# ---------------------------
# 7. 创建 Geometry Nodes 让点云变成实体并赋予材质
# ---------------------------
def create_geometry_nodes(obj):
    try:
        # 添加 Geometry Nodes 修改器
        geo_mod = obj.modifiers.new(name="GeoNodes", type='NODES')

        # 创建 Geometry Nodes 组
        geo_nodes = bpy.data.node_groups.new(name="PointCloud_GeoNodes", type='GeometryNodeTree')
        geo_mod.node_group = geo_nodes

        nodes = geo_nodes.nodes
        links = geo_nodes.links
        nodes.clear()

        # 创建输入/输出节点
        input_node = nodes.new(type="NodeGroupInput")
        output_node = nodes.new(type="NodeGroupOutput")

        geo_nodes.inputs.new("NodeSocketGeometry", "Geometry")
        geo_nodes.outputs.new("NodeSocketGeometry", "Geometry")

        # 创建 Icosahedron 作为点云球体
        point_instance = nodes.new(type="GeometryNodeInstanceOnPoints")
        mesh_ico = nodes.new(type="GeometryNodeMeshUVSphere")
        mesh_ico.inputs["Radius"].default_value = 0.003  # 小球大小

        # 添加 "Set Material" 节点，应用材质
        set_material = nodes.new(type="GeometryNodeSetMaterial")

        # 连接 Geometry Nodes
        links.new(input_node.outputs[0], point_instance.inputs[0])
        links.new(mesh_ico.outputs[0], point_instance.inputs[2])
        links.new(point_instance.outputs[0], set_material.inputs[0])  # 连接到 "Set Material"
        links.new(set_material.outputs[0], output_node.inputs[0])

        # 创建并赋予橙色材质
        orange_material = create_orange_material()
        set_material.inputs["Material"].default_value = orange_material

    except Exception as e:
        traceback.print_exc()

# ---------------------------
# 8. 处理 PLY 文件
# ---------------------------
for ply_file in ply_files:
    try:

        # 8.1 清空场景
        clean_scene()

        # 8.2 重新加载 PLY
        ply_path = os.path.join(ply_folder, ply_file)
        bpy.ops.import_mesh.ply(filepath=ply_path)

        # 8.3 获取导入的 PLY 对象
        obj = bpy.context.selected_objects[0]
        obj.name = "PointCloud"

        # 8.4 应用 Geometry Nodes（含 Set Material）
        create_geometry_nodes(obj)

        # 8.5 设置相机和光源
        setup_camera()
        #setup_light()

        # ---------------------------
        # 9. 设置当前 PLY 文件的输出渲染图片路径
        # ---------------------------
        output_path = os.path.join(output_folder, ply_file.replace('.ply', '.png'))
        scene.render.filepath = output_path
        
        # 9.1 渲染当前场景并保存图像
        bpy.ops.render.render(write_still=True)

    except Exception as e:
        traceback.print_exc()

