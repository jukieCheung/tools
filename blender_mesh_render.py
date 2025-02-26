import bpy
import os
import traceback

# ---------------------------
# 1. 设置 STL 文件路径（请修改）
# ---------------------------
stl_folder = "D:\\22"  # 你的 STL 文件夹路径
output_folder = "D:\\223"  # 你的 PNG 渲染输出路径

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

# 设置背景为白色
scene.world.color = (1, 1, 1)  # 纯白色背景

# 获取所有 STL 文件
stl_files = [f for f in os.listdir(stl_folder) if f.endswith(".stl")]

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
        bpy.ops.object.camera_add(location=(3, 0.35, 0.9))  # 远一点，确保物体完整
    camera = bpy.data.objects["Camera"]
    camera.rotation_euler = (1.0, 1, 1.2)  # 调整角度
    scene.camera = camera

# ---------------------------
# 6. 创建橙色材质
# ---------------------------
def create_orange_material():
    material = bpy.data.materials.new(name="OrangeMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (1, 0.5, 0, 1)  # 橙色
    return material

# ---------------------------
# 7. 处理 STL 文件并赋予材质
# ---------------------------
for stl_file in stl_files:
    try:
        print(f"📂 正在处理: {stl_file}")

        # 7.1 清空场景
        clean_scene()

        # 7.2 重新加载 STL
        stl_path = os.path.join(stl_folder, stl_file)
        bpy.ops.import_mesh.stl(filepath=stl_path)

        # 7.3 获取导入的 STL 对象
        obj = bpy.context.selected_objects[0]
        obj.name = "MeshObject"

        # 7.4 给网格赋予橙色材质
        orange_material = create_orange_material()
        obj.data.materials.append(orange_material)

        # 7.5 设置相机和光源
        setup_camera()
        #setup_light()

        # ---------------------------
        # 8. 设置当前 STL 文件的输出渲染图片路径
        # ---------------------------
        output_path = os.path.join(output_folder, stl_file.replace('.stl', '.png'))
        scene.render.filepath = output_path
        
        # 8.1 渲染当前场景并保存图像
        bpy.ops.render.render(write_still=True)
        print(f"✅ 渲染完成并保存: {output_path}")

    except Exception as e:
        print(f"❌ 处理 {stl_file} 时发生错误: {str(e)}")
        traceback.print_exc()

print("🎉 所有 STL 文件处理完成！")
