import os
import re
import json
import time
import uuid
import random
import base64
import uvicorn
import requests
from typing import Optional
from urllib.parse import quote, unquote
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response

# 全局变量：保存固定种子值（初始为None）
FIXED_SEED = None  # 种子固定模式下，复用此值
JUPYTER_URL = "https://a*****"  #在autodl中jupyter的链接地址前面是a

# ====================== 1. 核心配置（你需要修改的部分） ======================
class Config:
    # 种子参数名配置：支持所有工作流的种子命名，可自由添加（如seed/noise_seed/random_seed等）
    SEED_PARAM_NAMES = ["seed", "noise_seed", "random_seed", "latent_seed"]  # 你可根据需要添加，比如新增"random_seed"

    # 本地LLM配置（Ollama）
    OLLAMA_API_URL = "http://localhost:11434/v1/chat/completions"
    OLLAMA_MODEL = "llava:7b"
    
    # 远程ComfyUI配置（公网地址）
    COMFYUI_API_URL = "https://u*****"  # 替换为你的ComfyUI公网地址，在autodl中comfyui的链接地址前面是u
    COMFYUI_UPLOAD_URL = f"{COMFYUI_API_URL}ComfyUI/input"
    COMFYUI_PROMPT_URL = f"{COMFYUI_API_URL}/prompt"
    COMFYUI_HISTORY_URL = f"{COMFYUI_API_URL}/history"
    
    # 工作流文件路径（替换为你的本地工作流JSON路径）
    WORKFLOW_PATHS = {
        "z_image": r"D:\浏览器下载\Z-Image_双重采样工作流.json",               # Z-Image双重采样工作流
        "qwen_edit": r"D:\浏览器下载\Qwen-Imag-Eedit-2511图像编辑.json"       # Qwen图像编辑工作流
    }
    
    # Z-Image模型路径映射（12种组合，替换为你的实际模型路径）
    Z_IMAGE_BASE_MODELS = {
        1: "ComfyUI/models/diffusion_models/z_image_bf16.safetensors",
        2: "ComfyUI/models/diffusion_models/zib/moodyWildMix_v10Base50steps.safetensors",
        3: "ComfyUI/models/diffusion_models/zib/radianceZ_v10.safetensors"
    }
    Z_IMAGE_TURBO_MODELS = {
        1: "ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors",
        2: "ComfyUI/models/diffusion_models/zit/moodyPornMix_zitV8.safetensors",
        3: "ComfyUI/models/diffusion_models/zit/pornmasterZImage_turboV1.safetensors",
        4: "ComfyUI/models/diffusion_models/zit/zImageTurboNSFW_43BF16Diffusion.safetensors"
    }
    
    # 生成结果保存目录（ComfyUI的输出目录，需和远程一致）
    COMFYUI_OUTPUT_DIR = "ComfyUI/output"

# ====================== 2. 工具函数：ComfyUI API调用核心 ======================
def load_workflow(workflow_type: str) -> dict:
    """加载指定类型的工作流JSON"""
    workflow_path = Config.WORKFLOW_PATHS.get(workflow_type)
    if not workflow_path or not os.path.exists(workflow_path):
        raise FileNotFoundError(f"工作流文件不存在：{workflow_path}")
    with open(workflow_path, "r", encoding="utf-8") as f:
         workflow = json.load(f)
    
    # 关键修复：ComfyUI导出的API工作流直接是prompt内容，需包装成{"prompt": ...}
    if "prompt" not in workflow:
        workflow = {"prompt": workflow}
    
    return workflow

def replace_z_image_model(workflow: dict, base_model_id: int, turbo_model_id: int) -> dict:
    """替换Z-Image工作流中的base/turbo模型路径"""
    # 遍历所有节点，找到UNetLoader节点并替换模型路径
    base_path = Config.Z_IMAGE_BASE_MODELS.get(base_model_id)
    turbo_path = Config.Z_IMAGE_TURBO_MODELS.get(turbo_model_id)
    if not base_path or not turbo_path:
        raise ValueError(f"模型ID无效：base={base_model_id}, turbo={turbo_model_id}")

    field_name = "unet_name"
    
    for node_id, node in workflow["prompt"].items():
        if node.get("class_type") == "UNETLoader":
            # 区分base和turbo模型节点（根据节点名称/参数特征，你可根据实际JSON调整）
            if "base" in node["inputs"].get("model_name", "").lower() or len(workflow["prompt"][node_id]["inputs"]) == 1:
                workflow["prompt"][node_id]["inputs"]["model_name"] = base_path
            elif "turbo" in node["inputs"].get("model_name", "").lower():
                workflow["prompt"][node_id]["inputs"]["model_name"] = turbo_path
    return workflow

def replace_prompt(workflow: dict, prompt_text: str, negative_prompt: str = "") -> dict:
    """
    替换工作流中的提示词
    兼容传统的文生图，以及全新精简版的 Qwen 图生图工作流
    """
    # 获取真实节点集合（兼容 workflow 本身就是节点字典 或 包含在 "prompt" 键中的情况）
    nodes = workflow.get("prompt", workflow)
    
    for node_id, node in nodes.items():
        class_type = node.get("class_type")
        meta = node.get("_meta", {})
        inputs = node.get("inputs", {})
        
        # 1. 兼容图生图：Qwen 图像编辑工作流
        if class_type == "TextEncodeQwenImageEditPlus":
            # 精准寻找节点：只要您的节点标题叫做 "正向" 或者 默认填了 "123444" 都会被命中
            if meta.get("title") == "正向" or inputs.get("prompt") == "123444":
                # 将大模型处理好的提示词覆盖掉原有的 "123444"
                nodes[node_id]["inputs"]["prompt"] = prompt_text
                
        # 2. 兼容文生图：传统的 CLIP 节点工作流
        elif class_type == "CLIPTextEncode":
            if "text" in inputs:
                # 正向提示词节点（检测为空时替换）
                if not inputs["text"]:
                    nodes[node_id]["inputs"]["text"] = prompt_text
                # 负向提示词节点（检测到"泛黄"时替换）
                elif "泛黄" in inputs["text"]:
                    nodes[node_id]["inputs"]["text"] = negative_prompt

    return workflow

def upload_image_to_comfyui(image_file: UploadFile) -> str:
    """上传图片到ComfyUI，返回图片文件名"""
    base_url = Config.COMFYUI_API_URL.rstrip('/')
    upload_url = f"{base_url}/upload/image"
    try:
        # 重置文件指针
        image_file.file.seek(0)
        # 读取完整二进制流
        file_bytes = image_file.file.read()
        
        # 按照 ComfyUI 标准接口构建 multipart/form-data
        files = {
            "image": (
                image_file.filename or "uploaded_image.png", 
                file_bytes, 
                image_file.content_type or "image/png"
            )
        }
        
        # 向真正的云端地址发送 POST 请求
        response = requests.post(upload_url, files=files, timeout=60)

        if response.status_code == 200:
            result = response.json()
            return result.get("name")
        else:
            err_msg = f"{response.status_code}: 图片上传失败, 详情: {response.text}\n请求地址: {upload_url}"
            raise Exception(err_msg)
        
    except Exception as e:
        raise Exception(f"请求ComfyUI上传接口发生异常: {str(e)}")

def run_comfyui_workflow(workflow: dict, image_filename: str = None) -> str:
    """提交工作流到ComfyUI并等待生成完成，返回结果文件路径"""
    # 1. 提交工作流
    prompt_id = str(uuid.uuid4())
    payload = {
        "prompt_id": prompt_id,
        "prompt": workflow["prompt"],
        "client_id": "comfy_agent"
    }
    # 如果是图生图，添加图片输入（根据工作流节点ID调整）
    if image_filename:
        for node_id, node in workflow["prompt"].items():
            if node.get("class_type") == "LoadImage":
                workflow["prompt"][node_id]["inputs"]["image"] = image_filename
    
    response = requests.post(Config.COMFYUI_PROMPT_URL, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"提交工作流失败：{response.text}")
    
    # 2. 轮询等待生成完成
    max_wait = 300  # 最大等待5分钟
    start_time = time.time()
    while time.time() - start_time < max_wait:
        history_response = requests.get(f"{Config.COMFYUI_HISTORY_URL}/{prompt_id}")
        if history_response.status_code == 200 and history_response.json():
            history = history_response.json()
            if prompt_id in history and "outputs" in history[prompt_id]:
                # 提取生成的图片路径
                outputs = history[prompt_id]["outputs"]
                for node_output in outputs.values():
                    if "images" in node_output:
                        img_info = node_output["images"][0]
                        img_path = os.path.join(Config.COMFYUI_OUTPUT_DIR, img_info["subfolder"], img_info["filename"])
                        return img_path
        time.sleep(2)  # 每2秒查询一次
    raise TimeoutError("生成超时（超过5分钟）")

# ====================== 3. 工具函数：LLM聊天功能 ======================
def llm_chat(message: str) -> str:
    """调用本地Ollama的llava:7b进行聊天"""
    payload = {
        "model": Config.OLLAMA_MODEL,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.7,
        "stream": False
    }
    try:
        response = requests.post(Config.OLLAMA_API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"LLM调用失败：{response.status_code} - {response.text}"
    except Exception as e:
        return f"LLM聊天异常：{str(e)}"

# ====================== 4. 核心：指令解析与Agent处理 ======================
def parse_user_command(command: str) -> dict:
    """解析用户指令，返回指令类型、提示词、模型参数、种子参数等"""
    command = command.strip()
    result = {
        "type": "chat",          # 指令类型：chat / text2img / img2img
        "prompt": "",            # 生成提示词
        "base_model_id": 1,      # Z-Image base模型ID（默认1）
        "turbo_model_id": 1,     # Z-Image turbo模型ID（默认1）
        "seed_mode": "random",   # 种子模式：fixed（固定）/specify（指定数字）/random（随机）
        "seed_value": None,      # 指定的种子数字（仅seed_mode=specify时有效）
        "width": 1080,            # 默认宽度
        "height": 1920,           # 默认高度
        "error": ""
    }
    
    # 解析文生图/图生图指令
    if command.startswith("文生图：") or command.startswith("图生图："):
        # 标记指令类型
        result["type"] = "text2img" if command.startswith("文生图：") else "img2img"
        # 提取核心内容（去掉前缀）
        content = command.replace("文生图：", "").replace("图生图：", "").strip()
        
        # 拆分「提示词 + 参数（模型/种子）」（用|分隔）
        prompt_part = content
        param_part = ""
        if "|" in content:
            prompt_part, param_part = content.split("|", 1)
            prompt_part = prompt_part.strip()
            param_part = param_part.strip()
        
        result["prompt"] = prompt_part
        
        # 解析参数（模型+种子）
        if param_part:
            # 兼容处理：替换中文逗号，且去掉多余的“模型：”字样
            param_part = param_part.replace("，", ",").replace("模型：", "")
            params = param_part.split(",")
            for param in params:
                param = param.strip()
                # 解析模型参数
                if "base=" in param:
                    try:
                        result["base_model_id"] = int(param.replace("base=", "").strip())
                    except:
                        result["error"] = "base模型ID必须是数字（1-3）"
                elif "turbo=" in param:
                    try:
                        result["turbo_model_id"] = int(param.replace("turbo=", "").strip())
                    except:
                        result["error"] = "turbo模型ID必须是数字（1-4）"
                # 解析种子参数
                elif "种子：" in param:
                    try:
                        seed_num = int(param.replace("种子：", "").strip())
                        result["seed_mode"] = "specify"
                        result["seed_value"] = seed_num
                    except:
                        result["error"] = "种子必须是整数数字（如：种子：123456）"
                elif param == "种子固定":
                    result["seed_mode"] = "fixed"
                elif param == "种子随机":
                    result["seed_mode"] = "random"
                # 解析分辨率参数
                elif "分辨率：" in param:
                    try:
                        res_str = param.replace("分辨率：", "").strip()
                        width, height = res_str.split("x")
                        result["width"] = int(width)
                        result["height"] = int(height)
                        # 校验分辨率（可选：限制为偶数，ComfyUI通常要求）
                        if result["width"] % 2 != 0 or result["height"] % 2 != 0:
                            result["error"] = "分辨率宽高必须为偶数（如1920x1080）"
                    except:
                        result["error"] = "分辨率格式错误（正确示例：分辨率：1920x1080）"
        
        # 校验模型ID
        if result["base_model_id"] not in [1,2,3] or result["turbo_model_id"] not in [1,2,3,4]:
            result["error"] = f"模型ID超出范围：base(1-3)/turbo(1-4)"
    
    # 返回解析结果
    return result

def replace_seed(workflow: dict, seed_mode: str, seed_value: int = None) -> dict:
    """
    替换工作流中的种子值
    :param workflow: 工作流字典
    :param seed_mode: 种子模式：fixed（固定）/specify（指定数字）/random（随机）
    :param seed_value: 指定的种子数字（仅seed_mode=specify时有效）
    :return: 修改后的工作流
    """
    global FIXED_SEED  # 引用全局固定种子变量
    
    # 1. 确定最终使用的种子值
    final_seed = None
    if seed_mode == "specify":
        final_seed = seed_value
        FIXED_SEED = final_seed  # 指定种子后，更新固定种子值
    elif seed_mode == "fixed":
        if FIXED_SEED is None:
            # 首次固定种子，随机生成一个并保存
            final_seed = random.randint(1, 999999999999)
            FIXED_SEED = final_seed
        else:
            # 复用已保存的固定种子
            final_seed = FIXED_SEED
    elif seed_mode == "random":
        # 随机生成种子（范围：1~12位数字）
        final_seed = random.randint(1, 999999999999)
        FIXED_SEED = None  # 随机模式下，清空固定种子
    
    # 2. 遍历工作流节点，替换所有seed参数
    #for node_id, node in workflow["prompt"].items():
        # 匹配包含种子的节点（KSampler、RandomSeed、Seed等）
        #if "seed" in node["inputs"]:
            #workflow["prompt"][node_id]["inputs"]["seed"] = final_seed
        # 兼容部分节点的"noise_seed"参数
        #elif "noise_seed" in node["inputs"]:
            #workflow["prompt"][node_id]["inputs"]["noise_seed"] = final_seed

    # 2. 核心修改：遍历所有配置的种子参数名，批量替换（适配所有工作流）
    for seed_param in Config.SEED_PARAM_NAMES:  # 遍历你配置的所有种子参数名
        for node_id, node in workflow["prompt"].items():
            if seed_param in node["inputs"]:  # 匹配到该种子参数名则替换
                workflow["prompt"][node_id]["inputs"][seed_param] = final_seed
    
    # 返回修改后的工作流+种子值（用于提示）
    return workflow, final_seed

def replace_resolution(workflow: dict, width: int, height: int) -> dict:
    """替换工作流中的分辨率（适配EmptyLatentImage节点）"""
    # 遍历节点，找到EmptyLatentImage节点替换宽高
    for node_id, node in workflow["prompt"].items():
        if node.get("class_type") == "EmptyLatentImage":
            workflow["prompt"][node_id]["inputs"]["width"] = width
            workflow["prompt"][node_id]["inputs"]["height"] = height
    return workflow

def agent_handle(command: str, image_file: Optional[UploadFile] = None) -> dict:
    """Agent核心处理逻辑"""
    # 1. 解析指令
    parsed = parse_user_command(command)
    if parsed["error"]:
        return {"status": "error", "message": parsed["error"]}
    
    # 2. 聊天指令
    if parsed["type"] == "chat":
        reply = llm_chat(command)
        return {"status": "success", "message": reply}
    
    # 3. 文生图/图生图指令
    try:
        # ===== 修改开始：根据指令类型加载不同的工作流并处理专属参数 =====
        if parsed["type"] == "text2img":
            # 加载Z-Image工作流
            workflow = load_workflow("z_image")
            
            # 替换模型路径
            workflow = replace_z_image_model(workflow, parsed["base_model_id"], parsed["turbo_model_id"])
        elif parsed["type"] == "img2img":
            # 图生图：加载Qwen图像编辑工作流
            workflow = load_workflow("qwen_edit")
            # 图生图：不替换分辨率，也不替换模型，保留默认配置直接进入公用流程

        # 替换提示词
        workflow = replace_prompt(workflow, parsed["prompt"])
        
        # ===== 替换种子值 =====
        workflow, final_seed = replace_seed(
            workflow=workflow,
            seed_mode=parsed["seed_mode"],
            seed_value=parsed["seed_value"]
        )
        # ===== 替换分辨率 =====
        workflow = replace_resolution(
            workflow=workflow,
            width=parsed["width"],
            height=parsed["height"]
        )
        
        # 图生图：上传图片
        image_filename = None
        if parsed["type"] == "img2img" and image_file:
            image_filename = upload_image_to_comfyui(image_file)

             # 确保将上传后的文件名注入进图生图工作流内的“LoadImage”节点
            for node_id, node in workflow["prompt"].items():
                if node.get("class_type") == "LoadImage":
                    workflow["prompt"][node_id]["inputs"]["image"] = image_filename
        
        # 运行ComfyUI工作流
        img_path = run_comfyui_workflow(workflow, image_filename)

        # === 读取本地原图直接转码 ===
        base64_img_data = ""
        if img_path and os.path.exists(img_path):
            with open(img_path, "rb") as f:
                # 把图片转换成Base64数据形式
                encoded_string = base64.b64encode(f.read()).decode("utf-8")
                # 拼接成标准的 Data URI 格式供网页渲染
                base64_img_data = f"data:image/png;base64,{encoded_string}"
        else:
            print(f"[错误] 找不到生成的图片路径: {img_path}")
        
        # 正确解析图片URL（保留之前的路径修复逻辑）
        img_dir = os.path.dirname(img_path).replace(Config.COMFYUI_OUTPUT_DIR, "").strip(os.sep)
        # 将操作系统的路径分隔符统一转为URL使用的斜杠（避免跨平台错误）
        img_dir_url = img_dir.replace("\\", "/")
        img_name = os.path.basename(img_path)
        
        # 你的云端 Jupyter Lab 前缀地址
        CLOUD_FILES_URL = f"{JUPYTER_URL}jupyter/files/ComfyUI/output"
        # ⚠️ 将你抓到的完整 _xsrf 值写在这里（若之后失效不显示图片了，替换这里的字符串即可）
        XSRF_TOKEN = "*******"

        # 动态拼接最终提取链接
        if img_dir_url:
            target_url = f"{CLOUD_FILES_URL}/{img_dir_url}/{img_name}?_xsrf={XSRF_TOKEN}"
        else:
            target_url = f"{CLOUD_FILES_URL}/{img_name}?_xsrf={XSRF_TOKEN}"

        # 2. 优化：不再直接向前端丢原链接，而是将目标路径做 URL 编码并导向我们的代理路由
        # 注意此处的 8000 端口需要和你的后端的端口保持一致
        preview_url = f"http://192.168.*.*:8000/proxy-image?url={quote(target_url)}"

         # 提供一个供日常点击跳转查看的常规云端目录地址
        CLOUD_TREE_URL = f"{JUPYTER_URL}jupyter/lab/tree/ComfyUI/output"
        real_url = f"{CLOUD_TREE_URL}/{img_dir_url}/{img_name}" if img_dir_url else f"{CLOUD_TREE_URL}/{img_name}"

        # 返回结果（种子信息提示）
        seed_tips = f"种子：{final_seed}（模式：{parsed['seed_mode']}）"

        # === 新增代码：如果当前是文生图模式，在种子信息后加上具体的模型名称 ===
        if parsed.get("type") == "text2img":
            base_id = parsed.get("base_model_id", 1)
            turbo_id = parsed.get("turbo_model_id", 1)
            
            # 根据ID从Config中获取对应的完整路径字典，如果没有找到则默认显示原来的数字
            base_path = Config.Z_IMAGE_BASE_MODELS.get(base_id, str(base_id))
            turbo_path = Config.Z_IMAGE_TURBO_MODELS.get(turbo_id, str(turbo_id))
            
            # 使用 split("/")[-1] 截取路径最后一段，仅保留模型文件名（例：z_image_bf16.safetensors）
            base_name = base_path.split("/")[-1] if isinstance(base_path, str) else base_path
            turbo_name = turbo_path.split("/")[-1] if isinstance(turbo_path, str) else turbo_path
            
            seed_tips += f"\n生成模型：base={base_name} | turbo={turbo_name}"

        return {
            "status": "success",
            "message": f"✅ 生成成功！\n{seed_tips}",
            "file_path": img_path,
            "preview_url": preview_url,  # 这里将直接输出带有 _xsrf 的图片直链
            "seed": final_seed,
            "seed_mode": parsed["seed_mode"]
        }
    except Exception as e:
        return {"status": "error", "message": f"❌ 生成失败：{str(e)}"}

# ====================== 5. FastAPI接口：适配OpenAI标准（关键修改） ======================
app = FastAPI(title="ComfyUI Agent", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 允许所有人访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#代理下载和携带 Cookie
@app.get("/proxy-image")
async def proxy_image(url: str):
    """
    优化：统一图片请求中转代理，使用后端携带 Cookie 访问，防止前端因未登录被禁止
    """
    target_url = unquote(url)
    
    # 将此 headers 里的字符串替换为你能在浏览器正常预览时抓包得到的实际有效 Cookie
    headers = {
        "Cookie": '*****', 
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": f"{JUPYTER_URL}"
    }
    
    try:
        # 携带有效 Cookie 通过后端下载原图二进制数据流（无转码）
        res = requests.get(target_url, headers=headers, timeout=15)
        res.raise_for_status() 
        content_type = res.headers.get("Content-Type", "image/png")
        
        # 提取原文件名（例如 20260305_220547_Z-Image_00020_.png）
        from urllib.parse import urlparse
        filename = os.path.basename(urlparse(target_url).path)
        if not filename:
            filename = "image.png"
            
        # 核心：使用 inline 属性能在聊天框正常显示原图，同时注入 filename，确保下载/拖拽到桌面时名字完全不变    
        return Response(
            content=res.content, 
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'}
        )
    except Exception as e:
        return Response(content=f"图片代理解析报错: {str(e)}", status_code=500)

# 保留健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Agent运行正常"}

# 保留模型列表接口（解决ChatBox获取模型失败）
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": Config.OLLAMA_MODEL,
                "object": "model",
                "created": 1710000000,
                "owned_by": "local",
                "root": Config.OLLAMA_MODEL,
                "parent": None
            }
        ]
    }

# OpenAI标准聊天接口（ChatBox原生适配，修复无响应问题）
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: dict):
    try:
        # 1. 提取用户最后一条消息（ChatBox标准格式）
        if not request.get("messages"):
            raise ValueError("未收到任何消息")
        user_message = request["messages"][-1]["content"]
        print(f"[Agent日志] 收到用户指令：{user_message}")  # 控制台打印，方便调试

        # 过滤ChatBox自动发送的无效指令
        user_message = request["messages"][-1]["content"]
        # 过滤关键词：命名、图片检测、自动生成标题等
        filter_keywords = ["give this conversation a name", "What color is in this image", "data:image/png;base64"]
        if any(keyword in user_message for keyword in filter_keywords):
            return JSONResponse(
                content={
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": Config.OLLAMA_MODEL,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                },
                headers={"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}
            )

        # 2. 调用Agent核心逻辑（图片暂时传None，先跑通文本）
        result = agent_handle(user_message, image_file=None)
        print(f"[Agent日志] 处理结果：{result}")

        # 3. 构建ChatBox能识别的OpenAI标准响应
        response_content = result["message"]
        # 修复ChatBox图片显示：替换Markdown图片为HTML/img标签 + 直接URL
        if result.get("preview_url"):
            # 移除原Markdown图片语法，替换为HTML标签（ChatBox兼容更好）
            response_content = re.sub(r"!\[AI生成结果\]\(.*?\)", "", response_content)
            # 添加HTML图片标签 + 可点击URL
            response_content += f"\n\n✅ 生成预览：\n<img src='{result['preview_url']}' width='500'>\n\n直接访问：{result['preview_url']}"

         # 构建响应时，明确指定编码和响应头
        return JSONResponse(
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": Config.OLLAMA_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(user_message),
                    "completion_tokens": len(response_content),
                    "total_tokens": len(user_message) + len(response_content)
                }
            },
            headers={
                "Content-Type": "application/json; charset=utf-8",  # 强制UTF-8
                "Access-Control-Allow-Origin": "*"  # 解决跨域（关键！）
            }
        )

    except Exception as e:
        # 异常响应也加响应头
        return JSONResponse(
            content={
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": Config.OLLAMA_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"❌ 处理失败：{str(e)}"
                        },
                        "finish_reason": "error"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            },
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Access-Control-Allow-Origin": "*"
            }
        )

# 保留原自定义/chat接口（兼容curl测试）
@app.post("/chat")
async def chat(
    command: str = Form(...),
    image_file: Optional[UploadFile] = File(None)
):
    result = agent_handle(command, image_file)
    return JSONResponse(content=result)

# 兼容根路径
@app.get("/v1")
async def api_root():
    return {"message": "ComfyUI Agent API is running"}

# ====================== 6. 启动入口 ======================
if __name__ == "__main__":
    # 启动FastAPI服务，暴露公网可访问
    uvicorn.run(
        "comfy_agent:app",
        host="0.0.0.0",  # 允许外部访问
        port=8000,       # 端口可自定义
        reload=True      # 开发模式，修改代码自动重启
    )