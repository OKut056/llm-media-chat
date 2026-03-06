# llm-media-chat
连接本地模型与云端示例（本地也可以）comfyui的agnet实现，可以在本地网页聊天生图等

只推荐有编程基础和对web有了解的人用。或者至少会问ai的。

在python中运行setup_env.py即可一键配置环境，下载此程序需要的依赖包。（requirements.txt列出了所需的依赖包）


start.bat 是windos一键启动服务的程序

start.sh 是Linux/Mac的一键启动服务的程序

index目前是测试用的，只有简单的聊天和生图指令传输，图片传输。

你需要将这个“DEFAULT_IP”变量改为你的本地ip

agent是用来连接本地ollama服务中的模型的，如果你要连接远程的（如我所用的autodl实例，经过我的实验，是可以的）模型，更改开头的”OLLAMA_API_URL“和”OLLAMA_MODEL“变量即可。

至于comfyui同理，想要运行本地的或自己云端的只要更改”COMFYUI_API_URL“链接地址即可

“WORKFLOW_PATHS”注释有说明，是要使用的工作流地址

目前只实现了我本地的两个工作流，对其他的适配并未进行（除非我自己使用到了。。。

“XSRF_TOKEN”是autodl实例中图片下载链接后缀的跨站请求伪造保护。


headers = {

        "Cookie": '*****',    这是使用云端实例才填写的，你需要找到你的cookie放入这里
        
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",    伪装成电脑，防止被ban
       
        "Referer": f"{COMFYUI_API_URL}"
    }
