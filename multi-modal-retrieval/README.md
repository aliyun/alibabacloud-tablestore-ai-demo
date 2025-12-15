# Tablestore 多模态向量检索

## 使用帮助

1. 下载代码 `git clone <git 地址>`
2. `cd <本项目的目录，该目录包含该Readme，且包含 pyproject.toml>`
3. 安装相关依赖 `uv sync`。
    1. 本项目使用 `uv` 管理依赖，如果没有 `uv` 可以自行安装 `pip3 install uv`
4. 使用虚拟环境 `source .venv/bin/activate` 使用虚拟环境
5. 设置环境变量

```bash

export DASHSCOPE_API_KEY=自己的百炼API-KEY

export tablestore_end_point=xxxxx自己的
export tablestore_instance_name=xxxxx自己的
export tablestore_access_key_id=xxxxx自己的
export tablestore_access_key_secret=xxxxx自己的
```

6. 启动 notebook

```shell
python -m jupyterlab --notebook-dir=./ --no-browser --allow-root
```

7. 启动查询的前端界面:

```shell
python src/gradio_app.py
```