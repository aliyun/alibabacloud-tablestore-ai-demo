# Tablestore Multi-modal Vector Retrieval

## Usage Guide

1. Clone the repository: `git clone <git address>`
2. Change directory to project root: `cd <directory containing this Readme and pyproject.toml>`
3. Install dependencies using `uv sync`.
   1. This project uses `uv` for dependency management; if not installed, use `pip3 install uv`.
4. Activate virtual environment: `source .venv/bin/activate`
5. Set environment variables

```bash

export DASHSCOPE_API_KEY=your DashScope API KEY

export tablestore_end_point=xxxxx_your_own
export tablestore_instance_name=xxxxx_your_own
export tablestore_access_key_id=xxxxx_your_own
export tablestore_access_key_secret=xxxxx_your_own
```

6. Launch JupyterLab Notebook

```shell
python -m jupyterlab --notebook-dir=./ --no-browser --allow-root
```

7. Start the frontend query interface:

```shell
python src/gradio_app.py
```