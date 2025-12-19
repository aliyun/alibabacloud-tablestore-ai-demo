# -*- coding: utf-8 -*-

import base64
import json
import logging
import os
from pathlib import Path

import dashscope
import gradio as gr
import gradio_rangeslider
import tablestore
from PIL import Image
from dashscope import MultiModalEmbeddingItemText

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Util:
    import os
    import tablestore

    endpoint = os.getenv("tablestore_end_point")
    instance_name = os.getenv("tablestore_instance_name")
    access_key_id = os.getenv("tablestore_access_key_id")
    access_key_secret = os.getenv("tablestore_access_key_secret")

    # 创建 tablestore 的 sdk client
    client = tablestore.OTSClient(
        endpoint,
        access_key_id,
        access_key_secret,
        instance_name,
        retry_policy=tablestore.WriteRetryPolicy(),
    )

    table_name = "multi_modal_retrieval"
    index_name = "index"
    dimension = 1024

    @staticmethod
    def embedding_text(text) -> list[float]:
        """文本向量化"""
        return dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[MultiModalEmbeddingItemText(text=text, factor=1.0)]
        ).output["embeddings"][0]["embedding"]

    @staticmethod
    def embedding_image(image_path: str) -> list[float]:
        """
        本地图片向量化
        参考: https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
        """
        # 将本地图片转换为 base64
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # 获取图片格式
        suffix = Path(image_path).suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif suffix == ".png":
            mime_type = "image/png"
        elif suffix == ".gif":
            mime_type = "image/gif"
        elif suffix == ".webp":
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"

        # 构造 data URI
        data_uri = f"data:{mime_type};base64,{base64_image}"

        # 调用多模态向量化 API
        resp = dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[{"image": data_uri, "factor": 1.0}]
        )

        if resp.status_code == 200:
            return resp.output["embeddings"][0]["embedding"]
        else:
            raise Exception(f"图片向量化失败: {resp.code} - {resp.message}")

    @staticmethod
    def search_by_vector(
            query_vector: list[float],
            top_k: int = 10,
            city: list[str] = None,
            height_range_tuple: tuple[float, float] = None,
            width_range_tuple: tuple[float, float] = None,
    ) -> list[tuple[Image.Image, str]]:
        """使用向量进行搜索"""
        logger.info(f"search by vector, top_k:{top_k}, city:{city}, height:{height_range_tuple}, width:{width_range_tuple}")

        height_from = None if height_range_tuple is None else int(height_range_tuple[0])
        height_to = None if height_range_tuple is None else int(height_range_tuple[1])
        width_from = None if width_range_tuple is None else int(width_range_tuple[0])
        width_to = None if width_range_tuple is None else int(width_range_tuple[1])
        must_queries = []
        if city is not None and len(city) > 0:
            must_queries.append(tablestore.TermsQuery(field_name='city', column_values=city))
        if height_from is not None or height_to is not None:
            must_queries.append(tablestore.RangeQuery(
                field_name='height',
                range_from=height_from,
                range_to=height_to,
                include_lower=True,
                include_upper=True
            ))
        if width_from is not None or width_to is not None:
            must_queries.append(tablestore.RangeQuery(
                field_name='width',
                range_from=width_from,
                range_to=width_to,
                include_lower=True,
                include_upper=True
            ))
        vector_filter = None if len(must_queries) == 0 else tablestore.BoolQuery(must_queries=must_queries)
        query = tablestore.KnnVectorQuery(
            field_name='vector',
            top_k=top_k,
            float32_query_vector=query_vector,
            filter=vector_filter,
        )
        # 按照分数排序
        sort = tablestore.Sort(sorters=[tablestore.ScoreSort(sort_order=tablestore.SortOrder.DESC)])
        search_response: tablestore.SearchResponse = Util.client.search(
            table_name=Util.table_name,
            index_name=Util.index_name,
            search_query=tablestore.SearchQuery(query, limit=top_k, get_total_count=False, sort=sort),
            columns_to_get=tablestore.ColumnsToGet(
                column_names=["image_id", "city", "height", "width"],
                return_type=tablestore.ColumnReturnType.SPECIFIED)
        )
        search_hits: list[tablestore.metadata.SearchHit] = search_response.search_hits

        # 封装 gradio 界面的Response
        gallery_data = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for hit in search_hits:
            # 提取单行结果
            row_item = {}
            primary_key = hit.row[0]
            row_item["image_id"] = primary_key[0][1]
            attribute_columns = hit.row[1]
            for col in attribute_columns:
                key = col[0]
                val = col[1]
                row_item[key] = val
            # 构造 gradio 的 gallery_data
            file_path = os.path.join(current_dir, "../data/photograph/", row_item["image_id"])
            img = Image.open(file_path)
            gallery_data.append((img, json.dumps(row_item)))
        logger.info(f"search by vector, request_id:{search_response.request_id}, results:{len(gallery_data)}")
        return gallery_data

    @staticmethod
    def query_text(
            text: str,
            top_k: int = 5,
            city: list[str] = None,
            height_range_tuple: tuple[float, float] = None,
            width_range_tuple: tuple[float, float] = None,
    ) -> list[tuple[Image.Image, str]]:
        """文本搜索"""
        logger.info(f"search text:{text}, top_k:{top_k}, city:{city}, height:{height_range_tuple}, width:{width_range_tuple}")
        query_vector = Util.embedding_text(text)
        return Util.search_by_vector(query_vector, top_k, city, height_range_tuple, width_range_tuple)

    @staticmethod
    def query_image(
            image_path: str,
            top_k: int = 10,
            city: list[str] = None,
            height_range_tuple: tuple[float, float] = None,
            width_range_tuple: tuple[float, float] = None,
    ) -> list[tuple[Image.Image, str]]:
        """图片搜索 - 以图搜图"""
        if image_path is None:
            return []
        logger.info(f"search image:{image_path}, top_k:{top_k}, city:{city}, height:{height_range_tuple}, width:{width_range_tuple}")
        query_vector = Util.embedding_image(image_path)
        return Util.search_by_vector(query_vector, top_k, city, height_range_tuple, width_range_tuple)

    @staticmethod
    def on_gallery_box_select(evt: gr.SelectData):
        result = ""
        img_data = evt.value["caption"]
        img_data = json.loads(img_data)
        for key in img_data:
            img_data_item = img_data[key]
            if type(img_data_item) is str:
                img_data_item = img_data_item.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
            if type(img_data_item) is dict:
                for sub_key in img_data_item:
                    img_data_item[sub_key] = img_data_item[sub_key].replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")
                    result += f' - **{sub_key}**: &nbsp; {img_data_item[sub_key]}\r\n'
                continue
            result += f' - **{key}**: &nbsp; {img_data_item}\r\n'
        return result


custom_css = """
/* 图片详情面板样式 */
.image_detail {
    min-height: 200px;
    padding: 10px;
}

/* 以图搜图按钮样式 - 增加高度与同行组件对齐 */
.image-search-btn {
    height: 90px !important;
    min-height: 90px !important;
}
.image-search-btn button {
    height: 100% !important;
}
"""

with gr.Blocks(title="Tablestore 多模态检索 Demo", css=custom_css) as demo:
    # 状态变量：存储上传的图片路径
    uploaded_image_path = gr.State(value=None)

    with gr.Tab("Tablestore 多模态检索 Demo") as search_tab:
        # 搜索输入区域 - 第一行：以图搜图 + 自然语言搜索
        with gr.Row():
            # 图片上传按钮 - 使用 UploadButton
            image_upload_btn = gr.UploadButton(
                label="📷 以图搜图",
                file_types=["image"],
                file_count="single",
                scale=0,
                min_width=120,
                elem_classes="image-search-btn",
            )
            # 文本输入框
            query_text_box = gr.Textbox(
                label='自然语言搜索',
                interactive=True,
                value="远处是白雪覆盖的山峰",
                placeholder="请输入搜索内容...",
                scale=1
            )
        # 第二行：top_k + height_range_slider + width_range_slider
        with gr.Row():
            top_k_box = gr.Slider(minimum=1, maximum=30, value=10, step=1, label='top_k', interactive=True, scale=1)
            with gr.Column(scale=1):
                height_range_slider = gradio_rangeslider.RangeSlider(label="Height", minimum=0, maximum=1024, step=1, value=(0, 1024))
                height_range_slider.change(lambda s: height_text.format(min=int(s[0]), max=int(s[1])), height_range_slider)
            with gr.Column(scale=1):
                width_range_slider = gradio_rangeslider.RangeSlider(label="Width", minimum=0, maximum=1024, step=1, value=(0, 1024))
                width_range_slider.change(lambda s: width_text.format(min=int(s[0]), max=int(s[1])), width_range_slider)
        # 第三行：city
        with gr.Row():
            city_box = gr.Dropdown(label='city', multiselect=True, choices=["hangzhou", "shanghai", "beijing", "shenzhen", "guangzhou"], scale=3)
        with gr.Row():
            query_button = gr.Button(value="搜索", variant='primary')
            clear_image_button = gr.Button(value="清除图片", variant='secondary')

        # 搜索结果展示区域 - 全宽画廊
        gallery_box = gr.Gallery(
            columns=5,
            rows=2,
            show_label=False,
            preview=False,
            allow_preview=False,
            visible=True,
            show_download_button=False,
            object_fit="contain",
            height="auto"
        )

        # 图片详情（点击图片后显示）
        with gr.Accordion("图片详情", open=False):
            md_box = gr.Markdown(value="*点击上方图片查看详情*", elem_classes="image_detail")

        gallery_box.select(Util.on_gallery_box_select, [], [md_box])


        # 图片上传后自动搜索
        def on_image_upload(file, top_k, city, height_range, width_range):
            """图片上传后自动进行以图搜图"""
            if file is None:
                return [], None

            # UploadButton 返回的是文件路径字符串
            image_path = file if isinstance(file, str) else file.name

            # 执行图片搜索
            results = Util.query_image(image_path, top_k, city, height_range, width_range)
            return results, image_path


        image_upload_btn.upload(
            on_image_upload,
            inputs=[image_upload_btn, top_k_box, city_box, height_range_slider, width_range_slider],
            outputs=[gallery_box, uploaded_image_path],
        )


        # 清除图片按钮 - 重置状态
        def clear_image():
            return None


        clear_image_button.click(
            clear_image,
            inputs=[],
            outputs=[uploaded_image_path],
        )

        # 文本搜索按钮
        query_button.click(
            Util.query_text,
            inputs=[
                query_text_box,
                top_k_box,
                city_box,
                height_range_slider,
                width_range_slider,
            ],
            outputs=[
                gallery_box,
            ],
            concurrency_limit=1,
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
