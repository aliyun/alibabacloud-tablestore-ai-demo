# -*- coding: utf-8 -*-

import json
import logging
import os

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
    def embedding(text) -> list[float]:
        return dashscope.MultiModalEmbedding.call(
            model="multimodal-embedding-v1",
            input=[MultiModalEmbeddingItemText(text=text, factor=1.0)]
        ).output["embeddings"][0]["embedding"]

    @staticmethod
    def query_text(
            text: str,
            top_k: int = 5,
            city: list[str] = None,
            height_range_tuple: tuple[float, float] = None,
            width_range_tuple: tuple[float, float] = None,
    ) -> list[tuple[Image.Image, str]]:
        logger.info(f"search text:{text}, top_k:{top_k}, city:{city}, height:{height_range_tuple}, width:{width_range_tuple}")

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
            float32_query_vector=Util.embedding(text),
            filter=vector_filter,
        )
        # 按照分数排序。
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
        ret = gallery_data
        logger.info(f"search text:{text}, top_k:{top_k}, request_id:{search_response.request_id}, ret:{len(ret)}")
        return ret

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


with gr.Blocks(title="Tablestore 多模态检索 Demo") as demo:
    with gr.Tab("Tablestore 多模态检索 Demo") as search_tab:
        with gr.Row():
            query_text_box = gr.Textbox(label='query_text', interactive=True, value="狗狗")
            top_k_box = gr.Slider(minimum=1, maximum=30, value=10, step=1, label='top_k', interactive=True)
        with gr.Row():
            city_box = gr.Dropdown(label='city', multiselect=True, choices=["hangzhou", "shanghai", "beijing", "shenzhen", "guangzhou"])
        with gr.Row():
            with gr.Column(scale=8):
                height_text = "Height from: **{min}** to: **{max}**"
                height_range_slider = gradio_rangeslider.RangeSlider(label="Height", minimum=0, maximum=1024, step=1, value=(0, 1024))
                height_range_display = gr.Markdown(value=height_text.format(min=0, max=1024))
                height_range_slider.change(lambda s: height_text.format(min=int(s[0]), max=int(s[1])), height_range_slider, height_range_display)
            with gr.Column(scale=8):
                width_text = "Width from: **{min}** to: **{max}**"
                width_range_slider = gradio_rangeslider.RangeSlider(label="Width", minimum=0, maximum=1024, step=1, value=(0, 1024))
                width_range_display = gr.Markdown(value=width_text.format(min=0, max=1024))
                width_range_slider.change(lambda s: width_text.format(min=int(s[0]), max=int(s[1])), width_range_slider, width_range_display)
        with gr.Row():
            query_button = gr.Button(value="query", variant='primary')
        with gr.Row():
            with gr.Column(scale=8):
                gallery_box = gr.Gallery(columns=5, show_label=False, preview=False, allow_preview=False, visible=True, show_download_button=False)
            with gr.Column(scale=2):
                with gr.Row(variant="panel"):
                    md_box = gr.Markdown(visible=True, elem_classes="image_detail")
            gallery_box.select(Util.on_gallery_box_select, [], [md_box])
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
