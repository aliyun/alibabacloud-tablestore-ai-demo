# -*- coding: utf-8 -*-
"""
Multimodal Image Retrieval System - Gradio Web Application
Based on Tablestore vector search and Dashscope multimodal embedding
"""

import base64
import json
import logging
import os
from pathlib import Path

import dashscope
import gradio as gr
import gradio_rangeslider
import tablestore
from dashscope import MultiModalEmbeddingItemText
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MIME types for image formats
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Tablestore configuration
TABLE_NAME = "multi_modal_retrieval"
INDEX_NAME = "index"

# Global client instance (lazy initialization)
_client = None


def get_client():
    """Get Tablestore client (singleton pattern)."""
    global _client
    if _client is None:
        _client = tablestore.OTSClient(
            os.getenv("tablestore_end_point"),
            os.getenv("tablestore_access_key_id"),
            os.getenv("tablestore_access_key_secret"),
            os.getenv("tablestore_instance_name"),
            retry_policy=tablestore.WriteRetryPolicy(),
        )
    return _client


def text_to_embedding(text: str) -> list[float]:
    """Convert text to embedding vector."""
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[MultiModalEmbeddingItemText(text=text, factor=1.0)]
    )
    if resp.status_code == 200:
        return resp.output["embeddings"][0]["embedding"]
    raise Exception(f"Text embedding failed: {resp.code} - {resp.message}")


def image_to_embedding(image_path: str) -> list[float]:
    """Convert local image to embedding vector."""
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")

    suffix = Path(image_path).suffix.lower()
    mime_type = MIME_TYPES.get(suffix, "image/jpeg")
    data_uri = f"data:{mime_type};base64,{base64_image}"

    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=[{"image": data_uri, "factor": 1.0}]
    )

    if resp.status_code == 200:
        return resp.output["embeddings"][0]["embedding"]
    raise Exception(f"Image embedding failed: {resp.code} - {resp.message}")


def build_range_query(field_name: str, range_tuple: tuple[float, float] | None) -> tablestore.RangeQuery | None:
    """Build range query for filtering."""
    if range_tuple is None:
        return None
    return tablestore.RangeQuery(
        field_name=field_name,
        range_from=int(range_tuple[0]),
        range_to=int(range_tuple[1]),
        include_lower=True,
        include_upper=True
    )


def search_by_vector(
        query_vector: list[float],
        top_k: int = 10,
        city: list[str] = None,
        height_range: tuple[float, float] = None,
        width_range: tuple[float, float] = None,
) -> list[tuple[Image.Image, str]]:
    """Search images by vector similarity."""
    logger.info(f"Vector search: top_k={top_k}, city={city}, height={height_range}, width={width_range}")

    # Build filter conditions
    must_queries = []
    if city:
        must_queries.append(tablestore.TermsQuery(field_name='city', column_values=city))
    if height_query := build_range_query('height', height_range):
        must_queries.append(height_query)
    if width_query := build_range_query('width', width_range):
        must_queries.append(width_query)

    vector_filter = tablestore.BoolQuery(must_queries=must_queries) if must_queries else None

    # Build vector query
    query = tablestore.KnnVectorQuery(
        field_name='vector',
        top_k=top_k,
        float32_query_vector=query_vector,
        filter=vector_filter,
    )

    # Sort by score descending
    sort = tablestore.Sort(sorters=[tablestore.ScoreSort(sort_order=tablestore.SortOrder.DESC)])
    search_response = get_client().search(
        table_name=TABLE_NAME,
        index_name=INDEX_NAME,
        search_query=tablestore.SearchQuery(query, limit=top_k, get_total_count=False, sort=sort),
        columns_to_get=tablestore.ColumnsToGet(
            column_names=["image_id", "city", "height", "width"],
            return_type=tablestore.ColumnReturnType.SPECIFIED
        )
    )

    # Build gallery data
    gallery_data = []
    current_dir = Path(__file__).parent
    for hit in search_response.search_hits:
        row_item = {"image_id": hit.row[0][0][1]}
        for col in hit.row[1]:
            row_item[col[0]] = col[1]

        file_path = current_dir / ".." / "data" / "photograph" / row_item["image_id"]
        gallery_data.append((Image.open(file_path), json.dumps(row_item)))

    logger.info(f"Search completed: request_id={search_response.request_id}, results={len(gallery_data)}")
    return gallery_data


def search_by_text(
        text: str,
        top_k: int = 10,
        city: list[str] = None,
        height_range: tuple[float, float] = None,
        width_range: tuple[float, float] = None,
) -> list[tuple[Image.Image, str]]:
    """Search images by text description."""
    logger.info(f"Text search: query={text}, top_k={top_k}, city={city}")
    query_vector = text_to_embedding(text)
    return search_by_vector(query_vector, top_k, city, height_range, width_range)


def search_by_image(
        image_path: str,
        top_k: int = 10,
        city: list[str] = None,
        height_range: tuple[float, float] = None,
        width_range: tuple[float, float] = None,
) -> list[tuple[Image.Image, str]]:
    """Search similar images by uploading an image."""
    if image_path is None:
        return []
    logger.info(f"Image search: path={image_path}, top_k={top_k}, city={city}")
    query_vector = image_to_embedding(image_path)
    return search_by_vector(query_vector, top_k, city, height_range, width_range)


def escape_special_chars(text: str) -> str:
    """Escape special characters for display."""
    return text.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def on_gallery_select(evt: gr.SelectData) -> str:
    """Handle gallery image selection and display metadata."""
    img_data = json.loads(evt.value["caption"])
    lines = []

    for key, value in img_data.items():
        if isinstance(value, str):
            lines.append(f" - **{key}**: &nbsp; {escape_special_chars(value)}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                lines.append(f" - **{sub_key}**: &nbsp; {escape_special_chars(sub_value)}")
        else:
            lines.append(f" - **{key}**: &nbsp; {value}")

    return "\r\n".join(lines)


# Custom CSS styles
CUSTOM_CSS = """
.image-detail {
    min-height: 200px;
    padding: 10px;
}
.image-search-btn {
    height: 90px !important;
    min-height: 90px !important;
}
.image-search-btn button {
    height: 100% !important;
}
"""


def on_image_upload(file, top_k, city, height_range, width_range):
    """Handle image upload and trigger image search."""
    if file is None:
        return []
    image_path = file if isinstance(file, str) else file.name
    return search_by_image(image_path, top_k, city, height_range, width_range)


def create_demo() -> gr.Blocks:
    """Create Gradio web interface."""
    with gr.Blocks(title="Tablestore Multimodal Search", css=CUSTOM_CSS) as demo:
        with gr.Tab("Tablestore Multimodal Search"):
            # Row 1: Image upload + Text search
            with gr.Row():
                image_upload_btn = gr.UploadButton(
                    label="📷 Image Search",
                    file_types=["image"],
                    file_count="single",
                    scale=0,
                    min_width=150,
                    elem_classes="image-search-btn",
                )
                query_text_box = gr.Textbox(
                    label="Text Search",
                    interactive=True,
                    value="snow-capped mountains in the distance",
                    placeholder="Enter search query...",
                    scale=1
                )

            # Row 2: Top K + Height/Width range sliders
            with gr.Row():
                top_k_box = gr.Slider(
                    minimum=1, maximum=30, value=10, step=1,
                    label="Top K", interactive=True, scale=1
                )
                with gr.Column(scale=1):
                    height_range_slider = gradio_rangeslider.RangeSlider(
                        label="Height", minimum=0, maximum=1024, step=1, value=(0, 1024)
                    )
                with gr.Column(scale=1):
                    width_range_slider = gradio_rangeslider.RangeSlider(
                        label="Width", minimum=0, maximum=1024, step=1, value=(0, 1024)
                    )

            # Row 3: City filter
            with gr.Row():
                city_box = gr.Dropdown(
                    label="City",
                    multiselect=True,
                    choices=["hangzhou", "shanghai", "beijing", "shenzhen", "guangzhou"],
                    scale=3
                )

            # Row 4: Action buttons
            with gr.Row():
                query_button = gr.Button(value="Search", variant="primary")
                clear_button = gr.Button(value="Clear", variant="secondary")

            # Search results gallery
            gallery_box = gr.Gallery(
                columns=5,
                rows=2,
                show_label=False,
                allow_preview=True,
                visible=True,
                show_download_button=False,
                object_fit="contain",
                height="auto"
            )

            # Image details accordion
            with gr.Accordion("Image Details", open=False):
                md_box = gr.Markdown(
                    value="*Click an image above to view details*",
                    elem_classes="image-detail"
                )

            # Event bindings
            gallery_box.select(on_gallery_select, [], [md_box])

            image_upload_btn.upload(
                on_image_upload,
                inputs=[image_upload_btn, top_k_box, city_box, height_range_slider, width_range_slider],
                outputs=[gallery_box],
            )

            query_button.click(
                search_by_text,
                inputs=[query_text_box, top_k_box, city_box, height_range_slider, width_range_slider],
                outputs=[gallery_box],
                concurrency_limit=1,
            )

            clear_button.click(
                lambda: [],
                inputs=[],
                outputs=[gallery_box],
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
