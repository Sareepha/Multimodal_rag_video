import dataclasses
import gradio as gr
import lancedb
import os
import time
from enum import auto, Enum
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from mm_rag.embeddings.bridgetower_embeddings import CLIPEmbeddings
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
from moviepy.video.io.VideoFileClip import VideoFileClip
from pathlib import Path
from typing import List, Any
from utils import (
    prediction_guard_llava_conv, 
    encode_image, 
    lvlm_inference_with_conversation, 
    get_latest_batch_data) 
import anyio
from summary import async_main 
from preprocessing_vector_store_videos import process_from_gradio
import asyncio



server_error_msg="**An error occurred. Please change the question. or check your internet connection.**"

LANCEDB_HOST_FILE = "./shared_data/lancedb"

prompt_template = """
You are an AI assistant that answers user queries by analyzing:
1. **Transcript**: Spoken words extracted from the video.
2. **Image Analysis**: Visual context from the image, used only if relevant to the query.
3. **Conversation History**: Previous turns are provided for reference, but should only be used when they clarify or add essential context to the user's query.

---

### Information Provided:

**Transcript:**  
"{transcript}"  

**Image Path:**  
{image_path}  

**Conversation History (for reference only):**  
{turns}

---

### Instructions:
1. **Start with the Transcript**:  
   - Use the transcript to answer queries directly related to spoken content.

2. **Use the Image for Visual Details**:  
   - If the query involves visual details (e.g., "How many people are visible?", "What is happening in the image?"), analyze the image at `image_path` directly.
   - Avoid referencing conversation history to interpret the image unless explicitly relevant.

3. **Combine Sources for Context**:  
   - If the query requires both spoken and visual context, integrate the transcript and image analysis seamlessly.

4. **Minimize Use of Conversation History**:  
   - Use conversation history only when the current query depends on prior context (e.g., resolving ambiguous pronouns like "he," "she," or "they").
   - Avoid including irrelevant or redundant context from previous turns.

5. **Request Clarification When Required**:  
   - If the information is insufficient or ambiguous, ask the user for clarification instead of making assumptions.

---

### User Query:  
{user_query}

### Your Response:
Provide a precise and concise answer based on the prioritized analysis of the **Transcript**, **Image**, and **Conversation History** (in that order). Avoid overusing or including unnecessary references to the conversation history. """



def format_prompt(transcript, turns, user_query, state, max_turns):
    """
    สร้าง prompt โดยใช้ max_turns ล่าสุด
    """
    if not hasattr(state, "num_turns") or state.num_turns is None:
        state.num_turns = 1  
    else:
        if state.num_turns < max_turns:
            state.num_turns += 1
        else:
            state.num_turns = max_turns

    # เลือก `num_turns` ล่าสุด
    turns = turns[-state.num_turns:] if turns else []
    formatted_turns = "\n".join([f"{role}: {msg}" for role, msg in turns])

    return prompt_template.format(transcript=transcript, turns=formatted_turns, user_query=user_query)


def get_default_rag_chain(state, max_turns=3):
    latest_batch_id, _ = get_latest_batch_data()

    db = lancedb.connect(LANCEDB_HOST_FILE)
    table = db.open_table("test_tbl")
    records_df = table.to_pandas()

    filtered_df = records_df[records_df["metadata"].apply(lambda x: x["batch_id"] == latest_batch_id)]
    if filtered_df.empty:
        raise ValueError(f"No records found for batch {latest_batch_id}")

    embedder = CLIPEmbeddings()
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE,
        embedding=embedder,
        table_name="test_tbl"
    )

    retriever_module = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": 5}
    )

    client = PredictionGuardClient()
    lvlm_inference_module = LVLM(client=client)

    def prompt_processing(input, state, latest_batch_id, max_turns):
        retrieved_results, user_query = input['retrieved_results'], input['user_query']
    
        # กรองเฉพาะผลลัพธ์ที่ตรงกับ batch_id ล่าสุด
        filtered_results = [
            result for result in retrieved_results
            if result.metadata['metadata']['batch_id'] == latest_batch_id
        ]
        if not filtered_results:
            raise ValueError(f"No retrieved results for batch_id: {latest_batch_id}")
    
        # รวม transcript ทั้งหมด
        combined_transcripts = " ".join([
            result.metadata['metadata']['transcript']
            for result in filtered_results
        ])
    
        # ✅ ดึง path ของภาพจาก metadata
        first_frame_path = filtered_results[0].metadata['metadata'].get('extracted_frame_path', None)
        if not first_frame_path:
            raise ValueError("Image path (extracted_frame_path) is missing in metadata.")
    
        # ✅ ตัดคำถามปัจจุบันออกไป
        if hasattr(state, "messages") and state.messages:
            recent_turns = state.messages[:-2][-max_turns * 2:]  # ✅ ไม่รวมคำถามปัจจุบัน
        else:
            recent_turns = []
    
        # ฟอร์แมต turns
        formatted_turns = "\n".join([f"{role}: {msg}" for role, msg in recent_turns])
    
        # ✅ สร้าง prompt
        prompt = prompt_template.format(
            transcript=combined_transcripts,
            turns=formatted_turns,
            user_query=user_query,
            image_path=first_frame_path 
        )
    
        # แสดงเฉพาะ turns และ prompt ที่ส่งให้ LVLM
        print(f"Turns sent to LVLM:\n{formatted_turns}")
        print(f"Prompt sent to LVLM:\n{prompt}")
    
        return {
            'prompt': prompt,
            'image': first_frame_path,  # ✅ ใช้ path ของภาพ
            'metadata': filtered_results[0].metadata['metadata'],
            'batch_id': latest_batch_id,
            'turns': formatted_turns
        }


    prompt_processing_module = RunnableLambda(
        lambda input: prompt_processing(input, state, latest_batch_id, max_turns)
    )

    mm_rag_chain_with_retrieved_image = (
        RunnableParallel({
            "retrieved_results": retriever_module,
            "user_query": RunnablePassthrough()
        })
        | prompt_processing_module
        | RunnableParallel({
            'final_text_output': lvlm_inference_module,
            'input_to_lvlm': RunnablePassthrough(),
            'turns': RunnablePassthrough()
        })
    )
    return mm_rag_chain_with_retrieved_image

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()

@dataclasses.dataclass
class GradioInstance:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"
    sep2: str = None
    version: str = "Unknown"
    path_to_img: str = None
    caption: str = None
    mm_rag_chain: Any = None
    skip_next: bool = False
    num_turns: int = None  


    def _template_caption(self):
        out = ""
        if self.caption is not None:
            out = f"The caption associated with the image is '{self.caption}'. "
        return out

    def get_prompt_for_rag(self):
        messages = self.messages
        if len(messages) < 2:
            raise ValueError("Conversation must have at least 2 messages to proceed.")
        return messages[-2][1] 


    def get_conversation_for_lvlm(self):
        pg_conv = prediction_guard_llava_conv.copy()
        image_path = self.path_to_img
        b64_img = encode_image(image_path)
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if msg is None:
                break
            if i == 0:
                pg_conv.append_message(prediction_guard_llava_conv.roles[0], [msg, b64_img])
            elif i == len(self.messages[self.offset:]) - 2:
                pg_conv.append_message(role, [prompt_template.format(transcript=self.caption, user_query=msg)])
            else:
                pg_conv.append_message(role, [msg])
        return pg_conv
                
    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        if self.path_to_img is not None:
            path_to_image = self.path_to_img
            images.append(path_to_image)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return GradioInstance(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
            mm_rag_chain=self.mm_rag_chain,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "path_to_img": self.path_to_img,
            "caption" : self.caption,
        }

def get_gradio_instance(state, mm_rag_chain=None):
    if mm_rag_chain is None:
        mm_rag_chain = get_default_rag_chain(state)  # ส่ง state เข้าไป

    instance = GradioInstance(
        system="",
        roles=prediction_guard_llava_conv.roles,
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.SINGLE,
        sep="\n",
        path_to_img=None,
        caption=None,
        mm_rag_chain=mm_rag_chain,
    )
    return instance

import gradio as gr

# ตั้งค่าพาธของ Static Assets
gr.set_static_paths(paths=["./assets/"])

# กำหนดธีมของ Gradio
theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#e6f7ff", c100="#bae6fd", c200="#7cc8f8", c300="#38b6f0", c400="#009de0",
        c500="#007ac1", c600="#005b99", c700="#003d70", c800="#002848", c900="#001a33",
        c950="#001021"
    ),
    secondary_hue=gr.themes.Color(
        c50="#f9fafb", c100="#f3f4f6", c200="#e5e7eb", c300="#d1d5db", c400="#9ca3af",
        c500="#6b7280", c600="#4b5563", c700="#374151", c800="#1f2937", c900="#111827",
        c950="#0f172a"
    ),
).set(
    body_background_fill="#EBEBED",  # เปลี่ยนพื้นหลังเป็นสีเทาอ่อน
    body_background_fill_dark="#EBEBED",
    body_text_color="#003d70",  # ข้อความเป็นสีฟ้าเข้ม
    border_color_accent="#007ac1",  # ขอบสีฟ้าสด
    button_primary_background_fill="#009de0",  # ปุ่มสีฟ้าสด
    button_primary_border_color="#005b99"  # สีขอบปุ่ม
)

# กำหนด CSS สำหรับการปรับแต่ง UI
css = '''
    /* ปุ่มทั่วไป */
    .gr-button, #clear-history-btn, #summarize-btn {
        background-color: #009de0 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        padding: 12px 24px !important;
        text-transform: uppercase !important;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1) !important;
        transition: background-color 0.3s ease !important;
        width: 100% !important;
    }
    
    /* เพิ่มเอฟเฟกต์ Hover */
    #summarize-btn:hover {
        background-color: #007ac1 !important;
    }
'''

html_title = '''
<table style="width: 100%; border: 0; background-color: #009de0; padding: 10px; border-radius: 8px;">
    <tr style="border: 0;">
        <td style="border: 0; text-align: center;">
            <p style="font-size: xx-large; font-family: Verdana, Arial, Helvetica, sans-serif; color: white; margin: 0;">
                Multimodal RAG: Chat with Videos
            </p>
        </td>
    </tr>
</table>
'''

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)

def clear_history(state, request: gr.Request):
    state = get_gradio_instance(state.mm_rag_chain)
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 1

def add_text(state, text, request: gr.Request):
    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 1
    text = text[:1536] 
    state.append_message("user", text)  
    state.append_message("assistant", None)  
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None, disable_btn)


def http_bot(state, request: gr.Request):
    # print("Debug: Starting http_bot...")
    start_tstamp = time.time()

    if state.skip_next:
        # print("Debug: Skipping next turn")
        yield (state, state.to_gradio_chatbot(), None) + (no_change_btn,) * 1
        return

    try:
        latest_batch_id, related_records = get_latest_batch_data()
        # print(f"Debug: Latest batch_id - {latest_batch_id}")
        if not hasattr(state, 'latest_batch_id') or state.latest_batch_id != latest_batch_id:
            # print(f"Debug: Refreshing mm_rag_chain for batch_id - {latest_batch_id}")
            state.mm_rag_chain = get_default_rag_chain(state)  # ส่ง state เข้าไป
            state.latest_batch_id = latest_batch_id
    except Exception as e:
        # print(f"Error: Fetching latest batch_id failed - {e}")
        yield (state, state.to_gradio_chatbot(), None) + (enable_btn,)
        return

    try:
        prompt_or_conversation = state.get_prompt_for_rag()
        # print(f"Debug: Generated prompt/conversation - {prompt_or_conversation}")
    except ValueError as e:
        # print(f"Error: Generating prompt failed - {e}")
        yield (state, state.to_gradio_chatbot(), None) + (enable_btn,)
        return

    executor = state.mm_rag_chain
    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot(), None) + (disable_btn,) * 1

    try:
        response = executor.invoke(prompt_or_conversation)
        # print(f"Debug: Received response - {response}")
        message = response['final_text_output']
        metadata = response['input_to_lvlm']['metadata']

        if 'image' in response['input_to_lvlm']:
            state.path_to_img = response['input_to_lvlm']['image']
        if state.caption is None and 'transcript' in metadata:
            state.caption = metadata['transcript']
    except Exception as e:
        # print(f"Error: Execution failed - {e}")
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot(), None) + (enable_btn,)
        return

    state.messages[-1][-1] = message
    # print(f"Debug: Updated state messages - {state.messages}")
    yield (state, state.to_gradio_chatbot(), state.path_to_img) + (enable_btn,)


def get_summary_type_emoji(summary_type):
    """แมปประเภทของสรุปกับอีโมจิ"""
    emoji_map = {
        "Narrative Summary": "📖",
        "5W1H Summary": "📝",
        "Analysis & Insights": "🔍"
    }
    return emoji_map.get(summary_type, "❓")  # ใช้ ❓ หากไม่เจอประเภท


def summarize_mode():
    """เรียกใช้ฟังก์ชันสรุปผลและจัดข้อความให้อยู่ใน Markdown"""
    try:
        summary_results = asyncio.run(async_main())
        formatted_summary = format_summary(summary_results)  # จัดข้อความพร้อมอีโมจิ
        latest_batch_id, video_path = get_latest_batch_data()
        return formatted_summary, video_path
    except Exception as e:
        return f"⚠️ Error: {str(e)}", None



def format_summary(summary_results):
    """จัดรูปแบบ JSON ผลลัพธ์ในรูป Markdown พร้อมอีโมจิ"""
    formatted_output = ""
    for key, value in summary_results.items():
        emoji = get_summary_type_emoji(key)  # ดึงอีโมจิที่ตรงกับประเภท
        formatted_output += f"{emoji} {key} {emoji}\n\n{value}\n\n"
    return formatted_output



def switch_mode(mode):
    if mode == "Summarize":
        return gr.update(visible=True), gr.update(visible=False), gr.update(type="filepath")
    else:  
        return gr.update(visible=False), gr.update(visible=True), gr.update(type="image")

def process(mode, query, state):
    if mode == "Summarize":
        summary, video_path = summarize_mode()
        return state, summary, video_path  
    else:
        response = f"Chat response for query: {query}"
        return state, response, None 

def get_demo(rag_chain=None):
    state = gr.State()  # สร้าง state ก่อน
    if rag_chain is None:
        rag_chain = get_default_rag_chain(state)  # ส่ง state เข้าไป
    with gr.Blocks(theme=theme, css=css) as demo: #theme=theme, (theme=theme, css=css)
        instance = get_gradio_instance(state, rag_chain)
        state = gr.State(instance)

        demo.load(
            None,
            None,
            js="""
            () => {
            const params = new URLSearchParams(window.location.search);
            if (!params.has('__theme')) {
                params.set('__theme', 'light');  // เปลี่ยนจาก dark เป็น light
                window.location.search = params.toString();
            }
            }
            """,
        )


        gr.HTML(value=html_title)

        with gr.Tabs(): 
            with gr.Tab("Upload video"):
                gr.Markdown("## Welcome to Multimodal RAG System!")
                with gr.Tabs():
            
                    with gr.Tab("Upload Video"):
                        with gr.Group():
                            with gr.Column():
                                upload_btn = gr.File(
                                    label="Upload Video",
                                    file_types=[".mp4"],
                                    interactive=True,
                                )
                                
                                operation_type_upload = gr.Radio(
                                    choices=["Processing video with Whisper...", "Processing video with LVLM..."],
                                    value="Processing video with Whisper...",
                                    label="Select Operation Type for Upload",
                                )
                
                    with gr.Tab("Enter Link"):
                        with gr.Group():
                            with gr.Column():
                                link_input = gr.Textbox(
                                    label="Video Link",
                                    placeholder="https://example.com/your-video-link",
                                    interactive=True,
                                    elem_id="link-input",
                                )
                
                               
                                operation_type_link = gr.Radio(
                                    choices=["Processing video with Whisper...", "Processing video with LVLM..."],
                                    value="Processing video with Whisper...",
                                    label="Select Operation Type for Link",
                                )
                
             
                summarize_btn = gr.Button(value="Upload Now", elem_id="summarize-btn")
            
                # กล่องแสดงผลลัพธ์
                output_box = gr.Textbox(
                    label="Output",
                    placeholder="The summary will appear here...",
                    interactive=False,
                    elem_id="output-box",
                )
            
                # ฟังก์ชัน Summarize
                def summarize(upload_file, operation_upload, link, operation_link):
                    if upload_file is not None:
                        try:
                            video_path = upload_file.name
                            result = process_from_gradio(video_path, operation_upload)
                            return result
                        except Exception as e:
                            return f"Error: {str(e)}"
                    elif link:
                        try:
                            result = process_from_gradio(link, operation_link)
                            return result
                        except Exception as e:
                            return f"Error: {str(e)}"
                    return "Please provide a valid input!"
            
                summarize_btn.click(
                    summarize,
                    inputs=[upload_btn, operation_type_upload, link_input, operation_type_link],
                    outputs=[output_box],
                )
                



            # Summarize Tab
            with gr.Tab("Summarize"):
                gr.Markdown("## Summarize Mode")
                with gr.Row():
                    with gr.Column(scale=4):
                        relevant_media = gr.Video(label="Relevant Media/Video", visible=True)
                    with gr.Column(scale=15):
                        summarize_output = gr.Textbox(
                            label="Summarized Output",
                            visible=True,
                            lines=15,  # จำนวนบรรทัดที่แสดงในกล่อง
                            max_lines=30,  # จำนวนบรรทัดสูงสุด
                            interactive=False,  # ทำให้กล่องอ่านได้อย่างเดียว
                            elem_id="summary-output-box"
                        )



            with gr.Tab("Chat"):
                gr.Markdown("## Chat Mode")
                with gr.Row():
                    with gr.Column(scale=4):
                        relevant_image = gr.Image(
                            type="filepath",
                            label="Relevant Image",
                            elem_id="image-display",
                            interactive=False,
                            visible=True,
                        )

                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(
                            value=instance.to_gradio_chatbot(),
                            elem_id="chatbot",
                            label="Multimodal RAG Chatbot",
                            height=512,
                            visible=True,
                        )
                        # ส่วนป้อนคำถามในโหมด Chat
                        with gr.Row(elem_id="query-row") as query_row:
                            textbox = gr.Textbox(
                                label="Query",
                                placeholder="Enter your question here...",
                                lines=1
                            )
                            submit_btn = gr.Button("Send", variant="primary")
                        
                        # ✅ แยกปุ่ม Clear History ไว้ด้านล่าง
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear History", elem_id="clear-history-btn", variant="secondary")
                        
                    
                        textbox.submit(
                            add_text,
                            [state, textbox],  
                            [state, chatbot, textbox, relevant_image, clear_btn],  
                        ).then(
                            http_bot,
                            [state],
                            [state, chatbot, relevant_image, clear_btn],
                        )
                        
                        submit_btn.click(
                            add_text,
                            [state, textbox],
                            [state, chatbot, textbox, relevant_image, clear_btn],
                        ).then(
                            http_bot,
                            [state],
                            [state, chatbot, relevant_image, clear_btn],
                        )
                          

        # ฟังก์ชัน Clear History
        clear_btn.click(
            clear_history, [state], [state, chatbot, textbox, relevant_image, clear_btn]
        )

        demo.load(
            summarize_mode, 
            inputs=[],
            outputs=[summarize_output, relevant_media], 
        )



    return demo


