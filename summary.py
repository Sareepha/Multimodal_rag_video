import requests
import json
import time
import lancedb
from mm_rag.MLM.client import PredictionGuardClient
from mm_rag.MLM.lvlm import LVLM
from utils import get_latest_batch_data
import asyncio
from concurrent.futures import ThreadPoolExecutor
LANCEDB_HOST_FILE = "./shared_data/lancedb"
TBL_NAME = "test_tbl"
API_BASE = "http://192.168.10.234:11434"
MODEL_NAME = "llama3.2:latest"

api_cache = {}
def call_api(prompt, max_tokens):
    """ เรียก API และแคชผลลัพธ์เพื่อลดการเรียกซ้ำ """
    cache_key = f"{prompt[:50]}-{max_tokens}"
   
    if cache_key in api_cache:
        print(f"🔹 ใช้ค่าจากแคช: {cache_key}")
        return api_cache[cache_key]

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{API_BASE}/v1/chat/completions", headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        json_response = response.json()
        if "choices" in json_response and json_response["choices"]:
            result = json_response["choices"][0]["message"]["content"]
            api_cache[cache_key] = result 
            return result
        else:
            return "Error: Invalid response from API."

    except requests.exceptions.RequestException as e:
        print(f"⚠️ API Request Error: {str(e)}")
        return f"Error: {str(e)}"


def batch_data(data, batch_size=10):
    """แบ่งข้อมูลเป็น batch ละ batch_size"""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def generate_caption(lvlm_module, img):
    """สร้างคำบรรยายสั้น ๆ (Image Caption) สำหรับภาพแต่ละเฟรม"""
    input_data = {"prompt": "Generate the shortest possible caption for this image.", "image": img}
    try:
        return lvlm_module.invoke(input_data).strip()
    except Exception as e:
        print(f" Error in LVLM.invoke() for {img}: {e}")
        return None

caption_cache = {}  # แคชสำหรับ caption

def process_batch(batch):
    """ แปลงภาพเป็น Caption ที่สั้นที่สุด แล้วรวมกับ Transcript """
    client = PredictionGuardClient()
    lvlm_module = LVLM(client=client)
    responses = []

    for img, transcript in batch:
        if img in caption_cache:
            caption = caption_cache[img]
        else:
            caption = generate_caption(lvlm_module, img)
            if caption:  
                caption_cache[img] = caption

        if not transcript.strip():
            transcript = "No transcript available"

        combined_text = f"{caption} - {transcript}" if caption else f"- {transcript}"
        responses.append({'image': img, 'captioned_transcript': combined_text})
        print(combined_text)

    return responses



def call_api_with_retry(prompt, max_tokens, retries=3, delay=5):
    """ เรียก API และลองใหม่ถ้าล้มเหลว """
    for attempt in range(retries):
        response = call_api(prompt, max_tokens)
        if "Error" not in response:  # ตรวจสอบว่า API ตอบกลับปกติ
            return response
        print(f" Retry {attempt + 1}/{retries} after {delay} seconds...")
        time.sleep(delay)  # หน่วงเวลาแล้วลองใหม่

    return "API failed after multiple attempts."

async def summarize_text(text):
    """ เรียก API แบบขนาน (Parallel) เพื่อลดเวลารอ """
    if not text.strip():
        return {"Error": "No valid text provided for summarization."}

    summarization_types = [
        {
            "type": "Narrative Summary",
            "prompt": f"""Tell the story of the text in a clear, engaging, and human-like manner. 
            Focus on making it easy to understand while capturing the main ideas:\n\n{text}""",
            "max_tokens": 2500
        },
        {
            "type": "5W1H Summary",
            "prompt": f"Summarize this text using 5W1H format:\n\n{text}",
            "max_tokens": 3000
        },
        {
            "type": "Analysis & Insights",
            "prompt": f"""Provide analysis and insights for the following text:
            **Risks/Challenges**: Identify potential risks and obstacles.
            **Feasibility**: Evaluate the likelihood of success or impact.
            **Recommendations**: Provide actionable suggestions or alternatives.
            
            {text}""",
            "max_tokens": 2500
        }
    ]


    results = {}

    async def fetch_summary(summary):
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor() as pool:
            try:
                response = await loop.run_in_executor(pool, call_api, summary["prompt"], summary["max_tokens"])
                results[summary["type"]] = response
            except Exception as e:
                results[summary["type"]] = f"Error: {str(e)}"

    await asyncio.gather(*(fetch_summary(summary) for summary in summarization_types))
    return results



async def async_main():
    """กระบวนการหลัก: ดึงข้อมูล, แบ่ง batch, สร้าง caption และสรุปผล"""
    print("🔹 Connecting to database...")
    latest_batch_id, _ = get_latest_batch_data()
    db = lancedb.connect(LANCEDB_HOST_FILE)
    table = db.open_table(TBL_NAME)
    records = table.to_pandas().to_dict('records')

    print(f"🔹 Fetching records for batch ID: {latest_batch_id}")
    video_data = [
        (record['metadata']['extracted_frame_path'], record['metadata']['transcript'])
        for record in records if record['metadata'].get('batch_id') == latest_batch_id
    ]

    if not video_data:
        raise ValueError(f" No records found for batch {latest_batch_id}")

    print(f"🔹 Processing {len(video_data)} records into batches...")
    batches = batch_data(video_data, batch_size=10)

    final_combined_text = " ".join(
        captioned_transcript["captioned_transcript"]
        for batch in batches for captioned_transcript in process_batch(batch)
    )

    print("🔹 Summarizing final combined text...")
    summary_results = await summarize_text(final_combined_text)

    print("\n🔹 Final AI Summary Result:")
    for summary_type, summary_text in summary_results.items():
        print(f"\n🔹 **{summary_type}**:\n{summary_text}")

    return summary_results



def main():
    """ ใช้ asyncio.run() เพื่อเรียกใช้ async_main() """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()



