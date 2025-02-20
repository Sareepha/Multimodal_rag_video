from utils import (
    download_video, lvlm_inference, encode_image, 
    maintain_aspect_ratio_resize, str2time, getSubs,
    display_retrieved_results, load_json_file
)
import os
import cv2
from moviepy.editor import VideoFileClip
import whisper
import json
import os.path as osp
from pathlib import Path
from urllib.request import urlretrieve
import torch
from webvtt import read
import lancedb
from mm_rag.embeddings.bridgetower_embeddings import CLIPEmbeddings
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
import time



device = "cuda" if torch.cuda.is_available() else "cpu"
lvlm_prompt = "Briefly describe the image in 1-2 sentences, identifying the main object, person, or event that occurred. Use natural language"
LANCEDB_HOST_FILE = "./shared_data/lancedb"
TBL_NAME = "test_tbl"


def generate_transcript_with_whisper(path_to_video):
    """ ใช้ Whisper ถอดเสียงจากวิดีโอเป็นไฟล์ VTT """
    audio_file = osp.join(osp.dirname(path_to_video), 'audio.mp3')
    clip = VideoFileClip(path_to_video)
    clip.audio.write_audiofile(audio_file)

    model = whisper.load_model("large-v3")
    options = dict(task="translate", best_of=1, language='en')
    results = model.transcribe(audio_file, **options)

    vtt = getSubs(results["segments"], "vtt")
    path_to_transcript = osp.join(osp.dirname(path_to_video), 'generated_transcript.vtt')
    with open(path_to_transcript, 'w') as f:
        f.write(vtt)

    return path_to_transcript


def extract_and_save_frames_and_metadata(path_to_video, path_to_transcript, 
                                         path_to_save_extracted_frames, path_to_save_metadatas):
    """ ดึงเฟรมจากวิดีโอและสร้าง Metadata โดยเชื่อมโยงกับคำบรรยายจาก VTT """
    metadatas = []
    video = cv2.VideoCapture(path_to_video)
    trans = read(path_to_transcript)

    for idx, transcript in enumerate(trans):
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        text = transcript.text.replace("\n", ' ')

        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        success, frame = video.read()
        if success:
            image = maintain_aspect_ratio_resize(frame, height=350)
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(path_to_save_extracted_frames, img_fname).replace("\\", "/")
            cv2.imwrite(img_fpath, image)

            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

    metadata_file = osp.join(path_to_save_metadatas, 'metadatas.json').replace("\\", "/")
    with open(metadata_file, 'w') as outfile:
        json.dump(metadatas, outfile)

    return metadatas


def process_video_with_whisper(path_to_video, path_to_save):
    """ ใช้ Whisper เพื่อสร้างคำบรรยายและดึงเฟรมจากวิดีโอ พร้อมสร้าง metadata """
    extracted_frames_path = osp.join(path_to_save, 'extracted_frames')
    metadatas_path = path_to_save
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
    Path(metadatas_path).mkdir(parents=True, exist_ok=True)

    path_to_transcript = generate_transcript_with_whisper(path_to_video)
    return extract_and_save_frames_and_metadata(path_to_video, path_to_transcript, extracted_frames_path, metadatas_path)


def extract_and_save_frames_and_metadata_with_fps(path_to_video, path_to_save_extracted_frames, 
                                                  path_to_save_metadatas, num_of_extracted_frames_per_second=1):
    """ ดึงเฟรมจากวิดีโอและสร้าง Metadata โดยใช้คำบรรยายจาก LVLM """
    metadatas = []
    video = cv2.VideoCapture(path_to_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    hop = round(fps / num_of_extracted_frames_per_second)
    curr_frame = 0
    idx = -1

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if curr_frame % hop == 0:
            idx += 1
            image = maintain_aspect_ratio_resize(frame, height=350)
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(path_to_save_extracted_frames, img_fname).replace("\\", "/")
            cv2.imwrite(img_fpath, image)

            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)

            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)

        curr_frame += 1

    metadata_file = osp.join(path_to_save_metadatas, 'metadatas.json').replace("\\", "/")
    with open(metadata_file, 'w') as outfile:
        json.dump(metadatas, outfile)

    return metadatas


def ingest_data_to_lancedb(metadata_paths):
    """ บันทึกข้อมูลเข้าสู่ LanceDB """
    db = lancedb.connect(LANCEDB_HOST_FILE)
    embedder = CLIPEmbeddings()

    all_texts, all_image_paths, all_metadatas = [], [], []
    for metadata_path in metadata_paths:
        metadata = load_json_file(metadata_path)
        all_texts.extend([item['transcript'] for item in metadata])
        all_image_paths.extend([item['extracted_frame_path'] for item in metadata])
        all_metadatas.extend(metadata)

    MultimodalLanceDB.from_text_image_pairs(
        texts=all_texts,
        image_paths=all_image_paths,
        embedding=embedder,
        metadatas=all_metadatas,
        connection=db,
        table_name=TBL_NAME,
        mode="append",
    )


def process_from_gradio(video_url, operation_type):
    """
    ฟังก์ชันใหม่สำหรับเรียกจาก Gradio
    - video_url: ลิงก์วิดีโอจาก Gradio
    - operation_type: ประเภทการประมวลผล (Whisper หรือ LVLM)
    """
    batch_id = f"batch_{int(time.time())}_{os.urandom(4).hex()}"  # Unique batch ID
    vid_dir = f"./shared_data/videos/{'whisper_video' if operation_type == 'Processing video with Whisper...' else 'lvlm_video'}/{batch_id}"
    Path(vid_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        vid_filepath = download_video(video_url, vid_dir)
    except Exception as e:
        return f"Error downloading video: {e}"

    try:
        if operation_type == "Processing video with Whisper...":
            process_video_with_whisper(vid_filepath, vid_dir)
        elif operation_type == "Processing video with LVLM...":
            extracted_frames_path = osp.join(vid_dir, 'extracted_frames').replace("\\", "/")
            Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
            extract_and_save_frames_and_metadata_with_fps(
                vid_filepath, extracted_frames_path, vid_dir, num_of_extracted_frames_per_second=0.1
            )
        else:
            raise ValueError("Invalid operation type.")
    except Exception as e:
        return f"Error processing video: {e}"
        
    metadata_path = osp.join(vid_dir, 'metadatas.json')
    if not osp.exists(metadata_path):
        return f"Metadata file not created: {metadata_path}"
        
    expected_fields = {
        'batch_id': batch_id,
        'extracted_frame_path': None,
        'mid_time_ms': None,
        'transcript': None,
        'video_path': None,
        'video_segment_id': None
    }

    try:
        metadata = load_json_file(metadata_path)
        normalized_metadata = []
        for item in metadata:
            normalized_item = {field: item.get(field, default) for field, default in expected_fields.items()}
            normalized_metadata.append(normalized_item)

        with open(metadata_path, 'w') as outfile:
            json.dump(normalized_metadata, outfile)
    except Exception as e:
        return f"Error normalizing metadata: {e}"

    try:
        ingest_data_to_lancedb([metadata_path])
    except Exception as e:
        return f"Error ingesting data into LanceDB: {e}"
    # return f"Processing completed successfully! Data is stored under batch ID: {batch_id}"
    return f"Processing completed successfully!"








    
