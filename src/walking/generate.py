from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types
from walking.models import Dataset, Session
from typing import List
from pathlib import Path
import uuid
import asyncio

async def generate_session(user_id: str) -> Session:
    client = genai.Client()
    aclient = client.aio

    prompt = """
    あなたは北海道函館市の美原・神山・富岡町・中道エリアをフォトウォークするエージェントである。
    以下の条件に従って、フォトウォークを行いなさい。
    - 10〜30枚程度の画像を撮影すること。
    - 徒歩30分程度の距離を歩くこと。
    - 出発地点と到着地点が同一であること。
    - 撮影する場所が地図の道路上であること。
    - 緯度・経度は、小数点以下6桁の精度で記録すること。
    - 撮影する場所は、北海道函館市の美原・神山・富岡町・中道エリア内であること。
    """

    response = await aclient.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=Session.model_json_schema(),
        )
    )
    session = Session.model_validate_json(response.text)
    session.user_id = user_id
    return session

def save(sessions: List[Session]):
    dataset = Dataset(sessions=sessions)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"{timestamp}.json"
    output_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    print(f"Dataset saved to {output_path}")

async def generate_dataset():
    user_ids = [str(uuid.uuid4()) for _ in range(10)]
    total_users = len(user_ids)
    sessions_per_user = 5
    total_sessions = total_users * sessions_per_user
    
    print(f"Starting generation: {total_users} users × {sessions_per_user} sessions = {total_sessions} sessions total")
    print("Generating all sessions in parallel...")
    
    # Create all tasks for parallel execution
    tasks = []
    for user_id in user_ids:
        for _ in range(sessions_per_user):
            tasks.append(generate_session(user_id))
    
    # Execute all tasks in parallel
    sessions = await asyncio.gather(*tasks)
    
    print(f"\nAll {total_sessions} sessions generated. Saving to file...")
    save(sessions)
    print("Complete!")

def main():
    load_dotenv()
    asyncio.run(generate_dataset())

if __name__ == "__main__":
    main()
