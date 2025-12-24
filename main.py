from datetime import datetime, timezone
from dotenv import load_dotenv
from google import genai
from google.genai import types
from models import Dataset
from pathlib import Path

def main():
  client = genai.Client()

  prompt = """
  あなたは北海道函館市の美原・神山・富岡町・中道エリアをフォトウォークするデータセットを作成するエージェントです。
  ダミーデータセットを作成してください。
  データセットは、5人の撮影者が5回のセッションを行い、それぞれのセッションにおいて、10件の撮影データを生成してください。
  1回のセッションは、徒歩で30分間程度で、各セッションで出発地点と到着地点は同じになるようにしてください。
  """

  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_maps=types.GoogleMaps())],
        response_mime_type="application/json",
        response_json_schema=Dataset.model_json_schema(),
    ),
  )

  dataset = Dataset.model_validate_json(response.text)

  timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  output_path = Path("data") / f"{timestamp}.json"
  output_path.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
  print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
  load_dotenv()
  main()
