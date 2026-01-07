from pydantic import BaseModel, Field
from typing import List

class Record(BaseModel):
  id: str = Field(description="画像ID: UUID v4")
  timestamp: str = Field(description="撮影時刻: ISO 8601 形式")
  latitude: float = Field(description="緯度: 小数点以下6桁の精度で, 北海道函館市の美原・神山・富岡町・中道エリア内")
  longitude: float = Field(description="経度: 小数点以下6桁の精度で, 北海道函館市の美原・神山・富岡町・中道エリア内")
  category: str = Field(description="被写体のカテゴリ: landscape, building, vehicle, person, animal, plant")

class Session(BaseModel):
  id: str = Field(description="セッションID: UUID v4")
  user_id: str = Field(description="ユーザーID: UUID v4")
  records: List[Record] = Field(description="セッション内の撮影データ")

class Dataset(BaseModel):
  sessions: List[Session] = Field(description="セッションデータ")
