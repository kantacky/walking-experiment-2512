from pydantic import BaseModel, Field
from typing import List

class Record(BaseModel):
  id: str = Field(description="画像ID: UUID v4")
  timestamp: str = Field(description="撮影時刻: ISO 8601 形式")
  latitude: float = Field(description="緯度: 北海道函館市の美原・神山・富岡町・中道エリア内")
  longitude: float = Field(description="経度: 北海道函館市の美原・神山・富岡町・中道エリア内")
  category: str = Field(description="被写体のカテゴリ: traffic_light, crosswalk, vending_machine, shop_sign, bus_stop, manhole, bench")

class Session(BaseModel):
  id: str = Field(description="セッションID: UUID v4")
  user_id: str = Field(description="ユーザーID: UUID v4")
  records: List[Record] = Field(description="セッション内の撮影データ")

class Dataset(BaseModel):
  sessions: List[Session] = Field(description="セッションデータ")
