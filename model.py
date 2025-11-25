from pydantic import BaseModel

class RootRequest(BaseModel):
    word: str
