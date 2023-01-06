from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TextData(_message.Message):
    __slots__ = ["image", "text"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    text: str
    def __init__(self, text: _Optional[str] = ..., image: _Optional[bytes] = ...) -> None: ...

class TextRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
