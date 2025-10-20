from typing import Generator, Union, List

from .types.tool import Tool
from .types.file import File
from .types.message import Conversation
from .exceptions import ChatError, ModelOverloadedError
import json


class ResponseTypes:
    FINAL = "finalAnswer"
    STREAM = "stream"
    STATUS = "status"
    METADATA = "routerMetadata"
    TITLE = "title"



class MessageStatus:
    PENDING = 0
    RESOLVED = 1
    REJECTED = 2


MSGTYPE_ERROR = "error"



class Message(Generator):
    """
    :Args:
        * g: Generator
        * _stream_yield_all: bool = False
        - text: str = ""
        - msg_status: int = MessageStatus.PENDING
        - error: Union[Exception, None] = None

    A wrapper of `Generator` that receives and process the response

    :Example:
    .. code-block:: python

        msg = bot.chat(...)

        # stream process
        for res in msg:
            ... # process
        else:
            if msg.done() == MessageStatus.REJECTED:
                raise msg.error

        # or simply use:
        final = msg.wait_until_done()
    """

    _stream_yield_all: bool = False
    _result_text: str = ""

    gen: Generator

    msg_status: int = MessageStatus.PENDING
    error: Union[Exception, None] = None

    def __init__(
        self,
        g: Generator,
        _stream_yield_all: bool = True,
        conversation: Conversation = None
    ) -> None:
        self.gen = g
        self._stream_yield_all = _stream_yield_all
        self.conversation = conversation

    @property
    def text(self) -> str:
        self._result_text = self.wait_until_done()
        return self._result_text

    @text.setter
    def text(self, v: str) -> None:
        self._result_text = v

    def _filterResponse(self, obj: dict):
        if not obj.__contains__("type"):
            if obj.__contains__("message"):
                raise ChatError(f"Server returns an error: {obj['message']}")
            else:
                raise ChatError(f"No `type` and `message` returned: {obj}")

    def __next__(self) -> dict:
        if self.msg_status == MessageStatus.RESOLVED:
            raise StopIteration

        elif self.msg_status == MessageStatus.REJECTED:
            if self.error is not None:
                raise self.error
            else:
                raise Exception("Message status is `Rejected` but no error found")

        try:
            data: dict = next(self.gen)
            self._filterResponse(data)
            data_type: str = data["type"]
            message_type: str = ""

            # set _result_text if this is the final iteration of the chat message
            if data_type == ResponseTypes.FINAL:
                # self._result_text = data["text"]
                self.msg_status = MessageStatus.RESOLVED
            # replace null characters with an empty string
            elif data_type == ResponseTypes.STREAM:
                data["token"] = data["token"].replace('\u0000', '')
                self._result_text += data["token"]

            elif "messageType" in data:
                message_type: str = data["messageType"]
                if message_type == MSGTYPE_ERROR:
                    self.error = ChatError(data["message"])
                    self.msg_status = MessageStatus.REJECTED
            else:
                if "Model is overloaded" in str(data):
                    self.error = ModelOverloadedError(
                        "Model is overloaded, please try again later or switch to another model."
                    )
                    self.msg_status = MessageStatus.REJECTED
                elif data.__contains__(MSGTYPE_ERROR):
                    self.error = ChatError(data[MSGTYPE_ERROR])
                    self.msg_status = MessageStatus.REJECTED
                else:
                    self.error = ChatError(f"Unknown json response: {data}")

            # If _stream_yield_all is True, yield all responses from the server.
            if self._stream_yield_all or data_type == ResponseTypes.STREAM:
                return data
            else:
                return self.__next__()
        except StopIteration:
            if self.msg_status == MessageStatus.PENDING:
                self.error = ChatError(
                    "Stream of responses has abruptly ended (final answer has not been received)."
                )
                raise self.error
            pass
        except Exception as e:
            self.error = e
            self.msg_status = MessageStatus.REJECTED
            raise self.error

    def __iter__(self):
        return self

    def throw(
        self,
        __typ,
        __val=None,
        __tb=None,
    ):
        return self.gen.throw(__typ, __val, __tb)

    def send(self, __value):
        return self.gen.send(__value)

    def get_final_text(self) -> str:
        """
        :Return:
            - self.text
        """
        return self.text

    def wait_until_done(self) -> str:
        """
        :Return:
            - self.text if resolved else raise error

        wait until every response is resolved
        """
        while not self.is_done():
            self.__next__()

        if self.is_done() == MessageStatus.RESOLVED:
            return self._result_text

        elif self.error is not None:
            raise self.error

        else:
            raise Exception("Rejected but no error captured!")

    def is_done(self):
        """
        :Return:
            - self.msg_status

        3 status:
        - MessageStatus.PENDING = 0    # running
        - MessageStatus.RESOLVED = 1   # done with no error(maybe?)
        - MessageStatus.REJECTED = 2   # error raised
        """
        return self.msg_status

    def __str__(self):
        return self.wait_until_done()

    def __getitem__(self, key: str) -> str:
        self.wait_until_done()
        if key == "text":
            return self.text

    def __add__(self, other: str) -> str:
        self.wait_until_done()
        return self.text + other

    def __radd__(self, other: str) -> str:
        self.wait_until_done()
        return other + self.text

    def __iadd__(self, other: str) -> str:
        self.wait_until_done()
        self.text += other
        return self.text
