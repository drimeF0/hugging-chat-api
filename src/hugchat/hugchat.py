import requests
import json
import os
import datetime
import logging
import time
import typing
import traceback

from typing import Union, List
from requests import Session
from requests.sessions import RequestsCookieJar

from .message import Message
from . import exceptions
from .types.model import Model
from .types.message import MessageNode, Conversation


conversation = Conversation
model = Model


class ChatBot:
    cookies: dict
    """Cookies for authentication"""

    session: Session
    """HuggingChat session"""

    def __init__(
        self,
        cookies: Union[dict, RequestsCookieJar] = None,
        default_llm: Union[int, str] = 0,
        system_prompt: str = "",
    ) -> None:
        """
        Returns a ChatBot object
        default_llm: name or index
        """
        if cookies is None:
            raise exceptions.ChatBotInitError(
                "Authentication is required now, but no cookies provided. See tutorial at https://github.com/Soulter/hugging-chat-api"
            )
        self.cookies = cookies

        self.hf_base_url = "https://huggingface.co"
        self.json_header = {"Content-Type": "application/json"}
        self.session = self.get_hc_session()
        self.conversation_list = []
        self.sharing = True
        self.accepted_welcome_modal = (
            False  # It is no longer required to accept the welcome modal
        )

        self.llms = self.get_remote_llms()

        if isinstance(default_llm, str):
            self.active_model = self.get_llm_from_name(default_llm)
            if self.active_model is None:
                raise Exception(
                    f"Given model is not in llms list. LLM list: {[model.id for model in self.llms]}"
                )
        else:
            self.active_model = self.llms[default_llm]

        self.current_conversation = self.new_conversation(
            system_prompt=system_prompt)

    def get_hc_session(self) -> Session:
        session = Session()
        # set cookies
        session.cookies.update(self.cookies)
        session.get(self.hf_base_url + "/chat")
        return session

    def get_headers(self, ref=True, ref_cid: Conversation = None) -> dict:
        _h = {
            "Accept": "*/*",
            "Connection": "keep-alive",
            "Host": "huggingface.co",
            "Origin": "https://huggingface.co",
            "Sec-Fetch-Site": "same-origin",
            "Content-Type": "application/json",
            "Sec-Ch-Ua-Platform": "Windows",
            "Sec-Ch-Ua": 'Chromium";v="116", "Not)A;Brand";v="24", "Microsoft Edge";v="116',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Dest": "empty",
            "Accept-Encoding": "gzip, deflate, br",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        }

        if ref:
            if ref_cid is None:
                ref_cid = self.current_conversation
            _h["Referer"] = f"https://huggingface.co/chat/conversation/{ref_cid}"
        return _h

    def get_cookies(self) -> dict:
        return self.session.cookies.get_dict()

    # NOTE: To create a copy when calling this, call it inside of list().
    #       If not, when updating or altering the values in the variable will
    #       also be applied to this class's variable.
    #       This behavior is with any function returning self.<var_name>. It
    #       acts as a pointer to the data in the object.
    #
    # Returns a pointer to this objects list that contains id of conversations.
    def get_conversation_list(self) -> list:
        return list(self.conversation_list)

    def get_active_llm_index(self) -> int:
        return self.llms.index(self.active_model)


    def new_conversation(
        self,
        modelIndex: int = None,
        system_prompt: str = "",
        switch_to: bool = True,
    ) -> Conversation:
        """
        Create a new conversation. Return a conversation object. 

        modelIndex: int, get it from get_available_llm_models(). If None, use the default model.

        - You should change the conversation by calling change_conversation() after calling this method. Or set param switch_to to True.
        - if you use assistant, the parameter `system_prompt` will be ignored.

        """

        if modelIndex is None:
            model = self.active_model
        else:
            if modelIndex < 0 or modelIndex >= len(self.llms):
                raise IndexError("Out of range of llm index")

            model = self.llms[modelIndex]
        # Create new conversation and get a conversation id.

        _header = self.get_headers(ref=False)
        _header["Referer"] = "https://huggingface.co/chat/"

        request = {
            "model": model.id,
            "preprompt": system_prompt if system_prompt != "" else model.preprompt
        }


        resp = self.session.post(
            self.hf_base_url + "/chat/conversation",
            json=request,
            headers=_header,
            cookies=self.get_cookies(),
        )

        logging.debug(resp.text)
        cid = json.loads(resp.text)["conversationId"]

        c = Conversation(id=cid, system_prompt=system_prompt, model=model)

        self.conversation_list.append(c)

        if switch_to:
            self.change_conversation(c)

        # we need know the root message id (a.k.a system prompt message id).
        self.get_conversation_info(c)

        return c


    def change_conversation(self, conversation_object: Conversation):
        """
        Change the current conversation to another one.
        """

        local_conversation = self.get_conversation_from_id(
            conversation_object.id)

        if local_conversation is None:
            raise exceptions.InvalidConversationIDError(
                "Invalid conversation id, not in conversation list."
            )

        self.get_conversation_info(local_conversation)

        self.current_conversation = local_conversation
        
        return self.current_conversation

    def delete_all_conversations(self) -> None:
        """
        Deletes ALL conversations on the HuggingFace account
        """

        settings = {"": ("", "")}

        r = self.session.delete(
            f"{self.hf_base_url}/chat/api/v2/conversations/",
            headers={"Referer": "https://huggingface.co/chat/settings/application", "Origin": "https://huggingface.co", "Content-Type": "application/json"},
            cookies=self.get_cookies(),
            allow_redirects=True,
            files=settings,
        )

        if r.status_code != 200:
            raise exceptions.DeleteConversationError(
                f"Failed to delete ALL conversations with status code: {r.status_code}"
            )

        self.conversation_list = []
        self.current_conversation = None

    def delete_conversation(self, conversation_object: Conversation = None) -> None:
        """
        Delete a HuggingChat conversation by conversation.
        """

        if conversation_object is None:
            conversation_object = self.current_conversation

        headers = self.get_headers()

        r = self.session.delete(
            f"{self.hf_base_url}/chat/api/v2/conversations/{conversation_object}",
            headers=headers,
            cookies=self.get_cookies(),
        )

        if r.status_code != 200:
            raise exceptions.DeleteConversationError(
                f"Failed to delete conversation with status code: {r.status_code}"
            )
        else:
            self.conversation_list.pop(
                self.get_conversation_from_id(
                    conversation_object.id, return_index=True)
            )

            if conversation_object is self.current_conversation:
                self.current_conversation = None

    def get_available_llm_models(self) -> list:
        """
        Get all available models that are available in huggingface.co/chat.
        """
        return self.llms

    def switch_llm(self, index: int) -> bool:
        """
        Attempts to change current conversation's Large Language Model.
        Requires an index to indicate the model you want to switch.
        See self.llms for available models.

        Note: 1. The effect of switch is limited to the current conversation,
        You can manually switch the llm when you start a new conversation.

        2. Only works *after creating a new conversation.*
        :)
        """
        # TODO: I will work on making it have a model for each conversation that is changeable. - @Zekaroni

        if index < len(self.llms) and index >= 0:
            self.active_model = self.llms[index]
            return True
        else:
            raise IndexError("Out of range of llm index")

    def get_llm_from_name(self, name: str) -> Union[Model, None]:
        for model in self.llms:
            if model.name == name:
                return model

    # Gives information such as name, websiteUrl, description, displayName, parameters, etc.
    # We can use it in the future if we need to get information about models
    def get_remote_llms(self) -> list:
        """
        Fetches all possible LLMs that could be used. Returns the LLMs in a list
        """

        r = self.session.get(
            self.hf_base_url + "/chat/api/v2/models",
            headers=self.get_headers(ref=False),
            cookies=self.get_cookies(),
        )

        if r.status_code != 200:
            raise Exception(
                f"Failed to get remote LLMs with status code: {r.status_code}"
            )
        

        try:
            models_data_list = json.loads(r.text)['json']
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            models_data_list = []

        model_list = []

        for model_data_dict in models_data_list:
            if model_data_dict.get("unlisted", False): # Default to False if 'unlisted' key is missing
                print(f"Skipping unlisted model: {model_data_dict.get('id', 'Unknown ID')}")
                continue
            m = Model(
                id=model_data_dict.get("id"),
                name=model_data_dict.get("name"),
                displayName=model_data_dict.get("displayName"),
                preprompt=model_data_dict.get("preprompt", ""),
                websiteUrl=model_data_dict.get("websiteUrl"),
                description=model_data_dict.get("description"),
                modelUrl=model_data_dict.get("modelUrl"),
                unlisted=model_data_dict.get("unlisted", False), # Also store this if needed
                logoUrl=model_data_dict.get("logoUrl"),
                reasoning=model_data_dict.get("reasoning"),
                multimodal=model_data_dict.get("multimodal"),
                hasInferenceAPI=model_data_dict.get("hasInferenceAPI")
            )

            prompt_examples_list = model_data_dict.get("promptExamples")
            if prompt_examples_list:
                m.promptExamples = prompt_examples_list
            else:
                m.promptExamples = []

            parameters_dict = model_data_dict.get("parameters")
            if parameters_dict:
                m.parameters = parameters_dict
            else:
                m.parameters = {}


            model_list.append(m)
        return model_list

    def get_remote_conversations(self, replace_conversation_list=True):
        """
        Returns all the remote conversations for the active account. Returns the conversations in a list.
        """

        r = self.session.get(
            self.hf_base_url + "/chat/api/v2/conversations?p=0",
            headers=self.get_headers(ref=False),
            cookies=self.get_cookies(),
        )

        if r.status_code != 200:
            raise Exception(
                f"Failed to get remote conversations with status code: {r.status_code}"
            )
            
        # temporary workaround for #267
        data = r.json()["json"] 

        conversations = data[0]["conversations"]

        for conversation_data in conversations:
            c = Conversation(
                id=data[conversation_data["id"]],
                title=data[conversation_data["title"]],
                model=data[conversation_data["model"]],
            )

            conversations.append(c)

        if replace_conversation_list:
            self.conversation_list = conversations

        return conversations

    def parse_datetime_to_timestamp(self, datetime_str):
        if not datetime_str:
            return None
        try:
            # Python 3.7+ can often handle 'Z' directly, but for older or more robust parsing:
            if datetime_str.endswith('Z'):
                datetime_str = datetime_str[:-1] + '+00:00' # Replace Z with UTC offset
            dt_obj = datetime.datetime.fromisoformat(datetime_str)
            # If it's timezone-naive, assume UTC. If timezone-aware, convert to UTC then get timestamp.
            if dt_obj.tzinfo is None or dt_obj.tzinfo.utcoffset(dt_obj) is None:
                 dt_obj = dt_obj.replace(tzinfo=datetime.timezone.utc) # Assume UTC if naive
            else:
                dt_obj = dt_obj.astimezone(datetime.timezone.utc)
            return dt_obj.timestamp()
        except ValueError as e:
            print(f"Error parsing datetime string '{datetime_str}': {e}")
            return None

    def get_conversation_info(self, conversation: Union[Conversation, str] = None) -> Conversation:
        """
        Fetches information related to the specified conversation. Returns the conversation object.
        conversation: Conversation object that has the conversation id Or None to use the current conversation.
        """

        if conversation is None:
            conversation = self.current_conversation

        if isinstance(conversation, str):
            conversation = Conversation(id=conversation)

        r = self.session.get(
            self.hf_base_url +
            f"/chat/api/v2/conversations/{conversation.id}",
            headers=self.get_headers(ref=False),
            cookies=self.get_cookies(),
        )

        if r.status_code != 200:
            raise Exception(
                f"Failed to get conversation info with status code: {r.status_code}"
            )
            
        # you'll never understand the following codes until you try to debug huggingchat in person.
        try:
            data = r.json()['json']
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON: {e}")
            return None

        conversation.model = data.get("model")
        conversation.system_prompt = data.get("preprompt")
        conversation.title = data.get("title")

        messages_data_list = data.get("messages", [])
        conversation.history = []

        for msg_data_dict in messages_data_list:
            created_at_ts = self.parse_datetime_to_timestamp(msg_data_dict.get("createdAt"))
            updated_at_ts = self.parse_datetime_to_timestamp(msg_data_dict.get("updatedAt"))
            ancestor_ids = msg_data_dict.get("ancestors", [])
            children_ids = msg_data_dict.get("children", [])

            conversation.history.append(MessageNode(
                id=msg_data_dict.get("id"),
                role=msg_data_dict.get("from"),
                content=msg_data_dict.get("content"),
                ancestors=ancestor_ids, # Storing IDs
                children=children_ids,   # Storing IDs
                created_at=created_at_ts,
                updated_at=updated_at_ts
            ))

        logging.debug(f"Conversation {conversation.id} history (count): {len(conversation.history)}")
        if conversation.history:
            logging.debug(f"First message in history: {conversation.history[0]}")

        return conversation

    def get_conversation_from_id(self, conversation_id: str, return_index=False) -> Conversation:
        """
        Returns a conversation object that is already in the conversation list.
        """

        for i, conversation in enumerate(self.conversation_list):
            if conversation.id == conversation_id:
                if return_index:
                    return i
                return conversation


    def _stream_query(
        self,
        text: str,
        is_retry: bool = False,
        retry_count: int = 5,
        conversation: Conversation = None,
        message_id: str = None,
    ) -> typing.Generator[dict, None, None]:
    
        if conversation is None:
            conversation = self.current_conversation
    
        if retry_count <= 0:
            raise Exception(
                "the parameter retry_count must be greater than 0.")
        if len(conversation.history) == 0:
            raise Exception(
                "conversation history is empty, but we need the root message id of this conversation to continue.")
            
        if not message_id:
            # get last message id
            message_id = conversation.history[-1].id
            
        logging.debug(f'message_id: {message_id}')
    
        req_json = {
            "id": message_id,
            "inputs": text,
            "is_retry": is_retry,
        }

        req_json = json.dumps(req_json)
        headers = {
            'authority': 'huggingface.co',
            'accept': '*/*',
            'accept-language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7,en-GB;q=0.6',
            'origin': 'https://huggingface.co',
            'sec-ch-ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        }
        obj = {}
        break_flag = False
        has_started = False
    
        initial_retry_count = retry_count  # Track initial retry count
    
        while retry_count > 0:
            # Update is_retry flag for subsequent attempts
            if retry_count < initial_retry_count:
                req_json["is_retry"] = True
    
            resp = self.session.post(
                self.hf_base_url + f"/chat/conversation/{conversation}",
                files={ "data": (None, req_json)},
                stream=True,
                headers=headers,
                cookies=self.session.cookies.get_dict(),
            )
            resp.encoding = 'utf-8'

            if resp.status_code != 200:
                retry_count -= 1
                if retry_count <= 0:
                    raise exceptions.ChatError(
                        f"Failed to chat. ({resp.status_code})")
                continue  # Skip processing on non-200 status
    
            try:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    res = line
                    obj = json.loads(res)
                    if "type" in obj:
                        _type = obj["type"]
    
                        if _type == "finalAnswer":
                            break_flag = True
                            break
                        
                        if _type == "status" and obj["status"] == "started":
                            if has_started:
                                obj = {
                                    "type": "finalAnswer",
                                    "text": ""
                                }
                                break_flag = True
                                break
                            has_started = True
                        
                    else:
                        logging.error(f"No `type` found in response: {obj}")
                    yield obj
                # After processing all lines, mark completion
                break_flag = True
            except requests.exceptions.ChunkedEncodingError:
                pass
            except BaseException as e:
                traceback.print_exc()
                if "Model is overloaded" in str(e):
                    raise exceptions.ModelOverloadedError(
                        "Model is overloaded, please try again later or switch to another model."
                    )
                logging.debug(resp.headers)
                if "Conversation not found" in str(res):
                    raise exceptions.InvalidConversationIDError("Conversation id invalid")
                raise exceptions.ChatError(f"Failed to parse response: {res}")
            if break_flag:
                break
    
        # Update the history of current conversation
        self.get_conversation_info(conversation)
        yield obj
    
    def get_message_node(self, conversation: Conversation, message_id: str):
        for node in conversation.history:
            if node.id == message_id:
                return node
        raise Exception(f"no node found which id is {message_id}")

    def chat(
        self,
        text: str,
        _stream_yield_all: bool = True,
        retry_count: int = 5,
        conversation: Conversation = None,
        edit_user_node: MessageNode = None,
        *args,
        **kvargs,
    ) -> Message:
        """
        - Send a message to the current conversation. Return a Message object.
        
        - `Edit history`: pass `edit_user_node`. The history can be retrieved from `conversation.history`. 

        - About class `Message`:
            - `wait_until_done()`: Block until the response done processing or an error raised.
            - `__iter__()`: For loop call this Generator and get response.
            - `get_search_sources()`: The web search results. It is a list of WebSearchSource objects.

        - For more detail please see Message documentation(Message.__doc__)
        """
        if conversation is None:
            conversation = self.current_conversation
        
        if not text:
            raise Exception("don't support an empty string(I'm sure LLM cannot understand it)")
        if edit_user_node and edit_user_node.role != 'user':
            raise Exception("you must pass a user's message node to edit message")

        is_retry = True if edit_user_node else False
        edit_message_id = edit_user_node.id if is_retry else None

        msg = Message(
            g=self._stream_query(
                text=text,
                retry_count=retry_count,
                conversation=conversation,
                is_retry=is_retry,
                message_id=edit_message_id
            ),
            _stream_yield_all=_stream_yield_all,
            conversation=conversation
        )
        return msg
