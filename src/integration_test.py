"""
For test hugchat
"""

# import os
import logging

import pytest

from .hugchat import hugchat, cli
from .hugchat.login import Login
import sys
from unittest.mock import patch

logging.basicConfig(level=logging.DEBUG)

EMAIL = "just_a_temp_email@iubridge.com"
PASSWORD = "FOR_TEST_DO_NOT_LOGIN_a1"


@pytest.fixture(scope="session")
def login_to_chatbot():
    sign = Login(EMAIL, PASSWORD)
    cookies = sign.login()
    sign.saveCookiesToDir("./fortest")
    assert cookies is not None
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), default_llm=0)
    yield chatbot
    chatbot.session.close()


class TestAPI:
    """
    test hugchat api
    """

    @pytest.fixture(autouse=True)
    def setup(self, login_to_chatbot):
        """
        setup
        """
        self.chatbot = login_to_chatbot

    def test_create_conversation(self):
        """
        test create conversation module
        """
        global my_conversation
        res = self.chatbot.new_conversation()
        assert res is not None
        self.chatbot.change_conversation(res)
        my_conversation = res
        print("Test create conversation:", str(res))

    def test_chat_without_web_search(self):
        """
        test chat module without web search
        """
        res = str(self.chatbot.chat("Just reply me `test_ok`"))
        assert res is not None