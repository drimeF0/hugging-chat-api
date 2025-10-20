from typing import Dict
import requests
import os
import json
import logging
import re


class Login:
    def __init__(self, cookies: Dict) -> None:
        
        self.headers = {
            "Referer": "https://huggingface.co/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.64",
        }
        self.cookies = requests.sessions.RequestsCookieJar()

        for i in cookies.keys():
            self.cookies.set(i, cookies[i])
            logging.info(f"{i} loaded")


    def _request_get(self, url: str, params=None, allow_redirects=True) -> requests.Response:
        res = requests.get(
            url,
            params=params,
            headers=self.headers,
            cookies=self.cookies,
            allow_redirects=allow_redirects,
        )
        self._refresh_cookies(res.cookies)
        return res

    def _request_post(self, url: str, headers=None, params=None, data=None, stream=False,
                      allow_redirects=True) -> requests.Response:
        res = requests.post(
            url,
            stream=stream,
            params=params,
            data=data,
            headers=self.headers if headers is None else headers,
            cookies=self.cookies,
            allow_redirects=allow_redirects
        )
        self._refresh_cookies(res.cookies)
        return res

    def _refresh_cookies(self, cookies: requests.sessions.RequestsCookieJar):
        dic = cookies.get_dict()
        for i in dic:
            self.cookies.set(i, dic[i])