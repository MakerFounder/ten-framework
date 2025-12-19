#
# This file is part of TEN Framework, an open source project.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for more information.
#

import json
from typing import Any, AsyncGenerator, Optional
from ten_runtime import AsyncTenEnv, Cmd, CmdResult, Data, Loc, TenError


def is_punctuation(char):
    if char in [",", "，", ".", "。", "?", "？", "!", "！"]:
        return True
    return False


def parse_sentences(sentence_fragment, content):
    sentences = []
    current_sentence = sentence_fragment
    for char in content:
        current_sentence += char
        if is_punctuation(char):
            stripped_sentence = current_sentence
            if any(c.isalnum() for c in stripped_sentence):
                sentences.append(stripped_sentence)
            current_sentence = ""

    remain = current_sentence
    return sentences, remain


async def _send_cmd(
    ten_env: AsyncTenEnv, cmd_name: str, dest: str, payload: Any = None
) -> tuple[Optional[CmdResult], Optional[TenError]]:
    cmd = Cmd.create(cmd_name)
    loc = Loc("", "", dest)
    cmd.set_dests([loc])
    if payload is not None:
        cmd.set_property_from_json(None, json.dumps(payload))
    return await ten_env.send_cmd(cmd)


async def _send_cmd_ex(
    ten_env: AsyncTenEnv, cmd_name: str, dest: str, payload: Any = None
) -> AsyncGenerator[tuple[Optional[CmdResult], Optional[TenError]], None]:
    cmd = Cmd.create(cmd_name)
    loc = Loc("", "", dest)
    cmd.set_dests([loc])
    if payload is not None:
        cmd.set_property_from_json(None, json.dumps(payload))
    async for cmd_result, ten_error in ten_env.send_cmd_ex(cmd):
        if cmd_result:
            yield cmd_result, ten_error


async def _send_data(
    ten_env: AsyncTenEnv, data_name: str, dest: str, payload: Any = None
) -> Optional[TenError]:
    data = Data.create(data_name)
    loc = Loc("", "", dest)
    data.set_dests([loc])
    if payload is not None:
        data.set_property_from_json(None, json.dumps(payload))
    return await ten_env.send_data(data)
