"""Deterministic routing for unambiguous live-video Monitor creation.

Monitor requests with an explicit lifecycle (for example, "the first time"
or "every time") do not need a main model to decide between stream probes,
monitor listing, and creation. This module keeps that narrow intent gate beside
the multimodal capability so the conversation core can execute one
``set_monitor(create)`` call locally. Requests without a clear lifecycle leave
the fast path and let the main model choose ``once`` versus ``continuous``.
Neither decision depends on mutable conversation state or prompt rebuilding.

False positives are more costly than false negatives: a false positive creates
the wrong background job, while a false negative merely falls back to normal
agent routing.  The patterns therefore require a visual cue plus either a
future alert contract or an explicit Monitor action with a discrete condition,
while excluding status, management, current-frame, unresolved-reference,
research/watcher, non-visual, and compound-action intents.
"""

from __future__ import annotations

import re
from typing import Optional


_NEW_MONITOR_ACTION_RE = re.compile(
    r"(帮我|请|麻烦|可以|能否|能不能|给我|替我)?[\s,，。?？]*"
    r"(监控(?:一下)?|盯(?:着|一下|住)?|看着|守着|持续(?:监控|观察|看)|"
    r"一直(?:盯|看着)|\bmonitor\b|\bwatch\b|keep\s+an\s+eye|look\s+out\s+for)",
    re.IGNORECASE,
)
_EXPLICIT_NEW_RE = re.compile(
    r"(?:(?:新建|新增|创建|另开|再开|再加)"
    r"(?=[^，。,.!?！？\n]{0,20}监控)[^，。,.!?！？\n]{0,20}监控|"
    r"新监控|独立监控|"
    r"(?:create|start|open|add)\b(?=[^.!?\n]{0,30}\bmonitor\b)[^.!?\n]{0,30}\bmonitor\b|"
    r"new\s+monitor|separate\s+monitor)",
    re.IGNORECASE,
)
_VISUAL_RE = re.compile(
    r"(画面|屏幕|摄像头|相机|视频|直播|共享|帧|桌面|界面|窗口|"
    r"看到|看见|盯|看着|守着|视觉|"
    r"错误提示|弹窗|对话框|screen|camera|video|frame|dialog|"
    r"webcam|desktop|display|\bui\b|window|popup|(?:camera|video)\s+feed|"
    r"live\s+(?:stream|feed)|streaming\s+(?:video|screen)|\bsee\b|\bspot\b)",
    re.IGNORECASE,
)
_SPORTS_EVENT_RE = re.compile(
    r"(进球|得分|破门|(?:投篮|三分|罚球)命中|扣篮|球进(?:了|门)?|"
    r"\bgoals?\b|\bscor(?:e|es|ed|ing)\b|\bbaskets?\b|"
    r"\b(?:shot|three[- ]pointer|free throw)\s+(?:goes?\s+in|is\s+made)\b)",
    re.IGNORECASE,
)
_SPORTS_VIEWING_CONTEXT_RE = re.compile(
    r"(整场|全场|比赛|球赛|赛事|直播|转播|"
    r"每次|每当|每逢|每一次|"
    r"\bwhenever\b|\bevery\s+time\b|\beach\s+time\b|"
    r"\bwhole\s+(?:game|match)\b|\b(?:game|match|broadcast)\b)",
    re.IGNORECASE,
)
_FUTURE_EVENT_RE = re.compile(
    r"(看到|看见|拍到|捕捉到|出现|弹出|打开|点开|点击|点进|进入|离开|发现|"
    r"检测到|识别到|显示|变成|变化|消失|亮起|完成|结束|断开|恢复|"
    r"一旦|当|如果|等到|有没有|"
    r"\bwhen\b|\bwhenever\b|\bonce\b|\bif\b|\bappear(?:s)?\b|"
    r"\bpop(?:s)?\s+up\b|\bshow(?:s)?\s+up\b|\bopen(?:s)?\b|\benter(?:s)?\b|"
    r"\bdetect(?:s|ed)?\b|\bspot(?:s|ted)?\b|\bdisappear(?:s)?\b|"
    r"\bturn(?:s)?\b|\bfinish(?:es)?\b|\bcomplete(?:s)?\b|\bend(?:s)?\b|"
    r"\bdisconnect(?:s|ed)?\b|\bgo(?:es)?\s+offline\b)",
    re.IGNORECASE,
)
_DELIVERY_RE = re.compile(
    r"(告诉我|告我|告知我|提醒我|提醒一下|通知我|提示我|跟我说|"
    r"说一声|喊我|叫我|发给我|"
    r"tell\s+me|remind\s+me|notify\s+me|alert\s+me|let\s+me\s+know|"
    r"ping\s+me|message\s+me|warn\s+me|give\s+me\s+(?:a\s+)?heads[- ]up|send\s+me)",
    re.IGNORECASE,
)
_NON_CREATE_RE = re.compile(
    r"(监控列表|列出.{0,12}监控|哪些监控|有哪些监控|几个监控|现在盯着什么|"
    r"正在监控什么|有没有监控|监控还在吗|监控进展|监控状态|监控结果|监控报告|"
    r"监控.{0,12}(触发过吗|看到了吗|记录了什么)|汇总.{0,12}监控|"
    r"监控.{0,12}(有没有开|是否(?:启用|开启|运行)|启用了吗|启动了吗|运行吗|开着吗|状态)|"
    r"(?:停止|停掉|取消|删除|移除|暂停|恢复|继续|重新开启|开启|启用|禁用|关闭|"
    r"修改|更新|调整|改成|改为|换成|给已有监控加)[^，。,.!?！？\n]{0,30}监控|"
    r"(?:把|将)?\s*(?:这个|那个|已有)?\s*监控[^，。,.!?！？\n]{0,30}(?:停止|停掉|取消|删除|移除|暂停|恢复|继续|"
    r"重新开启|启用|禁用|关闭|修改|更新|调整|改成|改为|换成)|"
    r"(?:不要|不用|别)(?:再)?(?:继续)?(?:监控|盯|提醒|通知|告诉)|"
    r"这个监控|那个监控|刚才.{0,8}监控|原来.{0,8}监控|之前.{0,8}监控|"
    r"是否.{0,8}监控|监控.{0,8}(了吗|为什么|为何|怎么工作|如何工作|原理|慢|延迟|性能|优化)|"
    r"怎么设置.{0,8}监控|如何设置.{0,8}监控|监控教程|"
    r"给已有监控加|已有监控.{0,20}(?:也|加|改)|"
    r"list.{0,20}monitors?|which\s+monitors?|monitor\s+(?:list|status|progress)|"
    r"what\s+are\s+you\s+watching|any\s+monitors?|existing\s+monitor|previous\s+monitor|"
    r"(?:old|current|existing|previous)[^.!?\n]{0,24}\bmonitor\b|"
    r"\bmonitor\b[^.!?\n]{0,30}(?:\balso\b|\btoo\b|as\s+well)|"
    r"\bmonitors?\b.{0,20}\b(?:enabled|active|running|on|off|status)\b|"
    r"(?:stop|cancel|delete|remove|pause|resume|re-enable|enable|disable|update|edit|change|turn\s+on)"
    r"[^.!?\n]{0,40}\bmonitor\b|\badd\b[^.!?\n]{0,30}\bto\b[^.!?\n]{0,20}\bmonitor\b|"
    r"(?:the\s+)?(?:existing\s+|current\s+)?monitor[^.!?\n]{0,30}"
    r"(?:stop|cancel|delete|remove|pause|resume|re-enable|enable|disable|update|edit|change)|"
    r"why.{0,20}monitor|how\s+(?:does|do|to).{0,20}monitor|monitor.{0,20}(?:slow|latency|performance|optimi[sz]))",
    re.IGNORECASE,
)
_STRONG_FUTURE_RE = re.compile(
    r"(?:如果|一旦|等到|下次|再次|重新).{0,100}"
    r"(?:看到|看见|出现|弹出|打开|点开|点击|点进|进入|发现|检测到|识别到)|"
    r"当.{0,100}(?:时|时候)|"
    r"(?:看到|看见|拍到|捕捉到|出现|弹出|打开|点开|点击|点进|进入|离开|发现|"
    r"检测到|识别到|显示|变成|变化|消失|亮起|完成|结束|断开|恢复)"
    r".{0,100}(?:就|时|时候|之后|后|的话)|"
    r"(?:画面|屏幕|摄像头|视频|界面|窗口).{0,80}有.{1,80}就|"
    r"\b(?:when|whenever|once)\b|"
    r"\bif\b.{0,100}(?:\bappear(?:s)?\b|\bshow(?:s)?\s+up\b|"
    r"\bopen(?:s)?\b|\benter(?:s)?\b|\bdetect(?:s|ed)?\b|\bspot(?:s|ted)?\b)|"
    r"\b(?:again|next\s+time)\b",
    re.IGNORECASE,
)
_COMPACT_ALERT_RE = re.compile(
    r"(?:看到|看见|拍到|捕捉到|出现|弹出|点开|点击|点进|发现|检测到|识别到|"
    r"显示|变成|消失|亮起|完成|结束).{1,120}"
    r"(?:告诉我|告我|提醒我|提醒一下|通知我|喊我|叫我|说一声)|"
    r"(?:\bappear(?:s)?\b|\bpop(?:s)?\s+up\b|\bdisappear(?:s)?\b|"
    r"\bfinish(?:es)?\b|\bcomplete(?:s)?\b).{1,120}"
    r"(?:alert|notify|tell|message|warn|remind)\s+me",
    re.IGNORECASE,
)
_RECURRENCE_RE = re.compile(
    r"(再次|下次|重新.{0,20}出现|again|next\s+time)", re.IGNORECASE)
_START_NOW_RE = re.compile(
    r"(从现在开始|自现在起|接下来|以后|from\s+now\s+on|"
    r"starting\s+(?:right\s+)?now|going\s+forward)", re.IGNORECASE)
_CONTINUOUS_TRIGGER_RE = re.compile(
    r"(每次|每当|每逢|每一次|"
    r"每(?:看到|看见|发现|出现|检测到|识别到|拍到|捕捉到|新增)|"
    r"每(?:有|来|进)(?:一个|一位|一件|一类|一种)|"
    r"持续|一直|全程|整场|反复|循环|"
    r"任何时候|有一个算一个|进球|得分|"
    r"\bwhenever\b|\bevery\s+time\b|\beach\s+time\b|"
    r"\bcontinuously\b|\bcontinuous(?:ly)?\b|\bkeep\s+(?:on\s+)?watching\b|"
    r"\bthroughout\b|\bwhole\s+(?:game|match)\b|\bscore(?:s|d|ing)?\b|\bgoals?\b)",
    re.IGNORECASE,
)
_EXPLICIT_REPEAT_RE = re.compile(
    r"(每次|每当|每逢|每一次|"
    r"每(?:看到|看见|发现|出现|检测到|识别到|拍到|捕捉到|新增)|"
    r"每(?:有|来|进)(?:一个|一位|一件|一类|一种)|"
    r"持续|一直|全程|整场|反复|循环|"
    r"任何时候|有一个算一个|\bwhenever\b|\bevery\s+time\b|"
    r"\beach\s+time\b|\bcontinuously\b|\bcontinuous(?:ly)?\b|"
    r"\bkeep\s+(?:on\s+)?watching\b|\bthroughout\b|\bwhole\s+(?:game|match)\b)",
    re.IGNORECASE,
)
_PER_EVENT_REPEAT_RE = re.compile(
    r"(每次|每当|每逢|每一次|"
    r"每(?:看到|看见|发现|出现|检测到|识别到|拍到|捕捉到|新增)|"
    r"每(?:有|来|进)(?:一个|一位|一件|一类|一种)|"
    r"每(?:个|件|类|种|位)|"
    r"\bwhenever\b|\bevery\s+time\b|\beach\s+time\b)",
    re.IGNORECASE,
)
_ONE_SHOT_TRIGGER_RE = re.compile(
    r"(第一次|首次|下一次|只要一次|只等一次|一旦|等到|"
    r"\bfirst\s+time\b|\b(?:first|next)\s+(?:goal|score)\b|"
    r"\bnext\s+time\b|\bonce\b|"
    r"\bone[- ]shot\b)",
    re.IGNORECASE,
)
_EXPLICIT_ONCE_DELIVERY_RE = re.compile(
    r"((?:只|仅)(?:提醒|通知|告诉|告知|提示|播报|报告)(?:我)?(?:一|1)次|"
    r"(?:只|仅)(?:需要)?(?:提醒|通知|告诉|告知|提示|播报)(?:我)?一下|"
    r"\b(?:alert|notify|tell|remind|message)\s+me\s+(?:only\s+)?once\b|"
    r"\bonly\s+(?:alert|notify|tell|remind|message)\s+me\s+once\b)",
    re.IGNORECASE,
)
_GLOBAL_ONCE_SCOPE_RE = re.compile(
    r"((?:总共|总计).{0,12}(?<!每)(?:一|1)次|"
    r"(?:整个监控|全程).{0,12}(?:只|仅|总共|总计).{0,8}(?<!每)(?:一|1)次|"
    r"(?:提醒|通知|告诉|告知|提示|播报|报告)(?:我)?后(?:就)?"
    r"(?:停止|结束|关闭)|"
    r"(?:但|不过)只(?:提醒|通知|告诉|告知|提示|播报|报告)(?:我)?(?:一|1)次|"
    r"\b(?:overall|in\s+total)\b.{0,30}\bonce\b|"
    r"\bonce\b.{0,20}\b(?:overall|in\s+total)\b|"
    r"\bonce\b.{0,20}\b(?:then\s+)?(?:stop|finish|end)\b|"
    r"\bbut\s+(?:alert|notify|tell|remind|message)\s+me\s+(?:only\s+)?once\b|"
    r"\bbut\s+(?:only\s+)?(?:alert|notify|tell|remind|message)\s+me\s+once\b)",
    re.IGNORECASE,
)
_STREAM_STATUS_RE = re.compile(
    r"(视频流|共享|摄像头|相机).{0,24}"
    r"(是否|有没有|开着|开了|开启|关闭|活跃|正常|在线|连接|健康|工作|清晰|状态)|"
    r"(?:is|whether).{0,20}(?:screen\s+share|camera|video\s+stream).{0,20}"
    r"(?:on|off|active|running|online|connected|healthy|working)|"
    r"(?:screen\s+share|camera|video\s+stream|camera\s+feed).{0,24}"
    r"(?:is\s+on|is\s+off|is\s+active|status|online|connected|healthy|works?)|"
    r"(?:screen.{0,24})?sharing.{0,20}(?:enabled|on|off|active|running)", re.IGNORECASE)
_PAST_OR_CURRENT_RE = re.compile(
    r"(刚才|刚刚|方才|之前|先前|现在|当前|此刻|"
    r"earlier|just\s+now|previously|right\s+now|currently)", re.IGNORECASE)
_EXPLANATION_RE = re.compile(
    r"(什么意思|代表什么|为什么|为何|原因|解释|怎么修|怎么办|怎么做|怎么关闭|如何处理|如何关闭|"
    r"what\s+does.{0,30}mean|what\s+should\s+i\s+do|why\b|explain|"
    r"how\s+to\s+(?:fix|close|handle))", re.IGNORECASE)
_NON_VISUAL_RE = re.compile(
    r"(服务器|端口|进程|日志|数据库|构建|部署|温度|cpu|内存占用|价格|股价|股票|"
    r"邮件|邮箱|订单|航班|网站|接口|api|\bserver\b|\bport\b|"
    r"\bprocess\b|\blogs?\b|\bdatabase\b|\bbuild\b|\bci\b|\bdeployment\b|"
    r"\btemperature\b|\bcpu\b|\bstock\b|\bprice\b|\bemail\b|\border\b|"
    r"\bflight\b|\bwebsite\b)", re.IGNORECASE)
_VISIBLE_TECH_ON_SCREEN_RE = re.compile(
    r"(?:屏幕|画面|桌面|界面|窗口)(?:上|里|中|内)?[^,，。.!?\n]{0,50}"
    r"(?:日志|部署|构建|api|错误|error)|"
    r"(?:logs?|deployment|build|api|error)[^,.!?\n]{0,40}(?:on|in)\s+(?:the\s+)?"
    r"(?:screen|display|desktop|window|ui)", re.IGNORECASE)
_VISUAL_TECH_TARGET_RE = re.compile(
    r"(?:摄像头|相机|视频|帧|直播|共享)[\s_-]*"
    r"(?:api|上传|处理|编码|进程|管线|服务|数据库)|"
    r"(?:camera|video|frame|live\s+stream|screen[- ]sharing)[\s_-]*"
    r"(?:api|upload|process|pipeline|server|service|encoding|database)", re.IGNORECASE)
_WATCHER_RE = re.compile(
    r"(持续分析|分析.{0,12}(?:视频|画面|屏幕)|总结画面|研究|追踪趋势|整体变化|每隔.{0,12}(总结|汇报)|\bwatcher\b|"
    r"(?:analy[sz]e|summari[sz]e|research).{0,20}(?:screen|video|stream|frame)|"
    r"continuously\s+(?:analy[sz]e|summari[sz]e|research)|track\s+trends?|"
    r"every.{0,20}(?:summari[sz]e|report))", re.IGNORECASE)
_COMPLEX_OR_AMBIGUOUS_RE = re.compile(
    r"(别提醒我|不要提醒我|不用提醒我|只在后台|静默|"
    r"每.{0,10}(秒|分钟|小时)|看到(?:这个|那个)|像图里|照之前|"
    r"搜索|查询|下单|写文件|记录到|运行命令|执行命令|关闭应用|修改配置|改配置|"
    r"给.{1,20}发(?:消息|邮件|通知)|截图|保存|"
    r"发给(?!我)|(?:然后|并且|同时|顺便|再).{0,20}"
    r"(?:点击|复制|打开|关闭|关掉|上传|下载|调用|创建|写入|保存|搜索|查询|运行|执行|发送)|"
    r"(?:就|时|后|之后).{0,12}(?:点击|复制|调用|创建|关掉)|"
    r"复制|调用\s*webhook|创建工单|关掉窗口|"
    r"do\s+not\s+(?:create\s+(?:a\s+)?monitor|alert|notify|tell)|background.only|silent|"
    r"every.{0,12}(?:second|minute|hour)|once\s+per|search|look\s+up|write\s+(?:a\s+)?file|"
    r"run\s+(?:a\s+)?command|change\s+(?:the\s+)?config|send\s+to|"
    r"send\s+(?:a\s+)?(?:message|email|notification)\s+to|"
    r"(?:take|save|send\s+me)\s+(?:a\s+)?screenshot|\bemail\s+\w+|"
    r"(?:then|and|also).{0,20}(?:click|copy|open|close|upload|download|call|invoke|"
    r"create|write|save|search|send|run)\b)", re.IGNORECASE)
_META_RE = re.compile(
    r"(you\s+(?:just\s+)?said|did\s+you.{0,12}\bsay\b|quoted?|correct\?|"
    r"这句话|你刚才说|对吗|翻译|改写|润色|重复一遍|"
    r"举个.{0,40}例子|我可以说|写一句|提示词|怎么说|不要创建监控|"
    r"\btranslate\b|\brewrite\b|\bparaphrase\b|\brepeat\b|\bexample\b|"
    r"\bprompt\b|can\s+i\s+say|is.{0,120}natural)", re.IGNORECASE)
_REFERENTIAL_RE = re.compile(
    r"(这个|那个|刚才那|之前那|上一个|"
    r"\bthis\b|\bthat\s+(?:one|dialog|popup|error|item|object|thing|window|message)\b|"
    r"previous\s+one|same\s+one)", re.IGNORECASE)
_EXPLICIT_UI_SELECTION_RE = re.compile(
    r"(?:点开|点击|点进)[^,，。.!?\n]{0,30}"
    r"(?:第[一二三四五六七八九十\d]+个|\bfirst\b|\bsecond\b)|"
    r"(?:click|open|enter)[^,.!?\n]{0,30}\b(?:first|second)\b",
    re.IGNORECASE,
)
_PAST_EVENT_RE = re.compile(
    r"(什么时候.{0,40}(?:出现|弹出|打开).{0,8}的|"
    r"何时.{0,40}(?:出现|弹出|打开).{0,8}的|\bwhen\s+did\b|"
    r"\b(?:appeared|showed\s+up|popped\s+up|opened|entered|disappeared|finished|completed|ended)\b)",
    re.IGNORECASE,
)
_CURRENT_QA_RE = re.compile(
    r"(告诉我.{0,50}(?:是什么|是谁|有几个|多少|在哪|分辨率|亮度|是否)|"
    r"(?:看看|看一下).{0,40}(?:现在|当前|有没有)|"
    r"tell\s+me.{0,40}(?:what|who|how\s+many|where|whether|its\s+resolution)|"
    r"(?:what|who|how\s+many|where)\s+(?:do|can)\s+you\s+see)", re.IGNORECASE)


def direct_monitor_request_text(agent, text: str) -> str:
    """Return clean text for the zero-model Monitor create path, else ``""``."""
    try:
        if not getattr(agent, "_multimodal_session", False):
            return ""
        if "set_monitor" not in getattr(agent, "valid_tool_names", set()):
            return ""
        text = str(text or "").strip()
        if not text:
            return ""

        explicit_new = bool(_EXPLICIT_NEW_RE.search(text))
        strong_future = bool(
            _STRONG_FUTURE_RE.search(text)
            or _EXPLICIT_REPEAT_RE.search(text)
            or _ONE_SHOT_TRIGGER_RE.search(text)
        )
        if (
            _NON_CREATE_RE.search(text)
            or _WATCHER_RE.search(text)
            or _COMPLEX_OR_AMBIGUOUS_RE.search(text)
            or _EXPLANATION_RE.search(text)
            or _META_RE.search(text)
            or (
                _REFERENTIAL_RE.search(text)
                and not _EXPLICIT_UI_SELECTION_RE.search(text)
            )
            or _PAST_EVENT_RE.search(text)
            or (_STREAM_STATUS_RE.search(text) and not strong_future)
        ):
            return ""

        has_monitor_action = bool(_NEW_MONITOR_ACTION_RE.search(text))
        sports_visual_event = bool(
            _SPORTS_EVENT_RE.search(text)
            and _SPORTS_VIEWING_CONTEXT_RE.search(text)
        )
        has_visual = bool(_VISUAL_RE.search(text)) or sports_visual_event
        if _VISUAL_TECH_TARGET_RE.search(text):
            return ""
        if _NON_VISUAL_RE.search(text) and not _VISIBLE_TECH_ON_SCREEN_RE.search(text):
            return ""

        has_event = bool(_FUTURE_EVENT_RE.search(text)) or sports_visual_event
        has_delivery = bool(_DELIVERY_RE.search(text))
        past_or_current = bool(_PAST_OR_CURRENT_RE.search(text))
        if past_or_current and not (
            _RECURRENCE_RE.search(text)
            or _START_NOW_RE.search(text)
            or strong_future
        ):
            return ""

        current_qa = bool(_CURRENT_QA_RE.search(text))
        if current_qa and not strong_future:
            return ""
        compact_trigger = bool(_COMPACT_ALERT_RE.search(text)) and not past_or_current
        full_alert_contract = (
            has_visual
            and has_delivery
            and (strong_future or compact_trigger)
        )
        explicit_monitor_contract = (
            has_visual and has_event and has_monitor_action and not current_qa)
        explicit_new_contract = explicit_new and has_visual and has_event
        matched = (
            full_alert_contract
            or explicit_monitor_contract
            or explicit_new_contract
        )
        if not matched:
            return ""
        # Lifecycle conflicts are intentionally left to the main model.  The
        # deterministic path is only safe when it can also choose the mode
        # without guessing.
        return text if infer_monitor_trigger_mode(text) is not None else ""
    except Exception:
        return ""


def infer_monitor_trigger_mode(text: str) -> Optional[str]:
    """Infer a high-confidence lifecycle, or ``None`` when it is ambiguous.

    Explicit delivery cardinality wins because it directly states how many
    alerts the user wants.  Otherwise, conflicting one-shot and recurrence
    cues fall back to normal model routing instead of growing a brittle phrase
    dictionary.  Requests without any cardinality cue also fall back to the
    model: a bare "tell me when X appears" does not say whether the user wants
    only the first alert or every future occurrence.  Legacy persisted monitors
    choose their own compatibility default when restored; this helper is only
    for newly created requests.
    """
    value = str(text or "")
    per_event_repeat = bool(_PER_EVENT_REPEAT_RE.search(value))
    per_item_repeat = bool(re.search(
        r"\b(?:each|every)\s+(?:new\s+)?(?:object|item|person|thing|category)\b",
        value,
        re.IGNORECASE,
    ))
    explicit_repeat = bool(
        _EXPLICIT_REPEAT_RE.search(value)
        or per_event_repeat
        or per_item_repeat
    )
    ui_selection = bool(_EXPLICIT_UI_SELECTION_RE.search(value))
    explicit_once = bool(
        _ONE_SHOT_TRIGGER_RE.search(value)
        or (ui_selection and not per_event_repeat)
    )
    once_delivery = bool(_EXPLICIT_ONCE_DELIVERY_RE.search(value))
    if _GLOBAL_ONCE_SCOPE_RE.search(value):
        return "once"
    if once_delivery:
        # "每次 X 只提醒一次" can mean one alert per episode rather
        # than one alert for the monitor's entire lifetime.  Only an explicit
        # global scope/stop clause overrides recurrence locally; otherwise let
        # the model resolve the scope.
        if explicit_repeat:
            return None
        return "once"
    if explicit_repeat and explicit_once:
        return None
    if explicit_repeat:
        return "continuous"
    if explicit_once:
        return "once"
    if _CONTINUOUS_TRIGGER_RE.search(value):
        return "continuous"
    return None


__all__ = ["direct_monitor_request_text", "infer_monitor_trigger_mode"]
