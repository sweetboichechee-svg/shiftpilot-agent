import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PORT = int(os.environ.get("PORT", "9000"))
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
MODEL = os.environ.get("DASHSCOPE_MODEL", "deepseek-v4-flash").strip()
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
ENDPOINT = (
    BASE_URL
    if BASE_URL.endswith("/chat/completions")
    else BASE_URL.rstrip("/") + "/chat/completions"
)
ALLOWED_ORIGINS = {
    item.strip()
    for item in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if item.strip()
}


def is_object(value):
    return isinstance(value, dict)


def validate_config(config):
    if not is_object(config):
        return "缺少排班配置。"
    if not isinstance(config.get("days"), list) or not config["days"]:
        return "至少需要 1 个排班日期。"
    if not isinstance(config.get("shifts"), list) or not config["shifts"]:
        return "至少需要 1 个班次。"
    if not isinstance(config.get("employees"), list) or not config["employees"]:
        return "至少需要 1 名员工。"
    if (
        len(config["days"]) > 31
        or len(config["shifts"]) > 8
        or len(config["employees"]) > 120
    ):
        return "排班数据超过单次处理上限。"
    return None


def validate_payload_shape(value):
    if not is_object(value):
        return False
    array_fields = (
        "leave",
        "shiftLeave",
        "minimumStaff",
        "fixedAssignments",
        "employeeMaxShifts",
        "shiftPreferences",
        "understood",
        "warnings",
        "blockers",
    )
    return all(field not in value or isinstance(value[field], list) for field in array_fields)


class Handler(BaseHTTPRequestHandler):
    def cors_headers(self):
        origin = self.headers.get("Origin")
        allowed_origin = "*" if "*" in ALLOWED_ORIGINS else origin if origin in ALLOWED_ORIGINS else None
        headers = {
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Max-Age": "86400",
            "Vary": "Origin",
        }
        if allowed_origin:
            headers["Access-Control-Allow-Origin"] = allowed_origin
        return headers

    def send_json(self, status, body=None):
        content = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        for key, value in self.cors_headers().items():
            self.send_header(key, value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self):
        self.send_json(204)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/health", "/api/parse"):
            self.send_json(
                200,
                {
                    "configured": bool(API_KEY),
                    "provider": "阿里云百炼",
                    "model": MODEL,
                    "transport": "OpenAI-compatible Chat Completions",
                    "role": "自然语言转结构化约束",
                },
            )
            return
        self.send_json(404, {"error": "接口不存在。"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/parse":
            self.send_json(404, {"error": "接口不存在。"})
            return
        self.parse_with_model()

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("请求体过大。")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def parse_with_model(self):
        started_at = time.monotonic()
        if not API_KEY:
            self.send_json(503, {"error": "服务器尚未配置阿里云百炼 API。"})
            return
        try:
            body = self.read_json()
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(400, {"error": "请求不是有效的 JSON。"})
            return

        request_text = body.get("request", "") if is_object(body) else ""
        request_text = request_text.strip() if isinstance(request_text, str) else ""
        if not request_text or len(request_text) > 4000:
            self.send_json(400, {"error": "排班需求应为 1–4000 个字符。"})
            return
        config_error = validate_config(body.get("config"))
        if config_error:
            self.send_json(400, {"error": config_error})
            return

        config = body["config"]
        reference = {
            "scene": {"name": config.get("name"), "industry": config.get("industry")},
            "employees": [
                {"id": item.get("id"), "name": item.get("name")}
                for item in config["employees"]
            ],
            "days": config["days"],
            "shifts": [
                {"id": item.get("id"), "label": item.get("label")}
                for item in config["shifts"]
            ],
            "hardRuleNotice": "不得在自然语言解析阶段新增员工、技能、日期或班次，也不得放宽正式配置中的硬规则。",
        }
        output_shape = {
            "leave": [{"employeeId": "字符串", "dayIds": ["字符串"]}],
            "shiftLeave": [{"employeeId": "字符串", "dayId": "字符串", "shiftId": "字符串"}],
            "minimumStaff": [{"dayId": "字符串", "shiftId": "字符串", "count": 0}],
            "fixedAssignments": [{"employeeId": "字符串", "dayId": "字符串", "shiftId": "字符串"}],
            "employeeMaxShifts": [{"employeeId": "字符串", "count": 0}],
            "shiftPreferences": [{"employeeId": "字符串", "shiftIds": ["字符串"]}],
            "optimizePreferences": True,
            "understood": ["逐条概括已识别的约束"],
            "warnings": ["模糊但不阻塞的信息"],
            "blockers": ["无法可靠映射或试图放宽硬规则的信息"],
        }
        system_prompt = (
            "你是企业排班系统的需求解析器。你只把自然语言转成结构化增量约束，不生成排班，"
            "也不修改正式员工技能。只能使用参考数据中的 ID。全天请假写入 leave，只请某个班次"
            "写入 shiftLeave，不得同时写成全天请假。无法可靠映射时写入 blockers，不得猜测。"
            "只输出一个 JSON 对象，不要 Markdown。输出字段严格遵循："
            + json.dumps(output_shape, ensure_ascii=False)
        )
        upstream_body = json.dumps(
            {
                "model": MODEL,
                "temperature": 0,
                "max_tokens": 1800,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "参考数据："
                        + json.dumps(reference, ensure_ascii=False)
                        + "\n\n用户需求："
                        + request_text,
                    },
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        upstream_request = urllib.request.Request(
            ENDPOINT,
            data=upstream_body,
            headers={
                "Authorization": "Bearer " + API_KEY,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(upstream_request, timeout=20) as upstream:
                data = json.loads(upstream.read().decode("utf-8"))
                request_id = upstream.headers.get("x-request-id")
        except urllib.error.HTTPError as error:
            try:
                failure = json.loads(error.read().decode("utf-8"))
                message = failure.get("message") or failure.get("code")
            except Exception:
                message = None
            self.send_json(error.code, {"error": message or "百炼接口调用失败。"})
            return
        except Exception as error:
            print("Upstream request failed: %s: %s" % (type(error).__name__, error))
            self.send_json(502, {"error": "百炼请求失败或超时。"})
            return

        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            self.send_json(502, {"error": "模型没有返回可解析内容。"})
            return
        try:
            cleaned = content.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            payload = json.loads(cleaned.strip())
        except (TypeError, json.JSONDecodeError):
            self.send_json(502, {"error": "模型返回的结构不是有效 JSON。"})
            return
        if not validate_payload_shape(payload):
            self.send_json(502, {"error": "模型返回字段不符合约束协议。"})
            return

        usage = data.get("usage") or {}
        choice = choices[0] if choices else {}
        self.send_json(
            200,
            {
                "payload": payload,
                "provider": "阿里云百炼",
                "model": MODEL,
                "latencyMs": round((time.monotonic() - started_at) * 1000),
                "requestId": request_id or data.get("request_id") or data.get("id"),
                "finishReason": choice.get("finish_reason"),
                "usage": {
                    "inputTokens": usage.get("prompt_tokens", 0),
                    "outputTokens": usage.get("completion_tokens", 0),
                    "totalTokens": usage.get("total_tokens", 0),
                },
                "receivedAt": datetime.now(timezone.utc).isoformat(),
            },
        )

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("ShiftPilot parser API listening on %s" % PORT)
    server.serve_forever()
