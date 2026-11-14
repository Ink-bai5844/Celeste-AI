import hashlib
import json
import math
import os
import struct
import threading
import time
import urllib.error
import urllib.request

from CelestePythonInterface import SessionData


PLAN_FEATURE_COUNT = 13
ACTION_NAMES = ["run", "jump", "dash", "climb", "wait", "other"]


class RoomPlan:
    def __init__(self, map_key, map_summary, payload, source):
        self.map_key = map_key
        self.map_summary = map_summary
        self.payload = payload
        self.source = source
        self.created_at = time.time()

    @property
    def waypoints(self):
        waypoints = self.payload.get("waypoints", [])
        if isinstance(waypoints, list):
            return waypoints
        return []

    @property
    def confidence(self):
        try:
            return float(self.payload.get("confidence", 0.35))
        except (TypeError, ValueError):
            return 0.35

    def to_json(self):
        payload = dict(self.payload)
        payload["source"] = self.source
        payload["map_key"] = self.map_key
        return json.dumps(payload, ensure_ascii=False)


class CelesteRoutePlanner:
    def __init__(
        self,
        enabled=False,
        model=None,
        base_url=None,
        api_key=None,
        failure_threshold=5,
        timeout=45,
        display_plans=False,
        max_tokens=4096,
        retries=1,
        retry_delay=2,
        response_format=True,
        thinking_enabled=None,
        thinking_effort=None,
        wire_api="chat_completions",
    ):
        self.enabled = enabled
        self.model = model or os.environ.get("CELESTE_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.display_plans = display_plans
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_delay = retry_delay
        self.response_format = response_format
        self.thinking_enabled = thinking_enabled
        self.thinking_effort = thinking_effort
        self.wire_api = normalize_wire_api(wire_api)
        self.lock = threading.Lock()
        self.display_lock = threading.Lock()
        self.plan_cache = {}
        self.failures = {}

    @classmethod
    def from_env(cls):
        enabled = os.environ.get("CELESTE_LLM_ENABLED", "").lower() in ("1", "true", "yes", "on")
        failure_threshold = int(os.environ.get("CELESTE_LLM_FAILURE_THRESHOLD", "5"))
        timeout = int(os.environ.get("CELESTE_LLM_TIMEOUT", "45"))
        display_plans = os.environ.get("CELESTE_LLM_DISPLAY_PLANS", "").lower() in ("1", "true", "yes", "on")
        max_tokens = int(os.environ.get("CELESTE_LLM_MAX_TOKENS", "4096"))
        retries = int(os.environ.get("CELESTE_LLM_RETRIES", "1"))
        retry_delay = int(os.environ.get("CELESTE_LLM_RETRY_DELAY", "2"))
        response_format = os.environ.get("CELESTE_LLM_RESPONSE_FORMAT", "true").lower() in ("1", "true", "yes", "on")
        thinking_enabled = parse_optional_bool(os.environ.get("CELESTE_LLM_THINKING_ENABLED"))
        thinking_effort = os.environ.get("CELESTE_LLM_THINKING_EFFORT")
        wire_api = os.environ.get("CELESTE_LLM_WIRE_API", "chat_completions")
        return cls(
            enabled=enabled,
            failure_threshold=failure_threshold,
            timeout=timeout,
            display_plans=display_plans,
            max_tokens=max_tokens,
            retries=retries,
            retry_delay=retry_delay,
            response_format=response_format,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
            wire_api=wire_api,
        )

    @classmethod
    def from_config_file(cls, config_path):
        config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as config_file:
                config = json.load(config_file)

        enabled = bool(config.get("enabled", False))
        model = config.get("model")
        base_url = config.get("base_url")
        api_key = config.get("api_key")
        failure_threshold = int(config.get("failure_threshold", 5))
        timeout = int(config.get("timeout", 45))
        display_plans = bool(config.get("display_plans", True))
        max_tokens = int(config.get("max_tokens", 4096))
        retries = int(config.get("retries", 1))
        retry_delay = int(config.get("retry_delay", 2))
        response_format = bool(config.get("response_format", True))
        thinking_enabled = parse_optional_bool(config.get("thinking_enabled"))
        thinking_effort = config.get("thinking_effort")
        wire_api = config.get("wire_api", "chat_completions")

        if "CELESTE_LLM_ENABLED" in os.environ:
            enabled = os.environ.get("CELESTE_LLM_ENABLED", "").lower() in ("1", "true", "yes", "on")
        model = os.environ.get("CELESTE_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or model
        base_url = os.environ.get("OPENAI_BASE_URL") or base_url
        api_key = os.environ.get("OPENAI_API_KEY") or api_key
        failure_threshold = int(os.environ.get("CELESTE_LLM_FAILURE_THRESHOLD", failure_threshold))
        timeout = int(os.environ.get("CELESTE_LLM_TIMEOUT", timeout))
        if "CELESTE_LLM_DISPLAY_PLANS" in os.environ:
            display_plans = os.environ.get("CELESTE_LLM_DISPLAY_PLANS", "").lower() in ("1", "true", "yes", "on")
        max_tokens = int(os.environ.get("CELESTE_LLM_MAX_TOKENS", max_tokens))
        retries = int(os.environ.get("CELESTE_LLM_RETRIES", retries))
        retry_delay = int(os.environ.get("CELESTE_LLM_RETRY_DELAY", retry_delay))
        if "CELESTE_LLM_RESPONSE_FORMAT" in os.environ:
            response_format = os.environ.get("CELESTE_LLM_RESPONSE_FORMAT", "").lower() in ("1", "true", "yes", "on")
        if "CELESTE_LLM_THINKING_ENABLED" in os.environ:
            thinking_enabled = parse_optional_bool(os.environ.get("CELESTE_LLM_THINKING_ENABLED"))
        thinking_effort = os.environ.get("CELESTE_LLM_THINKING_EFFORT") or thinking_effort
        wire_api = os.environ.get("CELESTE_LLM_WIRE_API") or wire_api

        return cls(
            enabled=enabled,
            model=model,
            base_url=base_url,
            api_key=api_key,
            failure_threshold=failure_threshold,
            timeout=timeout,
            display_plans=display_plans,
            max_tokens=max_tokens,
            retries=retries,
            retry_delay=retry_delay,
            response_format=response_format,
            thinking_enabled=thinking_enabled,
            thinking_effort=thinking_effort,
            wire_api=wire_api,
        )

    def plan_for_map(self, map_info):
        map_summary = self.parse_map_summary(map_info)
        map_key = self.map_key(map_info, map_summary)

        with self.lock:
            cached = self.plan_cache.get(map_key)
            failures = list(self.failures.get(map_key, []))
            should_revise = cached is not None and len(failures) >= self.failure_threshold
            if cached is not None and not should_revise:
                if self.display_plans:
                    print(f"[LLM route plan] cache hit map_key={map_key} source={cached.source}", flush=True)
                return cached

        if self.enabled and self.api_key:
            try:
                payload = self.request_llm_plan_with_retries(map_summary, failures if should_revise else [])
                plan = RoomPlan(map_key, map_summary, self.normalize_plan(payload, map_summary), "llm")
            except Exception as exc:
                plan = RoomPlan(map_key, map_summary, self.fallback_plan(map_summary, str(exc)), "fallback_after_llm_error")
        else:
            plan = RoomPlan(map_key, map_summary, self.fallback_plan(map_summary, None), "fallback")

        with self.lock:
            self.plan_cache[map_key] = plan
            if should_revise:
                self.failures[map_key] = []
        self.display_plan(plan, failures if should_revise else [], should_revise)
        return plan

    def display_plan(self, plan, failures, revised):
        if not self.display_plans:
            return

        map_summary = plan.map_summary
        bounds = map_summary.get("bounds", {})
        objective = map_summary.get("objective", {})
        waypoints = plan.waypoints[:12]
        hazards = plan.payload.get("hazards", [])
        notes = plan.payload.get("notes", [])

        with self.display_lock:
            print("", flush=True)
            print("=" * 72, flush=True)
            print("[LLM route plan]", flush=True)
            print(f"source: {plan.source}", flush=True)
            print(f"map_key: {plan.map_key}", flush=True)
            print(f"level: {map_summary.get('level', '')}", flush=True)
            print(
                f"bounds: x={bounds.get('x')} y={bounds.get('y')} "
                f"w={bounds.get('width')} h={bounds.get('height')}",
                flush=True,
            )
            print(f"objective: x={objective.get('x')} y={objective.get('y')}", flush=True)
            print(f"revised_after_failures: {revised} failures_sent: {len(failures)}", flush=True)
            print(f"confidence: {plan.confidence:.3f}", flush=True)
            print(f"strategy: {plan.payload.get('strategy', '')}", flush=True)
            print("waypoints:", flush=True)
            for index, waypoint in enumerate(waypoints, start=1):
                print(
                    f"  {index}. x={waypoint.get('x')} y={waypoint.get('y')} "
                    f"radius={waypoint.get('radius')} action={waypoint.get('action')} "
                    f"reason={waypoint.get('reason', '')}",
                    flush=True,
                )
            if hazards:
                print("hazards:", flush=True)
                for hazard in hazards[:8]:
                    print(f"  - {hazard}", flush=True)
            if notes:
                print("notes:", flush=True)
                for note in notes[:8]:
                    print(f"  - {note}", flush=True)
            print("=" * 72, flush=True)
            print("", flush=True)

    def request_llm_plan_with_retries(self, map_summary, failures):
        last_error = None
        attempts = max(1, self.retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                if self.display_plans:
                    thinking = "default" if self.thinking_enabled is None else ("enabled" if self.thinking_enabled else "disabled")
                    print(
                        f"[LLM route plan] requesting plan attempt {attempt}/{attempts} "
                        f"model={self.model} wire_api={self.wire_api} thinking={thinking}",
                        flush=True,
                    )
                return self.request_llm_plan(map_summary, failures)
            except Exception as exc:
                last_error = exc
                if self.display_plans:
                    print(f"[LLM route plan] request attempt {attempt}/{attempts} failed: {exc}", flush=True)
                if self.response_format and "empty message.content" in str(exc):
                    try:
                        if self.display_plans:
                            print(
                                "[LLM route plan] retrying once without response_format=json_object "
                                "because the provider returned empty content",
                                flush=True,
                            )
                        return self.request_llm_plan(map_summary, failures, response_format=False)
                    except Exception as fallback_exc:
                        last_error = fallback_exc
                        if self.display_plans:
                            print(f"[LLM route plan] non-json-mode retry failed: {fallback_exc}", flush=True)
                if attempt < attempts:
                    time.sleep(self.retry_delay)
        raise last_error

    def record_result(self, plan, final_state, fitness):
        if plan is None:
            return

        finished = final_state[SessionData.NUMBER_OF_LEVELS_FINISHED] >= 1
        if finished:
            with self.lock:
                self.failures[plan.map_key] = []
            return

        map_summary = plan.map_summary
        bounds = map_summary.get("bounds", {})
        player_x = bounds.get("x", 0) + final_state[SessionData.NORMALIZED_X] * max(bounds.get("width", 1), 1)
        player_y = bounds.get("y", 0) + final_state[SessionData.NORMALIZED_Y] * max(bounds.get("height", 1), 1)
        failure = {
            "x": round(player_x, 2),
            "y": round(player_y, 2),
            "distance_to_objective": round(final_state[SessionData.DISTANCE_TO_OBJECTIVE], 2),
            "best_distance_to_objective": round(final_state[SessionData.BEST_DISTANCE_TO_OBJECTIVE], 2),
            "seconds_elapsed": round(final_state[SessionData.SECONDS_ELAPSED], 2),
            "died": bool(final_state[SessionData.DIED] >= 0.5),
            "stuck": bool(final_state[SessionData.IS_STUCK] >= 0.5),
            "fitness": round(fitness, 2),
            "plan_source": plan.source,
            "plan_waypoints": plan.waypoints[:8],
        }

        with self.lock:
            failures = self.failures.setdefault(plan.map_key, [])
            failures.append(failure)
            if len(failures) > self.failure_threshold:
                del failures[0 : len(failures) - self.failure_threshold]

    def features_for_state(self, player_state, plan):
        if plan is None:
            return [0.0] * PLAN_FEATURE_COUNT

        map_summary = plan.map_summary
        bounds = map_summary.get("bounds", {})
        width = max(float(bounds.get("width", 1)), 1.0)
        height = max(float(bounds.get("height", 1)), 1.0)
        diagonal = math.hypot(width, height)
        left = float(bounds.get("x", 0))
        top = float(bounds.get("y", 0))
        player_x = left + player_state[SessionData.NORMALIZED_X] * width
        player_y = top + player_state[SessionData.NORMALIZED_Y] * height

        waypoint_index, waypoint = self.next_waypoint(player_x, player_y, plan)
        if waypoint is None:
            return [0.0] * PLAN_FEATURE_COUNT

        waypoint_x = float(waypoint.get("x", player_x))
        waypoint_y = float(waypoint.get("y", player_y))
        radius = float(waypoint.get("radius", 24))
        dx = waypoint_x - player_x
        dy = waypoint_y - player_y
        distance = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        action = str(waypoint.get("action", "other")).lower()

        action_features = [0.0] * len(ACTION_NAMES)
        if action not in ACTION_NAMES:
            action = "other"
        action_features[ACTION_NAMES.index(action)] = 1.0

        route_progress = 0.0
        if plan.waypoints:
            route_progress = waypoint_index / max(len(plan.waypoints) - 1, 1)

        return [
            clamp(dx / width),
            clamp(dy / height),
            clamp(distance / max(diagonal, 1.0), 0.0, 1.0),
            (angle + math.pi) / (2 * math.pi),
            clamp(radius / 64.0, 0.0, 1.0),
            clamp(route_progress, 0.0, 1.0),
            clamp(plan.confidence, 0.0, 1.0),
            *action_features,
        ]

    def next_waypoint(self, player_x, player_y, plan):
        waypoints = plan.waypoints
        for index, waypoint in enumerate(waypoints):
            waypoint_x = float(waypoint.get("x", player_x))
            waypoint_y = float(waypoint.get("y", player_y))
            radius = float(waypoint.get("radius", 24))
            if math.hypot(waypoint_x - player_x, waypoint_y - player_y) > radius:
                return index, waypoint
        if waypoints:
            return len(waypoints) - 1, waypoints[-1]
        return 0, None

    def request_llm_plan(self, map_summary, failures, response_format=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a Celeste level route planner. Analyze the room map and entity semantics. "
                    "Return only a valid JSON object with strategy, confidence, waypoints, hazards, and notes. "
                    "Waypoints must be sparse, physically plausible, and useful as hints for a local controller. "
                    "Your final assistant message content must not be empty and must contain only the JSON object.\n"
                    "Example JSON output:\n"
                    "{"
                    "\"strategy\":\"reach the exit using safe platforms\","
                    "\"confidence\":0.7,"
                    "\"waypoints\":[{\"x\":64,\"y\":128,\"radius\":20,\"action\":\"jump\",\"reason\":\"first platform\"}],"
                    "\"hazards\":[\"avoid spikes\"],"
                    "\"notes\":[\"correct locally if the jump undershoots\"]"
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Plan a route through this Celeste room for an AI controller.",
                        "required_json_schema": {
                            "strategy": "short route explanation",
                            "confidence": "0.0-1.0",
                            "waypoints": [
                                {
                                    "x": "float world coordinate",
                                    "y": "float world coordinate",
                                    "radius": "float acceptance radius",
                                    "action": "run|jump|dash|climb|wait|other",
                                    "reason": "short reason",
                                }
                            ],
                            "hazards": ["short hazard notes"],
                            "notes": ["short controller hints"],
                        },
                        "map_summary": map_summary,
                        "recent_failures": failures,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        if self.wire_api == "responses":
            return self.request_responses_plan(messages, response_format)

        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        use_response_format = self.response_format if response_format is None else response_format
        if use_response_format:
            payload["response_format"] = {"type": "json_object"}
        self.add_thinking_options(payload)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM planner HTTP {exc.code}: {error_body}") from exc

        if not raw_response.strip():
            raise RuntimeError("LLM planner returned an empty HTTP response body")

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM planner returned non-JSON HTTP body: {preview_text(raw_response)}") from exc

        content = self.extract_message_content(response_payload)
        return parse_json_object(content)

    def request_responses_plan(self, messages, response_format=None):
        url = f"{self.base_url}/responses"
        prompt_parts = []
        for message in messages:
            prompt_parts.append(f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}")
        payload = {
            "model": self.model,
            "input": "\n\n".join(prompt_parts),
            "temperature": 0.2,
            "max_output_tokens": self.max_tokens,
        }

        use_response_format = self.response_format if response_format is None else response_format
        if use_response_format:
            payload["text"] = {"format": {"type": "json_object"}}
        self.add_thinking_options(payload)

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM planner HTTP {exc.code}: {error_body}") from exc

        if not raw_response.strip():
            raise RuntimeError("LLM planner returned an empty HTTP response body")

        try:
            response_payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM planner returned non-JSON HTTP body: {preview_text(raw_response)}") from exc

        content = self.extract_responses_content(response_payload)
        return parse_json_object(content)

    def add_thinking_options(self, payload):
        if self.thinking_enabled is None:
            return

        if self.wire_api == "responses":
            if self.thinking_enabled and self.thinking_effort:
                payload["reasoning"] = {"effort": self.thinking_effort}
            return

        thinking = {"type": "enabled" if self.thinking_enabled else "disabled"}
        if self.thinking_enabled and self.thinking_effort:
            thinking["reasoning_effort"] = self.thinking_effort
        payload["thinking"] = thinking

    @staticmethod
    def extract_message_content(response_payload):
        try:
            choice = response_payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM planner response has no choices[0].message: {preview_text(json.dumps(response_payload, ensure_ascii=False))}") from exc

        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            content = "".join(parts)

        if content is None:
            content = ""
        content = str(content)

        if not content.strip():
            finish_reason = choice.get("finish_reason")
            reasoning_content = message.get("reasoning_content") or ""
            if str(reasoning_content).strip():
                raise RuntimeError(
                    "LLM planner returned empty message.content while reasoning_content was present "
                    f"(finish_reason={finish_reason}, reasoning_chars={len(str(reasoning_content))}). "
                    "Increase max_tokens further or switch to a non-reasoning/chat model."
                )
            response_preview = json.dumps(response_payload, ensure_ascii=False)
            raise RuntimeError(f"LLM planner returned empty message.content: {preview_text(response_preview)}")

        return content

    @staticmethod
    def extract_responses_content(response_payload):
        output_text = response_payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts = []
        output = response_payload.get("output", [])
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text") or content_item.get("output_text")
                    if text:
                        parts.append(str(text))

        content = "".join(parts)
        if content.strip():
            return content

        response_preview = json.dumps(response_payload, ensure_ascii=False)
        raise RuntimeError(f"LLM planner Responses API returned no output text: {preview_text(response_preview)}")

    def fallback_plan(self, map_summary, error):
        objective = map_summary.get("objective", {})
        player_start = map_summary.get("player_start", {})
        start_x = float(player_start.get("x", 0))
        start_y = float(player_start.get("y", 0))
        objective_x = float(objective.get("x", start_x))
        objective_y = float(objective.get("y", start_y))
        midpoint_x = start_x + (objective_x - start_x) * 0.55
        midpoint_y = min(start_y, objective_y) - 8
        notes = ["fallback straight-line waypoint plan"]
        if error:
            notes.append(f"llm_error: {error[:180]}")
        return {
            "strategy": "Move toward an intermediate waypoint, then the objective. Use local terrain inputs to correct jumps and hazards.",
            "confidence": 0.25,
            "waypoints": [
                {"x": midpoint_x, "y": midpoint_y, "radius": 28, "action": "run", "reason": "coarse midpoint toward objective"},
                {"x": objective_x, "y": objective_y, "radius": 24, "action": "jump", "reason": "final objective"},
            ],
            "hazards": [],
            "notes": notes,
        }

    def normalize_plan(self, payload, map_summary):
        if not isinstance(payload, dict):
            return self.fallback_plan(map_summary, "planner returned non-object JSON")
        payload.setdefault("strategy", "")
        payload.setdefault("confidence", 0.5)
        payload.setdefault("hazards", [])
        payload.setdefault("notes", [])

        waypoints = payload.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            return self.fallback_plan(map_summary, "planner returned no waypoints")

        normalized_waypoints = []
        for waypoint in waypoints[:16]:
            if not isinstance(waypoint, dict):
                continue
            try:
                x = float(waypoint["x"])
                y = float(waypoint["y"])
            except (KeyError, TypeError, ValueError):
                continue
            action = str(waypoint.get("action", "other")).lower()
            if action not in ACTION_NAMES:
                action = "other"
            normalized_waypoints.append(
                {
                    "x": x,
                    "y": y,
                    "radius": float(waypoint.get("radius", 24)),
                    "action": action,
                    "reason": str(waypoint.get("reason", ""))[:160],
                }
            )

        if not normalized_waypoints:
            return self.fallback_plan(map_summary, "planner returned unusable waypoints")
        payload["waypoints"] = normalized_waypoints
        return payload

    @staticmethod
    def parse_map_summary(map_info):
        try:
            return json.loads(map_info)
        except json.JSONDecodeError:
            return {"raw_map_summary": map_info}

    @staticmethod
    def map_key(map_info, map_summary=None):
        if map_summary is None:
            map_summary = CelesteRoutePlanner.parse_map_summary(map_info)

        try:
            bounds = map_summary.get("bounds", {})
            objective = map_summary.get("objective", {})
            grid = map_summary.get("grid", {})
            entities = map_summary.get("entities", [])
            entity_signature = []
            if isinstance(entities, list):
                for entity in entities:
                    if not isinstance(entity, dict):
                        continue
                    entity_signature.append(
                        (
                            str(entity.get("semantic", "")),
                            str(entity.get("type", "")),
                            round(float(entity.get("x", 0)) / 8.0),
                            round(float(entity.get("y", 0)) / 8.0),
                        )
                    )
            entity_signature.sort()
            stable_summary = {
                "schema_version": map_summary.get("schema_version"),
                "level": map_summary.get("level"),
                "bounds": {
                    "x": int(bounds.get("x", 0)),
                    "y": int(bounds.get("y", 0)),
                    "width": int(bounds.get("width", 0)),
                    "height": int(bounds.get("height", 0)),
                },
                "objective": {
                    "x": round(float(objective.get("x", 0)) / 4.0),
                    "y": round(float(objective.get("y", 0)) / 4.0),
                },
                "grid": {
                    "cell_pixels": grid.get("cell_pixels"),
                    "width": grid.get("width"),
                    "height": grid.get("height"),
                    "rows": grid.get("rows", []),
                },
                "entities": entity_signature,
            }
            serialized = json.dumps(stable_summary, sort_keys=True, separators=(",", ":"))
        except Exception:
            serialized = map_info
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def parse_optional_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("1", "true", "yes", "on", "enabled"):
        return True
    if lowered in ("0", "false", "no", "off", "disabled"):
        return False
    return None


def normalize_wire_api(value):
    normalized = str(value or "chat_completions").strip().lower().replace("-", "_")
    if normalized in ("responses", "response"):
        return "responses"
    return "chat_completions"


def parse_json_object(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"LLM planner message was not valid JSON: {preview_text(text)}") from exc
        raise RuntimeError(f"LLM planner message did not contain a JSON object: {preview_text(text)}")


def preview_text(text, limit=700):
    one_line = str(text).replace("\r", "\\r").replace("\n", "\\n")
    if len(one_line) <= limit:
        return one_line
    return one_line[:limit] + "...<truncated>"
