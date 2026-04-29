import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.config import config
from app.flow.base import BaseFlow
from app.flow.planning import PlanStepStatus
from app.logger import logger
from app.tool.cdc_plan_export import CDCPlanExportTool
from app.tool.planning import PlanningTool


class CDCPlanFlow(BaseFlow):
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    active_plan_id: str = Field(default_factory=lambda: f"cdc_plan_{int(time.time())}")
    output_docx: str = "cdc_plan_end_to_end.docx"
    manual_fix_path: str = "manual_fix_plan.json"
    max_manual_fix_rounds: int = 2
    step_max_retries: int = 1
    max_rollbacks: int = 1

    ctx: Dict[str, Any] = Field(default_factory=dict)
    step_attempts: Dict[int, int] = Field(default_factory=dict)
    rollback_count: int = 0
    ctx_snapshots: Dict[int, Dict[str, Any]] = Field(default_factory=dict)

    @staticmethod
    def _extract_first_json_object(text: str) -> Dict[str, Any]:
        if not isinstance(text, str):
            return {}
        s = text.strip()
        if not s:
            return {}

        if s.startswith("```"):
            s = re.sub(r"^```(?:json)?\s*\n", "", s)
            s = re.sub(r"\n```$", "", s)

        start = s.find("{")
        if start < 0:
            return {}

        in_string = False
        escape = False
        depth = 0
        end = None
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            return {}

        return CDCPlanFlow._safe_json_loads(s[start:end])

    def _extract_step_json(self, run_text: str, step_no: int = 1) -> Dict[str, Any]:
        if not isinstance(run_text, str) or not run_text.strip():
            return {}
        marker = f"Step {step_no}:"
        if marker in run_text:
            segment = run_text.split(marker, 1)[1]
        else:
            segment = run_text
        return self._extract_first_json_object(segment)

    @staticmethod
    def _safe_json_loads(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _require_nonempty_list(v: Any, label: str) -> list:
        if isinstance(v, list) and len(v) > 0:
            return v
        raise RuntimeError(f"{label} is missing or empty")

    async def _get_current_step(self) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        if self.active_plan_id not in self.planning_tool.plans:
            return None, None
        plan = self.planning_tool.plans[self.active_plan_id]
        steps = plan.get("steps", [])
        statuses = plan.get("step_statuses", [])
        for i, step in enumerate(steps):
            status = (
                statuses[i] if i < len(statuses) else PlanStepStatus.NOT_STARTED.value
            )
            if status in PlanStepStatus.get_active_statuses():
                step_info: Dict[str, Any] = {"text": step}
                m = re.search(r"\[([A-Z_]+)\]", step)
                if m:
                    step_info["type"] = m.group(1).lower()
                await self.planning_tool.execute(
                    command="mark_step",
                    plan_id=self.active_plan_id,
                    step_index=i,
                    step_status=PlanStepStatus.IN_PROGRESS.value,
                )
                return i, step_info
        return None, None

    async def _mark_step(self, idx: int, status: str, notes: str = "") -> None:
        await self.planning_tool.execute(
            command="mark_step",
            plan_id=self.active_plan_id,
            step_index=idx,
            step_status=status,
            step_notes=notes or None,
        )

    def _snapshot_ctx(self, idx: int) -> None:
        try:
            self.ctx_snapshots[idx] = json.loads(
                json.dumps(self.ctx, ensure_ascii=False)
            )
        except Exception:
            self.ctx_snapshots[idx] = dict(self.ctx)

    def _restore_ctx(self, idx: int) -> None:
        snap = self.ctx_snapshots.get(idx)
        if isinstance(snap, dict):
            self.ctx = snap

    async def _rollback_from(self, idx: int, reason: str) -> None:
        plan = self.planning_tool.plans.get(self.active_plan_id) or {}
        steps = plan.get("steps", [])
        statuses = plan.get("step_statuses", [])
        notes = plan.get("step_notes", [])
        while len(statuses) < len(steps):
            statuses.append(PlanStepStatus.NOT_STARTED.value)
        while len(notes) < len(steps):
            notes.append("")
        for i in range(idx, len(steps)):
            statuses[i] = PlanStepStatus.NOT_STARTED.value
            notes[i] = f"rolled back: {reason}" if i == idx else ""
        plan["step_statuses"] = statuses
        plan["step_notes"] = notes
        for k in list(self.step_attempts.keys()):
            if k >= idx:
                self.step_attempts.pop(k, None)
        self._restore_ctx(idx)

    async def _return_to_manual_fix(self, reason: str) -> None:
        plan = self.planning_tool.plans.get(self.active_plan_id) or {}
        steps = plan.get("steps", [])
        statuses = plan.get("step_statuses", [])
        notes = plan.get("step_notes", [])
        while len(statuses) < len(steps):
            statuses.append(PlanStepStatus.NOT_STARTED.value)
        while len(notes) < len(steps):
            notes.append("")

        manual_idx = None
        reval_idx = None
        for i, s in enumerate(steps):
            if "[MANUAL_FIX]" in s:
                manual_idx = i
            if "[REVALIDATION]" in s:
                reval_idx = i

        if manual_idx is None or reval_idx is None:
            return

        for i in range(manual_idx, len(steps)):
            statuses[i] = PlanStepStatus.NOT_STARTED.value
            notes[i] = ""
        notes[manual_idx] = f"revalidation failed: {reason}"
        plan["step_statuses"] = statuses
        plan["step_notes"] = notes
        for k in list(self.step_attempts.keys()):
            if k >= manual_idx:
                self.step_attempts.pop(k, None)

    async def _print_plan(self) -> None:
        res = await self.planning_tool.execute(
            command="get", plan_id=self.active_plan_id
        )
        if res.output:
            try:
                print(res.output)
            except UnicodeEncodeError:
                enc = getattr(sys.stdout, "encoding", None) or "utf-8"
                data = str(res.output).encode(enc, errors="replace")
                if hasattr(sys.stdout, "buffer"):
                    sys.stdout.buffer.write(data + b"\n")
                else:
                    print(data.decode(enc, errors="replace"))

    def _build_base_steps(self) -> List[str]:
        return [
            "[RISK] 风险评估",
            "[MEASURES] 防控措施生成（含规范检索）",
            "[RESOURCES] 资源调配（含库存/病例接口）",
            "[VALIDATION] 预案校验（含规范检索）",
            "[REVIEW] 预案草稿导览与确认",
            "[EXPORT] 导出文件",
        ]

    async def _review_draft(self) -> None:
        draft = self.ctx.get("draft_plan")
        if not isinstance(draft, dict):
            draft = self.ctx.get("plan")
        if not isinstance(draft, dict):
            raise RuntimeError("missing draft plan for review")

        p = Path(config.workspace_root) / f"{self.active_plan_id}_draft_plan.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.write_text(
                json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        self.ctx["draft_path"] = str(p)

        review_mode = str(self.ctx.get("review_mode") or "terminal").lower()
        if review_mode == "api":
            self.ctx["review_pending"] = True
            event = self.ctx.get("review_event")
            if not isinstance(event, asyncio.Event):
                event = asyncio.Event()
                self.ctx["review_event"] = event
            await event.wait()
            self.ctx["review_pending"] = False
            action = str(self.ctx.pop("review_action", "approve")).strip().lower()
            override = self.ctx.pop("review_plan_override", None)
            if action == "cancel":
                raise RuntimeError("review canceled by user")
            if action == "edit":
                if isinstance(override, dict):
                    self.ctx["plan"] = override
                else:
                    self.ctx["plan"] = draft
                return
            self.ctx["plan"] = draft
            return

        print(f"\n已生成预案草稿：{p}")
        print(
            "请选择： [A] 接受草稿并导出（默认） / [E] 编辑草稿 JSON 后导出 / [Q] 终止"
        )
        choice = "a"
        if sys.stdin and sys.stdin.isatty():
            choice = (
                await asyncio.to_thread(input, "请输入选项 [A/E/Q]：")
            ).strip().lower() or "a"
        if choice == "q":
            raise RuntimeError("review canceled by user")
        if choice == "e":
            print("请编辑草稿 JSON（保存后回到终端），然后按回车继续导出。")
            await asyncio.to_thread(input, "")
            raw = p.read_text(encoding="utf-8")
            updated = json.loads(raw)
            if not isinstance(updated, dict):
                raise RuntimeError("draft plan is not a JSON object")
            self.ctx["plan"] = updated
            return
        self.ctx["plan"] = draft

    async def _ensure_branch_steps(self) -> None:
        plan = self.planning_tool.plans.get(self.active_plan_id) or {}
        steps = plan.get("steps", [])
        if any("[MANUAL_FIX]" in s for s in steps):
            return
        new_steps = []
        for s in steps:
            if s.startswith("[EXPORT]"):
                new_steps.append("[MANUAL_FIX] 人工修改预案（必要时）")
                new_steps.append("[REVALIDATION] 重新校验")
            new_steps.append(s)
        await self.planning_tool.execute(
            command="update",
            plan_id=self.active_plan_id,
            steps=new_steps,
        )

    async def _manual_fix(self) -> None:
        plan_obj = self.ctx.get("plan")
        if not isinstance(plan_obj, dict):
            raise RuntimeError("missing plan for manual fix")
        p = Path(config.workspace_root) / self.manual_fix_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(plan_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manual_fix_mode = str(self.ctx.get("manual_fix_mode") or "terminal").lower()
        if manual_fix_mode == "api":
            self.ctx["manual_fix_pending"] = True
            self.ctx["manual_fix_path"] = str(p)
            event = self.ctx.get("manual_fix_event")
            if not isinstance(event, asyncio.Event):
                event = asyncio.Event()
                self.ctx["manual_fix_event"] = event

            await event.wait()
            self.ctx["manual_fix_pending"] = False
            action = str(self.ctx.pop("manual_fix_action", "s")).strip().lower()
            override = self.ctx.pop("manual_fix_plan_override", None)
            if action == "q":
                raise RuntimeError("manual fix aborted by user")
            if action == "x":
                self.ctx["manual_fix_bypass_revalidation"] = True
                self._append_manual_fix_note(
                    "用户选择不修改预案并跳过重新校验，直接导出。"
                )
                return
            if action == "s":
                self._append_manual_fix_note(
                    "用户进入人工修改节点，但选择不修改预案，继续重新校验。"
                )
                return
            if action == "e":
                if isinstance(override, dict):
                    self.ctx["plan"] = override
                else:
                    raw = p.read_text(encoding="utf-8")
                    try:
                        updated = json.loads(raw)
                    except Exception as e:
                        raise RuntimeError(f"manual plan json parse failed: {e}")
                    if not isinstance(updated, dict):
                        raise RuntimeError("manual plan is not a JSON object")
                    self.ctx["plan"] = updated
                self._append_manual_fix_note("用户已人工编辑预案，并继续重新校验。")
                return

            self._append_manual_fix_note(
                "用户进入人工修改节点，但未提供有效操作，默认继续重新校验。"
            )
            return

        print(f"\n已写入待人工修改预案：{p}")
        print("请选择后续动作：")
        print("  [E] 编辑 JSON 后重新校验")
        print("  [S] 不修改，直接重新校验（默认）")
        print("  [X] 不修改，跳过重新校验并继续导出（会写入说明）")
        print("  [Q] 终止流程")

        choice = "s"
        if sys.stdin and sys.stdin.isatty():
            choice = (await asyncio.to_thread(input, "请输入选项 [E/S/X/Q]：")).strip()
            choice = (choice or "s").lower()
        else:
            default_choice = str(
                self.ctx.get("manual_fix_default_action") or "s"
            ).lower()
            choice = default_choice if default_choice in {"e", "s", "x", "q"} else "s"
            print(f"检测到非交互终端，默认执行 [{choice.upper()}]。")

        if choice == "q":
            raise RuntimeError("manual fix aborted by user")

        if choice == "x":
            self.ctx["manual_fix_bypass_revalidation"] = True
            self._append_manual_fix_note("用户选择不修改预案并跳过重新校验，直接导出。")
            return

        if choice == "s":
            self._append_manual_fix_note(
                "用户进入人工修改节点，但选择不修改预案，继续重新校验。"
            )
            return

        print("请编辑该 JSON（保存后回到终端），然后按回车继续重新校验。")
        await asyncio.to_thread(input, "")
        raw = p.read_text(encoding="utf-8")
        try:
            updated = json.loads(raw)
        except Exception as e:
            raise RuntimeError(f"manual plan json parse failed: {e}")
        if not isinstance(updated, dict):
            raise RuntimeError("manual plan is not a JSON object")
        self.ctx["plan"] = updated
        self._append_manual_fix_note("用户已人工编辑预案，并继续重新校验。")

    def _append_manual_fix_note(self, note: str) -> None:
        plan_obj = self.ctx.get("plan")
        if not isinstance(plan_obj, dict):
            return
        sections = plan_obj.get("sections")
        if not isinstance(sections, list):
            sections = []
        if sections:
            last = sections[-1] if isinstance(sections[-1], dict) else {}
            if (
                isinstance(last, dict)
                and last.get("title") == "人工修改说明"
                and (last.get("paragraphs") or []) == [note]
            ):
                plan_obj["sections"] = sections
                self.ctx["plan"] = plan_obj
                return
        sections.append(
            {
                "title": "人工修改说明",
                "paragraphs": [note],
                "subsections": [],
            }
        )
        plan_obj["sections"] = sections
        self.ctx["plan"] = plan_obj

    async def _run_risk(self, input_text: str) -> None:
        agent = self.get_agent("risk")
        if agent is None:
            raise RuntimeError("missing risk agent")
        out = await agent.run(input_text)
        step = self._extract_step_json(out, 1)
        assessment = step.get("assessment") if isinstance(step, dict) else {}
        thinking = self._require_nonempty_list(
            (assessment or {}).get("thinking_summary"),
            "RiskAssessment.thinking_summary",
        )
        print("\n==== RiskAssessment.thinking_summary ====")
        print(json.dumps(thinking, ensure_ascii=False, indent=2))
        self.ctx["risk_step"] = step

    async def _run_measures(
        self, disease_type: str, location: str, risk_level: str
    ) -> None:
        agent = self.get_agent("measures")
        if agent is None:
            raise RuntimeError("missing measures agent")
        key_points = str(self.ctx.get("region_profile") or "").strip()
        key_points = self._sanitize_region_profile(disease_type, key_points)
        base = (
            f"disease_type: {disease_type}；location: {location}；risk_level: {risk_level}；"
            f"key_points: {key_points}；"
        )
        if key_points:
            prompt = (
                base
                + "请基于规范检索生成防控措施：至少10条，其中核心措施(core)不少于6条，补充措施(supplementary)不少于2条；"
                "每条core措施必须绑定>=1条引用(citations)；supplementary必须体现区域适配并明确写出适配因素；并输出thinking_summary。"
            )
        else:
            prompt = (
                base
                + "请基于规范检索生成防控措施：核心措施(core)不少于6条；不要输出补充措施(supplementary)；"
                "每条core措施必须绑定>=1条引用(citations)；并输出thinking_summary。"
            )
        out = await agent.run(prompt)
        step = self._extract_step_json(out, 1)
        output_obj = step.get("output") if isinstance(step, dict) else {}
        thinking = self._require_nonempty_list(
            (output_obj or {}).get("thinking_summary"),
            "ControlMeasures.thinking_summary",
        )
        print("\n==== ControlMeasures.thinking_summary ====")
        print(json.dumps(thinking, ensure_ascii=False, indent=2))
        self.ctx["measures_step"] = step

    async def _run_resources(
        self,
        disease_type: str,
        location: str,
        risk_level: str,
        population: int,
        cases: int,
        days: int,
    ) -> None:
        agent = self.get_agent("resources")
        if agent is None:
            raise RuntimeError("missing resources agent")
        prompt = (
            f"disease_type: {disease_type}；location: {location}；risk_level: {risk_level}；"
            f"population: {population}；cases: {cases}；days: {days}；"
            "请给出物资需求清单（按7天），并尝试从库存中分配；输出shortages（若有）与thinking_summary。"
        )
        out = await agent.run(prompt)
        step = self._extract_step_json(out, 1)
        output_obj = step.get("output") if isinstance(step, dict) else {}
        demands_thinking = self._require_nonempty_list(
            (output_obj or {}).get("demands_thinking_summary"),
            "ResourceAllocation.demands_thinking_summary",
        )
        thinking = self._require_nonempty_list(
            (output_obj or {}).get("thinking_summary"),
            "ResourceAllocation.thinking_summary",
        )
        print("\n==== ResourceAllocation.demands_thinking_summary ====")
        print(json.dumps(demands_thinking, ensure_ascii=False, indent=2))
        print("\n==== ResourceAllocation.thinking_summary ====")
        print(json.dumps(thinking, ensure_ascii=False, indent=2))
        self.ctx["resources_step"] = step

    async def _run_validation(self, plan_obj: Dict[str, Any]) -> Dict[str, Any]:
        agent = self.get_agent("validation")
        if agent is None:
            raise RuntimeError("missing validation agent")
        out = await agent.run(json.dumps({"plan": plan_obj}, ensure_ascii=False))
        step = self._extract_step_json(out, 1)
        output_obj = step.get("output") if isinstance(step, dict) else {}
        thinking = self._require_nonempty_list(
            (output_obj or {}).get("thinking_summary"),
            "PlanValidation.thinking_summary",
        )
        print("\n==== PlanValidation.improved_plan_validation_errors ====")
        print(
            json.dumps(
                (output_obj or {}).get("improved_plan_validation_errors") or [],
                ensure_ascii=False,
                indent=2,
            )
        )
        print("\n==== PlanValidation.improved_plan_rule_issues ====")
        print(
            json.dumps(
                (output_obj or {}).get("improved_plan_rule_issues") or [],
                ensure_ascii=False,
                indent=2,
            )
        )
        print("\n==== PlanValidation.thinking_summary ====")
        print(json.dumps(thinking, ensure_ascii=False, indent=2))
        self.ctx["validation_step"] = step
        return step

    def _ensure_supplementary_measures(self, measures: Any, disease_type: str) -> list:
        key_points = str(self.ctx.get("region_profile") or "").strip()
        key_points = self._sanitize_region_profile(str(disease_type or ""), key_points)
        key_points = re.sub(r"[。！？；;]+$", "", key_points).strip()
        items = measures if isinstance(measures, list) else []
        fixed: List[Dict[str, Any]] = []
        for m in items:
            if isinstance(m, dict):
                fixed.append(m)
        if not key_points:
            return fixed
        supp = [
            m for m in fixed if str(m.get("level") or "").strip() == "supplementary"
        ]
        if len(supp) >= 2:
            return fixed
        templates: List[Dict[str, Any]] = [
            {
                "title": "区域适配：校园结构与活动管理",
                "content": "适用场景/触发条件：当校园人员密集或活动频繁、需结合校内布局细化防控时。结合属地政策与本地条件细化执行。",
                "level": "supplementary",
                "citations": [],
            },
            {
                "title": "区域适配：医疗资源与检测保障",
                "content": "适用场景/触发条件：当校医院接诊压力上升或检测/药物存在缺口风险时。结合属地政策与本地条件细化执行。",
                "level": "supplementary",
                "citations": [],
            },
            {
                "title": "区域适配：属地政策与预警联动",
                "content": "适用场景/触发条件：当达到属地预警阈值或需与疾控会商调整策略时。结合属地政策与本地条件细化执行。",
                "level": "supplementary",
                "citations": [],
            },
        ]
        for t in templates:
            if len(supp) >= 2:
                break
            fixed.append(t)
            supp.append(t)
        return fixed

    @staticmethod
    def _sanitize_region_profile(disease_type: str, text: str) -> str:
        dt = str(disease_type or "").strip().lower()
        s = str(text or "").strip()
        if not s:
            return s
        if dt == "covid19":
            s = re.sub(r"奥司他韦[^，。,；;\n]*", "抗病毒药物（按属地目录）", s)
            s = re.sub(
                r"(甲流|流感|influenza)[^，。,；;\n]*", "", s, flags=re.IGNORECASE
            )
        elif dt == "influenza":
            s = re.sub(
                r"(paxlovid|奈玛特韦|利托那韦)[^，。,；;\n]*",
                "",
                s,
                flags=re.IGNORECASE,
            )
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"[，,。；;]\s*[，,。；;]+", "；", s)
        s = re.sub(r"[；;]\s*$", "", s)
        return s

    @staticmethod
    def _validation_needs_branch(step: Dict[str, Any]) -> bool:
        if not isinstance(step, dict):
            return True
        if isinstance(step.get("valid"), bool) and step["valid"] is False:
            return True
        output_obj = step.get("output") or {}
        if not isinstance(output_obj, dict):
            return True
        v_errors = output_obj.get("improved_plan_validation_errors") or []
        r_issues = output_obj.get("improved_plan_rule_issues") or []
        return bool(v_errors) or bool(r_issues)

    def _build_plan_skeleton(
        self,
        *,
        disease_type: str,
        location: str,
        population: int,
        reported_cases: int,
        r0: Optional[float],
        incubation_days: Optional[float],
        infectious_days: Optional[float],
        risk_level: str,
        risk_summary: str,
        predicted_cases_7d: Optional[int],
        measures: list,
        resources_items: list,
        jurisdiction: str,
        region_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        created_at = time.strftime("%Y-%m-%d", time.localtime())
        rp = str(region_profile or "").strip()
        return {
            "meta": {
                "title": f"{location}疫情应急处置预案",
                "jurisdiction": jurisdiction,
                "created_at": created_at,
            },
            "input": {
                "event_type": disease_type,
                "location": location,
                "population": population,
                "reported_cases": reported_cases,
                "report_date": created_at,
                **({"region_profile": rp} if rp else {}),
                "transmission": {
                    "r0": r0,
                    "incubation_days": incubation_days,
                    "infectious_days": infectious_days,
                },
            },
            "risk": {
                "level": risk_level,
                "summary": risk_summary,
                "predicted_cases_7d": predicted_cases_7d,
            },
            "measures": measures,
            "resources": {"items": resources_items},
            "sections": [],
        }

    async def execute(self, input_text: str) -> str:
        await self.planning_tool.execute(
            command="create",
            plan_id=self.active_plan_id,
            title="CDC Plan Flow",
            steps=self._build_base_steps(),
        )
        await self._print_plan()
        self.ctx["input_text"] = input_text

        while True:
            idx, step_info = await self._get_current_step()
            if idx is None or step_info is None:
                break
            step_type = (step_info.get("type") or "").lower()
            print(f"\n>> 执行节点 {idx}: {step_info.get('text')}")
            self._snapshot_ctx(idx)
            try:
                if step_type == "risk":
                    await self._run_risk(input_text)
                elif step_type == "measures":
                    risk_step = self.ctx.get("risk_step") or {}
                    disease_type = (
                        (risk_step.get("input") or {}).get("disease_type") or ""
                    ).strip()
                    location = (
                        (risk_step.get("input") or {}).get("location") or ""
                    ).strip()
                    risk_level = (
                        (risk_step.get("assessment") or {}).get("risk_level")
                        or "medium"
                    ).strip()
                    await self._run_measures(disease_type, location, risk_level)
                elif step_type == "resources":
                    risk_step = self.ctx.get("risk_step") or {}
                    disease_type = (
                        (risk_step.get("input") or {}).get("disease_type") or ""
                    ).strip()
                    location = (
                        (risk_step.get("input") or {}).get("location") or ""
                    ).strip()
                    risk_level = (
                        (risk_step.get("assessment") or {}).get("risk_level")
                        or "medium"
                    ).strip()
                    population = int(
                        (risk_step.get("input") or {}).get("population") or 0
                    )
                    cases = int(
                        (risk_step.get("input") or {}).get("reported_cases") or 0
                    )
                    days = int((risk_step.get("input") or {}).get("days") or 7)
                    await self._run_resources(
                        disease_type, location, risk_level, population, cases, days
                    )
                elif step_type == "validation":
                    risk_step = self.ctx.get("risk_step") or {}
                    measures_step = self.ctx.get("measures_step") or {}
                    resources_step = self.ctx.get("resources_step") or {}
                    disease_type = (
                        (risk_step.get("input") or {}).get("disease_type") or ""
                    ).strip()
                    location = (
                        (risk_step.get("input") or {}).get("location") or ""
                    ).strip()
                    population = int(
                        (risk_step.get("input") or {}).get("population") or 0
                    )
                    reported_cases = int(
                        (risk_step.get("input") or {}).get("reported_cases") or 0
                    )
                    r0 = (risk_step.get("input") or {}).get("r0")
                    incubation_days = (risk_step.get("input") or {}).get(
                        "incubation_days"
                    )
                    infectious_days = (risk_step.get("input") or {}).get(
                        "infectious_days"
                    )
                    risk_level = (
                        (risk_step.get("assessment") or {}).get("risk_level")
                        or "medium"
                    ).strip()
                    risk_summary = (
                        (risk_step.get("assessment") or {}).get("summary")
                        or "已完成风险评估。"
                    ).strip()
                    predicted = (risk_step.get("assessment") or {}).get(
                        "predicted_cases_7d"
                    )
                    measures = (measures_step.get("output") or {}).get("measures") or []
                    measures = self._ensure_supplementary_measures(
                        measures, disease_type
                    )
                    demands = resources_step.get("demands") or []
                    allocation_result = (
                        resources_step.get("allocation_result")
                        if isinstance(resources_step, dict)
                        else None
                    ) or {}
                    resources_output = (
                        resources_step.get("output")
                        if isinstance(resources_step, dict)
                        else None
                    ) or {}
                    resources_items = []
                    if isinstance(demands, list):
                        for d in demands:
                            if not isinstance(d, dict):
                                continue
                            name = str(d.get("name") or "").strip()
                            if not name:
                                continue
                            unit = str(d.get("unit") or "").strip()
                            if not unit or unit.lower() == "unit":
                                unit = "个"
                            resources_items.append(
                                {
                                    "name": name,
                                    "unit": unit,
                                    "quantity": float(d.get("quantity") or 0),
                                }
                            )
                    allocation_sec: Dict[str, Any] = {
                        "title": "资源调配与缺口",
                        "paragraphs": [],
                        "subsections": [],
                    }
                    allocation_sec["paragraphs"].append(
                        "说明：resources 表为系统基于病例数与天数估算的 7 天物资“需求清单”（不是库存清单）；系统已尝试从库存接口按“先市级库、后区级库”顺序分配。缺口部分需通过应急调拨、紧急采购或同级单位借用补齐。"
                    )
                    if isinstance(resources_output, dict):
                        summary = str(resources_output.get("summary") or "").strip()
                        if summary:
                            allocation_sec["paragraphs"].append(f"调配摘要：{summary}")
                        actions = resources_output.get("actions")
                        if isinstance(actions, list) and actions:
                            allocation_sec["paragraphs"].append(
                                "建议动作："
                                + "；".join(
                                    [str(a) for a in actions if str(a).strip()]
                                )[:300]
                            )
                    if isinstance(allocation_result, dict):
                        alloc_rows = allocation_result.get("allocations")
                        if isinstance(alloc_rows, list) and alloc_rows:
                            details = []
                            for a in alloc_rows[:12]:
                                if not isinstance(a, dict):
                                    continue
                                nm = str(a.get("name") or "").strip()
                                unit = str(a.get("unit") or "").strip() or "个"
                                req_qty = a.get("requested_quantity")
                                alloc_qty = a.get("allocated_quantity")
                                try:
                                    req_f = (
                                        float(req_qty) if req_qty is not None else 0.0
                                    )
                                except Exception:
                                    req_f = 0.0
                                try:
                                    alloc_f = (
                                        float(alloc_qty)
                                        if alloc_qty is not None
                                        else 0.0
                                    )
                                except Exception:
                                    alloc_f = 0.0
                                if not nm:
                                    continue
                                shortage_f = max(0.0, req_f - alloc_f)
                                wh_brief = []
                                alloc_list = a.get("allocations")
                                if isinstance(alloc_list, list) and alloc_list:
                                    for it in alloc_list[:3]:
                                        if not isinstance(it, dict):
                                            continue
                                        wn = str(
                                            it.get("warehouse_name")
                                            or it.get("warehouse_id")
                                            or ""
                                        ).strip()
                                        q = it.get("quantity")
                                        try:
                                            qf = float(q) if q is not None else 0.0
                                        except Exception:
                                            qf = 0.0
                                        if wn and qf > 0:
                                            wh_brief.append(f"{wn}{qf:g}{unit}")
                                source_text = (
                                    f"；来源：{', '.join(wh_brief)}" if wh_brief else ""
                                )
                                status_text = (
                                    "已满足"
                                    if shortage_f <= 1e-9
                                    else f"缺口{shortage_f:g}{unit}"
                                )
                                details.append(
                                    f"- {nm}：需求{req_f:g}{unit}，已分配{alloc_f:g}{unit}，{status_text}{source_text}"
                                )
                            if details:
                                allocation_sec["paragraphs"].append("分配明细：")
                                allocation_sec["paragraphs"].extend(details)
                        shortages = allocation_result.get("shortages")
                        if isinstance(shortages, list) and shortages:
                            brief = []
                            for s in shortages[:8]:
                                if not isinstance(s, dict):
                                    continue
                                nm = str(s.get("name") or "").strip()
                                unit = str(s.get("unit") or "").strip() or "个"
                                sh = s.get("shortage")
                                if nm and sh is not None:
                                    brief.append(f"{nm} 缺口 {sh}{unit}")
                            if brief:
                                allocation_sec["paragraphs"].append(
                                    "缺口概览：" + "；".join(brief)
                                )
                    plan_obj = self._build_plan_skeleton(
                        disease_type=disease_type,
                        location=location,
                        population=population,
                        reported_cases=reported_cases,
                        r0=r0,
                        incubation_days=incubation_days,
                        infectious_days=infectious_days,
                        risk_level=risk_level,
                        risk_summary=risk_summary,
                        predicted_cases_7d=(
                            int(predicted) if predicted is not None else None
                        ),
                        measures=measures if isinstance(measures, list) else [],
                        resources_items=resources_items,
                        jurisdiction="某市疾控中心",
                        region_profile=str(self.ctx.get("region_profile") or "").strip()
                        or None,
                    )
                    plan_obj["sections"] = [allocation_sec]
                    self.ctx["plan"] = plan_obj
                    validation_step = await self._run_validation(plan_obj)
                    output_obj = (
                        validation_step.get("output")
                        if isinstance(validation_step, dict)
                        else {}
                    )
                    improved_plan = (output_obj or {}).get("improved_plan")
                    draft_plan = (
                        improved_plan if isinstance(improved_plan, dict) else plan_obj
                    )
                    self.ctx["draft_plan"] = draft_plan
                    self.ctx["validation_result"] = validation_step
                elif step_type == "review":
                    await self._review_draft()
                elif step_type == "export":
                    plan_obj = self.ctx.get("plan")
                    if not isinstance(plan_obj, dict):
                        raise RuntimeError("missing plan for export")
                    tool = CDCPlanExportTool()
                    out = str(self.output_docx or "").strip()
                    fmt = "pdf" if out.lower().endswith(".pdf") else "docx"
                    res = await tool.execute(
                        plan=plan_obj, output_path=self.output_docx, output_format=fmt
                    )
                    if res.error:
                        raise RuntimeError(res.error)
                    self.ctx["export"] = res.output
                else:
                    raise RuntimeError(f"unknown step type: {step_type}")
                await self._mark_step(idx, PlanStepStatus.COMPLETED.value)
            except Exception as e:
                attempts = self.step_attempts.get(idx, 0) + 1
                self.step_attempts[idx] = attempts

                if attempts <= self.step_max_retries:
                    self._restore_ctx(idx)
                    await self._mark_step(
                        idx,
                        PlanStepStatus.NOT_STARTED.value,
                        f"retry {attempts}/{self.step_max_retries} after error: {e}",
                    )
                    await self._print_plan()
                    await asyncio.sleep(min(1.0 * attempts, 3.0))
                    continue

                if self.rollback_count < self.max_rollbacks:
                    self.rollback_count += 1
                    await self._rollback_from(idx, f"{e}")
                    await self._print_plan()
                    continue

                await self._mark_step(idx, PlanStepStatus.BLOCKED.value, str(e))
                await self._print_plan()
                raise

            await self._print_plan()

        export_out = self.ctx.get("export")
        if export_out:
            return str(export_out)
        plan_obj = self.ctx.get("plan")
        return json.dumps({"plan": plan_obj}, ensure_ascii=False, indent=2)
