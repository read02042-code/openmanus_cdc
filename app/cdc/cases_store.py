import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import config
from app.schema import CDCEventType


class CaseReport(BaseModel):
    report_date: str
    event_type: CDCEventType
    location: str
    confirmed_cases: int = Field(ge=0)
    suspected_cases: int = Field(ge=0)
    severe_cases: int = Field(ge=0)
    deaths: int = Field(ge=0)
    notes: Optional[str] = None


class CaseStore(BaseModel):
    reports: List[CaseReport] = Field(default_factory=list)

    @staticmethod
    def default() -> "CaseStore":
        reports = [
            CaseReport(
                report_date="2026-04-10",
                event_type="influenza",
                location="某中学",
                confirmed_cases=12,
                suspected_cases=5,
                severe_cases=0,
                deaths=0,
                notes="学校出现聚集性流感样病例，已启动晨检与缺课追踪。",
            ),
            CaseReport(
                report_date="2026-04-11",
                event_type="influenza",
                location="某中学",
                confirmed_cases=20,
                suspected_cases=8,
                severe_cases=0,
                deaths=0,
                notes="病例数上升，建议开展流感样病例采样检测与健康宣教。",
            ),
            CaseReport(
                report_date="2026-04-12",
                event_type="influenza",
                location="某中学",
                confirmed_cases=25,
                suspected_cases=10,
                severe_cases=1,
                deaths=0,
                notes="出现重症病例，需评估停课阈值与医疗救治资源。",
            ),
            CaseReport(
                report_date="2026-04-10",
                event_type="influenza",
                location="某大学",
                confirmed_cases=35,
                suspected_cases=18,
                severe_cases=0,
                deaths=0,
                notes="高校宿舍区出现聚集性流感样病例，需加强晨午检与校内医疗点分诊。",
            ),
            CaseReport(
                report_date="2026-04-11",
                event_type="influenza",
                location="某大学",
                confirmed_cases=52,
                suspected_cases=26,
                severe_cases=1,
                deaths=0,
                notes="病例持续上升，建议开展重点人群采样检测与集体活动风险评估。",
            ),
            CaseReport(
                report_date="2026-04-12",
                event_type="influenza",
                location="某大学",
                confirmed_cases=60,
                suspected_cases=30,
                severe_cases=2,
                deaths=0,
                notes="出现多学院分布的病例，需评估分区停课与宿舍消毒通风措施。",
            ),
            CaseReport(
                report_date="2026-04-10",
                event_type="covid19",
                location="某社区",
                confirmed_cases=8,
                suspected_cases=6,
                severe_cases=0,
                deaths=0,
                notes="社区发现散发新冠病例，需开展重点人群健康监测与风险沟通。",
            ),
            CaseReport(
                report_date="2026-04-11",
                event_type="covid19",
                location="某社区",
                confirmed_cases=15,
                suspected_cases=10,
                severe_cases=0,
                deaths=0,
                notes="病例数增加，需评估聚集活动风险并强化医疗机构发热门诊预检分诊。",
            ),
            CaseReport(
                report_date="2026-04-12",
                event_type="covid19",
                location="某社区",
                confirmed_cases=22,
                suspected_cases=14,
                severe_cases=1,
                deaths=0,
                notes="出现家庭聚集病例，需加强密接管理与重点场所消毒。",
            ),
            CaseReport(
                report_date="2026-04-10",
                event_type="norovirus",
                location="某乡镇",
                confirmed_cases=0,
                suspected_cases=18,
                severe_cases=0,
                deaths=0,
                notes="乡镇发生急性胃肠炎聚集性症状，需开展流行病学调查与采样检测。",
            ),
            CaseReport(
                report_date="2026-04-11",
                event_type="norovirus",
                location="某乡镇",
                confirmed_cases=2,
                suspected_cases=30,
                severe_cases=0,
                deaths=0,
                notes="疑似诺如聚集持续，需加强饮用水与食品卫生、重点场所消毒。",
            ),
            CaseReport(
                report_date="2026-04-12",
                event_type="norovirus",
                location="某乡镇",
                confirmed_cases=6,
                suspected_cases=38,
                severe_cases=0,
                deaths=0,
                notes="确诊诺如病例增加，需落实隔离休息、呕吐物规范处置与环境消毒。",
            ),
        ]
        return CaseStore(reports=reports)

    @staticmethod
    def default_path() -> Path:
        return config.workspace_root / "cdc_data" / "cases.json"

    @classmethod
    def load_or_create(cls, path: Optional[Path] = None) -> "CaseStore":
        p = path or cls.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            store = cls.default()
            store.save(p)
            return store
        data = json.loads(p.read_text(encoding="utf-8"))
        reports = data.get("reports") if isinstance(data, dict) else None
        if isinstance(reports, list):
            for r in reports:
                if not isinstance(r, dict):
                    continue
                et = str(r.get("event_type") or "").strip()
                legacy = {
                    "influenza_school": "influenza",
                    "covid_community": "covid19",
                    "norovirus_cluster": "norovirus",
                }
                if et in legacy:
                    r["event_type"] = legacy[et]
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or self.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return p

    @staticmethod
    def _date_in_range(date_str: str, start: Optional[str], end: Optional[str]) -> bool:
        d = (date_str or "").strip()
        if not d:
            return False
        if start and d < start:
            return False
        if end and d > end:
            return False
        return True

    def query(
        self,
        *,
        event_type: Optional[str] = None,
        location_contains: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 200,
    ) -> List[CaseReport]:
        et = (event_type or "").strip()
        loc = (location_contains or "").strip()
        out: List[CaseReport] = []
        for r in self.reports:
            if et and r.event_type != et:
                continue
            if loc and loc not in r.location:
                continue
            if not self._date_in_range(r.report_date, start_date, end_date):
                continue
            out.append(r)
            if len(out) >= max(1, limit):
                break
        return out

    def append_report(self, report: CaseReport) -> None:
        self.reports.append(report)

    def summarize(
        self,
        *,
        event_type: Optional[str] = None,
        location_contains: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        rows = self.query(
            event_type=event_type,
            location_contains=location_contains,
            start_date=start_date,
            end_date=end_date,
            limit=10_000,
        )
        rows.sort(key=lambda r: r.report_date)
        totals = {
            "confirmed_cases": 0,
            "suspected_cases": 0,
            "severe_cases": 0,
            "deaths": 0,
        }
        series: List[Dict[str, Any]] = []
        for r in rows:
            totals["confirmed_cases"] += int(r.confirmed_cases)
            totals["suspected_cases"] += int(r.suspected_cases)
            totals["severe_cases"] += int(r.severe_cases)
            totals["deaths"] += int(r.deaths)
            series.append(
                {
                    "report_date": r.report_date,
                    "location": r.location,
                    "confirmed_cases": int(r.confirmed_cases),
                    "suspected_cases": int(r.suspected_cases),
                    "severe_cases": int(r.severe_cases),
                    "deaths": int(r.deaths),
                }
            )
        return {"totals": totals, "series": series, "count": len(rows)}
