import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.config import config


class MaterialItem(BaseModel):
    sku: str
    name: str
    unit: str = "unit"
    category: Optional[str] = None
    spec: Optional[str] = None
    safety_stock: Optional[float] = None


class Warehouse(BaseModel):
    warehouse_id: str
    name: str
    location: Optional[str] = None


class StockRecord(BaseModel):
    warehouse_id: str
    sku: str
    quantity: float = Field(ge=0)
    updated_at: Optional[str] = None


def _utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        if isinstance(v, bool):
            return default
        return float(v)
    except Exception:
        return default


class MaterialStore(BaseModel):
    items: List[MaterialItem] = Field(default_factory=list)
    warehouses: List[Warehouse] = Field(default_factory=list)
    stock: List[StockRecord] = Field(default_factory=list)

    @staticmethod
    def default() -> "MaterialStore":
        items = [
            MaterialItem(
                sku="mask_surgical",
                name="一次性医用外科口罩",
                unit="只",
                category="PPE",
            ),
            MaterialItem(sku="mask_n95", name="N95 防护口罩", unit="只", category="PPE"),
            MaterialItem(sku="goggles", name="护目镜", unit="副", category="PPE"),
            MaterialItem(sku="face_shield", name="防护面屏", unit="个", category="PPE"),
            MaterialItem(sku="gloves", name="一次性医用手套", unit="双", category="PPE"),
            MaterialItem(sku="protective_suit", name="防护服", unit="套", category="PPE"),
            MaterialItem(sku="isolation_gown", name="隔离衣", unit="件", category="PPE"),
            MaterialItem(sku="shoe_cover", name="一次性鞋套", unit="双", category="PPE"),
            MaterialItem(sku="antigen_test", name="抗原检测试剂", unit="人份", category="Test"),
            MaterialItem(sku="pcr_reagent", name="核酸检测试剂", unit="人份", category="Test"),
            MaterialItem(sku="sample_swab", name="采样拭子", unit="根", category="Sampling"),
            MaterialItem(sku="vtm_tube", name="病毒保存液采样管", unit="管", category="Sampling"),
            MaterialItem(sku="sample_bag", name="样本密封袋", unit="个", category="Sampling"),
            MaterialItem(sku="biohazard_bag", name="医疗废物袋", unit="个", category="Disposal"),
            MaterialItem(sku="sharps_box", name="利器盒", unit="个", category="Disposal"),
            MaterialItem(
                sku="disinfectant",
                name="含氯消毒液",
                unit="瓶",
                category="Disinfection",
            ),
            MaterialItem(
                sku="hand_sanitizer",
                name="速干手消毒剂",
                unit="瓶",
                category="Disinfection",
            ),
            MaterialItem(
                sku="chlorine_tablet",
                name="含氯消毒片",
                unit="片",
                category="Disinfection",
            ),
            MaterialItem(sku="sprayer", name="消毒喷雾器", unit="台", category="Disinfection"),
            MaterialItem(sku="thermometer", name="体温计", unit="支", category="Equipment"),
            MaterialItem(sku="temp_gun", name="红外测温枪", unit="把", category="Equipment"),
            MaterialItem(sku="cooler_box", name="冷链周转箱", unit="个", category="ColdChain"),
            MaterialItem(sku="ice_pack", name="冰排", unit="个", category="ColdChain"),
            MaterialItem(sku="transport_box", name="样本转运箱", unit="个", category="Logistics"),
            MaterialItem(sku="warning_sign", name="警示标识", unit="张", category="Logistics"),
            MaterialItem(sku="megaphone", name="扩音器", unit="个", category="Logistics"),
        ]
        warehouses = [
            Warehouse(warehouse_id="wh_city_cdc", name="市疾控中心库", location="市疾控中心"),
            Warehouse(warehouse_id="wh_district_cdc", name="区疾控中心库", location="区疾控中心"),
        ]
        stock = [
            StockRecord(warehouse_id="wh_city_cdc", sku="mask_surgical", quantity=20000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="mask_n95", quantity=3000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="goggles", quantity=600, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="face_shield", quantity=1000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="gloves", quantity=8000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="protective_suit", quantity=1200, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="isolation_gown", quantity=1500, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="shoe_cover", quantity=8000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="pcr_reagent", quantity=15000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="sample_swab", quantity=20000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="vtm_tube", quantity=8000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="sample_bag", quantity=15000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="biohazard_bag", quantity=6000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="sharps_box", quantity=300, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="disinfectant", quantity=500, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="hand_sanitizer", quantity=900, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="chlorine_tablet", quantity=4000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="sprayer", quantity=25, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="thermometer", quantity=120, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="temp_gun", quantity=40, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="cooler_box", quantity=60, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="ice_pack", quantity=600, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="transport_box", quantity=90, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="warning_sign", quantity=300, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_city_cdc", sku="megaphone", quantity=12, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="mask_surgical", quantity=6000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="antigen_test", quantity=3000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="goggles", quantity=120, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="face_shield", quantity=220, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="gloves", quantity=2500, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="protective_suit", quantity=260, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="disinfectant", quantity=120, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="hand_sanitizer", quantity=200, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="sample_swab", quantity=5000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="vtm_tube", quantity=1200, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="sample_bag", quantity=4000, updated_at=_utc_ts()),
            StockRecord(warehouse_id="wh_district_cdc", sku="ice_pack", quantity=120, updated_at=_utc_ts()),
        ]
        return MaterialStore(items=items, warehouses=warehouses, stock=stock)

    @staticmethod
    def default_path() -> Path:
        return config.workspace_root / "cdc_data" / "materials.json"

    @classmethod
    def load_or_create(cls, path: Optional[Path] = None) -> "MaterialStore":
        p = path or cls.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            store = cls.default()
            store.save(p)
            return store
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        p = path or self.default_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def _find_item_by_sku(self, sku: str) -> Optional[MaterialItem]:
        s = (sku or "").strip()
        if not s:
            return None
        for it in self.items:
            if it.sku == s:
                return it
        return None

    def find_item(self, *, sku: Optional[str] = None, name: Optional[str] = None) -> Optional[MaterialItem]:
        if sku:
            it = self._find_item_by_sku(sku)
            if it:
                return it
        q = (name or "").strip()
        if not q:
            return None
        for it in self.items:
            if it.name == q:
                return it
        for it in self.items:
            if q in it.name:
                return it
        return None

    def list_stock(self, *, warehouse_id: Optional[str] = None) -> List[Dict[str, Any]]:
        wh_filter = (warehouse_id or "").strip()
        stock = [s for s in self.stock if (not wh_filter or s.warehouse_id == wh_filter)]
        item_map = {i.sku: i for i in self.items}
        wh_map = {w.warehouse_id: w for w in self.warehouses}
        out: List[Dict[str, Any]] = []
        for s in stock:
            it = item_map.get(s.sku)
            wh = wh_map.get(s.warehouse_id)
            out.append(
                {
                    "warehouse_id": s.warehouse_id,
                    "warehouse_name": wh.name if wh else s.warehouse_id,
                    "sku": s.sku,
                    "name": it.name if it else s.sku,
                    "unit": it.unit if it else "unit",
                    "quantity": float(s.quantity),
                    "updated_at": s.updated_at,
                }
            )
        out.sort(key=lambda x: (x["warehouse_id"], x["sku"]))
        return out

    def get_total_stock(self, *, sku: str) -> float:
        s = (sku or "").strip()
        if not s:
            return 0.0
        return float(sum(r.quantity for r in self.stock if r.sku == s))

    def upsert_item(
        self,
        *,
        sku: str,
        name: str,
        unit: str = "unit",
        category: Optional[str] = None,
        spec: Optional[str] = None,
        safety_stock: Optional[float] = None,
    ) -> MaterialItem:
        s = (sku or "").strip()
        if not s:
            raise ValueError("sku is required")
        n = (name or "").strip()
        if not n:
            raise ValueError("name is required")
        unit = (unit or "unit").strip() or "unit"
        item = self._find_item_by_sku(s)
        if item is None:
            item = MaterialItem(
                sku=s,
                name=n,
                unit=unit,
                category=category,
                spec=spec,
                safety_stock=safety_stock,
            )
            self.items.append(item)
            return item
        item.name = n
        item.unit = unit
        item.category = category
        item.spec = spec
        item.safety_stock = safety_stock
        return item

    def set_stock(
        self,
        *,
        warehouse_id: str,
        sku: str,
        quantity: float,
        updated_at: Optional[str] = None,
    ) -> StockRecord:
        wh = (warehouse_id or "").strip()
        s = (sku or "").strip()
        if not wh:
            raise ValueError("warehouse_id is required")
        if not s:
            raise ValueError("sku is required")
        qty = max(0.0, _safe_float(quantity, 0.0))
        now = updated_at or _utc_ts()
        for r in self.stock:
            if r.warehouse_id == wh and r.sku == s:
                r.quantity = qty
                r.updated_at = now
                return r
        rec = StockRecord(warehouse_id=wh, sku=s, quantity=qty, updated_at=now)
        self.stock.append(rec)
        return rec

    def allocate(
        self,
        *,
        sku: str,
        quantity: float,
        warehouse_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        s = (sku or "").strip()
        if not s:
            raise ValueError("sku is required")
        need = max(0.0, _safe_float(quantity, 0.0))
        if need <= 0:
            return [], 0.0

        wh_filter = (warehouse_id or "").strip()
        candidates = [
            r for r in self.stock if r.sku == s and (not wh_filter or r.warehouse_id == wh_filter)
        ]
        candidates.sort(key=lambda r: (r.warehouse_id, r.updated_at or ""))

        allocations: List[Dict[str, Any]] = []
        allocated = 0.0
        for r in candidates:
            if allocated >= need:
                break
            avail = float(r.quantity)
            if avail <= 0:
                continue
            take = min(avail, need - allocated)
            r.quantity = float(max(0.0, avail - take))
            r.updated_at = _utc_ts()
            allocated += float(take)
            allocations.append({"warehouse_id": r.warehouse_id, "sku": r.sku, "quantity": float(take)})
        return allocations, float(allocated)
