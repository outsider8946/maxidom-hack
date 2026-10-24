from dataclasses import dataclass, field

from pydantic import BaseModel

# ---------- модели ----------


@dataclass
class Room:
    length: float  # м
    width: float  # м
    height: float = 2.7  # м
    purpose: str = "living"  # bathroom | kitchen | living | hallway
    humidity: str = "medium"  # low | medium | high
    doors: int = 1
    windows: int = 1
    door_area: float = 1.8
    window_area: float = 1.5

    @property
    def floor_area(self) -> float:
        return self.length * self.width

    @property
    def perimeter(self) -> float:
        return 2 * (self.length + self.width)

    @property
    def wall_area_gross(self) -> float:
        return self.perimeter * self.height

    @property
    def openings_area(self) -> float:
        return self.doors * self.door_area + self.windows * self.window_area

    @property
    def wall_area(self) -> float:
        return max(self.wall_area_gross - self.openings_area, 0.0)

    @property
    def ceiling_area(self) -> float:
        return self.floor_area


@dataclass
class CatalogProduct:
    id: int
    name: str
    price: float  # result_price за упаковку
    unit: str  # "упак.", "шт.", "рул."
    package_area: float | None = None  # м² в упаковке (плитка, ламинат)
    quantity_in_package: float | None = None  # кг/л/шт в упаковке (клей, краска)
    consumption_per_m2: float | None = None  # норма расхода (кг/м², л/м²)
    moisture_resistant: bool = False
    washable: bool = False
    category: str = ""

    def price_per_package(self) -> float:
        return self.price


@dataclass
class LineItem:
    product_id: int
    name: str
    category: str
    qty: int  # число упаковок
    unit: str
    unit_price: float
    total: float
    reason: str = ""


@dataclass
class WorkEstimate:
    work: str  # floor_tile | wall_paint | ...
    area: float
    items: list[LineItem] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(i.total for i in self.items)


@dataclass
class Estimate:
    room: Room
    works: list[WorkEstimate] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(w.total for w in self.works)

    def to_dict(self) -> dict:
        return {
            "room": {
                "length": self.room.length,
                "width": self.room.width,
                "height": self.room.height,
                "purpose": self.room.purpose,
                "humidity": self.room.humidity,
                "floor_area": round(self.room.floor_area, 2),
                "wall_area": round(self.room.wall_area, 2),
                "perimeter": round(self.room.perimeter, 2),
            },
            "works": [
                {
                    "work": w.work,
                    "area": round(w.area, 2),
                    "total": round(w.total, 2),
                    "items": [
                        {
                            "product_id": i.product_id,
                            "name": i.name,
                            "category": i.category,
                            "qty": i.qty,
                            "unit": i.unit,
                            "unit_price": round(i.unit_price, 2),
                            "total": round(i.total, 2),
                            "reason": i.reason,
                        }
                        for i in w.items
                    ],
                }
                for w in self.works
            ],
            "total": round(self.total, 2),
        }


class EstimateRequest(BaseModel):
    """Тело запроса для полной сметы помещения."""

    room: Room
    variant: str = "capital"


class ProductRequest(BaseModel):
    """Тело запроса для расчета одного товара из каталога."""

    product_id: int
    area: float
    waste: float = 0.05
    layers: int | None = None
