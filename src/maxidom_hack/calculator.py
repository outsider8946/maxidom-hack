from math import ceil
from .utils.models import Room, CatalogProduct, LineItem, WorkEstimate, Estimate



class Calculator:
    @staticmethod
    def filter_for_room(products: list[CatalogProduct], room: Room) -> list[CatalogProduct]:
        """Отсекает материалы, не подходящие под влажность и назначение."""
        out = []
        for p in products:
            if room.humidity == "high" and p.category in ("wall_tile", "wall_paint"):
                if not p.moisture_resistant:
                    continue
            if room.purpose == "kitchen" and p.category == "wall_paint":
                if not p.washable:
                    continue
            out.append(p)
        return out

    @staticmethod
    def pick_best(products: list[CatalogProduct]) -> CatalogProduct | None:
        """Из подходящих выбирает один — самый дешёвый на м²/единицу.

        На MVP — цена, а не рейтинг. Позже можно подмешать recsys_rating.
        """
        if not products:
            return None
        def metric(p: CatalogProduct) -> float:
            if p.package_area and p.package_area > 0:
                return p.price / p.package_area
            if p.quantity_in_package and p.quantity_in_package > 0:
                return p.price / p.quantity_in_package
            return p.price
        return min(products, key=metric)


# ---------- формулы по работам ----------

    def calc_tiles(
        self,
        room: Room, products: list[CatalogProduct], *,
        surface: str,              # "floor" | "wall"
        waste: float = 0.10,       # 10% на подрезку, 0.15 для диагонали
        glue: CatalogProduct | None = None,
        grout: CatalogProduct | None = None,
    ) -> WorkEstimate:
        area = room.floor_area if surface == "floor" else room.wall_area
        category = "floor_tile" if surface == "floor" else "wall_tile"

        tile = self.pick_best([p for p in self.filter_for_room(products, room) if p.category == category])
        we = WorkEstimate(work=f"{surface}_tile", area=area)
        if not tile or not tile.package_area:
            return we

        qty = ceil(area * (1 + waste) / tile.package_area)
        we.items.append(LineItem(
            product_id=tile.id, name=tile.name, category=category,
            qty=qty, unit=tile.unit, unit_price=tile.price,
            total=qty * tile.price,
            reason=f"запас {int(waste*100)}% на подрезку",
        ))

        if glue and glue.consumption_per_m2 and glue.quantity_in_package:
            kg = area * glue.consumption_per_m2
            q = ceil(kg / glue.quantity_in_package)
            we.items.append(LineItem(
                product_id=glue.id, name=glue.name, category="tile_glue",
                qty=q, unit=glue.unit, unit_price=glue.price,
                total=q * glue.price,
                reason=f"норма {glue.consumption_per_m2} кг/м²",
            ))

        if grout and grout.consumption_per_m2 and grout.quantity_in_package:
            kg = area * grout.consumption_per_m2
            q = ceil(kg / grout.quantity_in_package)
            we.items.append(LineItem(
                product_id=grout.id, name=grout.name, category="tile_grout",
                qty=q, unit=grout.unit, unit_price=grout.price,
                total=q * grout.price,
                reason="норма на затирку швов",
            ))
        return we


    def calc_laminate(
        self,
        room: Room, products: list[CatalogProduct], *, waste: float = 0.05,
    ) -> WorkEstimate:
        area = room.floor_area
        lam = self.pick_best([p for p in self.filter_for_room(products, room) if p.category == "floor_laminate"])
        we = WorkEstimate(work="floor_laminate", area=area)
        if not lam or not lam.package_area:
            return we
        qty = ceil(area * (1 + waste) / lam.package_area)
        we.items.append(LineItem(
            product_id=lam.id, name=lam.name, category="floor_laminate",
            qty=qty, unit=lam.unit, unit_price=lam.price,
            total=qty * lam.price,
            reason=f"запас {int(waste*100)}%",
        ))
        return we


    def calc_paint(
        self,
        room: Room, products: list[CatalogProduct], *,
        surface: str = "wall",     # wall | ceiling
        layers: int = 2,
        coverage_per_liter: float = 10.0,   # м²/л на слой (акрил)
    ) -> WorkEstimate:
        area = room.wall_area if surface == "wall" else room.ceiling_area
        paint = self.pick_best([p for p in self.filter_for_room(products, room) if p.category == "wall_paint"])
        we = WorkEstimate(work=f"{surface}_paint", area=area)
        if not paint or not paint.quantity_in_package:
            return we
        liters = area * layers / coverage_per_liter
        qty = ceil(liters / paint.quantity_in_package)
        we.items.append(LineItem(
            product_id=paint.id, name=paint.name, category="wall_paint",
            qty=qty, unit=paint.unit, unit_price=paint.price,
            total=qty * paint.price,
            reason=f"{layers} слоя, {coverage_per_liter} м²/л",
        ))
        return we


    def calc_wallpaper(
        self,
        room: Room, products: list[CatalogProduct], *,
        roll_width: float = 0.53, roll_length: float = 10.05,
        pattern_repeat: float = 0.0,   # 0.64 для обоев с раппортом
    ) -> WorkEstimate:
        area = room.wall_area
        wp = self.pick_best([p for p in self.filter_for_room(products, room) if p.category == "wall_wallpaper"])
        we = WorkEstimate(work="wall_wallpaper", area=area)
        if not wp:
            return we

        strips_per_room = ceil(room.perimeter / roll_width)
        usable_h = room.height + pattern_repeat
        strips_per_roll = max(int(roll_length // usable_h), 1)
        qty = ceil(strips_per_room / strips_per_roll)

        we.items.append(LineItem(
            product_id=wp.id, name=wp.name, category="wall_wallpaper",
            qty=qty, unit=wp.unit, unit_price=wp.price,
            total=qty * wp.price,
            reason=f"{strips_per_roll} полос в рулоне, раппорт {pattern_repeat} м",
        ))
        return we


    def calc_skirting(
        self,
        room: Room, products: list[CatalogProduct], *, strip_length: float = 2.5,
    ) -> WorkEstimate:
        length = max(room.perimeter - room.doors * 0.9, 0.0)
        sk = self.pick_best([p for p in self.filter_for_room(products, room) if p.category == "skirting"])
        we = WorkEstimate(work="skirting", area=length)
        if not sk:
            return we
        qty = ceil(length / strip_length)
        we.items.append(LineItem(
            product_id=sk.id, name=sk.name, category="skirting",
            qty=qty, unit=sk.unit, unit_price=sk.price,
            total=qty * sk.price,
            reason=f"за вычетом дверных проёмов",
        ))
        return we


    def calc_electrics(
        self,
        room: Room, products: list[CatalogProduct],
    ) -> WorkEstimate:
        """Эвристика: розетки и выключатели на комнату, кабель на площадь."""
        area = room.floor_area
        sockets_needed = max(2, ceil(area / 6))
        switches_needed = 2 if room.doors > 1 else 1
        cable_length = max(10, area * 1.5)

        def add(cat: str, qty: float, reason: str):
            p = self.pick_best([p for p in products if p.category == cat])
            if not p:
                return
            items.append(LineItem(
                product_id=p.id, name=p.name, category=cat,
                qty=int(qty) if p.unit in ("шт.", "упак.") else ceil(qty),
                unit=p.unit, unit_price=p.price,
                total=ceil(qty) * p.price, reason=reason,
            ))

        items: list[LineItem] = []
        add("socket", sockets_needed, "1 розетка на 6 м²")
        add("switch", switches_needed, "по числу входов")
        add("cable", cable_length, "1.5 м кабеля на м²")

        return WorkEstimate(work="electrics", area=area, items=items)


    # ---------- сценарии ремонта ----------

    def calc_renovation(
        self,
        room: Room,
        products: list[CatalogProduct],
        variant: str = "capital",   # cosmetic | capital
    ) -> Estimate:
        est = Estimate(room=room)

        if variant == "cosmetic":
            est.works.append(self.calc_paint(room, products))
            est.works.append(self.calc_wallpaper(room, products))
            est.works.append(self.calc_laminate(room, products))
            est.works.append(self.calc_skirting(room, products))
            return est

        # capital
        if room.purpose in ("bathroom", "kitchen"):
            est.works.append(self.calc_tiles(room, products, surface="floor"))
            est.works.append(self.calc_tiles(room, products, surface="wall"))
        else:
            est.works.append(self.calc_laminate(room, products))
            est.works.append(self.calc_paint(room, products, surface="wall"))
            est.works.append(self.calc_paint(room, products, surface="ceiling"))

        est.works.append(self.calc_skirting(room, products))
        est.works.append(self.calc_electrics(room, products))
        return est