"""Ядро расчетов количества материалов и формирования сметы ремонта."""

from math import ceil

from .utils.coefficients import COEFFICIENTS
from .utils.models import CatalogProduct, Estimate, LineItem, Room, WorkEstimate


class Calculator:
    """Рассчитывает необходимое количество упаковок для товаров и помещений.

    Калькулятор не зависит от хранилища каталога и ML-сервисов. Он получает
    нормализованные объекты ``CatalogProduct`` и возвращает доменные модели.
    """

    def calc_by_product(
        self,
        product: CatalogProduct,
        area: float,
        waste: float = 0.05,
        layers: int | None = None,
    ) -> LineItem:
        """Рассчитывает количество упаковок ОДНОГО товара на заданную площадь.

        Логика делится на два типа:
        1. Жидкости и смеси (краска, клей, шпаклевка): Считаем по расходу на м².
        2. Штучные/площадные (плитка, ламинат, обои): Считаем по площади упаковки с запасом..
        """
        # Тип 1: Жидкости и сыпучие смеси
        if product.category in (
            "wall_paint",
            "tile_glue",
            "tile_grout",
            "primer",
            "putty",
        ):
            # Берем расход из карточки товара, если нет — из справочника
            consumption = product.consumption_per_m2 or COEFFICIENTS[
                "default_consumption"
            ].get(product.category, 0)
            package_quantity = product.quantity_in_package or 0

            # Для краски учитываем количество слоев, для остального — 1 слой
            actual_layers = (
                (layers or COEFFICIENTS["paint_layers"])
                if product.category == "wall_paint"
                else 1
            )

            # Формула: Общий_расход = Площадь * Расход_на_м2 * Слои
            needed = area * consumption * actual_layers

            # Формула: Упаковки = Общий_расход / Объем_упаковки (округляем вверх)
            qty = ceil(needed / package_quantity) if package_quantity else 0
            reason = f"Площадь {area:g} м2 x расход {consumption:g} x {actual_layers} сл. / упаковка {package_quantity:g} = {qty} шт. упак."

        # Тип 2: Площадные товары (плитка, ламинат, обои)
        else:
            package_area = product.package_area or 0
            # Формула: Площадь с запасом = Площадь * (1 + Коэффициент_запаса)
            covered = area * (1 + waste)

            # Формула: Упаковки = Площадь с запасом / Площадь_упаковки (округляем вверх)
            qty = ceil(covered / package_area) if package_area else 0
            reason = f"Площадь {area:g} м2 + запас {waste:.0%} ({area * waste:g} м2) / упаковка {package_area:g} м2 = {qty} шт. упак."
        return LineItem(
            product.id,
            product.name,
            product.category,
            qty,
            product.unit,
            product.price,
            qty * product.price,
            reason,
        )

    def calc_estimate(
        self, room: Room, variant: str, products: list[CatalogProduct]
    ) -> Estimate:
        """
        Формирует полную смету для помещения и варианта ремонта.
        Точка входа для общего калькулятора (Сценарий 1).
        """
        return self.calc_renovation(room, products, variant)

    @staticmethod
    def filter_for_room(
        products: list[CatalogProduct], room: Room
    ) -> list[CatalogProduct]:
        """
        Сценарий 2: Фильтрует каталог, отсекая материалы, не подходящие под комнату.

        - Если влажность высокая (санузел), оставляем только влагостойкую плитку и краску.
        - Если кухня, оставляем только моющуюся краску.
        """
        out = []
        for p in products:
            # Для влажных помещений отсекаем не влагостойкие материалы
            if room.humidity == "high" and p.category in ("wall_tile", "wall_paint"):
                if not p.moisture_resistant:
                    continue
            # Для кухни отсекаем немоющуюся краску
            if room.purpose == "kitchen" and p.category == "wall_paint":
                if not p.washable:
                    continue
            out.append(p)
        return out

    @staticmethod
    def pick_best(products: list[CatalogProduct]) -> CatalogProduct | None:
        """Выбирает самый дешевый товар из подходящих (по цене за м² или за упаковку).

        На MVP используем просто минимальную цену. В будущем можно заменить на
        алгоритм рекомендаций (recsys_rating).
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

    # ---------- формулы для отдельных работ ----------

    def calc_tiles(
        self,
        room: Room,
        products: list[CatalogProduct],
        *,
        surface: str,  # "floor" | "wall"
        waste: float | None = None,
        glue: CatalogProduct | None = None,
        grout: CatalogProduct | None = None,
    ) -> WorkEstimate:
        """Рассчитывает плитку для пола или стен, а также сопутствующий клей и затирку."""
        area = room.floor_area if surface == "floor" else room.wall_area
        # Берем запас из справочника: 5% для прямой укладки (по умолчанию)
        waste = COEFFICIENTS["tile_waste"]["straight"] if waste is None else waste
        category = "floor_tile" if surface == "floor" else "wall_tile"

        tile = self.pick_best(
            [p for p in self.filter_for_room(products, room) if p.category == category]
        )
        we = WorkEstimate(work=f"{surface}_tile", area=area)
        if not tile or not tile.package_area:
            return we

        # Плитка: Площадь * (1 + запас) / Площадь упаковки
        qty = ceil(area * (1 + waste) / tile.package_area)
        we.items.append(
            LineItem(
                product_id=tile.id,
                name=tile.name,
                category=category,
                qty=qty,
                unit=tile.unit,
                unit_price=tile.price,
                total=qty * tile.price,
                reason=f"Площадь {area:g} м2 + запас {waste:.0%} ({area * waste:g} м2) / упаковка {tile.package_area:g} м2 = {qty} шт. упак.",
            )
        )

        # Клей: Площадь * Норма_расхода(кг/м2) / Вес_мешка(кг)
        if glue and glue.consumption_per_m2 and glue.quantity_in_package:
            kg = area * glue.consumption_per_m2
            q = ceil(kg / glue.quantity_in_package)
            we.items.append(
                LineItem(
                    product_id=glue.id,
                    name=glue.name,
                    category="tile_glue",
                    qty=q,
                    unit=glue.unit,
                    unit_price=glue.price,
                    total=q * glue.price,
                    reason=f"Площадь {area:g} м² × норма {glue.consumption_per_m2} кг/м² ÷ {glue.quantity_in_package} кг (мешок) = {q} шт.",
                )
            )

        # Затирка: Площадь * Норма_расхода / Вес_упаковки
        if grout and grout.consumption_per_m2 and grout.quantity_in_package:
            kg = area * grout.consumption_per_m2
            q = ceil(kg / grout.quantity_in_package)
            we.items.append(
                LineItem(
                    product_id=grout.id,
                    name=grout.name,
                    category="tile_grout",
                    qty=q,
                    unit=grout.unit,
                    unit_price=grout.price,
                    total=q * grout.price,
                    reason=f"Площадь {area:g} м² × норма {grout.consumption_per_m2} кг/м² ÷ {grout.quantity_in_package} кг (упаковка) = {q} шт.",
                )
            )
        return we

    def calc_laminate(
        self,
        room: Room,
        products: list[CatalogProduct],
        *,
        waste: float = 0.05,
    ) -> WorkEstimate:
        """Рассчитывает количество упаковок ламината для пола помещения. (с запасом на подрезку)."""
        area = room.floor_area
        lam = self.pick_best(
            [
                p
                for p in self.filter_for_room(products, room)
                if p.category == "floor_laminate"
            ]
        )
        we = WorkEstimate(work="floor_laminate", area=area)
        if not lam or not lam.package_area:
            return we

        # Ламинат: Площадь * (1 + запас) / Площадь упаковки
        qty = ceil(area * (1 + waste) / lam.package_area)
        we.items.append(
            LineItem(
                product_id=lam.id,
                name=lam.name,
                category="floor_laminate",
                qty=qty,
                unit=lam.unit,
                unit_price=lam.price,
                total=qty * lam.price,
                reason=f"Площадь {area:g} м² + запас {waste:.0%} ÷ {lam.package_area:g} м² = {qty} упак.",
            )
        )
        return we

    def calc_paint(
        self,
        room: Room,
        products: list[CatalogProduct],
        *,
        surface: str = "wall",  # wall | ceiling
        layers: int | None = None,
        coverage_per_liter: float | None = None,
    ) -> WorkEstimate:
        """Рассчитывает количество упаковок краски для стен или потолка (с учетом количества слоев)."""
        area = room.wall_area if surface == "wall" else room.ceiling_area

        # Краска: (Площадь * Слои) / Расход(м2/л) = Литры
        layers = layers or COEFFICIENTS["paint_layers"]
        coverage_per_liter = (
            coverage_per_liter or COEFFICIENTS["default_consumption"]["wall_paint"]
        )
        paint = self.pick_best(
            [
                p
                for p in self.filter_for_room(products, room)
                if p.category == "wall_paint"
            ]
        )
        we = WorkEstimate(work=f"{surface}_paint", area=area)
        if not paint or not paint.quantity_in_package:
            return we
        liters = area * layers / coverage_per_liter

        # Упаковки: Литры / Объем_банки
        qty = ceil(liters / paint.quantity_in_package)
        we.items.append(
            LineItem(
                product_id=paint.id,
                name=paint.name,
                category="wall_paint",
                qty=qty,
                unit=paint.unit,
                unit_price=paint.price,
                total=qty * paint.price,
                reason=f"Площадь {area:g} м2 x {layers} слоя / расход {coverage_per_liter:g} м2/л = {qty} упак.",
            )
        )
        return we

    def calc_wallpaper(
        self,
        room: Room,
        products: list[CatalogProduct],
        *,
        roll_width: float = 0.53,
        roll_length: float = 10.05,
        pattern_repeat: float = 0.0,  # 0.64 для обоев с раппортом
    ) -> WorkEstimate:
        """Рассчитывает количество рулонов обоев по периметру, высоте и раппорту (повторяющемуся рисунку)."""
        area = room.wall_area
        wp = self.pick_best(
            [
                p
                for p in self.filter_for_room(products, room)
                if p.category == "wall_wallpaper"
            ]
        )
        we = WorkEstimate(work="wall_wallpaper", area=area)
        if not wp:
            return we

        # Сколько полос нужно на всю комнату: Периметр / Ширина рулона
        strips_per_room = ceil(room.perimeter / roll_width)

        # Полезная высота полосы: Высота стены + Раппорт (подгонка)
        usable_h = room.height + pattern_repeat

        # Сколько полос выходит из одного рулона: Длина рулона / Полезная высота
        strips_per_roll = max(int(roll_length // usable_h), 1)
        # Итог: Нужно полос / Полос в рулоне
        qty = ceil(strips_per_room / strips_per_roll)

        we.items.append(
            LineItem(
                product_id=wp.id,
                name=wp.name,
                category="wall_wallpaper",
                qty=qty,
                unit=wp.unit,
                unit_price=wp.price,
                total=qty * wp.price,
                reason=f"Периметр {room.perimeter:g} м ÷ {roll_width} м = {strips_per_room} полос. В рулоне {strips_per_roll} полос (с учетом раппорта {pattern_repeat} м). Итого: {qty} рул.",
            )
        )
        return we

    def calc_skirting(
        self,
        room: Room,
        products: list[CatalogProduct],
        *,
        strip_length: float = 2.5,
    ) -> WorkEstimate:
        """Рассчитывает количество плинтусов за вычетом дверных проемов (0.9 м на дверь)."""
        length = max(room.perimeter - room.doors * 0.9, 0.0)
        sk = self.pick_best(
            [
                p
                for p in self.filter_for_room(products, room)
                if p.category == "skirting"
            ]
        )
        we = WorkEstimate(work="skirting", area=length)
        if not sk:
            return we
        qty = ceil(length / strip_length)
        we.items.append(
            LineItem(
                product_id=sk.id,
                name=sk.name,
                category="skirting",
                qty=qty,
                unit=sk.unit,
                unit_price=sk.price,
                total=qty * sk.price,
                reason=f"Периметр {room.perimeter:g} м - {room.doors} двер(и) × 0.9 м = {length:g} м. Планок по {strip_length} м: {qty} шт.",
            )
        )
        return we

    def calc_electrics(
        self,
        room: Room,
        products: list[CatalogProduct],
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

            # ИСПРАВЛЕНО: всегда используем ceil для округления вверх
            # int(qty) обрезает дробь вниз (например, 2.9 станет 2), а нам нужно округление вверх (ceil).
            final_qty = ceil(qty)

            items.append(
                LineItem(
                    product_id=p.id,
                    name=p.name,
                    category=cat,
                    qty=final_qty,
                    unit=p.unit,
                    unit_price=p.price,
                    total=final_qty * p.price,
                    reason=reason,
                )
            )

        items: list[LineItem] = []
        add(
            "socket",
            sockets_needed,
            f"Эвристика: 1 розетка на 6 м² ({area} м² / 6 = {sockets_needed} шт.)",
        )
        add("switch", switches_needed, "Эвристика: по 1 выключателю на вход")
        add("cable", cable_length, "Эвристика: 1.5 м кабеля на 1 м² пола")

        return WorkEstimate(work="electrics", area=area, items=items)

    # ---------- сценарии ремонта ----------

    def calc_renovation(
        self,
        room: Room,
        products: list[CatalogProduct],
        variant: str = "capital",  # косметический | капитальный
    ) -> Estimate:
        """Собирает смету работ, комбинируя работы в зависимости от типа ремонта (для косметического или капитального ремонта)"""
        est = Estimate(room=room)

        if variant == "cosmetic":
            # Косметика: только покраска/обои, полы и плинтус
            est.works.append(self.calc_paint(room, products))
            est.works.append(self.calc_wallpaper(room, products))
            est.works.append(self.calc_laminate(room, products))
            est.works.append(self.calc_skirting(room, products))
            return est

        # Капитальный ремонт.
        if room.purpose in ("bathroom", "kitchen"):
            # В санузле и на кухне — плитка на пол и стены
            # Клей и затирка считаются вместе с плиткой: без явной передачи
            # calc_tiles не добавляет их в смету вовсе.
            glue = self.pick_best([p for p in products if p.category == "tile_glue"])
            grout = self.pick_best([p for p in products if p.category == "tile_grout"])
            est.works.append(
                self.calc_tiles(
                    room, products, surface="floor", glue=glue, grout=grout
                )
            )
            est.works.append(
                self.calc_tiles(
                    room, products, surface="wall", glue=glue, grout=grout
                )
            )
        else:
            # В жилых комнатах — ламинат и покраска стен/потолка
            est.works.append(self.calc_laminate(room, products))
            est.works.append(self.calc_paint(room, products, surface="wall"))
            est.works.append(self.calc_paint(room, products, surface="ceiling"))

        est.works.append(self.calc_skirting(room, products))
        est.works.append(self.calc_electrics(room, products))
        return est
