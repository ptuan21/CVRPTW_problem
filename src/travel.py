"""Giao thông phụ thuộc thời điểm (Time-Dependent travel) — giờ cao điểm.

Thời gian di chuyển thực = thời gian thông thoáng × hệ số_tắc(giờ khởi hành).
Hồ sơ tắc có hai đỉnh (sáng & chiều) dạng Gauss. Vì tốc độ = quãng đường /
thời gian, giờ cao điểm làm tốc độ giảm -> mô hình phát thải PRP cho ra nhiên
liệu cao hơn. Đây là cách khép kín "tắc đường -> tốn nhiên liệu".

Lưu ý FIFO: hệ số áp theo GIỜ KHỞI HÀNH của từng chặng (xấp xỉ rời rạc của
TD-VRP, Ichoua-Gendreau-Potvin 2003). Đủ cho mục tiêu mô phỏng.
"""
from __future__ import annotations

import math

from config import TrafficConfig


class Congestion:
    """Hàm hệ số tắc đường theo thời điểm (giây kể từ đầu ca)."""

    def __init__(self, cfg: TrafficConfig):
        self.cfg = cfg

    def factor(self, depart_s: float) -> float:
        c = self.cfg
        if not c.enabled:
            return 1.0
        hour = c.shift_start_hour + depart_s / 3600.0
        bump = 0.0
        for peak in (c.morning_peak_hour, c.evening_peak_hour):
            z = (hour - peak) / c.peak_width_h
            bump += math.exp(-0.5 * z * z)
        bump = min(bump, 1.0)  # khi hai đỉnh gần nhau không cộng dồn quá mức
        return c.base_multiplier + (c.peak_multiplier - c.base_multiplier) * bump

    def __call__(self, depart_s: float) -> float:
        return self.factor(depart_s)


def flat_congestion(_depart_s: float) -> float:
    """Hệ số tắc cố định = 1.0 (tắt time-dependent)."""
    return 1.0
