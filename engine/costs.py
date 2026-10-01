"""Transaction cost model for Indian equity trades (NSE/BSE).

Every rate is configurable; defaults mirror a typical Indian discount-broker
schedule. Rates are fractions (0.001 = 0.1%).

Costs are the difference between a fantasy backtest and a real one, so this
module is deliberately explicit: each leg (buy/sell) is charged separately.
"""
from dataclasses import dataclass


@dataclass
class CostModel:
    # Brokerage per order leg
    brokerage_flat: float = 20.0    # Rs per order (intraday)
    brokerage_pct: float = 0.0005   # 0.05% -- flat is capped at this
    delivery_brokerage: float = 0.0  # Rs 0 delivery at most discount brokers
    # Securities Transaction Tax
    stt_intraday_sell: float = 0.00025  # 0.025% on the sell leg
    stt_delivery: float = 0.001        # 0.1% on buy and sell legs
    # NSE transaction charges
    exchange_pct: float = 0.0000345
    # Stamp duty (buy leg only)
    stamp_intraday: float = 0.00003
    stamp_delivery: float = 0.00015
    # GST on brokerage + exchange charges
    gst: float = 0.18
    # SEBI charges: Rs 10 per crore
    sebi_pct: float = 0.000001
    # Slippage in basis points per side, applied adversely to the price
    slippage_bps: float = 5.0

    def leg(self, value: float, side: str, intraday: bool = True) -> float:
        """Cost in Rs for one leg of `value` rupees. side: 'buy' or 'sell'."""
        assert side in ("buy", "sell")
        if intraday:
            brokerage = min(self.brokerage_flat, self.brokerage_pct * value)
            stt = self.stt_intraday_sell * value if side == "sell" else 0.0
            stamp = self.stamp_intraday * value if side == "buy" else 0.0
        else:
            brokerage = self.delivery_brokerage
            stt = self.stt_delivery * value
            stamp = self.stamp_delivery * value if side == "buy" else 0.0
        exchange = self.exchange_pct * value
        gst = self.gst * (brokerage + exchange)
        sebi = self.sebi_pct * value
        return brokerage + stt + exchange + stamp + gst + sebi

    def round_trip(self, buy_value: float, sell_value: float,
                   intraday: bool = True) -> float:
        return (self.leg(buy_value, "buy", intraday)
                + self.leg(sell_value, "sell", intraday))
