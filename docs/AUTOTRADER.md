# Autotrader: trading autónomo con TradingAgents

`tradingagents autotrade` convierte las decisiones de los agentes (Buy, Overweight,
Hold, Underweight, Sell) en órdenes reales o simuladas, sin intervención humana.

> **Aviso importante.** Ningún sistema puede garantizar rentabilidad. Las
> calificaciones las produce un LLM y pueden equivocarse. Empieza siempre en
> modo `paper` (simulado) durante semanas, revisa el diario de operaciones y
> solo después considera dinero real, con un monto que puedas perder.

## Cómo funciona

1. **Cada día hábil a `analysis_time`** (15:30 hora de Nueva York por defecto) los
   agentes analizan cada ticker de la `watchlist`, viendo tu cartera actual.
2. **Dimensionamiento** (determinista, sin LLM): cada calificación tiene un peso
   objetivo (`target_weights`). La posición se limita a `max_position_pct` del
   capital, se reduce si el activo es más volátil que `target_annual_vol`, y la
   exposición total no supera `max_gross_exposure_pct`. Hold no opera; REVIEW
   (decisión ilegible) nunca opera.
3. **Salidas cada `stop_check_minutes`**: stop dinámico (trailing) a
   `stop_loss_atr` ATRs por debajo del máximo desde la entrada, y toma de
   ganancias opcional (`take_profit_atr`). No usa el LLM.
4. **Cortacircuitos**: si el día pierde más de `max_daily_loss_pct`, no hay más
   compras ese día. Si la caída desde el máximo supera `max_drawdown_pct`, las
   compras se detienen hasta que tú lo reinicies con `--reset-halt`.
5. **Auditoría**: todo queda en `state_dir/journal.jsonl` (decisiones, órdenes,
   motivos, errores).

## Uso

```bash
pip install .
cp autotrader.example.json mi_autotrader.json   # edita la watchlist y límites
# Configura la clave de tu LLM en .env (ver README), p.ej. OPENAI_API_KEY

tradingagents autotrade mi_autotrader.json --once   # un ciclo ahora, para probar
tradingagents autotrade mi_autotrader.json          # bucle autónomo (Ctrl+C para parar)
```

Para dejarlo corriendo solo en un servidor: `nohup tradingagents autotrade mi_autotrader.json &`,
un servicio `systemd`, o el `docker-compose.yml` del repositorio.

## Brokers

- `"broker": "paper"` (por defecto): libro simulado local con comisión y
  deslizamiento; precios de Yahoo Finance. No envía órdenes a ningún lado.
- `"broker": "alpaca"`: define `ALPACA_API_KEY` y `ALPACA_SECRET_KEY`. Usa la
  cuenta *paper* de Alpaca salvo que pongas `"live": true` **y** exportes
  `TRADINGAGENTS_ALLOW_LIVE=1`. Ambos son obligatorios para dinero real.

## Antes de arriesgar dinero

- Mide la calidad de las decisiones con `tradingagents backtest NVDA,AAPL --start ... --end ...`.
- Cada análisis llama al LLM varias veces por ticker: vigila el costo de la API.
- Una watchlist diversificada y límites prudentes importan más que cualquier señal.
