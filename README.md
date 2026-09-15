# macdscanner

## خادم MCP

`mcp_server.py` يعرض الماسحات الثلاثة (MACD Cascade، EMA Cross، SMC Breakout)
كأدوات MCP تُرجع النتائج مباشرة بلا تنبيهات تيليجرام ولا حفظ حالة.

```bash
pip install -r requirements.txt
python mcp_server.py                 # stdio — للاستخدام المحلي مع عميل MCP

MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8000 python mcp_server.py
```

الأدوات المتاحة: `macd_cascade_scan`، `ema_cross_scan`، `smc_breakout_scan` —
كل واحدة تقبل قائمة رموز اختيارية (`tickers`)، وإن تُركت فارغة تُستخدم قائمة
الماسح الافتراضية.